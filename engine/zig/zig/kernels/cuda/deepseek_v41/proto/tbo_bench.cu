// Two-batch overlap (TBO) probe on the real GB10 (DGX Spark: sm_121, 48 SMs, ~240 GB/s): one layer's x3gm routed
// experts (rotation + gateup2 + down2, DRAM-bound) and one prefill layer's dense EXL3 GEMMs (pfdense, lanes form,
// compute-bound), serial on one stream vs concurrent on two. One rank of TP 2: 384 experts top-6, D 5,120, I 1,152.
//
// Modes (cudaEvent timing, median of --reps after one warm-up rep, a 64 MB memset before every rep for a cold-ish L2):
//   serial      x3gm (grid = every SM x per_sm) then the dense set, one stream
//   x3gm  S     x3gm alone with its persistent grid capped at S x per_sm CTAs (the rotation is not capped)
//   dense       the dense set alone (pfdense's grid: row tiles x 128-column blocks)
//   streams S   x3gm capped at S on a low-priority stream, the dense set on a high-priority stream, launched back to
//               back; wall = first launch to both done (the CTA scheduler places CTAs: the cap bounds x3gm's CTAs,
//               it does not pin SMs)
//   green S     (--green) green contexts: x3gm on S SMs (cuDevSmResourceSplitByCount), the dense set on the rest;
//               host wall clock (events do not cross contexts), compared with the serial run timed the same way
// Bits: gm2's items are claimed by ticket, so Y (fp32) must not depend on the grid: every capped / concurrent run's Y
// is compared with the uncapped run's, word for word.
//
//   tbo_bench [--rows 2048,4096] [--k2 6] [--reps 5] [--green] [--tiles] [--sms 48,44,40,36,32,28,24] [--zipf 0]
//   --tiles: x3gm v2 alone on every SM at x3gm.cu's 128-member tiles (GU2-4 / DN2-4, one pass an expert at ~64
//   members) against G6's GU1 / DN0, Y bit for bit; "TILES R=..." lines
#define DSV41_X3GM_NO_INSTANCES
#include "../x3gm.cu"
#include "../pfdense.cuh"

#include <cuda.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); exit(1); } } while (0)

namespace {

constexpr int E = 384, SLOTS = 6, D = 5120, I = 1152;

// -- fills (x3gm_bench's) ------------------------------------------------------------------------------------------
__global__ void fill_words(uint32_t* p, size_t n, uint32_t seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 0x9E3779B1u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        p[i] = x;
    }
}
__global__ void fill_half(half* p, size_t n, uint32_t seed, float lo, float hi) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 0x9E3779B1u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
        p[i] = __float2half(lo + (hi - lo) * (x & 0xffffff) / 16777216.f);
    }
}
// first index where a and b differ (bit patterns), atomicMin into *first (preset to n)
__global__ void first_diff(const uint32_t* a, const uint32_t* b, long long n, unsigned long long* first) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        if (a[i] != b[i]) atomicMin(first, (unsigned long long)i);
}

// -- x3gm: the v2 kernels prod launches (gateup2 with two rotated inputs at GM_GU1, down2 at GM_DN0) -------------
using GmKern = void (*)(const dsv41_x3gm::Args);
struct GmLaunch { GmKern k; int threads; int smem; int per_sm; };

template <int MATS, int XM, int MTL, int NG, int KS, int NSA, int NGR>
GmLaunch gm_make() {
    using S2 = dsv41_x3gm::Smem2<MATS, XM, MTL, NG, KS, NSA, NGR>;   // kernels_exl3.zig gm2Shape: widest width's smem
    GmKern k = dsv41_x3gm::gm2_kernel<MATS, XM, MTL, NG, KS, NSA, NGR>;
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)S2::SMEM));
    CK(cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    int per = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k, S2::THREADS, S2::SMEM));
    if (per < 1) { fprintf(stderr, "gm2 configuration does not fit an SM\n"); exit(2); }
    return {k, S2::THREADS, (int)S2::SMEM, per};
}

// --tiles: gm2_kernel at x3gm.cu's 128-member tiles (NG 16: BM 128) beside G6's (NG 8: BM 64). Tiles move data
// only (x3gm.cu's header): the same Y bit for bit, which the run checks. per_sm 0: does not fit an SM (skipped)
template <int MATS, int XM, int MTL, int NG, int KS, int NSA, int NGR>
GmLaunch gm_try() {
    using S2 = dsv41_x3gm::Smem2<MATS, XM, MTL, NG, KS, NSA, NGR>;
    GmKern k = dsv41_x3gm::gm2_kernel<MATS, XM, MTL, NG, KS, NSA, NGR>;
    if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)S2::SMEM) != cudaSuccess) {
        cudaGetLastError();
        return {k, S2::THREADS, (int)S2::SMEM, 0};
    }
    CK(cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    int per = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k, S2::THREADS, S2::SMEM) != cudaSuccess) { cudaGetLastError(); per = 0; }
    return {k, S2::THREADS, (int)S2::SMEM, per};
}
struct TilePair { const char* name; GmLaunch gu, dn; int bm_gu, bm_dn; };

struct Plan { std::vector<int> pick, order, pe, poff, pcnt; int npass = 0; };

// routing (x3gm_bench's): each row picks 6 distinct experts; zipf > 0 skews popularity (rank^-zipf). Passes of <= bm
// members an expert, the integers x3gm.plan makes (order stable by expert)
Plan route(int rows, double zipf, int bm, uint32_t seed) {
    Plan p; std::mt19937 rng(seed);
    std::vector<double> w(E);
    for (int e = 0; e < E; ++e) w[e] = zipf > 0 ? std::pow(e + 1.0, -zipf) : 1.0;
    std::shuffle(w.begin(), w.end(), rng);
    std::discrete_distribution<int> pickd(w.begin(), w.end());
    p.pick.resize((size_t)rows * SLOTS);
    for (int r = 0; r < rows; ++r) {
        int got = 0;
        while (got < SLOTS) {
            const int e = pickd(rng);
            bool dup = false;
            for (int j = 0; j < got; ++j) dup |= p.pick[(size_t)r * SLOTS + j] == e;
            if (!dup) p.pick[(size_t)r * SLOTS + got++] = e;
        }
    }
    const int P = rows * SLOTS;
    p.order.resize(P); std::iota(p.order.begin(), p.order.end(), 0);
    std::stable_sort(p.order.begin(), p.order.end(), [&](int a, int b) { return p.pick[a] < p.pick[b]; });
    std::vector<int> cnt(E, 0);
    for (int x : p.pick) ++cnt[x];
    int off = 0;
    for (int e = 0; e < E; ++e) {
        for (int j = 0; j < cnt[e]; j += bm) {
            p.pe.push_back(e); p.poff.push_back(off + j); p.pcnt.push_back(std::min(bm, cnt[e] - j));
        }
        off += cnt[e];
    }
    p.npass = (int)p.pe.size();
    return p;
}

// -- dense: pfdense lanes at pfdense.heuristic's choice (kernels_ops.zig pfd.heuristic, 48 SMs, group 8) ----------
using PfdKern = void (*)(const dsv41_pfd::Args);
struct PfdLaunch { PfdKern k; int threads; int smem; int bm; int ks; };

template <int K2, class C>
PfdLaunch pfd_make() {
    using L = dsv41_pfd::Lay<K2, true, C>;
    PfdKern k = dsv41_pfd::pfd_kernel<K2, true, C>;
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, L::SMEM));
    return {k, C::THREADS, L::SMEM, C::BM, C::KS};
}
// the heuristic's candidates only (cfg 0, 3, 6 of DSV41_PFD_CFGS); lanes are built for K2 8, 10, 12
template <int K2>
PfdLaunch pfd_cfg(int cfg) {
    switch (cfg) {
        case 0: return pfd_make<K2, dsv41_pfd::Cfg<128, 2, 4, 2, 3>>();
        case 3: return pfd_make<K2, dsv41_pfd::Cfg<64, 2, 2, 2, 4>>();
        default: return pfd_make<K2, dsv41_pfd::Cfg<32, 1, 4, 2, 4>>();
    }
}
PfdLaunch pfd_get(int k2, int cfg) {
    if (k2 == 8) return pfd_cfg<8>(cfg);
    if (k2 == 10) return pfd_cfg<10>(cfg);
    return pfd_cfg<12>(cfg);
}
int pfd_smem(int cfg, int k2) {                       // kernels_ops.zig pfd.smem
    static const int c[8][5] = {{128, 2, 4, 2, 3}, {128, 2, 4, 2, 4}, {128, 4, 2, 2, 3}, {64, 2, 2, 2, 4},
                                {64, 1, 4, 2, 4},  {128, 2, 4, 2, 2}, {32, 1, 4, 2, 4},  {256, 4, 2, 2, 3}};
    const int bm = c[cfg][0], wm = c[cfg][1], ks = c[cfg][3], nst = c[cfg][4];
    const int ring = nst * (bm * 16 * ks * 2 + ks * 8 * 4 * k2 * 4), main = ring + 2 * (ks * 8 * 32 * 16);
    return std::max(main, (bm / wm) * 128 * 2 * 2);
}
int pfd_heuristic(int k, int n, int k2, int m, int sms) {
    static const int bms[3] = {128, 64, 32}, ids[3] = {0, 3, 6};
    int best = -1; double best_eff = -1;
    for (int i = 0; i < 3; ++i) {
        if (k % 32) continue;                              // KS 2 for all three
        const int blocks = 2 * (pfd_smem(ids[i], k2) + 1024) <= 102400 ? 2 : 1;
        const long ctas = (long)((m + bms[i] - 1) / bms[i]) * (n / 128), slots = (long)sms * blocks;
        const double eff = (double)ctas / (double)(((ctas + slots - 1) / slots) * slots);
        if (eff > best_eff + 0.10) { best = ids[i]; best_eff = eff; }
    }
    return best < 0 ? 0 : best;
}

struct Proj { const char* name; int K, N, k2, count; };
// one layer's dense projections (rank of TP 2); the shared expert's are timed in lanes form too (a timing proxy)
const Proj PROJS[] = {
    {"wq_a", 5120, 1280, 10, 1}, {"wkv", 5120, 512, 10, 1},     {"wq_b", 1280, 16384, 10, 1},
    {"ix_wq_b", 1280, 2048, 10, 1}, {"wo_a", 4096, 1024, 10, 4}, {"wo_b", 4096, 5120, 10, 1},
    {"sh_w1", 5120, 1152, 12, 1}, {"sh_w3", 5120, 1152, 12, 1},  {"sh_w2", 1152, 5120, 12, 1},
};

struct DenseCall { PfdLaunch l; dsv41_pfd::Args a; int grid; };

template <typename T> T* dev(const std::vector<T>& v) {
    T* d; CK(cudaMalloc(&d, v.size() * sizeof(T) + 16));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
std::vector<int> parse_list(const char* s) {
    std::vector<int> v; std::string t(s); size_t p = 0;
    while (p <= t.size()) {
        size_t q = t.find(',', p); if (q == std::string::npos) q = t.size();
        if (q > p) v.push_back(atoi(t.substr(p, q - p).c_str()));
        p = q + 1;
    }
    return v;
}

// -- green contexts (CUDA >= 12.4) ----------------------------------------------------------------------------------
struct Green { bool ok = false; int sms_x = 0, sms_d = 0; cudaStream_t sx = nullptr, sd = nullptr; std::string why; };
#define CU_TRY(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* m_ = nullptr; cuGetErrorString(r_, &m_); \
    g.why = std::string(#x).substr(0, 40) + ": " + (m_ ? m_ : "?"); return g; } } while (0)
Green green_make(int want, int prio_lo, int prio_hi) {
    Green g;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12040
    CUdevice dev; CU_TRY(cuDeviceGet(&dev, 0));
    CUdevResource all{}, part[1]{}, rest{};
    CU_TRY(cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM));
    unsigned n = 1;
    CU_TRY(cuDevSmResourceSplitByCount(part, &n, &all, &rest, 0, (unsigned)want));
    if (n < 1) { g.why = "split gave no group"; return g; }
    if (rest.sm.smCount == 0) { g.why = "no SMs left for the dense set"; return g; }
    CUdevResourceDesc dx, dd;
    CU_TRY(cuDevResourceGenerateDesc(&dx, &part[0], 1));
    CU_TRY(cuDevResourceGenerateDesc(&dd, &rest, 1));
    CUgreenCtx gx, gd;
    CU_TRY(cuGreenCtxCreate(&gx, dx, dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CU_TRY(cuGreenCtxCreate(&gd, dd, dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CUstream sx, sd;
    CU_TRY(cuGreenCtxStreamCreate(&sx, gx, CU_STREAM_NON_BLOCKING, prio_lo));
    CU_TRY(cuGreenCtxStreamCreate(&sd, gd, CU_STREAM_NON_BLOCKING, prio_hi));
    g.ok = true; g.sms_x = (int)part[0].sm.smCount; g.sms_d = (int)rest.sm.smCount;
    g.sx = (cudaStream_t)sx; g.sd = (cudaStream_t)sd;
    // contexts / streams are kept for the run (one set a split; a few KB each)
#else
    (void)want; (void)prio_lo; (void)prio_hi;
    g.why = "built against CUDA < 12.4";
#endif
    return g;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> rows_list{2048, 4096}, caps{48, 44, 40, 36, 32, 28, 24};
    int k2 = 6, reps = 5;
    bool green = false, tiles = false;
    double zipf = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has = i + 1 < argc;
        if (a == "--green") green = true;
        else if (a == "--tiles") tiles = true;
        else if (a == "--rows" && has) rows_list = parse_list(argv[++i]);
        else if (a == "--sms" && has) caps = parse_list(argv[++i]);
        else if (a == "--k2" && has) k2 = atoi(argv[++i]);
        else if (a == "--reps" && has) reps = atoi(argv[++i]);
        else if (a == "--zipf" && has) zipf = atof(argv[++i]);
        else { fprintf(stderr, "usage: tbo_bench [--rows 2048,4096] [--k2 6] [--reps 5] [--green] [--sms 48,...] [--zipf 0]\n"); return 2; }
    }
    if (k2 != 3 && k2 != 4 && k2 != 5 && k2 != 6 && k2 != 7 && k2 != 8 && k2 != 10) {
        fprintf(stderr, "--k2 %d: gm2_kernel dispatches K2 3-8, 10\n", k2); return 2;
    }
    if (reps < 1) reps = 1;
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);           // no lazy load inside a timed or concurrent launch
    CK(cudaSetDevice(0));
    CK(cudaFree(nullptr));
    cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
    const int sms = prop.multiProcessorCount;
    for (int& s : caps) s = std::min(std::max(s, 1), sms);
    int prio_lo = 0, prio_hi = 0;                         // least (numerically largest), greatest
    CK(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
    cudaStream_t st, sx, sd;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithPriority(&sx, cudaStreamNonBlocking, prio_lo));
    CK(cudaStreamCreateWithPriority(&sd, cudaStreamNonBlocking, prio_hi));

    const GmLaunch GU = gm_make<2, 2, GM_GU1>();          // gateup2, two rotated inputs (x3gm.tuned2: shx false -> 1)
    const GmLaunch DN = gm_make<1, 1, GM_DN0>();          // down2
    std::vector<TilePair> tilepairs;
    if (tiles) {
        tilepairs = {
            {"GU1/DN0 (G6, prod)", GU, DN, 64, 64},
            {"GU2/DN0", gm_try<2, 2, GM_GU2>(), DN, 128, 64},
            {"GU3/DN0", gm_try<2, 2, GM_GU3>(), DN, 128, 64},
            {"GU4/DN0", gm_try<2, 2, GM_GU4>(), DN, 128, 64},
            {"GU1/DN2", GU, gm_try<1, 1, GM_DN2>(), 64, 128},
            {"GU1/DN3", GU, gm_try<1, 1, GM_DN3>(), 64, 128},
            {"GU1/DN4", GU, gm_try<1, 1, GM_DN4>(), 64, 128},
            {"GU2/DN2", gm_try<2, 2, GM_GU2>(), gm_try<1, 1, GM_DN2>(), 128, 128},
            {"GU4/DN4", gm_try<2, 2, GM_GU4>(), gm_try<1, 1, GM_DN4>(), 128, 128},
        };
    }
    constexpr int BM = 8 * 8;                             // NG 8 for GU1 and DN0: 64-member passes
    const int RMAX = *std::max_element(rows_list.begin(), rows_list.end()), PMAX = RMAX * SLOTS;

    // -- allocations, once at the largest row count --------------------------------------------------------------
    const size_t mw = (size_t)E * (D / 16) * (I / 16) * 4 * k2;          // words of one stacked expert matrix
    uint32_t *tg, *tu, *td;
    CK(cudaMalloc(&tg, mw * 4)); CK(cudaMalloc(&tu, mw * 4)); CK(cudaMalloc(&td, mw * 4));
    fill_words<<<1024, 256>>>(tg, mw, 1); fill_words<<<1024, 256>>>(tu, mw, 2); fill_words<<<1024, 256>>>(td, mw, 3);
    half *x, *suh_g, *suh_u, *svg, *svu, *sud, *svd, *xg, *xu, *xd;
    float *y, *yref;
    CK(cudaMalloc(&x, (size_t)RMAX * D * 2));
    CK(cudaMalloc(&suh_g, (size_t)E * D * 2)); CK(cudaMalloc(&suh_u, (size_t)E * D * 2));
    CK(cudaMalloc(&svg, (size_t)E * I * 2)); CK(cudaMalloc(&svu, (size_t)E * I * 2));
    CK(cudaMalloc(&sud, (size_t)E * I * 2)); CK(cudaMalloc(&svd, (size_t)E * D * 2));
    CK(cudaMalloc(&xg, (size_t)PMAX * D * 2)); CK(cudaMalloc(&xu, (size_t)PMAX * D * 2));
    CK(cudaMalloc(&xd, (size_t)PMAX * I * 2));
    CK(cudaMalloc(&y, (size_t)PMAX * D * 4)); CK(cudaMalloc(&yref, (size_t)PMAX * D * 4));
    fill_half<<<1024, 256>>>(x, (size_t)RMAX * D, 4, -1.f, 1.f);
    fill_half<<<1024, 256>>>(suh_g, (size_t)E * D, 5, -1.f, 1.f);
    fill_half<<<1024, 256>>>(suh_u, (size_t)E * D, 15, -1.f, 1.f);
    fill_half<<<1024, 256>>>(svg, (size_t)E * I, 6, 0.5f, 1.5f);
    fill_half<<<1024, 256>>>(svu, (size_t)E * I, 16, 0.5f, 1.5f);
    fill_half<<<1024, 256>>>(sud, (size_t)E * I, 26, 0.5f, 1.5f);
    fill_half<<<1024, 256>>>(svd, (size_t)E * D, 7, 0.5f, 1.5f);
    int* k2e = dev(std::vector<int>(E, k2));
    int* ticket; CK(cudaMalloc(&ticket, 8));
    unsigned long long* diff; CK(cudaMalloc(&diff, 8));
    void* flush; const size_t FLUSH = 64ull << 20; CK(cudaMalloc(&flush, FLUSH));

    // dense: the heuristic's configuration per (shape, rows); inputs shared (xh [R, <= 5120], one output buffer),
    // every projection its own words (the weights stream from DRAM as in a layer)
    std::vector<uint32_t*> dense_T; std::vector<const Proj*> dense_p;
    size_t out_max = 0;
    for (const Proj& p : PROJS) {
        const bool ok = (p.k2 == 8 || p.k2 == 10 || p.k2 == 12) && p.K % 128 == 0 && p.N % 128 == 0 && p.K % 32 == 0;
        if (!ok) { printf("dense: skip %s [%d -> %d] k2 %d (not pfdense-lanes eligible)\n", p.name, p.K, p.N, p.k2); continue; }
        for (int c = 0; c < p.count; ++c) {
            const size_t words = (size_t)(p.K / 16) * (p.N / 16) * 4 * p.k2;
            uint32_t* T; CK(cudaMalloc(&T, words * 4));
            fill_words<<<1024, 256>>>(T, words, 100 + (uint32_t)dense_T.size());
            dense_T.push_back(T); dense_p.push_back(&p);
        }
        out_max = std::max(out_max, (size_t)RMAX * p.N);
    }
    half *dxh, *dsv, *dout;
    CK(cudaMalloc(&dxh, (size_t)RMAX * 5120 * 2)); CK(cudaMalloc(&dsv, 16384 * 2)); CK(cudaMalloc(&dout, out_max * 2));
    fill_half<<<1024, 256>>>(dxh, (size_t)RMAX * 5120, 8, -1.f, 1.f);
    fill_half<<<64, 256>>>(dsv, 16384, 9, 0.5f, 1.5f);
    CK(cudaDeviceSynchronize());

    printf("device %s: %d SMs | x3gm k2 %d gateup2 %d thr %d B smem %d/SM, down2 %d thr %d B %d/SM | reps %d\n",
           prop.name, sms, k2, GU.threads, GU.smem, GU.per_sm, DN.threads, DN.smem, DN.per_sm, reps);

    cudaEvent_t ev[4];
    for (auto& e : ev) CK(cudaEventCreate(&e));
    bool bits_ok = true; std::string bits_msg;
    std::vector<std::string> best_lines;
    std::vector<Green> greens(caps.size());
    if (green) {
        int made = 0;
        for (size_t i = 0; i < caps.size(); ++i) {
            greens[i] = green_make(caps[i], prio_lo, prio_hi);
            if (greens[i].ok) ++made;
            else printf("green S=%d: unavailable (%s)\n", caps[i], greens[i].why.c_str());
        }
        if (!made) { printf("green: unavailable\n"); green = false; }
    }

    for (const int R : rows_list) {
        const int P = R * SLOTS;
        const Plan pl = route(R, zipf, BM, 1234 + R);
        int *pick = dev(pl.pick), *order = dev(pl.order), *pe = dev(pl.pe), *poff = dev(pl.poff), *pcnt = dev(pl.pcnt);
        int* np = dev(std::vector<int>{pl.npass});

        dsv41_x3gm::Args ag{};
        ag.x0 = xg; ag.x1 = xu; ag.t0 = tg; ag.t1 = tu; ag.k2e = k2e; ag.order = order; ag.pe = pe; ag.poff = poff;
        ag.pcnt = pcnt; ag.npass = np; ag.ticket = ticket; ag.sv0 = svg; ag.sv1 = svu; ag.sd = sud; ag.xd = xd;
        ag.K = D; ag.N = I; ag.limit = 10.f;
        dsv41_x3gm::Args ad{};
        ad.x0 = xd; ad.x1 = xd; ad.t0 = td; ad.t1 = td; ad.k2e = k2e; ad.order = order; ad.pe = pe; ad.poff = poff;
        ad.pcnt = pcnt; ad.npass = np; ad.ticket = ticket + 1; ad.sv0 = svd; ad.y = y; ad.K = I; ad.N = D;
        const int rot_items = P * (D / 128);

        // the x3gm layer on stream s with gm2 grids of cap SMs (tickets zeroed first, on the same stream)
        auto x3gm = [&](cudaStream_t s, int cap) {
            CK(cudaMemsetAsync(ticket, 0, 8, s));
            dsv41_x3gm::rot_kernel<half, 2><<<(rot_items + 7) / 8, 256, 0, s>>>(x, D, pick, suh_g, suh_u, xg, xu, D,
                                                                               SLOTS, rot_items);
            GU.k<<<cap * GU.per_sm, GU.threads, GU.smem, s>>>(ag);
            DN.k<<<cap * DN.per_sm, DN.threads, DN.smem, s>>>(ad);
        };
        std::vector<DenseCall> calls;
        for (size_t i = 0; i < dense_T.size(); ++i) {
            const Proj& p = *dense_p[i];
            const int cfg = pfd_heuristic(p.K, p.N, p.k2, R, 48);
            DenseCall c;
            c.l = pfd_get(p.k2, cfg);
            const long long tw = 4LL * p.k2;
            c.a = {};
            c.a.xh = dxh; c.a.T = dense_T[i]; c.a.stride_k = 8 * tw; c.a.stride_nb = (long long)(p.K / 16) * 8 * tw;
            c.a.svh = dsv; c.a.bias = nullptr; c.a.out = dout; c.a.o_stride = p.N;
            c.a.M = R; c.a.K = p.K; c.a.N = p.N; c.a.group = 8; c.a.out_type = dsv41_pfd::OUT_F16;
            c.grid = ((R + c.l.bm - 1) / c.l.bm) * (p.N / 128);
            calls.push_back(c);
        }
        auto dense = [&](cudaStream_t s) {
            for (const DenseCall& c : calls) c.l.k<<<c.grid, c.l.threads, c.l.smem, s>>>(c.a);
        };
        {
            printf("\nR=%d: P %d pairs, %d passes; dense:", R, P, pl.npass);
            for (size_t i = 0; i < calls.size(); ++i)
                if (i == 0 || dense_p[i] != dense_p[i - 1])
                    printf(" %s[cfg bm %d, %d CTAs]", dense_p[i]->name, calls[i].l.bm, calls[i].grid);
            printf("\n");
        }
        auto flush_on = [&](cudaStream_t s, int r) { CK(cudaMemsetAsync(flush, r & 0xff, FLUSH, s)); };
        auto check_bits = [&](const char* what, int cap) {
            CK(cudaDeviceSynchronize());
            const unsigned long long n = (unsigned long long)P * D;
            CK(cudaMemcpy(diff, &n, 8, cudaMemcpyHostToDevice));
            first_diff<<<1024, 256>>>((const uint32_t*)y, (const uint32_t*)yref, (long long)n, diff);
            unsigned long long got = n;
            CK(cudaMemcpy(&got, diff, 8, cudaMemcpyDeviceToHost));
            if (got != n && bits_ok) {
                bits_ok = false;
                char b[200];
                snprintf(b, sizeof b, "R=%d %s cap %d: first differing Y index %llu (pair %llu col %llu)", R, what, cap,
                         got, got / D, got % D);
                bits_msg = b;
            }
        };

        // one timed rep of a single-stream body: flush, start, body, end
        auto time1 = [&](cudaStream_t s, int r, auto&& body) {
            flush_on(s, r);
            CK(cudaEventRecord(ev[0], s));
            body(s);
            CK(cudaEventRecord(ev[1], s));
            CK(cudaEventSynchronize(ev[1]));
            CK(cudaGetLastError());
            float ms; CK(cudaEventElapsedTime(&ms, ev[0], ev[1]));
            return (double)ms;
        };
        auto med1 = [&](auto&& body) {
            std::vector<double> t;
            for (int r = 0; r <= reps; ++r) { const double v = time1(st, r, body); if (r) t.push_back(v); }
            return median(t);
        };

        // reference Y: uncapped x3gm
        x3gm(st, sms);
        CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
        CK(cudaMemcpy(yref, y, (size_t)P * D * 4, cudaMemcpyDeviceToDevice));

        const double t_serial = med1([&](cudaStream_t s) { x3gm(s, sms); dense(s); });
        const double t_dense = med1([&](cudaStream_t s) { dense(s); });
        std::vector<double> t_x(caps.size()), t_c(caps.size()), t_cx(caps.size()), t_cd(caps.size()), t_g(caps.size(), -1);
        for (size_t i = 0; i < caps.size(); ++i) {
            t_x[i] = med1([&](cudaStream_t s) { x3gm(s, caps[i]); });
            check_bits("alone", caps[i]);
        }
        // concurrent on two priority streams: start on sx, sd waits for it; wall = start to the later end
        for (size_t i = 0; i < caps.size(); ++i) {
            std::vector<double> w, ex, ed;
            for (int r = 0; r <= reps; ++r) {
                flush_on(sx, r);
                CK(cudaEventRecord(ev[0], sx));
                CK(cudaStreamWaitEvent(sd, ev[0], 0));
                x3gm(sx, caps[i]);
                dense(sd);
                CK(cudaEventRecord(ev[1], sx));
                CK(cudaEventRecord(ev[2], sd));
                CK(cudaEventSynchronize(ev[1])); CK(cudaEventSynchronize(ev[2]));
                CK(cudaGetLastError());
                float a, b; CK(cudaEventElapsedTime(&a, ev[0], ev[1])); CK(cudaEventElapsedTime(&b, ev[0], ev[2]));
                if (r) { w.push_back(std::max(a, b)); ex.push_back(a); ed.push_back(b); }
            }
            t_c[i] = median(w); t_cx[i] = median(ex); t_cd[i] = median(ed);
            check_bits("streams", caps[i]);
        }
        // green contexts: host wall clock (both streams launched, both synced), serial timed the same way
        double t_serial_host = -1;
        if (green) {
            using clk = std::chrono::steady_clock;
            std::vector<double> t;
            for (int r = 0; r <= reps; ++r) {
                flush_on(st, r); CK(cudaStreamSynchronize(st));
                const auto t0 = clk::now();
                x3gm(st, sms); dense(st);
                CK(cudaStreamSynchronize(st));
                if (r) t.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
            }
            t_serial_host = median(t);
            for (size_t i = 0; i < caps.size(); ++i) {
                const Green& g = greens[i];
                if (!g.ok) continue;
                std::vector<double> w;
                for (int r = 0; r <= reps; ++r) {
                    flush_on(st, r); CK(cudaStreamSynchronize(st));
                    const auto t0 = clk::now();
                    x3gm(g.sx, g.sms_x);                  // persistent grid = the partition's SMs x per_sm
                    dense(g.sd);
                    const cudaError_t le = cudaGetLastError(); // runtime launches into a green stream: a launch
                    if (le != cudaSuccess) {                   // error (not sticky) skips the split, not the run
                        printf("green S=%d: launch failed (%s)\n", g.sms_x, cudaGetErrorString(le));
                        CK(cudaDeviceSynchronize());
                        break;
                    }
                    CK(cudaStreamSynchronize(g.sx)); CK(cudaStreamSynchronize(g.sd));
                    if (r) w.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
                }
                if ((int)w.size() < reps) { greens[i].ok = false; continue; }
                t_g[i] = median(w);
                check_bits("green", caps[i]);
            }
        }

        printf("%-14s | %9s | %9s | %9s | %13s | %9s | %9s | %8s\n", "mode/split", "x3gm ms", "dense ms", "serial ms",
               "concurrent ms", "x3gm end", "dense end", "saving");
        printf("%-14s | %9s | %9.3f | %9.3f | %13s | %9s | %9s | %8s\n", "serial", "-", t_dense, t_serial, "-", "-", "-", "-");
        // best split by saving: streams rows against the event-timed serial, green rows against the host-timed one
        double best_sv = -1e30, best = 0, best_base = 0; std::string best_mode; int best_s = 0;
        for (size_t i = 0; i < caps.size(); ++i) {
            char lab[32]; snprintf(lab, sizeof lab, "streams S=%d", caps[i]);
            const double sv = 100.0 * (t_serial - t_c[i]) / t_serial;
            printf("%-14s | %9.3f | %9.3f | %9.3f | %13.3f | %9.3f | %9.3f | %7.1f%%\n", lab, t_x[i], t_dense, t_serial,
                   t_c[i], t_cx[i], t_cd[i], sv);
            if (sv > best_sv) { best_sv = sv; best = t_c[i]; best_mode = "streams"; best_s = caps[i]; best_base = t_serial; }
        }
        if (green) {
            for (size_t i = 0; i < caps.size(); ++i) {
                if (!greens[i].ok) continue;
                char lab[32]; snprintf(lab, sizeof lab, "green S=%d/%d", greens[i].sms_x, greens[i].sms_d);
                const double sv = 100.0 * (t_serial_host - t_g[i]) / t_serial_host;
                printf("%-14s | %9s | %9s | %9.3f | %13.3f | %9s | %9s | %7.1f%%   (host wall)\n", lab, "-", "-",
                       t_serial_host, t_g[i], "-", "-", sv);
                if (sv > best_sv) { best_sv = sv; best = t_g[i]; best_mode = "green"; best_s = greens[i].sms_x; best_base = t_serial_host; }
            }
        }
        char b[256];
        snprintf(b, sizeof b, "BEST R=%d: x3gm %d SMs, %s, %.3f ms vs serial %.3f ms (-%.1f %%)%s", R, best_s,
                 best_mode.c_str(), best, best_base, 100.0 * (best_base - best) / best_base,
                 best_mode == "green" ? " [host wall]" : "");
        best_lines.push_back(b);
        printf("%s\n", b);
        // --tiles: x3gm alone (every SM) at each tile pair, its own plan per BM (the same picks), Y against G6's
        if (tiles) {
            const Plan p128 = route(R, zipf, 128, 1234 + R);
            int *pe2 = dev(p128.pe), *poff2 = dev(p128.poff), *pcnt2 = dev(p128.pcnt);
            int* np2 = dev(std::vector<int>{p128.npass});
            printf("tiles R=%d (%d passes at BM 64, %d at BM 128):\n", R, pl.npass, p128.npass);
            double t_ref = 0, best_t = 1e30; const char* best_n = "";
            for (const TilePair& tp : tilepairs) {
                if (tp.gu.per_sm == 0 || tp.dn.per_sm == 0) { printf("   %-20s does not fit an SM\n", tp.name); continue; }
                dsv41_x3gm::Args g2 = ag, d2 = ad;
                if (tp.bm_gu == 128) { g2.pe = pe2; g2.poff = poff2; g2.pcnt = pcnt2; g2.npass = np2; }
                if (tp.bm_dn == 128) { d2.pe = pe2; d2.poff = poff2; d2.pcnt = pcnt2; d2.npass = np2; }
                auto body = [&](cudaStream_t s) {
                    CK(cudaMemsetAsync(ticket, 0, 8, s));
                    dsv41_x3gm::rot_kernel<half, 2><<<(rot_items + 7) / 8, 256, 0, s>>>(x, D, pick, suh_g, suh_u, xg, xu, D,
                                                                                       SLOTS, rot_items);
                    tp.gu.k<<<sms * tp.gu.per_sm, tp.gu.threads, tp.gu.smem, s>>>(g2);
                    tp.dn.k<<<sms * tp.dn.per_sm, tp.dn.threads, tp.dn.smem, s>>>(d2);
                };
                const double t = med1(body);
                body(st);
                check_bits(tp.name, sms);
                if (t_ref == 0) t_ref = t;
                if (t < best_t) { best_t = t; best_n = tp.name; }
                printf("   %-20s %8.3f ms  (%+.1f %% vs G6)  %.3f ms / 1K rows\n", tp.name, t, 100.0 * (t - t_ref) / t_ref,
                       1024.0 * t / R);
            }
            char tb[200];
            snprintf(tb, sizeof tb, "TILES R=%d: best %s %.3f ms vs G6 %.3f ms (%+.1f %%)", R, best_n, best_t, t_ref,
                     100.0 * (best_t - t_ref) / t_ref);
            best_lines.push_back(tb);
            printf("%s\n", tb);
            for (int* q : {pe2, poff2, pcnt2, np2}) CK(cudaFree(q));
        }
        for (int* p : {pick, order, pe, poff, pcnt, np}) CK(cudaFree(p));
    }

    printf("\n");
    for (const auto& l : best_lines) printf("%s\n", l.c_str());
    if (bits_ok) printf("x3gm bits: PASS (same at every cap)\n");
    else printf("x3gm bits: FAIL %s\n", bits_msg.c_str());
    return bits_ok ? 0 : 1;
}
