// Device code of src/tensorfold/families/deepseek_v41/cuda/x3gm.cu (git blob 7aed2b027afc at dsv41-x3gm1; lines 1-39, 45-628, 649-666, 732-733; torch includes and host code cut),
// generated and checked by zig/kernels/cuda/deepseek_v41/sync.py. Do not edit: change the Python source.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
// x3gm: routed EXL3 (mul1) experts for fast prefill segments (TF_DSV41_FAST_EXPERTS=gm; x3gm.py has the design and
// the cost model). Two persistent kernels over items (expert pass of <= 64 members, one 128-column Hadamard block):
//
//   gateup  Xd[p] = fp16(H(act * suh_d) / sqrt 128),  act = fp32 SwiGLU(H(Xg[p] Wg) * svh_g, H(Xu[p] Wu) * svh_u)
//   down    Y[p]  = H(Xd[p] Wd) / sqrt 128 * svh_d                                                      (fp32)
//
// Data movement (the GLM spark engine's "fat" kernel, patches/0170, at this family's widths): a CTA claims items by
// ticket; an NSA-stage cp.async ring holds, per stage, the item's member rows (gathered by pair index, XOR-swizzled
// for ldmatrix) and the item's trellis words (16 K2 bytes a 16x16 tile, copied as they are stored). Each warp decodes
// its MTL column tiles from shared memory straight into the m16n8k16 A fragment (the weights are A: W^T, so upstream's
// decode lane layout IS the A fragment) and multiplies it by up to 8 n8 groups of member rows (ldmatrix B): a weight
// tile is fetched from DRAM once a segment and decoded once a pass of 64 members, never written back.
//
// Numerics (the fast tag's rule, prefill_mm): every output element is one chain of mma over ascending k tiles from
// +0.0 (no split-K, no atomics), then upstream's fwht128 butterflies and its ACT_F32 formulas; a pair's output depends
// on its own row only (an mma column is one member; members past the count only fill zero / unused B columns), and
// which CTA runs an item changes nothing. The decode is upstream's (cb_pair / LaneMap / Fmt from
// experts_grouped.cuh, fwht128 from decode.cuh: included, never copied).

//
// G16 4K (TF_DSV41_PF_4K): 4,096-row blocks put ~64 members on an average expert (24,576 pairs / 384), so a 64-member
// pass splits most experts in two (each pass streams and decodes the expert's tiles again). The tile table below adds
// 128-member passes (NG 16: BM = 8 NG) for gate/up and down; only data movement changes (which warp holds which
// member columns, how many k tiles a stage, how deep the ring), never a chain: every element is still one mma chain
// over ascending k tiles from +0.0 and the same epilogue, so every configuration gives the same bits (x3gm.py TILES,
// tests/test_dsv41_chunk4k_x3gm.py on the emulator, tests/cuda/test_dsv41_chunk4k_gpu.py on the GPU).
//
// E1 / E2 (docs/ENGINE-E1E2.md of the quant repo): the widths K2 3, 5, 7 and 10 (1.5 / 2.5 / 3.5 / 5 bits, every
// routed width the allocator emits) are instances of the same kernels (upstream's Fmt / LaneMap cover every K2; a tile
// is 16 K2 bytes, so the 16-byte stage copies hold);
// and an expert's trellis may come from a per-expert pointer table (tp0 / tp1, a ragged stack: each expert its own
// width) instead of the stack's base + e x its stride. The host runs one launch per width present, each over the
// passes of that width's experts only; the address of an expert's words is the same either way, so a uniform stack
// gives the same loads and the same bits through either form.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "decode.cuh"
#include "experts_grouped.cuh"

namespace dsv41_x3gm {

using tf_exl3x::cb_pair;
using tf_exl3x::Fmt;
using tf_exl3x::LaneMap;
using tf_exl3x::mma16816;

constexpr float HAD = 0.08838834764831845f;          // 1 / sqrt(128)
constexpr int NB = 8;                                 // column tiles an item: 128 columns, one Hadamard block
constexpr int MUL1 = 2;

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(uint32_t (&a)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}

// the 16-byte chunk c of member row r in a stage (CPR chunks a row): 8 rows of an ldmatrix phase in 8 bank groups
template <int CPR>
__device__ __forceinline__ int swz(int r, int c) {
    static_assert(CPR == 8 || CPR == 4, "rows of 64 or 32 halves");
    return CPR == 8 ? (c ^ (r & 7)) : (c ^ ((r >> 1) & 3));
}

// One 16x16 tile (k rows, n columns) of W_q from its words in shared memory, as the A fragment of W^T: upstream's
// decode_tile (the same LaneMap windows and cb_pair values; b0 = column g, b1 = column g + 8), registers reordered.
template <int K2>
__device__ __forceinline__ void decode_a(const uint32_t* tile, const LaneMap<K2>& m, uint32_t (&a)[4]) {
    constexpr int GV = Fmt<K2>::GV, NG = Fmt<K2>::NG;
    uint32_t st[8];
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        const uint32_t whi = tile[m.hi[g]], wlo = tile[m.lo[g]];
        const uint64_t mm = ((((uint64_t)wlo) << 32) | whi) >> m.sh[g];
#pragma unroll
        for (int j = 0; j < GV; ++j) st[g * GV + j] = (uint32_t)(mm >> Fmt<K2>::off(j)) & 0xffffu;
    }
    a[0] = cb_pair<MUL1>(st[0], st[1]);      // b0[0]: k 2t, 2t+1 of column g      -> A row g,     k 2t..
    a[1] = cb_pair<MUL1>(st[4], st[5]);      // b1[0]: k 2t, 2t+1 of column g + 8  -> A row g + 8, k 2t..
    a[2] = cb_pair<MUL1>(st[2], st[3]);      // b0[1]: k 2t+8, 2t+9 of column g    -> A row g,     k 2t+8..
    a[3] = cb_pair<MUL1>(st[6], st[7]);      // b1[1]                              -> A row g + 8, k 2t+8..
}

template <int MATS, int XM, int K2, int MTL, int NG, int KS, int NSA, int NGR>
struct Cfg {
    static constexpr int BM = NG * 8, WPM = NB / MTL, W = MATS * WPM, THREADS = W * 32;
    static constexpr int CPR = KS * 2, LDA = KS * 16, TW = 4 * K2;
    static constexpr int LDE = 132, ER = NGR * 8, ROUNDS = NG / NGR;
    static constexpr size_t A_BYTES = (size_t)XM * BM * LDA * sizeof(half);
    static constexpr size_t W_BYTES = (size_t)MATS * KS * NB * TW * sizeof(uint32_t);
    static constexpr size_t STAGE = A_BYTES + W_BYTES;
    static constexpr size_t RING = (size_t)NSA * STAGE;
    static constexpr size_t EPI = (size_t)MATS * ER * LDE * sizeof(float);
    static constexpr size_t SMEM = RING > EPI ? RING : EPI;
    // CTAs an SM the registers are capped for: two when two rings fit, a CTA has at most 8 warps and a lane at most
    // 64 accumulators (every G6 configuration: unchanged); else one (4K's 16-warp tiles get 128 registers a thread,
    // its 128-accumulator tiles 255)
    static constexpr int MINB = 2 * (SMEM + 1536) <= 102400 && THREADS <= 256 && MTL * NG <= 16 ? 2 : 1;
    static_assert(NG % 2 == 0 && NG % NGR == 0 && NGR % 2 == 0 && NB % MTL == 0, "tiling");
    static_assert(A_BYTES % 16 == 0 && (TW * 4) % 16 == 0, "16-byte stages");
    static_assert(XM == 1 || XM == MATS, "one shared input or one a matrix");
};

struct Args {
    const half* x0;           // [P, K] rows of matrix 0 (gate / down input)
    const half* x1;           // [P, K] rows of matrix 1 (up) when XM == 2
    const uint32_t* t0;       // [E, K/16, N/16, 4 K2] trellis words, stacked (when tp0 is null)
    const uint32_t* t1;
    const int64_t* tp0;       // [E] expert e's words [K/16, N/16, 4 K2] (a ragged stack), or null
    const int64_t* tp1;
    const int* k2e;           // [E] each expert's K2 (gm2_kernel; the launch's width table)
    const int* order;         // pair indices by expert (stable: rows ascending)
    const int* pe;            // pass -> expert
    const int* poff;          // pass -> first index into order
    const int* pcnt;          // pass -> members (<= BM)
    const int* npass;         // [1] passes in all
    int* ticket;              // [1] zeroed, or null (static stride)
    const half* sv0;          // gate/up: svh_g [E, N]; down: svh_d [E, N]
    const half* sv1;          // svh_u [E, N]
    const half* sd;           // suh_d [E, N]
    half* xd;                 // gate/up out [P, N]
    float* y;                 // down out [P, N]
    int K, N;
    float limit;
};

template <int MATS, int XM, int K2, int MTL, int NG, int KS, int NSA, int NGR>
__global__ void __launch_bounds__(Cfg<MATS, XM, K2, MTL, NG, KS, NSA, NGR>::THREADS,
                                  Cfg<MATS, XM, K2, MTL, NG, KS, NSA, NGR>::MINB) gm_kernel(const Args a) {
    using C = Cfg<MATS, XM, K2, MTL, NG, KS, NSA, NGR>;
    constexpr int THREADS = C::THREADS, BM = C::BM, LDA = C::LDA, LDE = C::LDE, ER = C::ER, CPR = C::CPR,
                  TW = C::TW, W = C::W;
    extern __shared__ __align__(16) unsigned char smem[];
    float* ep = reinterpret_cast<float*>(smem);               // [MATS][ER][LDE] after the K loop (aliases the ring)
    __shared__ int rows_sh[BM];
    __shared__ int item_sh;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int mat = warp / C::WPM, slice = warp % C::WPM;     // columns slice 16 MTL .. + 16 MTL of the item
    const int xmat = XM == 1 ? 0 : mat;
    const int g = lane >> 2, t = lane & 3;
    const int K = a.K, N = a.N, KT = K >> 4, NTILES = N >> 4, S = KT / KS, NBLK = N >> 7;
    const size_t se = (size_t)KT * NTILES * TW;               // words of one expert's matrix
    const int total = a.npass[0] * NBLK;
    const int lr = (lane & 7) + ((lane >> 4) << 3), lc = (lane >> 3) & 1;
    const LaneMap<K2> map(lane);

    int next = blockIdx.x;
    for (;;) {
        __syncthreads();                                      // the last item is done with rows_sh and shared memory
        if (threadIdx.x == 0) item_sh = a.ticket ? atomicAdd(a.ticket, 1) : next;
        next += gridDim.x;
        __syncthreads();
        const int item = item_sh;
        if (item >= total) break;
        const int tt = item / NBLK, nb = item % NBLK;
        const int e = a.pe[tt], off = a.poff[tt], cnt = a.pcnt[tt];
        const uint32_t* te0 = a.tp0 ? reinterpret_cast<const uint32_t*>(a.tp0[e]) : a.t0 + (size_t)e * se;
        const uint32_t* te1 = a.tp1 ? reinterpret_cast<const uint32_t*>(a.tp1[e]) : a.t1 + (size_t)e * se;
        for (int i = threadIdx.x; i < BM; i += THREADS) rows_sh[i] = i < cnt ? a.order[off + i] : -1;
        __syncthreads();
        const int cnt16 = (cnt + 15) & ~15;                   // rows past it are never read (warp-uniform skip)

        // stage s: member rows' k [16 KS s, 16 KS (s + 1)) (zero past the count), then the item's words of k tiles
        // KS s .. KS s + KS - 1, every matrix, the 8 column tiles of block nb
        auto load_stage = [&](int s) {
            unsigned char* st = smem + (size_t)(s % NSA) * C::STAGE;
            half* ra = reinterpret_cast<half*>(st);
            const int k0 = s * KS * 16;
            for (int i = threadIdx.x; i < XM * BM * CPR; i += THREADS) {
                const int m = i / (BM * CPR), r = (i / CPR) % BM, c = i % CPR;
                if (r < cnt16) {
                    const int row = rows_sh[r];
                    const half* src = (m ? a.x1 : a.x0) + (size_t)(row < 0 ? 0 : row) * K + k0 + c * 8;
                    cp_async16(ra + ((size_t)m * BM + r) * LDA + swz<CPR>(r, c) * 8, src, row >= 0);
                }
            }
            uint32_t* rw = reinterpret_cast<uint32_t*>(st + C::A_BYTES);
            constexpr int WCH = MATS * KS * NB * K2;          // 16-byte chunks: K2 a tile
            for (int i = threadIdx.x; i < WCH; i += THREADS) {
                const int q = i % K2, n = (i / K2) % NB, kk = (i / (K2 * NB)) % KS, m = i / (K2 * NB * KS);
                const uint32_t* src = (m ? te1 : te0) + ((size_t)(s * KS + kk) * NTILES + nb * NB + n) * TW + q * 4;
                cp_async16(rw + ((m * KS + kk) * NB + n) * TW + q * 4, src, true);
            }
        };

        float acc[MTL][NG][4];
#pragma unroll
        for (int l = 0; l < MTL; ++l)
#pragma unroll
            for (int n = 0; n < NG; ++n)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[l][n][c] = 0.f;

#pragma unroll
        for (int s = 0; s < NSA - 1; ++s) {
            if (s < S) load_stage(s);
            cp_commit();
        }
        cp_wait<NSA - 2>();
        __syncthreads();
        for (int s = 0; s < S; ++s) {
            if (s + NSA - 1 < S) load_stage(s + NSA - 1);
            cp_commit();
            const unsigned char* st = smem + (size_t)(s % NSA) * C::STAGE;
            const half* A = reinterpret_cast<const half*>(st) + (size_t)xmat * BM * LDA;
            const uint32_t* Wd = reinterpret_cast<const uint32_t*>(st + C::A_BYTES) + (size_t)mat * KS * NB * TW +
                                 slice * MTL * TW;
#pragma unroll
            for (int kk = 0; kk < KS; ++kk) {
                uint32_t af[MTL][4];
#pragma unroll
                for (int l = 0; l < MTL; ++l) decode_a<K2>(Wd + (kk * NB + l) * TW, map, af[l]);
                const int chunk = swz<CPR>(lr, kk * 2 + lc);
#pragma unroll
                for (int np = 0; np < NG / 2; ++np) {
                    if (16 * np < cnt) {                      // warp-uniform
                        uint32_t b[4];
                        ldsm4(b, A + (16 * np + lr) * LDA + chunk * 8);
                        const uint32_t b01[2] = {b[0], b[1]}, b23[2] = {b[2], b[3]};
#pragma unroll
                        for (int l = 0; l < MTL; ++l) {
                            mma16816(acc[l][2 * np], af[l], b01);
                            mma16816(acc[l][2 * np + 1], af[l], b23);
                        }
                    }
                }
            }
            cp_wait<NSA - 2>();
            __syncthreads();
        }

        // epilogue: ER members a round, accumulators -> ep[mat][member][column], then one warp a member row
#pragma unroll
        for (int rd = 0; rd < C::ROUNDS; ++rd) {
            if (rd) __syncthreads();
            if (ER * rd < cnt) {
#pragma unroll
                for (int n = 0; n < NGR; ++n) {
                    const int ng = rd * NGR + n;
                    float* E = ep + ((size_t)mat * ER + 8 * n + 2 * t) * LDE + slice * 16 * MTL + g;
#pragma unroll
                    for (int l = 0; l < MTL; ++l) {
                        E[l * 16] = acc[l][ng][0];
                        E[LDE + l * 16] = acc[l][ng][1];
                        E[l * 16 + 8] = acc[l][ng][2];
                        E[LDE + l * 16 + 8] = acc[l][ng][3];
                    }
                }
            }
            __syncthreads();
            for (int r = warp; r < ER && ER * rd + r < cnt; r += W) {
                const int row = rows_sh[ER * rd + r];
                const int c0 = 4 * lane;
                const size_t col = (size_t)e * N + nb * 128 + c0;
                float v[4];
                *reinterpret_cast<float4*>(v) = *reinterpret_cast<const float4*>(ep + (size_t)r * LDE + c0);
                tf_exl3::fwht128(v, lane);
                if constexpr (MATS == 2) {
                    float u[4];
                    *reinterpret_cast<float4*>(u) = *reinterpret_cast<const float4*>(ep + ((size_t)ER + r) * LDE + c0);
                    tf_exl3::fwht128(u, lane);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {                 // upstream gateup_epilogue, ACT_F32
                        const float gg = fminf(v[j] * HAD * __half2float(a.sv0[col + j]), a.limit);
                        const float uu = fminf(fmaxf(u[j] * HAD * __half2float(a.sv1[col + j]), -a.limit), a.limit);
                        const float act = gg / (1.f + expf(-gg)) * uu;
                        v[j] = act * __half2float(a.sd[col + j]);
                    }
                    tf_exl3::fwht128(v, lane);
                    half2* o = reinterpret_cast<half2*>(a.xd + (size_t)row * N + nb * 128 + c0);
                    o[0] = __halves2half2(__float2half_rn(v[0] * HAD), __float2half_rn(v[1] * HAD));
                    o[1] = __halves2half2(__float2half_rn(v[2] * HAD), __float2half_rn(v[3] * HAD));
                } else {
                    float4 o;                                     // upstream down_epilogue
                    o.x = v[0] * HAD * __half2float(a.sv0[col + 0]);
                    o.y = v[1] * HAD * __half2float(a.sv0[col + 1]);
                    o.z = v[2] * HAD * __half2float(a.sv0[col + 2]);
                    o.w = v[3] * HAD * __half2float(a.sv0[col + 3]);
                    *reinterpret_cast<float4*>(a.y + (size_t)row * N + nb * 128 + c0) = o;
                }
            }
        }
    }
}

// x3gm v2 (TF_DSV41_GM_V2): the same items, chains and epilogue as gm_kernel, with less issue work. Measured on 48
// SMs (docs/ENGINE-X3GM2.md of the quant repo) gm_kernel issues as many instructions as its tensor cores need cycles,
// and two of its costs are not arithmetic: every ldmatrix / mma of an item's absent member groups is issued and
// predicated off (an item of 33 members issues the 64-member pass's 16 mma a k tile), and every stage re-derives each
// cp.async address with divides. v2:
//   - an item's member groups are a template argument of its k loop, dispatched once an item: n8 groups,
//     ceil(cnt / 8) (1..8, then even counts); an odd last group runs one ldmatrix.x2 and one mma a tile;
//   - each thread's cp.async sources are computed once an item (row offsets; word offsets of stage 0), a stage
//     adds its k;
//   - the expert's width is read per item (a.k2e) and dispatched once an item, so a ragged stack runs one launch a
//     projection over one plan (gm_kernel: one launch and one plan a width present). Each width keeps the tile
//     configuration; its ring is cut (Ring2) where the widest width would otherwise cost the launch its second CTA
//     an SM; dynamic shared memory is the widest width's (Smem2).
// Every element is still one mma chain over ascending k tiles from +0.0 into a zeroed accumulator, then the same
// epilogue: a pair's Xd / Y are gm_kernel's bits (only which instructions are issued around a chain changes; an mma
// column is one member, so the groups a pass skips are columns no member owns).

__device__ __forceinline__ void ldsm2(uint32_t (&a)[2], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(a[0]), "=r"(a[1]) : "r"(s));
}

// The K loop of one item with N8 active n8 member groups (accumulators acc[l][0 .. N8 - 1]; the rest stay 0.0 and
// are never stored).
template <int N8, int MATS, int XM, int K2, int MTL, int NG, int KS, int NSA, int NGR, typename LoadFn>
__device__ __forceinline__ void kloop2(float (&acc)[MTL][NG][4], const unsigned char* smem, int S, int xmat, int mat,
                                       int slice, int lr, int lc, const LaneMap<K2>& map, LoadFn&& load_stage) {
    using C = Cfg<MATS, XM, K2, MTL, NG, KS, NSA, NGR>;
    constexpr int BM = C::BM, LDA = C::LDA, CPR = C::CPR, TW = C::TW, PAIRS = N8 / 2;
#pragma unroll
    for (int s = 0; s < NSA - 1; ++s) {
        if (s < S) load_stage(s);
        cp_commit();
    }
    cp_wait<NSA - 2>();
    __syncthreads();
    for (int s = 0; s < S; ++s) {
        if (s + NSA - 1 < S) load_stage(s + NSA - 1);
        cp_commit();
        const unsigned char* st = smem + (size_t)(s % NSA) * C::STAGE;
        const half* A = reinterpret_cast<const half*>(st) + (size_t)xmat * BM * LDA;
        const uint32_t* Wd = reinterpret_cast<const uint32_t*>(st + C::A_BYTES) + (size_t)mat * KS * NB * TW +
                             slice * MTL * TW;
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            uint32_t af[MTL][4];
#pragma unroll
            for (int l = 0; l < MTL; ++l) decode_a<K2>(Wd + (kk * NB + l) * TW, map, af[l]);
            const int chunk = swz<CPR>(lr, kk * 2 + lc);
#pragma unroll
            for (int np = 0; np < PAIRS; ++np) {
                uint32_t b[4];
                ldsm4(b, A + (16 * np + lr) * LDA + chunk * 8);
                const uint32_t b01[2] = {b[0], b[1]}, b23[2] = {b[2], b[3]};
#pragma unroll
                for (int l = 0; l < MTL; ++l) {
                    mma16816(acc[l][2 * np], af[l], b01);
                    mma16816(acc[l][2 * np + 1], af[l], b23);
                }
            }
            if constexpr ((N8 & 1) != 0) {                // members 16 PAIRS .. + 7 (x2: lanes 0-15's addresses)
                uint32_t b[2];
                ldsm2(b, A + (16 * PAIRS + lr) * LDA + chunk * 8);
#pragma unroll
                for (int l = 0; l < MTL; ++l) mma16816(acc[l][2 * PAIRS], af[l], b);
            }
        }
        cp_wait<NSA - 2>();
        __syncthreads();
    }
}

// One item (pass x 128-column block) at width K2: the caller claimed it and filled rows_sh; gm_kernel's body
// otherwise (the same stage layout, decode, mma chains and epilogue).
template <int MATS, int XM, int K2, int MTL, int NG, int KS, int NSA, int NGR>
__device__ __forceinline__ void gm2_item(const Args& a, unsigned char* smem, const int* rows_sh, int e, int cnt,
                                         int nb) {
    using C = Cfg<MATS, XM, K2, MTL, NG, KS, NSA, NGR>;
    constexpr int THREADS = C::THREADS, BM = C::BM, LDA = C::LDA, LDE = C::LDE, ER = C::ER, CPR = C::CPR,
                  TW = C::TW, W = C::W;
    constexpr int ACH = XM * BM * CPR, WCH = MATS * KS * NB * K2;    // 16-byte chunks a stage: member rows, words
    constexpr int APT = (ACH + THREADS - 1) / THREADS, WPT = (WCH + THREADS - 1) / THREADS;
    float* ep = reinterpret_cast<float*>(smem);               // [MATS][ER][LDE] after the K loop (aliases the ring)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int mat = warp / C::WPM, slice = warp % C::WPM;
    const int xmat = XM == 1 ? 0 : mat;
    const int g = lane >> 2, t = lane & 3;
    const int K = a.K, N = a.N, KT = K >> 4, NTILES = N >> 4, S = KT / KS;
    const size_t se = (size_t)KT * NTILES * TW;
    const int lr = (lane & 7) + ((lane >> 4) << 3), lc = (lane >> 3) & 1;
    const LaneMap<K2> map(lane);
    const uint32_t* te0 = a.tp0 ? reinterpret_cast<const uint32_t*>(a.tp0[e]) : a.t0 + (size_t)e * se;
    const uint32_t* te1 = a.tp1 ? reinterpret_cast<const uint32_t*>(a.tp1[e]) : a.t1 + (size_t)e * se;
    const int cnt16 = (cnt + 15) & ~15;                       // rows past it are never read

    // this thread's chunks: a row chunk's element offset at k 0 (-1: none, -2: zero fill), a word chunk's word
    // offset at stage 0; the destinations follow from the chunk index as in gm_kernel's load_stage
    int a_src[APT], w_src[WPT];
#pragma unroll
    for (int j = 0; j < APT; ++j) {
        const int i = threadIdx.x + j * THREADS, r = (i / CPR) % BM, c = i % CPR;
        int v = -1;
        if (i < ACH && r < cnt16) v = rows_sh[r] < 0 ? -2 : rows_sh[r] * K + c * 8;
        a_src[j] = v;
    }
#pragma unroll
    for (int j = 0; j < WPT; ++j) {
        const int i = threadIdx.x + j * THREADS, q = i % K2, n = (i / K2) % NB, kk = (i / (K2 * NB)) % KS;
        w_src[j] = (kk * NTILES + nb * NB + n) * TW + q * 4;
    }
    auto load_stage = [&](int s) {
        unsigned char* st = smem + (size_t)(s % NSA) * C::STAGE;
        half* ra = reinterpret_cast<half*>(st);
        uint32_t* rw = reinterpret_cast<uint32_t*>(st + C::A_BYTES);
        const int k0 = s * KS * 16;
#pragma unroll
        for (int j = 0; j < APT; ++j) {
            const int i = threadIdx.x + j * THREADS, v = a_src[j];
            if (v != -1) {
                const int m = i / (BM * CPR), r = (i / CPR) % BM, c = i % CPR;
                cp_async16(ra + ((size_t)m * BM + r) * LDA + swz<CPR>(r, c) * 8, (m ? a.x1 : a.x0) + (v < 0 ? 0 : v + k0),
                           v >= 0);
            }
        }
        const size_t ws = (size_t)s * KS * NTILES * TW;
#pragma unroll
        for (int j = 0; j < WPT; ++j) {
            const int i = threadIdx.x + j * THREADS;
            if (WCH % THREADS == 0 || i < WCH) {
                const int q = i % K2, n = (i / K2) % NB, kk = (i / (K2 * NB)) % KS, m = i / (K2 * NB * KS);
                cp_async16(rw + ((m * KS + kk) * NB + n) * TW + q * 4, (m ? te1 : te0) + ws + w_src[j], true);
            }
        }
    };

    float acc[MTL][NG][4];
#pragma unroll
    for (int l = 0; l < MTL; ++l)
#pragma unroll
        for (int n = 0; n < NG; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[l][n][c] = 0.f;

    static_assert(NG <= 16 && NG % 2 == 0, "n8 groups");
    const int n8 = (cnt + 7) >> 3;                            // >= 1 (a pass has members)
#define GM2_K(G) kloop2<G, MATS, XM, K2, MTL, NG, KS, NSA, NGR>(acc, smem, S, xmat, mat, slice, lr, lc, map, load_stage)
    switch (n8 <= 8 ? n8 : (n8 + 1) & ~1) {                   // 1..8, then even counts (NG 16)
        case 1: GM2_K(1); break;
        case 2: GM2_K(2); break;
        case 3: GM2_K(3); break;
        case 4: GM2_K(4); break;
        case 5: if constexpr (NG >= 6) GM2_K(5); break;
        case 6: if constexpr (NG >= 6) GM2_K(6); break;
        case 7: if constexpr (NG >= 8) GM2_K(7); break;
        case 8: if constexpr (NG >= 8) GM2_K(8); break;
        case 10: if constexpr (NG >= 10) GM2_K(10); break;
        case 12: if constexpr (NG >= 12) GM2_K(12); break;
        case 14: if constexpr (NG >= 14) GM2_K(14); break;
        default: if constexpr (NG >= 16) GM2_K(16); break;
    }
#undef GM2_K

    // epilogue: gm_kernel's
#pragma unroll
    for (int rd = 0; rd < C::ROUNDS; ++rd) {
        if (rd) __syncthreads();
        if (ER * rd < cnt) {
#pragma unroll
            for (int n = 0; n < NGR; ++n) {
                const int ng = rd * NGR + n;
                float* E = ep + ((size_t)mat * ER + 8 * n + 2 * t) * LDE + slice * 16 * MTL + g;
#pragma unroll
                for (int l = 0; l < MTL; ++l) {
                    E[l * 16] = acc[l][ng][0];
                    E[LDE + l * 16] = acc[l][ng][1];
                    E[l * 16 + 8] = acc[l][ng][2];
                    E[LDE + l * 16 + 8] = acc[l][ng][3];
                }
            }
        }
        __syncthreads();
        for (int r = warp; r < ER && ER * rd + r < cnt; r += W) {
            const int row = rows_sh[ER * rd + r];
            const int c0 = 4 * lane;
            const size_t col = (size_t)e * N + nb * 128 + c0;
            float v[4];
            *reinterpret_cast<float4*>(v) = *reinterpret_cast<const float4*>(ep + (size_t)r * LDE + c0);
            tf_exl3::fwht128(v, lane);
            if constexpr (MATS == 2) {
                float u[4];
                *reinterpret_cast<float4*>(u) = *reinterpret_cast<const float4*>(ep + ((size_t)ER + r) * LDE + c0);
                tf_exl3::fwht128(u, lane);
#pragma unroll
                for (int j = 0; j < 4; ++j) {                 // upstream gateup_epilogue, ACT_F32
                    const float gg = fminf(v[j] * HAD * __half2float(a.sv0[col + j]), a.limit);
                    const float uu = fminf(fmaxf(u[j] * HAD * __half2float(a.sv1[col + j]), -a.limit), a.limit);
                    const float act = gg / (1.f + expf(-gg)) * uu;
                    v[j] = act * __half2float(a.sd[col + j]);
                }
                tf_exl3::fwht128(v, lane);
                half2* o = reinterpret_cast<half2*>(a.xd + (size_t)row * N + nb * 128 + c0);
                o[0] = __halves2half2(__float2half_rn(v[0] * HAD), __float2half_rn(v[1] * HAD));
                o[1] = __halves2half2(__float2half_rn(v[2] * HAD), __float2half_rn(v[3] * HAD));
            } else {
                float4 o;                                     // upstream down_epilogue
                o.x = v[0] * HAD * __half2float(a.sv0[col + 0]);
                o.y = v[1] * HAD * __half2float(a.sv0[col + 1]);
                o.z = v[2] * HAD * __half2float(a.sv0[col + 2]);
                o.w = v[3] * HAD * __half2float(a.sv0[col + 3]);
                *reinterpret_cast<float4*>(a.y + (size_t)row * N + nb * 128 + c0) = o;
            }
        }
    }
}

// The widths a v2 launch dispatches (K2S of x3gm.py) and, per width, the ring depth: the configuration's, cut while
// the width's ring would cost a configuration that holds two CTAs an SM at 2 bits its second CTA (data movement only)
#define GM2_WIDTHS(X) X(3) X(4) X(5) X(6) X(7) X(8) X(10)
constexpr int GM2_NWIDTHS = 7;
template <int MATS, int XM, int K2, int MTL, int NG, int KS, int NSA, int NGR>
struct Ring2 {
    static constexpr bool fits2(int n) {
        return 2 * (Cfg<MATS, XM, K2, MTL, NG, KS, 2, NGR>::STAGE * (size_t)n + 1536) <= 102400;
    }
    static constexpr int cut(int n) { return n <= 2 || fits2(n) ? n : cut(n - 1); }
    static constexpr int value = Cfg<MATS, XM, 4, MTL, NG, KS, NSA, NGR>::MINB == 2 ? cut(NSA) : NSA;
};
template <int MATS, int XM, int MTL, int NG, int KS, int NSA, int NGR>
struct Smem2 {
#define GM2_SM(K) Cfg<MATS, XM, K, MTL, NG, KS, Ring2<MATS, XM, K, MTL, NG, KS, NSA, NGR>::value, NGR>::SMEM,
    static constexpr size_t list[GM2_NWIDTHS] = {GM2_WIDTHS(GM2_SM)};
#undef GM2_SM
    static constexpr size_t max_from(int i) {
        return i == GM2_NWIDTHS ? 0 : (list[i] > max_from(i + 1) ? list[i] : max_from(i + 1));
    }
    static constexpr size_t SMEM = max_from(0);
    static constexpr int THREADS = Cfg<MATS, XM, 4, MTL, NG, KS, NSA, NGR>::THREADS;
    static constexpr int MINB = 2 * (SMEM + 1536) <= 102400 && THREADS <= 256 && MTL * NG <= 16 ? 2 : 1;
};

// One v2 launch over the passes of every width (uniform stack: one width; ragged: each expert its own, a.k2e)
template <int MATS, int XM, int MTL, int NG, int KS, int NSA, int NGR>
__global__ void __launch_bounds__(Smem2<MATS, XM, MTL, NG, KS, NSA, NGR>::THREADS,
                                  Smem2<MATS, XM, MTL, NG, KS, NSA, NGR>::MINB) gm2_kernel(const Args a) {
    constexpr int BM = NG * 8, THREADS = Smem2<MATS, XM, MTL, NG, KS, NSA, NGR>::THREADS;
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int rows_sh[BM];
    __shared__ int item_sh;
    const int NBLK = a.N >> 7, total = a.npass[0] * NBLK;
    int next = blockIdx.x;
    for (;;) {
        __syncthreads();                                      // the last item is done with rows_sh and shared memory
        if (threadIdx.x == 0) item_sh = a.ticket ? atomicAdd(a.ticket, 1) : next;
        next += gridDim.x;
        __syncthreads();
        const int item = item_sh;
        if (item >= total) break;
        const int tt = item / NBLK, nb = item % NBLK;
        const int e = a.pe[tt], off = a.poff[tt], cnt = a.pcnt[tt];
        for (int i = threadIdx.x; i < BM; i += THREADS) rows_sh[i] = i < cnt ? a.order[off + i] : -1;
        __syncthreads();
        switch (a.k2e[e]) {                                   // block-uniform
#define GM2_CASE(K)                                                                                                   \
    case K:                                                                                                           \
        gm2_item<MATS, XM, K, MTL, NG, KS, Ring2<MATS, XM, K, MTL, NG, KS, NSA, NGR>::value, NGR>(a, smem, rows_sh, e, \
                                                                                                  cnt, nb);           \
        break;
            GM2_WIDTHS(GM2_CASE)
#undef GM2_CASE
            default: break;                                   // never: the host checks every width is in K2S
        }
    }
}

// Rotated inputs of the routed pairs: out_m[p] = fp16((x[p / slots] * suh_m[e]) H / sqrt 128), upstream rot_in's
// arithmetic (one warp a (pair, 128-block), 8 a block; the x values read once for both matrices).
template <typename TIN> __device__ __forceinline__ float to_f(TIN v);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }

template <typename TIN, int MATS>
__global__ void __launch_bounds__(256) rot_kernel(const TIN* __restrict__ x, int x_stride, const int* __restrict__ pick,
                                                  const half* __restrict__ suh0, const half* __restrict__ suh1,
                                                  half* __restrict__ out0, half* __restrict__ out1, int K, int slots,
                                                  int items) {
    const int item = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (item >= items) return;                                // warp-uniform
    const int KB = K >> 7, p = item / KB, blk = item % KB, lane = threadIdx.x & 31;
    const int e = pick[p];
    const TIN* xr = x + (size_t)(p / slots) * x_stride + blk * 128 + 4 * lane;
    float xv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) xv[j] = to_f<TIN>(xr[j]);
#pragma unroll
    for (int m = 0; m < MATS; ++m) {
        const half* suh = (m ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = xv[j] * __half2float(suh[j]);
        tf_exl3::fwht128(v, lane);
        half* o = (m ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD);
    }
}

// Tests: W_q [K, N] fp16 of one trellis through decode_a (words staged in shared memory as the gm kernels stage them).
template <int K2>
__global__ void __launch_bounds__(32) dequant_kernel(const uint32_t* __restrict__ T, half* __restrict__ out, int N) {
    constexpr int TW = 4 * K2;
    __shared__ __align__(16) uint32_t tile[TW];
    const int kt = blockIdx.x, nt = blockIdx.y, lane = threadIdx.x, NTILES = N >> 4;
    for (int i = lane; i < TW; i += 32) tile[i] = T[((size_t)kt * NTILES + nt) * TW + i];
    __syncwarp();
    const LaneMap<K2> map(lane);
    uint32_t af[4];
    decode_a<K2>(tile, map, af);
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int q = 0; q < 4; ++q) {                             // A row (q & 1 ? g + 8 : g) = W column; k 2t (+8)
        const half2 h = *reinterpret_cast<const half2*>(&af[q]);
        const int col = nt * 16 + g + 8 * (q & 1), row = kt * 16 + 2 * t + 8 * (q >> 1);
        out[(size_t)row * N + col] = __low2half(h);
        out[(size_t)(row + 1) * N + col] = __high2half(h);
    }
}


// The tile table (MTL, NG, KS, NSA, NGR), mirrored by x3gm.py GU_TILES / DN_TILES (a test parses these lines):
// cfg 0 = fat's (4 k tiles a stage), cfg 1 = half stages, one more in flight (G6, 64-member passes);
// cfg 2.. = G16 4K: 128-member passes (NG 16) for blocks of ~64+ members an expert, and a deeper 64-member ring.
#define GM_GU0 2, 8, 4, 3, 4
#define GM_GU1 2, 8, 2, 4, 4
#define GM_GU2 1, 16, 2, 4, 4
#define GM_GU3 1, 16, 4, 3, 4
#define GM_GU4 2, 16, 2, 3, 4
#define GM_GU5 2, 8, 2, 6, 4
#define GM_DN0 2, 8, 4, 4, 8
#define GM_DN1 1, 8, 4, 4, 8
#define GM_DN2 1, 16, 4, 4, 8
#define GM_DN3 1, 16, 2, 6, 8
#define GM_DN4 2, 16, 4, 3, 16
#define GM_NGU 6
#define GM_NDN 5


}  // namespace dsv41_x3gm

#ifndef DSV41_X3GM_NO_INSTANCES
// The instantiations x3gm's bindings launch: gate/up (shared input XM 1, own inputs XM 2) at GM_GU0-5,
// down at GM_DN0-4, every width K2 3-8, 10; the rotation (bf16 / fp16 input, one or two matrices); dequant.
#define DSV41_X3GM_GU(XM, K2, CFG) template __global__ void dsv41_x3gm::gm_kernel<2, XM, K2, CFG>(const dsv41_x3gm::Args);
#define DSV41_X3GM_DN(K2, CFG) template __global__ void dsv41_x3gm::gm_kernel<1, 1, K2, CFG>(const dsv41_x3gm::Args);
// v2 (gm2_kernel: every width in one launch), the tiles launch2 takes: gate/up GM_GU0 / GM_GU1 at either
// input mode, down GM_DN0
template __global__ void dsv41_x3gm::gm2_kernel<2, 1, GM_GU0>(const dsv41_x3gm::Args);
template __global__ void dsv41_x3gm::gm2_kernel<2, 1, GM_GU1>(const dsv41_x3gm::Args);
template __global__ void dsv41_x3gm::gm2_kernel<2, 2, GM_GU0>(const dsv41_x3gm::Args);
template __global__ void dsv41_x3gm::gm2_kernel<2, 2, GM_GU1>(const dsv41_x3gm::Args);
template __global__ void dsv41_x3gm::gm2_kernel<1, 1, GM_DN0>(const dsv41_x3gm::Args);
DSV41_X3GM_GU(1, 3, GM_GU0)
DSV41_X3GM_GU(1, 3, GM_GU1)
DSV41_X3GM_GU(1, 3, GM_GU2)
DSV41_X3GM_GU(1, 3, GM_GU3)
DSV41_X3GM_GU(1, 3, GM_GU4)
DSV41_X3GM_GU(1, 3, GM_GU5)
DSV41_X3GM_GU(2, 3, GM_GU0)
DSV41_X3GM_GU(2, 3, GM_GU1)
DSV41_X3GM_GU(2, 3, GM_GU2)
DSV41_X3GM_GU(2, 3, GM_GU3)
DSV41_X3GM_GU(2, 3, GM_GU4)
DSV41_X3GM_GU(2, 3, GM_GU5)
DSV41_X3GM_DN(3, GM_DN0)
DSV41_X3GM_DN(3, GM_DN1)
DSV41_X3GM_DN(3, GM_DN2)
DSV41_X3GM_DN(3, GM_DN3)
DSV41_X3GM_DN(3, GM_DN4)
DSV41_X3GM_GU(1, 4, GM_GU0)
DSV41_X3GM_GU(1, 4, GM_GU1)
DSV41_X3GM_GU(1, 4, GM_GU2)
DSV41_X3GM_GU(1, 4, GM_GU3)
DSV41_X3GM_GU(1, 4, GM_GU4)
DSV41_X3GM_GU(1, 4, GM_GU5)
DSV41_X3GM_GU(2, 4, GM_GU0)
DSV41_X3GM_GU(2, 4, GM_GU1)
DSV41_X3GM_GU(2, 4, GM_GU2)
DSV41_X3GM_GU(2, 4, GM_GU3)
DSV41_X3GM_GU(2, 4, GM_GU4)
DSV41_X3GM_GU(2, 4, GM_GU5)
DSV41_X3GM_DN(4, GM_DN0)
DSV41_X3GM_DN(4, GM_DN1)
DSV41_X3GM_DN(4, GM_DN2)
DSV41_X3GM_DN(4, GM_DN3)
DSV41_X3GM_DN(4, GM_DN4)
DSV41_X3GM_GU(1, 5, GM_GU0)
DSV41_X3GM_GU(1, 5, GM_GU1)
DSV41_X3GM_GU(1, 5, GM_GU2)
DSV41_X3GM_GU(1, 5, GM_GU3)
DSV41_X3GM_GU(1, 5, GM_GU4)
DSV41_X3GM_GU(1, 5, GM_GU5)
DSV41_X3GM_GU(2, 5, GM_GU0)
DSV41_X3GM_GU(2, 5, GM_GU1)
DSV41_X3GM_GU(2, 5, GM_GU2)
DSV41_X3GM_GU(2, 5, GM_GU3)
DSV41_X3GM_GU(2, 5, GM_GU4)
DSV41_X3GM_GU(2, 5, GM_GU5)
DSV41_X3GM_DN(5, GM_DN0)
DSV41_X3GM_DN(5, GM_DN1)
DSV41_X3GM_DN(5, GM_DN2)
DSV41_X3GM_DN(5, GM_DN3)
DSV41_X3GM_DN(5, GM_DN4)
DSV41_X3GM_GU(1, 6, GM_GU0)
DSV41_X3GM_GU(1, 6, GM_GU1)
DSV41_X3GM_GU(1, 6, GM_GU2)
DSV41_X3GM_GU(1, 6, GM_GU3)
DSV41_X3GM_GU(1, 6, GM_GU4)
DSV41_X3GM_GU(1, 6, GM_GU5)
DSV41_X3GM_GU(2, 6, GM_GU0)
DSV41_X3GM_GU(2, 6, GM_GU1)
DSV41_X3GM_GU(2, 6, GM_GU2)
DSV41_X3GM_GU(2, 6, GM_GU3)
DSV41_X3GM_GU(2, 6, GM_GU4)
DSV41_X3GM_GU(2, 6, GM_GU5)
DSV41_X3GM_DN(6, GM_DN0)
DSV41_X3GM_DN(6, GM_DN1)
DSV41_X3GM_DN(6, GM_DN2)
DSV41_X3GM_DN(6, GM_DN3)
DSV41_X3GM_DN(6, GM_DN4)
DSV41_X3GM_GU(1, 7, GM_GU0)
DSV41_X3GM_GU(1, 7, GM_GU1)
DSV41_X3GM_GU(1, 7, GM_GU2)
DSV41_X3GM_GU(1, 7, GM_GU3)
DSV41_X3GM_GU(1, 7, GM_GU4)
DSV41_X3GM_GU(1, 7, GM_GU5)
DSV41_X3GM_GU(2, 7, GM_GU0)
DSV41_X3GM_GU(2, 7, GM_GU1)
DSV41_X3GM_GU(2, 7, GM_GU2)
DSV41_X3GM_GU(2, 7, GM_GU3)
DSV41_X3GM_GU(2, 7, GM_GU4)
DSV41_X3GM_GU(2, 7, GM_GU5)
DSV41_X3GM_DN(7, GM_DN0)
DSV41_X3GM_DN(7, GM_DN1)
DSV41_X3GM_DN(7, GM_DN2)
DSV41_X3GM_DN(7, GM_DN3)
DSV41_X3GM_DN(7, GM_DN4)
DSV41_X3GM_GU(1, 8, GM_GU0)
DSV41_X3GM_GU(1, 8, GM_GU1)
DSV41_X3GM_GU(1, 8, GM_GU2)
DSV41_X3GM_GU(1, 8, GM_GU3)
DSV41_X3GM_GU(1, 8, GM_GU4)
DSV41_X3GM_GU(1, 8, GM_GU5)
DSV41_X3GM_GU(2, 8, GM_GU0)
DSV41_X3GM_GU(2, 8, GM_GU1)
DSV41_X3GM_GU(2, 8, GM_GU2)
DSV41_X3GM_GU(2, 8, GM_GU3)
DSV41_X3GM_GU(2, 8, GM_GU4)
DSV41_X3GM_GU(2, 8, GM_GU5)
DSV41_X3GM_DN(8, GM_DN0)
DSV41_X3GM_DN(8, GM_DN1)
DSV41_X3GM_DN(8, GM_DN2)
DSV41_X3GM_DN(8, GM_DN3)
DSV41_X3GM_DN(8, GM_DN4)
DSV41_X3GM_GU(1, 10, GM_GU0)
DSV41_X3GM_GU(1, 10, GM_GU1)
DSV41_X3GM_GU(1, 10, GM_GU2)
DSV41_X3GM_GU(1, 10, GM_GU3)
DSV41_X3GM_GU(1, 10, GM_GU4)
DSV41_X3GM_GU(1, 10, GM_GU5)
DSV41_X3GM_GU(2, 10, GM_GU0)
DSV41_X3GM_GU(2, 10, GM_GU1)
DSV41_X3GM_GU(2, 10, GM_GU2)
DSV41_X3GM_GU(2, 10, GM_GU3)
DSV41_X3GM_GU(2, 10, GM_GU4)
DSV41_X3GM_GU(2, 10, GM_GU5)
DSV41_X3GM_DN(10, GM_DN0)
DSV41_X3GM_DN(10, GM_DN1)
DSV41_X3GM_DN(10, GM_DN2)
DSV41_X3GM_DN(10, GM_DN3)
DSV41_X3GM_DN(10, GM_DN4)
#define DSV41_X3GM_ROT(T, M) template __global__ void dsv41_x3gm::rot_kernel<T, M>(const T*, int, const int*, const half*, \
    const half*, half*, half*, int, int, int);
DSV41_X3GM_ROT(__nv_bfloat16, 1)
DSV41_X3GM_ROT(__nv_bfloat16, 2)
DSV41_X3GM_ROT(half, 1)
DSV41_X3GM_ROT(half, 2)
template __global__ void dsv41_x3gm::dequant_kernel<3>(const uint32_t*, half*, int);
template __global__ void dsv41_x3gm::dequant_kernel<4>(const uint32_t*, half*, int);
template __global__ void dsv41_x3gm::dequant_kernel<5>(const uint32_t*, half*, int);
template __global__ void dsv41_x3gm::dequant_kernel<6>(const uint32_t*, half*, int);
template __global__ void dsv41_x3gm::dequant_kernel<7>(const uint32_t*, half*, int);
template __global__ void dsv41_x3gm::dequant_kernel<8>(const uint32_t*, half*, int);
template __global__ void dsv41_x3gm::dequant_kernel<10>(const uint32_t*, half*, int);
#endif  // DSV41_X3GM_NO_INSTANCES
