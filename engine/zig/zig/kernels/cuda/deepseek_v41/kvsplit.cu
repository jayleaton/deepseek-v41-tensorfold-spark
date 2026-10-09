// Split KV's exchange copies (zig/src/families/deepseek_v41/kv/split.zig: HostKernels is the reference): before the
// all-gather, the selected rows of a split family packed dense, and a row list gathered. Pure copies (bit-exact by
// construction), 8 bytes a thread (row bytes are multiples of 8: 584 FP8 KV rows, 384 / 132 + pad index rows), no
// host sync, graph-capturable.
//
//   dense:  for i < R K: t = max(sel[i], 0); s = rslot ? rslot[i / K] : 0; local = table[s pts + (t >> psh)]
//           send[i] = base row ((local << psh) | (t & (2^psh - 1)));  tok[i] = sel[i] < 0 ? -1 : ((t >> psh) % world) R K + i
//   gather: send[i] = base row phys[i]
//   pack:   (TF_DSV41_KV_SPLIT_COMPACT, kvsplit.py's dense_packed / kvsplit_pack.pack2) o_i = owner of entry i (none: -1), at_i =
//           its place among o_i's valid entries in list order; tok[i] = sel[i] < 0 ? -1 : o_i R K + at_i; send[at_i] = the dense
//           row of entry i where o_i == rank; lens[o] = (entries of o) row_bytes. One launch for any world <= 8: block b takes
//           entries [256 b, 256 b + 256), counts each owner's entries before them itself (no inter-block wait), scans its own
//           by warp ballots and copies its owned rows; the last block has seen every entry and writes lens.

#include <cstdint>

namespace dsv41_kvsplit {

// kv/split.zig DenseArgs (extern, by value): the same fields at the same offsets
struct DenseArgs {
    const int* sel;          // int32 [R, K] logical rows (-1: none)
    uint32_t rows, k;
    const int* table;        // int32 split page tables: [pages], or row mode's stacked [S, pts]
    uint32_t pts;
    const int* rslot;        // int32 [R] or null
    uint32_t psh;
    const uint8_t* base;     // the family's local rows
    uint32_t row_bytes, world;
    uint8_t* send;           // [R K, row bytes]
    int* tok;                // int32 [R, K]
};
static_assert(sizeof(DenseArgs) == 80, "kv/split.zig DenseArgs");

__global__ void __launch_bounds__(256) dense_kernel(const DenseArgs a) {
    const uint32_t wpr = a.row_bytes >> 3;                      // 8-byte words a row
    const long long n = (long long)a.rows * a.k;
    const long long words = n * wpr;
    for (long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x; w < words; w += (long long)gridDim.x * blockDim.x) {
        const long long i = w / wpr;
        const uint32_t q = (uint32_t)(w - i * wpr);
        const int sv = a.sel[i];
        const uint32_t t = sv > 0 ? (uint32_t)sv : 0u;
        const long long s = a.rslot ? (long long)a.rslot[i / a.k] : 0;
        const uint32_t local = (uint32_t)a.table[s * a.pts + (t >> a.psh)];
        const unsigned long long row = ((unsigned long long)local << a.psh) | (t & ((1u << a.psh) - 1u));
        const uint64_t* src = reinterpret_cast<const uint64_t*>(a.base + row * a.row_bytes);
        reinterpret_cast<uint64_t*>(a.send + (unsigned long long)i * a.row_bytes)[q] = src[q];
        if (q == 0) a.tok[i] = sv < 0 ? -1 : (int)((long long)((t >> a.psh) % a.world) * n + i);
    }
}

// kv/split.zig PackArgs: DenseArgs with rank in psh's padding and lens after tok
struct PackArgs {
    const int* sel;
    uint32_t rows, k;
    const int* table;
    uint32_t pts;
    const int* rslot;
    uint32_t psh, rank;
    const uint8_t* base;
    uint32_t row_bytes, world;
    uint8_t* send;           // [R K, row bytes]: this rank's owned rows in list order first
    int* tok;                // int32 [R, K]
    int* lens;               // int32 [world]: bytes
};
static_assert(sizeof(PackArgs) == 88, "kv/split.zig PackArgs");

constexpr int kMaxWorld = 8;      // kv/split.zig max_world
constexpr int kPackThreads = 256; // entries a block; the launch's block size

__device__ __forceinline__ uint32_t owner_of(int sv, uint32_t psh, uint32_t world) {
    return sv < 0 ? world : ((uint32_t)sv >> psh) % world;   // world: no owner
}

__global__ void __launch_bounds__(kPackThreads) pack_kernel(const PackArgs a) {
    __shared__ uint32_t s_before[kMaxWorld];                     // each owner's entries before this block's
    __shared__ uint32_t s_warp[kMaxWorld][kPackThreads / 32];    // each warp's entries of each owner
    __shared__ unsigned long long s_row[kPackThreads];           // this block's owned rows by place (from the first)
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t W = a.world;
    const long long n = (long long)a.rows * a.k;
    const long long start = (long long)blockIdx.x * kPackThreads;
    if (tid < kMaxWorld) s_before[tid] = 0u;

    // 1. the entries before this block, by owner
    uint32_t c[kMaxWorld];
#pragma unroll
    for (int q = 0; q < kMaxWorld; q++) c[q] = 0u;
    for (long long j = tid; j < start; j += kPackThreads) {
        const uint32_t o = owner_of(a.sel[j], a.psh, W);
#pragma unroll
        for (int q = 0; q < kMaxWorld; q++) c[q] += (uint32_t)(o == (uint32_t)q);
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < kMaxWorld; q++) {
        if ((uint32_t)q < W) {
            uint32_t v = c[q];
#pragma unroll
            for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
            if (lane == 0u && v != 0u) atomicAdd(&s_before[q], v);
        }
    }

    // 2. this block's entries: a ballot an owner gives the place inside the warp, the warps before add theirs
    const long long i = start + tid;
    const int sv = i < n ? a.sel[i] : -1;
    const uint32_t o = owner_of(sv, a.psh, W);
    const uint32_t lt = (1u << lane) - 1u;
    uint32_t below = 0u;
#pragma unroll
    for (int q = 0; q < kMaxWorld; q++) {
        if ((uint32_t)q < W) {
            const uint32_t m = __ballot_sync(0xffffffffu, o == (uint32_t)q);
            if (o == (uint32_t)q) below = __popc(m & lt);
            if (lane == 0u) s_warp[q][warp] = __popc(m);
        }
    }
    __syncthreads();
    if (sv >= 0) {
        uint32_t at = below;
        for (uint32_t w = 0; w < warp; w++) at += s_warp[o][w];
        a.tok[i] = (int)((long long)o * n + s_before[o] + at);
        if (o == a.rank) {
            const uint32_t t = (uint32_t)sv;
            const long long s = a.rslot ? (long long)a.rslot[i / a.k] : 0;
            const uint32_t local = (uint32_t)a.table[s * a.pts + (t >> a.psh)];
            s_row[at] = ((unsigned long long)local << a.psh) | (t & ((1u << a.psh) - 1u));
        }
    } else if (i < n) {
        a.tok[i] = -1;
    }
    uint32_t mine = 0u;
#pragma unroll
    for (int w = 0; w < kPackThreads / 32; w++) mine += s_warp[a.rank][w];
    __syncthreads();

    // 3. this block's owned rows to send[first place ..]
    const uint32_t wpr = a.row_bytes >> 3;
    const unsigned long long first = s_before[a.rank];
    for (uint32_t w = tid; w < mine * wpr; w += kPackThreads) {
        const uint32_t j = w / wpr;
        const uint32_t q = w - j * wpr;
        const uint64_t* src = reinterpret_cast<const uint64_t*>(a.base + s_row[j] * a.row_bytes);
        reinterpret_cast<uint64_t*>(a.send + (first + j) * a.row_bytes)[q] = src[q];
    }

    // 4. the last block has counted every entry
    if (blockIdx.x == gridDim.x - 1 && tid < W) {
        uint32_t total = s_before[tid];
#pragma unroll
        for (int w = 0; w < kPackThreads / 32; w++) total += s_warp[tid][w];
        a.lens[tid] = (int)(total * a.row_bytes);
    }
}

__global__ void __launch_bounds__(256) gather_kernel(const uint8_t* __restrict__ base, uint32_t row_bytes,
                                                     const uint32_t* __restrict__ phys, uint32_t n,
                                                     uint8_t* __restrict__ send) {
    const uint32_t wpr = row_bytes >> 3;
    const long long words = (long long)n * wpr;
    for (long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x; w < words; w += (long long)gridDim.x * blockDim.x) {
        const long long i = w / wpr;
        const uint32_t q = (uint32_t)(w - i * wpr);
        const uint64_t* src = reinterpret_cast<const uint64_t*>(base + (unsigned long long)phys[i] * row_bytes);
        reinterpret_cast<uint64_t*>(send + (unsigned long long)i * row_bytes)[q] = src[q];
    }
}

}  // namespace dsv41_kvsplit
