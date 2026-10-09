// Device code of src/tensorfold/families/deepseek_v41/cuda/csa2/attn_cuda.cu (git blob f4be064c7368 at 78b703d; the whole file, DSV41_ATTN_NO_TORCH defined),
// generated and checked by zig/kernels/cuda/deepseek_v41/sync.py. Do not edit: change the Python source.
#define DSV41_ATTN_NO_TORCH
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
// REWRITE-PLAN e2: CSA2's decode attention core (attn._chunks + attn._merge) as ONE launch, TF_DSV41_ATTN_CUDA=1.
// attn_cuda.py has the design notes and the cost model; attn_cuda_emu.py is this file's arithmetic in numpy.
//
// CTA (chunk c of 5, head tile hb of 16 heads, row r), 8 warps. The arithmetic is attn._chunks' (8 warps, the
// nvidia_mma v2 layout with warpsPerCTA [1, 8]) and attn._merge's, operation for operation, read off the PTX Triton
// emits for sm_121, so every output is the same bits as the Triton pair:
//   keys: chunk c < 4 = the row's compressed list entries [128 c, 128 c + 128) (paged), c = 4 the SWA window; the
//     FP8 rows land in shared memory once (cp.async, 8-byte pieces, zero-filled for masked keys: Triton's other=0);
//   S (4 tiles x 4 n8 blocks x K = 512): mma.sync m16n8k16 bf16 chains from +0.0 in Triton's k order (the kWidth
//     KW dot-operand layout: KW = 4 puts k {4t, 4t+1, 16+4t, 17+4t} of a 32-block in step 0 and {4t+2, ...} in
//     step 1), then mul.f32 by 512^-0.5 and -inf where masked;
//   the tile step: tm = max (exact in any order); nm, alpha = ex2.approx(rn((m - nm) * log2 e)), p = ex2(...),
//     o = mma(bf16(p), kv, rn(o * alpha)) in two k steps, l = fma(l, alpha, (W0 + W2) + (W1 + W3)) with
//     W = (a0 + a2) + (a1 + a3), a_t = p[8w + 2t] + p[8w + 2t + 1] (Triton's shuffle / shared-memory tree);
//   partials to Triton's scratch layout, a fence and a ticket per (row, head tile); the LAST CTA to arrive merges the
//     chunks in order (chunk 0: fma(b, co, +0); then fma(o, a, rn(co * b)), fma(l, a, rn(b * cl))), the sink,
//     div.full.f32, the inverse RoPE (fma(e, c, rn(o * s)), fma(o, c, -rn(e * s))) and one bf16 rounding.
// Rows never mix; which CTA arrives last never changes a bit. Chunks with no keys (c >= 1 past the row's count)
// exit at once: the merge uses Triton's empty partial for them (m = -inf, l = 0, o = +0).
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace dsv41_attn {

constexpr int D = 512, CH = 128, NCOMP = 4, NCH = 5, BMQ = 16, WINDOW = 128;   // tiles of 32 keys
constexpr int NOPE = 448, VB = 576, SB = 8, NT = 256;
constexpr int SM_Q = 0;                          // q A fragments [32 steps][32 lanes] uint4      16384
constexpr int SM_KV = 16384;                     // 128 rows x 576 B, 8-byte pieces swizzled by key & 7    73728
constexpr int SM_SC = SM_KV + CH * VB;           // 128 x 8 scale bytes                                    1024
constexpr int SM_P = SM_SC + CH * SB;            // P A fragments [4 tiles][2 steps][32 lanes] uint4       4096
constexpr int SM_TMAX = SM_P + 4096;             // [4 tiles][4 n8][16 rows] fp32                          1024
constexpr int SM_WSUM = SM_TMAX + 1024;          // same shape: the warps' row sums                         1024
constexpr int SM_ROW = SM_WSUM + 1024;           // [4 tiles][16 rows] nm, alpha, active                    768
constexpr int SM_PHYS = SM_ROW + 768;            // 128 physical rows (int32, -1: masked)                   512
constexpr int SM_MISC = SM_PHYS + 512;           // final m, l [16] each + the ticket flag                  256
constexpr int SMEM = SM_MISC + 256;              // 98,816 B (sm_121: <= 101,376 a block)
static_assert(SMEM <= 101376, "shared memory");
constexpr unsigned FULL = 0xffffffffu;

struct Args {
    const uint16_t* q;                            // [R, H, 512] bf16 (RoPE'd)
    const uint8_t* cv; const uint8_t* csc; long long cvs, css;   // compressed rows (values, scales), row strides
    const int* tok; long long ts; const int* cnt; // selection [R, ts] int32 (-1 pad), counts [R]
    const uint8_t* sv; const uint8_t* ssc; long long svs, sss;   // SWA rings
    const int* lo; const int* hi;                 // [R] window starts; window ends (HAS_HI) or null
    const void* pos;                              // int32 [1] (POS + r) or int64 [R] (ROWS)
    const long long* sl;                          // ROWS: the rows' slots [R]
    const int* pt; long long pts; int psh;        // page table of the compressed rows (psh 0: contiguous)
    const float* sink; const float* cs; long long csst;          // [H] fp32, RoPE table fp32 [max_pos, 64]
    float* po; float* pm; float* pl;              // partials (attn.Scratch layout)
    uint16_t* out;                                // [R, H, 512] bf16
    int* ticket;                                  // [R, H / 16] (self-cleaning)
    int R, H, ring, has_comp;
    int qrope;                                    // split_kernel: q arrives un-rotated, RoPE'd as it loads (R1c)
};

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float div_full(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}
__device__ __forceinline__ float texp(float x) { return ex2(__fmul_rn(x, __int_as_float(0x3FB8AA3B))); }
__device__ __forceinline__ float pow2b(uint32_t s) { return __int_as_float(static_cast<int>(s) << 23); }

__device__ __forceinline__ void mma(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// two e4m3 bytes (low byte = the lower k) with their own scales -> bf16x2 (exact: e4m3 x 2^e is a bf16)
__device__ __forceinline__ uint32_t deq2(uint32_t two, float s0, float s1) {
    uint32_t h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(static_cast<unsigned short>(two)));
    const __half2 h = *reinterpret_cast<const __half2*>(&h2);
    const __nv_bfloat162 b = __floats2bfloat162_rn(__fmul_rn(__low2float(h), s0), __fmul_rn(__high2float(h), s1));
    return *reinterpret_cast<const uint32_t*>(&b);
}

// a byte offset x of row `key` in the staged rows: 8-byte pieces swizzled within groups of 8 (bank-conflict free)
__device__ __forceinline__ int swz(int key, int x) {
    const int ch = x >> 3;
    return key * VB + ((((ch & ~7) | ((ch & 7) ^ (key & 7)))) << 3) + (x & 7);
}

// Triton's dot-operand k order: step s (16 k) of a 32-k block takes k = klo(s), klo + 1 and khi(s), khi + 1
template <int KW> __device__ __forceinline__ int klo(int s, int t) {
    return KW == 4 ? 32 * (s >> 1) + 4 * t + 2 * (s & 1) : 16 * s + 2 * t;
}
template <int KW> __device__ __forceinline__ int khi(int s, int t) { return klo<KW>(s, t) + (KW == 4 ? 16 : 8); }
// where element k (< 32) of a row sits in a 2-step A fragment: step, quad lane, high half (+16 / +8)
template <int KW> __device__ __forceinline__ void frag_of(int k, int& step, int& tq, int& hi) {
    if (KW == 4) { step = (k >> 1) & 1; hi = (k >> 4) & 1; tq = (k & 15) >> 2; }
    else { step = k >> 4; hi = (k >> 3) & 1; tq = (k & 7) >> 1; }
}

// the bf16 pair (k, k + 1) of staged row `key` (S's B operand: k along the row)
__device__ __forceinline__ uint32_t pair_k(const uint8_t* sm, int key, int k) {
    if (k < NOPE) {
        const uint32_t two = *reinterpret_cast<const uint16_t*>(sm + SM_KV + swz(key, k));
        const float s = pow2b(sm[SM_SC + key * SB + (k >> 6)]);
        return deq2(two, s, s);
    }
    return *reinterpret_cast<const uint32_t*>(sm + SM_KV + swz(key, NOPE + 2 * (k - NOPE)));
}

// value (key, n) of the staged rows as a bf16 bit pattern / its e4m3 byte (PV's B operand: k along the keys)
__device__ __forceinline__ uint32_t pair_n(const uint8_t* sm, int k0, int k1, int n) {
    if (n < NOPE) {
        const uint32_t two = sm[SM_KV + swz(k0, n)] | (static_cast<uint32_t>(sm[SM_KV + swz(k1, n)]) << 8);
        return deq2(two, pow2b(sm[SM_SC + k0 * SB + (n >> 6)]), pow2b(sm[SM_SC + k1 * SB + (n >> 6)]));
    }
    const int x = NOPE + 2 * (n - NOPE);
    return *reinterpret_cast<const uint16_t*>(sm + SM_KV + swz(k0, x)) |
           (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(sm + SM_KV + swz(k1, x))) << 16);
}

__device__ __forceinline__ void cp8(uint32_t dst, const void* src, bool ok) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(dst), "l"(src), "r"(ok ? 8 : 0));
}

template <bool ROWS> __device__ __forceinline__ int row_pos(const Args& a, int r) {
    return ROWS ? static_cast<int>(static_cast<const long long*>(a.pos)[r]) : static_cast<const int*>(a.pos)[0] + r;
}

// how many chunks of row r have keys (the merge's ticket target; the others are Triton's empty partial)
__device__ __forceinline__ bool chunk_works(const Args& a, int r, int c) {
    return c == NCOMP || (a.has_comp && c * CH < a.cnt[r]);
}

template <bool ROWS>
__device__ void merge(const Args& a, uint8_t* sm, int r, int hb) {
    const int tid = threadIdx.x, H = a.H, R = a.R;
    float* ca = reinterpret_cast<float*>(sm + SM_TMAX);          // [5][16] a, [5][16] b, [16] af, [16] den
    float* cb = ca + NCH * BMQ;
    float* fa = cb + NCH * BMQ;
    float* fd = fa + BMQ;
    if (tid < BMQ) {                                             // a head's scalar chain (attn._merge, unrolled)
        const int h = hb * BMQ + tid;
        float m = -INFINITY, l = 0.f;
        for (int c = 0; c < NCH; ++c) {
            const bool w = chunk_works(a, r, c);
            const long long base = (static_cast<long long>(c) * R + r) * H + h;
            const float cm = w ? __ldcg(a.pm + base) : -INFINITY, cl = w ? __ldcg(a.pl + base) : 0.f;
            const bool act = cl > 0.f;
            float aa, bb;
            if (c == 0) {
                const float nm = act ? fmaxf(cm, -INFINITY) : -INFINITY;
                bb = act ? texp(__fsub_rn(cm, nm)) : 0.f;
                aa = 0.f;
                l = __fmaf_rn(bb, cl, 0.f);
                m = nm;
            } else {
                const float nm = act ? fmaxf(m, cm) : m;
                aa = act ? (m == -INFINITY ? 0.f : texp(__fsub_rn(m, nm))) : 1.f;
                bb = act ? texp(__fsub_rn(cm, nm)) : 0.f;
                l = __fmaf_rn(l, aa, __fmul_rn(bb, cl));
                m = nm;
            }
            ca[c * BMQ + tid] = aa;
            cb[c * BMQ + tid] = bb;
        }
        const float sink = a.sink[h];
        const float top = fmaxf(m, sink);
        const float af = m == -INFINITY ? 0.f : texp(__fsub_rn(m, top));
        const float e = sink == -INFINITY ? 0.f : texp(__fsub_rn(sink, top));
        fa[tid] = af;
        fd[tid] = __fmaf_rn(l, af, e);
    }
    __syncthreads();
    const int at = row_pos<ROWS>(a, r);
    const float* csr = a.cs + static_cast<long long>(at) * a.csst;
    for (int u = tid; u < BMQ * (D / 4); u += NT) {               // float4 units: head hl, columns d .. d + 3
        const int hl = u / (D / 4), d = (u % (D / 4)) * 4, h = hb * BMQ + hl;
        float o[4];
        for (int c = 0; c < NCH; ++c) {
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (chunk_works(a, r, c))
                v = __ldcg(reinterpret_cast<const float4*>(a.po + ((static_cast<long long>(c) * R + r) * H + h) * D
                                                            + d));
            const float co[4] = {v.x, v.y, v.z, v.w};
            const float aa = ca[c * BMQ + hl], bb = cb[c * BMQ + hl];
            for (int i = 0; i < 4; ++i)
                o[i] = c == 0 ? __fmaf_rn(bb, co[i], 0.f) : __fmaf_rn(o[i], aa, __fmul_rn(co[i], bb));
        }
        float x[4];
        for (int i = 0; i < 4; ++i) x[i] = div_full(__fmul_rn(o[i], fa[hl]), fd[hl]);
        uint16_t ob[4];
        for (int i = 0; i < 4; i += 2) {                         // pair (d + i, d + i + 1): GPT-J, inverse
            const int p = (d + i) >> 1;
            const float c = p >= 224 ? csr[p - 224] : 1.f, s = p >= 224 ? csr[32 + p - 224] : 0.f;
            const float ne = __fmaf_rn(x[i], c, __fmul_rn(x[i + 1], s));
            const float no = __fmaf_rn(x[i + 1], c, -__fmul_rn(x[i], s));
            const __nv_bfloat16 b0 = __float2bfloat16_rn(ne), b1 = __float2bfloat16_rn(no);
            ob[i] = *reinterpret_cast<const uint16_t*>(&b0);
            ob[i + 1] = *reinterpret_cast<const uint16_t*>(&b1);
        }
        uint2 w;
        w.x = ob[0] | (static_cast<uint32_t>(ob[1]) << 16);
        w.y = ob[2] | (static_cast<uint32_t>(ob[3]) << 16);
        *reinterpret_cast<uint2*>(a.out + (static_cast<long long>(r) * H + h) * D + d) = w;
    }
}

template <int KW, bool ROWS, bool HAS_HI>
__global__ void __launch_bounds__(NT, 1) attn_kernel(Args a) {
    extern __shared__ __align__(16) uint8_t sm[];
    const int c = blockIdx.x, hb = blockIdx.y, r = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int H = a.H, R = a.R;
    if (!chunk_works(a, r, c)) return;
    int* phys = reinterpret_cast<int*>(sm + SM_PHYS);
    const long long sl = ROWS ? a.sl[r] : 0;
    if (tid < CH) {                                              // the chunk's 128 keys -> physical rows (-1: masked)
        int row = -1;
        if (c < NCOMP) {
            const int idx = c * CH + tid;
            if (idx < a.cnt[r]) {
                const int tk = a.tok[static_cast<long long>(r) * a.ts + idx];
                if (tk >= 0) {
                    row = tk;
                    if (a.psh > 0) {
                        const int* ptr = a.pt + (ROWS ? sl * a.pts : 0);
                        row = (ptr[tk >> a.psh] << a.psh) + (tk & ((1 << a.psh) - 1));
                    }
                }
            }
        } else {
            const int hi = HAS_HI ? a.hi[r] : row_pos<ROWS>(a, r);
            const int lo = max(a.lo[r], hi - (WINDOW - 1));
            const int p = hi - (WINDOW - 1) + tid;
            if (p >= lo && p >= 0) row = (p % a.ring) + (ROWS ? static_cast<int>(sl) * a.ring : 0);
        }
        phys[tid] = row;
    }
    {                                                            // q -> A fragments (Triton's k order)
        const uint16_t* qb = a.q + (static_cast<long long>(r) * H + hb * BMQ) * D;
        for (int i = tid; i < 32 * 32; i += NT) {
            const int s = i >> 5, ln = i & 31, gg = ln >> 2, tt = ln & 3;
            const int k0 = klo<KW>(s, tt), k1 = khi<KW>(s, tt);
            uint4 f;
            f.x = *reinterpret_cast<const uint32_t*>(qb + gg * D + k0);
            f.y = *reinterpret_cast<const uint32_t*>(qb + (gg + 8) * D + k0);
            f.z = *reinterpret_cast<const uint32_t*>(qb + gg * D + k1);
            f.w = *reinterpret_cast<const uint32_t*>(qb + (gg + 8) * D + k1);
            reinterpret_cast<uint4*>(sm + SM_Q)[i] = f;
        }
    }
    __syncthreads();
    {                                                            // the rows: 72 value pieces + 1 scale piece a key
        const bool comp = c < NCOMP;
        const uint8_t* V = comp ? a.cv : a.sv;
        const uint8_t* S = comp ? a.csc : a.ssc;
        const long long vs = comp ? a.cvs : a.svs, ss = comp ? a.css : a.sss;
        const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(sm));
        for (int i = tid; i < CH * 73; i += NT) {
            const int key = i / 73, pc = i % 73, row = phys[key];
            const bool ok = row >= 0;
            const long long rr = ok ? row : 0;
            if (pc < 72) cp8(base + SM_KV + swz(key, pc * 8), V + rr * vs + pc * 8, ok);
            else cp8(base + SM_SC + key * SB, S + rr * ss, ok);
        }
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    }
    __syncthreads();
    const float NINF = -INFINITY;
    const int n8 = warp & 3, ta = warp >> 2;                     // this warp: S of tiles ta and ta + 2, block n8
    float sv[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    {
        const int kA = 32 * ta + 8 * n8 + g, kB = kA + 64;
        const uint4* qf = reinterpret_cast<const uint4*>(sm + SM_Q);
#pragma unroll 4
        for (int s = 0; s < 32; ++s) {
            const uint4 af = qf[s * 32 + lane];
            const int k0 = klo<KW>(s, t), k1 = khi<KW>(s, t);
            mma(sv[0], af, pair_k(sm, kA, k0), pair_k(sm, kA, k1));
            mma(sv[1], af, pair_k(sm, kB, k0), pair_k(sm, kB, k1));
        }
    }
    float* tmax = reinterpret_cast<float*>(sm + SM_TMAX);
    float* wsum = reinterpret_cast<float*>(sm + SM_WSUM);
    float* rowv = reinterpret_cast<float*>(sm + SM_ROW);        // [tile][row]: nm, alpha, active at 0 / 64 / 128
    bool val[2][2];
    for (int j = 0; j < 2; ++j) {
        const int tile = ta + 2 * j, kc = 32 * tile + 8 * n8 + 2 * t;
        val[j][0] = phys[kc] >= 0;
        val[j][1] = phys[kc + 1] >= 0;
        for (int e = 0; e < 4; ++e)
            sv[j][e] = val[j][e & 1] ? __fmul_rn(sv[j][e], __int_as_float(0x3D3504F3)) : NINF;
        float m0 = fmaxf(sv[j][0], sv[j][1]), m1 = fmaxf(sv[j][2], sv[j][3]);
        m0 = fmaxf(m0, __shfl_xor_sync(FULL, m0, 2));
        m0 = fmaxf(m0, __shfl_xor_sync(FULL, m0, 1));
        m1 = fmaxf(m1, __shfl_xor_sync(FULL, m1, 2));
        m1 = fmaxf(m1, __shfl_xor_sync(FULL, m1, 1));
        if (t == 0) {
            tmax[(tile * 4 + n8) * 16 + g] = m0;
            tmax[(tile * 4 + n8) * 16 + g + 8] = m1;
        }
    }
    __syncthreads();
    if (tid < BMQ) {                                             // the running max over the tiles, a row a thread
        float m = NINF;
        for (int tile = 0; tile < 4; ++tile) {
            const float* tm4 = tmax + tile * 64 + tid;
            const float tm = fmaxf(fmaxf(tm4[0], tm4[16]), fmaxf(tm4[32], tm4[48]));
            const bool act = tm != NINF;
            const float nm = act ? fmaxf(m, tm) : m;
            const float al = act ? (m == NINF ? 0.f : texp(__fsub_rn(m, nm))) : 1.f;
            rowv[tile * 16 + tid] = nm;
            rowv[64 + tile * 16 + tid] = al;
            rowv[128 + tile * 16 + tid] = act ? 1.f : 0.f;
            m = nm;
        }
        reinterpret_cast<float*>(sm + SM_MISC)[tid] = m;
    }
    __syncthreads();
    uint32_t* pf = reinterpret_cast<uint32_t*>(sm + SM_P);       // [tile][step][lane][4]
    for (int j = 0; j < 2; ++j) {                                // p, the thread / quad sums, bf16(p) -> fragments
        const int tile = ta + 2 * j;
        float p[4];
        for (int e = 0; e < 4; ++e) {
            const int row = g + (e >> 1) * 8;
            const bool act = rowv[128 + tile * 16 + row] != 0.f;
            p[e] = (val[j][e & 1] && act) ? texp(__fsub_rn(sv[j][e], rowv[tile * 16 + row])) : 0.f;
        }
        float a0 = __fadd_rn(p[0], p[1]), a1 = __fadd_rn(p[2], p[3]);
        a0 = __fadd_rn(a0, __shfl_xor_sync(FULL, a0, 2));
        a0 = __fadd_rn(a0, __shfl_xor_sync(FULL, a0, 1));
        a1 = __fadd_rn(a1, __shfl_xor_sync(FULL, a1, 2));
        a1 = __fadd_rn(a1, __shfl_xor_sync(FULL, a1, 1));
        if (t == 0) {
            wsum[(tile * 4 + n8) * 16 + g] = a0;
            wsum[(tile * 4 + n8) * 16 + g + 8] = a1;
        }
        int step, tq, hi;
        frag_of<KW>(8 * n8 + 2 * t, step, tq, hi);
        const __nv_bfloat162 lo2 = __floats2bfloat162_rn(p[0], p[1]), hi2 = __floats2bfloat162_rn(p[2], p[3]);
        uint32_t* f = pf + ((tile * 2 + step) * 32 + g * 4 + tq) * 4;
        f[2 * hi] = *reinterpret_cast<const uint32_t*>(&lo2);    // reg 0 / 2: row g; 1 / 3: row g + 8
        f[2 * hi + 1] = *reinterpret_cast<const uint32_t*>(&hi2);
    }
    __syncthreads();
    if (tid < BMQ) {                                             // l: fma(l, alpha, (W0 + W2) + (W1 + W3))
        float l = 0.f;
        for (int tile = 0; tile < 4; ++tile) {
            const float* w4 = wsum + tile * 64 + tid;
            const float sum = __fadd_rn(__fadd_rn(w4[0], w4[32]), __fadd_rn(w4[16], w4[48]));
            l = __fmaf_rn(l, rowv[64 + tile * 16 + tid], sum);
        }
        reinterpret_cast<float*>(sm + SM_MISC)[16 + tid] = l;
    }
    float acc[8][4];
    for (int j = 0; j < 8; ++j)
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.f;
    for (int tile = 0; tile < 4; ++tile) {                       // o = mma(P, kv, rn(o * alpha)), 2 k steps a tile
        const float al0 = rowv[64 + tile * 16 + g], al1 = rowv[64 + tile * 16 + g + 8];
        for (int j = 0; j < 8; ++j) {
            acc[j][0] = __fmul_rn(acc[j][0], al0);
            acc[j][1] = __fmul_rn(acc[j][1], al0);
            acc[j][2] = __fmul_rn(acc[j][2], al1);
            acc[j][3] = __fmul_rn(acc[j][3], al1);
        }
        for (int step = 0; step < 2; ++step) {
            const uint4 af = reinterpret_cast<const uint4*>(pf)[(tile * 2 + step) * 32 + lane];
            const int k0 = 32 * tile + klo<KW>(step, t), k1 = 32 * tile + khi<KW>(step, t);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int n = 64 * j + 8 * warp + g;
                mma(acc[j], af, pair_n(sm, k0, k0 + 1, n), pair_n(sm, k1, k1 + 1, n));
            }
        }
    }
    const long long pbase = (static_cast<long long>(c) * R + r) * H + hb * BMQ;
    for (int j = 0; j < 8; ++j) {
        const int n = 64 * j + 8 * warp + 2 * t;
        *reinterpret_cast<float2*>(a.po + (pbase + g) * D + n) = make_float2(acc[j][0], acc[j][1]);
        *reinterpret_cast<float2*>(a.po + (pbase + g + 8) * D + n) = make_float2(acc[j][2], acc[j][3]);
    }
    if (tid < BMQ) {
        a.pm[pbase + tid] = reinterpret_cast<float*>(sm + SM_MISC)[tid];
        a.pl[pbase + tid] = reinterpret_cast<float*>(sm + SM_MISC)[16 + tid];
    }
    int* flag = reinterpret_cast<int*>(sm + SM_MISC) + 32;
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        int need = 0;
        for (int cc = 0; cc < NCH; ++cc) need += chunk_works(a, r, cc) ? 1 : 0;
        *flag = atomicAdd(a.ticket + r * (H / BMQ) + hb, 1) == need - 1;
    }
    __syncthreads();
    if (!*flag) return;
    __threadfence();
    if (tid == 0) atomicExch(a.ticket + r * (H / BMQ) + hb, 0);  // self-cleaning: the next launch / replay
    merge<ROWS>(a, sm, r, hb);
}

template <int KW, bool ROWS, bool HAS_HI>
cudaError_t launch_one(const Args& a, cudaStream_t stream) {
    static bool init = false;
    if (!init) {
        const cudaError_t e = cudaFuncSetAttribute(attn_kernel<KW, ROWS, HAS_HI>,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        if (e != cudaSuccess) return e;
        init = true;
    }
    attn_kernel<KW, ROWS, HAS_HI><<<dim3(NCH, a.H / BMQ, a.R), NT, SMEM, stream>>>(a);
    return cudaGetLastError();
}

// ---- R1b: the same work split over a CLUSTER of P CTAs a chunk (TF_DSV41_ATTN_SPLIT=P, P = 1 / 4) ----------------
// attn_kernel's time is one CTA's dependent chains: S (16 chains of 32 mma a CTA, 2 a warp), PV (8 n8 blocks a warp)
// and the last CTA's merge of 16 x 512 outputs (one L2 round trip a chunk a unit). split_kernel, rank k of a chunk's
// P CTAs (cluster dims (P, 1, 1)):
//   S of the chunk's tiles [TPC k, TPC k + TPC) (TPC = 4 / P; every chain whole: one (16 rows x 8 keys) block over
//   K = 512 in the same k order); the other ranks' tile maxima, then their row sums and bf16(p) fragments, come over
//   DSMEM (two cluster barriers), so the running max / alpha / l chains run whole and the same in every rank;
//   PV of the columns [W k, W k + W) (W = 512 / P: n8 blocks j in [J k, J k + J) of every warp, each chain whole over
//   the 4 tiles); the partials of those columns; a ticket per (row, head tile, rank): the last rank-k CTA of the row's
//   chunks merges the columns [W k, W k + W), every load issued before its chains (the proto's fast merge).
// Each output element keeps attn_kernel's operations in the same order (row map, mask, scale, maxima, the ex2 / fma
// chains, the bf16 roundings, the merge's chain): the SAME bits at any P. Loads: 16-byte cp.async pieces when the row
// pointers and strides allow (two 8-byte halves otherwise), swizzled per 16 bytes by fsw(key): conflict-free for S's
// 8 consecutive keys and PV's 4 keys (4t + c at kWidth 4, 2t + c at kWidth 2). P = 1 is attn_kernel with these loads
// and the fast merge, no cluster.
constexpr int MAXP = 4;                          // the ticket's ranks a (row, head tile)

__host__ __device__ constexpr int col2byte(int n) { return n <= NOPE ? n : NOPE + 2 * (n - NOPE); }

template <int P> struct Split {
    static constexpr int TPC = 4 / P;            // S tiles a CTA
    static constexpr int KS = 32 * TPC;          // S keys a CTA
    static constexpr int W = D / P;              // PV / merge columns a CTA
    static constexpr int J = 8 / P;              // n8 blocks a warp in PV
    static constexpr int NPW = TPC == 4 ? 2 : 1; // S chains a warp ((tile, n8) pairs: 4 TPC over 8 warps)
    static constexpr int SLB = P == 1 ? VB : col2byte(D) - col2byte(D - W);   // slice pitch (the widest rank's)
    static constexpr int SM_Q = 0;                           // q A fragments                                 16384
    static constexpr int SM_KP = 16384;                      // PV slice: 128 keys x SLB (P = 1: the whole rows)
    static constexpr int SM_KS = P == 1 ? SM_KP : SM_KP + CH * SLB;   // S rows: KS keys x 576 (P = 1: the slice)
    static constexpr int SM_SC = SM_KS + KS * VB;            // 128 keys x 8 scale bytes                       1024
    static constexpr int SM_KB = SM_SC + CH * SB;            // P > 1: S's B operands as bf16 pairs [KS][256] words
    static constexpr int SM_P = SM_KB + (P == 1 ? 0 : KS * D * 2);   // P A fragments [4][2][32] uint4         4096
    static constexpr int SM_TMAX = SM_P + 4096;              // [4 tiles][4 n8][16 rows] fp32 (merge: ca, cb, ...)
    static constexpr int SM_WSUM = SM_TMAX + 1024;
    static constexpr int SM_ROW = SM_WSUM + 1024;            // [4 tiles][16 rows] nm, alpha, active
    static constexpr int SM_PHYS = SM_ROW + 768;             // 128 physical rows (-1: masked)
    static constexpr int SM_MISC = SM_PHYS + 512;            // final m, l [16] each + the ticket flag
    static constexpr int SMEM = SM_MISC + 256;               // P = 1: 98,816 B; P = 4: 100,864 B
    static constexpr int SM_VB = SM_KB;                      // P > 1, after S: PV's B pairs [64 key pairs][W] words
    static_assert(P == 1 || (CH / 2) * W * 4 <= KS * D * 2, "PV's pairs fit where S's were");
    static_assert(SMEM <= 101376, "shared memory");
    static_assert(SLB % 64 == 0 && W % 128 == 0, "16-byte swizzle groups; whole n8 blocks a warp");
};

__device__ __forceinline__ int fsw(int key) { return ((key >> 1) ^ (key >> 3)) & 3; }
__device__ __forceinline__ int swz16(int key, int x, int pitch) {
    const int c = x >> 4;
    return key * pitch + (((c & ~3) | ((c & 3) ^ fsw(key))) << 4) + (x & 15);
}

// S's B pair (k, k + 1) of the S row `key` (local to the S rows; `ck`: its chunk key, for the scale)
template <int P>
__device__ __forceinline__ uint32_t spair_k(const uint8_t* sm, int key, int ck, int k) {
    using G = Split<P>;
    if (k < NOPE) {
        const uint32_t two = *reinterpret_cast<const uint16_t*>(sm + G::SM_KS + swz16(key, k, VB));
        const float s = pow2b(sm[G::SM_SC + ck * SB + (k >> 6)]);
        return deq2(two, s, s);
    }
    return *reinterpret_cast<const uint32_t*>(sm + G::SM_KS + swz16(key, NOPE + 2 * (k - NOPE), VB));
}

// P > 1: S's B operands dequantized once into bf16 pairs, word w = k / 2 of S row `key` at w ^ bsw(key): a warp's 8
// keys x 4 quad lanes hit 32 distinct banks (the quad lanes vary w's bits 1-2 at kWidth 4, bits 0-1 at kWidth 2)
template <int KW> __device__ __forceinline__ int bsw(int key) {
    return (KW == 4 ? (key & 1) : ((key & 1) << 2)) | (((key >> 1) & 3) << 3);
}
template <int KW, int P>
__device__ __forceinline__ uint32_t sbf_k(const uint8_t* sm, int key, int k) {
    return reinterpret_cast<const uint32_t*>(sm + Split<P>::SM_KB)[key * (D / 2) + ((k >> 1) ^ bsw<KW>(key))];
}

// PV's B pair: value (k0, n), (k1, n) of the slice (byte x0: the rank's first byte)
template <int P>
__device__ __forceinline__ uint32_t spair_n(const uint8_t* sm, int k0, int k1, int n, int x0) {
    using G = Split<P>;
    if (n < NOPE) {
        const int x = n - x0;
        const uint32_t two = sm[G::SM_KP + swz16(k0, x, G::SLB)] |
                             (static_cast<uint32_t>(sm[G::SM_KP + swz16(k1, x, G::SLB)]) << 8);
        return deq2(two, pow2b(sm[G::SM_SC + k0 * SB + (n >> 6)]), pow2b(sm[G::SM_SC + k1 * SB + (n >> 6)]));
    }
    const int x = col2byte(n) - x0;
    return *reinterpret_cast<const uint16_t*>(sm + G::SM_KP + swz16(k0, x, G::SLB)) |
           (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(sm + G::SM_KP + swz16(k1, x, G::SLB))) << 16);
}

// P > 1: PV's B pairs dequantized once, word (key pair kp, column x) at column x ^ vsw(kp): a warp's 8 columns x 4
// quad lanes hit 32 distinct banks (the quad lanes vary kp's bits 1-2 at kWidth 4, bits 0-1 at kWidth 2)
template <int KW> __device__ __forceinline__ int vsw(int kp) { return (KW == 4 ? (kp >> 1) & 3 : kp & 3) << 3; }

// R1c (TF_DSV41_ATTN_ROPE): rope_fused._rope's pair, in place, on q's word (k, k + 1) = (even, odd) of rope pair j:
// (rn(e c) - rn(o s), rn(o c) + rn(e s)) with no contraction, then bf16 (nearest even); cos / sin the table's row
__device__ __forceinline__ uint32_t rope_word(uint32_t w, const float* csr, int j) {
    const float e = __uint_as_float(w << 16), o = __uint_as_float(w & 0xffff0000u);
    const float c = csr[j], sn = csr[32 + j];
    const __nv_bfloat162 b = __floats2bfloat162_rn(__fsub_rn(__fmul_rn(e, c), __fmul_rn(o, sn)),
                                                   __fadd_rn(__fmul_rn(o, c), __fmul_rn(e, sn)));
    return *reinterpret_cast<const uint32_t*>(&b);
}

__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ uint32_t cl_rank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}
// store word v at shared address `p` of cluster rank q (DSMEM)
__device__ __forceinline__ void st_peer(void* p, uint32_t q, uint32_t v) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p)), b;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(b) : "r"(a), "r"(q));
    asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(b), "r"(v) : "memory");
}
// word i of shared array `p` in cluster rank q (DSMEM)
__device__ __forceinline__ uint32_t ld_peer(const void* p, uint32_t q) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p)), b, v;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(b) : "r"(a), "r"(q));
    asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(b) : "memory");
    return v;
}

// stage row `row` bytes [b0, b0 + 16 n) into a region of pitch `pitch` at key slot `key` (16-byte pieces, or 8-byte
// halves when `v16` is off); masked rows zero-fill (Triton's other=0)
__device__ __forceinline__ void stage_piece(uint32_t base, int region, int key, int pitch, int pc, const uint8_t* src,
                                            bool ok, bool v16) {
    const uint32_t dst = base + region + swz16(key, pc * 16, pitch);
    if (v16) {
        cp16(dst, src, ok);
    } else {
        cp8(dst, src, ok);
        cp8(dst + 8, src + 8, ok);
    }
}

// the merge of columns [c0, c0 + NC) of (row r, head tile hb): attn_kernel's merge per element, loads first
template <bool ROWS, int NC>
__device__ void merge_cols(const Args& a, uint8_t* sm, int smt, int r, int hb, int c0) {
    const int tid = threadIdx.x, H = a.H, R = a.R;
    float* ca = reinterpret_cast<float*>(sm + smt);              // [5][16] a, [5][16] b, [16] af, [16] den
    float* cb = ca + NCH * BMQ;
    float* fa = cb + NCH * BMQ;
    float* fd = fa + BMQ;
    bool cw[NCH];
#pragma unroll
    for (int c = 0; c < NCH; ++c) cw[c] = chunk_works(a, r, c);
    constexpr int UB = BMQ * (NC / 4) / NT;                      // float4 units a thread (8 at P = 1, 2 at P = 4)
    static_assert(UB * NT == BMQ * (NC / 4), "whole units");
    float4 pv[UB][NCH];
#pragma unroll
    for (int ub = 0; ub < UB; ++ub) {                            // the partials in flight first
        const int u = tid + ub * NT, hl = u / (NC / 4), d = c0 + (u % (NC / 4)) * 4, h = hb * BMQ + hl;
#pragma unroll
        for (int c = 0; c < NCH; ++c)
            pv[ub][c] = cw[c] ? __ldcg(reinterpret_cast<const float4*>(
                                    a.po + ((static_cast<long long>(c) * R + r) * H + h) * D + d))
                              : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    if (tid < BMQ) {                                             // a head's scalar chain (attn._merge, unrolled)
        const int h = hb * BMQ + tid;
        float pmv[NCH], plv[NCH];
#pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const long long base = (static_cast<long long>(c) * R + r) * H + h;
            pmv[c] = cw[c] ? __ldcg(a.pm + base) : -INFINITY;
            plv[c] = cw[c] ? __ldcg(a.pl + base) : 0.f;
        }
        const float sink = a.sink[h];
        float m = -INFINITY, l = 0.f;
#pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const float cm = pmv[c], cl = plv[c];
            const bool act = cl > 0.f;
            float aa, bb;
            if (c == 0) {
                const float nm = act ? fmaxf(cm, -INFINITY) : -INFINITY;
                bb = act ? texp(__fsub_rn(cm, nm)) : 0.f;
                aa = 0.f;
                l = __fmaf_rn(bb, cl, 0.f);
                m = nm;
            } else {
                const float nm = act ? fmaxf(m, cm) : m;
                aa = act ? (m == -INFINITY ? 0.f : texp(__fsub_rn(m, nm))) : 1.f;
                bb = act ? texp(__fsub_rn(cm, nm)) : 0.f;
                l = __fmaf_rn(l, aa, __fmul_rn(bb, cl));
                m = nm;
            }
            ca[c * BMQ + tid] = aa;
            cb[c * BMQ + tid] = bb;
        }
        const float top = fmaxf(m, sink);
        const float af = m == -INFINITY ? 0.f : texp(__fsub_rn(m, top));
        const float e = sink == -INFINITY ? 0.f : texp(__fsub_rn(sink, top));
        fa[tid] = af;
        fd[tid] = __fmaf_rn(l, af, e);
    }
    __syncthreads();
    const float* csr = a.cs + static_cast<long long>(row_pos<ROWS>(a, r)) * a.csst;
#pragma unroll
    for (int ub = 0; ub < UB; ++ub) {
        const int u = tid + ub * NT, hl = u / (NC / 4), d = c0 + (u % (NC / 4)) * 4, h = hb * BMQ + hl;
        float o[4];
#pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const float4 v = pv[ub][c];
            const float co[4] = {v.x, v.y, v.z, v.w};
            const float aa = ca[c * BMQ + hl], bb = cb[c * BMQ + hl];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                o[i] = c == 0 ? __fmaf_rn(bb, co[i], 0.f) : __fmaf_rn(o[i], aa, __fmul_rn(co[i], bb));
        }
        float x[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = div_full(__fmul_rn(o[i], fa[hl]), fd[hl]);
        uint16_t ob[4];
#pragma unroll
        for (int i = 0; i < 4; i += 2) {                         // pair (d + i, d + i + 1): GPT-J, inverse
            const int p = (d + i) >> 1;
            const float c = p >= 224 ? csr[p - 224] : 1.f, s = p >= 224 ? csr[32 + p - 224] : 0.f;
            const float ne = __fmaf_rn(x[i], c, __fmul_rn(x[i + 1], s));
            const float no = __fmaf_rn(x[i + 1], c, -__fmul_rn(x[i], s));
            const __nv_bfloat16 b0 = __float2bfloat16_rn(ne), b1 = __float2bfloat16_rn(no);
            ob[i] = *reinterpret_cast<const uint16_t*>(&b0);
            ob[i + 1] = *reinterpret_cast<const uint16_t*>(&b1);
        }
        uint2 w;
        w.x = ob[0] | (static_cast<uint32_t>(ob[1]) << 16);
        w.y = ob[2] | (static_cast<uint32_t>(ob[3]) << 16);
        *reinterpret_cast<uint2*>(a.out + (static_cast<long long>(r) * H + h) * D + d) = w;
    }
}

template <int KW, bool ROWS, bool HAS_HI, int P>
__global__ void __launch_bounds__(NT, 1) split_kernel(Args a) {
    using G = Split<P>;
    extern __shared__ __align__(16) uint8_t sm[];
    const int k = P == 1 ? 0 : static_cast<int>(cl_rank());
    const int c = blockIdx.x / P, hb = blockIdx.y, r = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int H = a.H, R = a.R;
    if (!chunk_works(a, r, c)) return;                           // the whole cluster: no barrier is pending
    uint4 qv[32 * 32 / NT];                                      // q's A fragments: every load in flight at once
    {
        const uint16_t* qb = a.q + (static_cast<long long>(r) * H + hb * BMQ) * D;
#pragma unroll
        for (int u = 0; u < 32 * 32 / NT; ++u) {
            const int i = tid + u * NT, s = i >> 5, ln = i & 31, gg = ln >> 2, tt = ln & 3;
            const int k0 = klo<KW>(s, tt), k1 = khi<KW>(s, tt);
            qv[u].x = *reinterpret_cast<const uint32_t*>(qb + gg * D + k0);
            qv[u].y = *reinterpret_cast<const uint32_t*>(qb + (gg + 8) * D + k0);
            qv[u].z = *reinterpret_cast<const uint32_t*>(qb + gg * D + k1);
            qv[u].w = *reinterpret_cast<const uint32_t*>(qb + (gg + 8) * D + k1);
        }
        if (a.qrope) {                                           // the rope dims (k >= 448): their pair's rotation
            const float* csr = a.cs + static_cast<long long>(row_pos<ROWS>(a, r)) * a.csst;
#pragma unroll
            for (int u = 0; u < 32 * 32 / NT; ++u) {
                const int i = tid + u * NT, s = i >> 5, tt = (i & 31) & 3;
                const int k0 = klo<KW>(s, tt), k1 = khi<KW>(s, tt);
                if (k0 >= NOPE) {
                    qv[u].x = rope_word(qv[u].x, csr, (k0 - NOPE) >> 1);
                    qv[u].y = rope_word(qv[u].y, csr, (k0 - NOPE) >> 1);
                }
                if (k1 >= NOPE) {
                    qv[u].z = rope_word(qv[u].z, csr, (k1 - NOPE) >> 1);
                    qv[u].w = rope_word(qv[u].w, csr, (k1 - NOPE) >> 1);
                }
            }
        }
    }
    int* phys = reinterpret_cast<int*>(sm + G::SM_PHYS);
    const long long sl = ROWS ? a.sl[r] : 0;
    if (tid < CH) {                                              // attn_kernel's row map
        int row = -1;
        if (c < NCOMP) {
            const int idx = c * CH + tid;
            if (idx < a.cnt[r]) {
                const int tk = a.tok[static_cast<long long>(r) * a.ts + idx];
                if (tk >= 0) {
                    row = tk;
                    if (a.psh > 0) {
                        const int* ptr = a.pt + (ROWS ? sl * a.pts : 0);
                        row = (ptr[tk >> a.psh] << a.psh) + (tk & ((1 << a.psh) - 1));
                    }
                }
            }
        } else {
            const int hi = HAS_HI ? a.hi[r] : row_pos<ROWS>(a, r);
            const int lo = max(a.lo[r], hi - (WINDOW - 1));
            const int p = hi - (WINDOW - 1) + tid;
            if (p >= lo && p >= 0) row = (p % a.ring) + (ROWS ? static_cast<int>(sl) * a.ring : 0);
        }
        phys[tid] = row;
    }
    __syncthreads();
    {                                                            // the rows in flight before q's fragments
        const bool comp = c < NCOMP;
        const uint8_t* V = comp ? a.cv : a.sv;
        const uint8_t* S = comp ? a.csc : a.ssc;
        const long long vs = comp ? a.cvs : a.svs, ss = comp ? a.css : a.sss;
        const bool v16 = ((reinterpret_cast<uintptr_t>(V) | static_cast<uintptr_t>(vs)) & 15) == 0;
        const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(sm));
        const int x0 = col2byte(G::W * k), nsl = (col2byte(G::W * k + G::W) - x0) / 16;   // the slice's pieces a key
        for (int i = tid; i < CH * nsl; i += NT) {
            const int key = i / nsl, pc = i % nsl, row = phys[key];
            const long long rr = row >= 0 ? row : 0;
            stage_piece(base, G::SM_KP, key, G::SLB, pc, V + rr * vs + x0 + pc * 16, row >= 0, v16);
        }
        if (P > 1) {                                             // the S tiles' whole rows
            for (int i = tid; i < G::KS * (VB / 16); i += NT) {
                const int key = i / (VB / 16), pc = i % (VB / 16), row = phys[G::KS * k + key];
                const long long rr = row >= 0 ? row : 0;
                stage_piece(base, G::SM_KS, key, VB, pc, V + rr * vs + pc * 16, row >= 0, v16);
            }
        }
        for (int key = tid; key < CH; key += NT) {
            const int row = phys[key];
            cp8(base + G::SM_SC + key * SB, S + (row >= 0 ? row : 0) * ss, row >= 0);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
#pragma unroll
    for (int u = 0; u < 32 * 32 / NT; ++u) reinterpret_cast<uint4*>(sm + G::SM_Q)[tid + u * NT] = qv[u];
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if (P > 1) {                                                 // S's operands -> bf16 pairs, 4 k a step (deq2)
        uint32_t* kb = reinterpret_cast<uint32_t*>(sm + G::SM_KB);
        for (int i = tid; i < G::KS * (D / 4); i += NT) {
            const int key = i / (D / 4), k4 = (i % (D / 4)) * 4, w = k4 >> 1;
            uint32_t w0, w1;
            if (k4 < NOPE) {
                const uint32_t b = *reinterpret_cast<const uint32_t*>(sm + G::SM_KS + swz16(key, k4, VB));
                const float sc = pow2b(sm[G::SM_SC + (key + G::KS * k) * SB + (k4 >> 6)]);
                w0 = deq2(b & 0xffffu, sc, sc);
                w1 = deq2(b >> 16, sc, sc);
            } else {
                const uint2 b = *reinterpret_cast<const uint2*>(sm + G::SM_KS + swz16(key, NOPE + 2 * (k4 - NOPE), VB));
                w0 = b.x;
                w1 = b.y;
            }
            kb[key * (D / 2) + (w ^ bsw<KW>(key))] = w0;
            kb[key * (D / 2) + ((w + 1) ^ bsw<KW>(key))] = w1;
        }
        __syncthreads();
    }
    const float NINF = -INFINITY;
    float* tmax = reinterpret_cast<float*>(sm + G::SM_TMAX);
    float* wsum = reinterpret_cast<float*>(sm + G::SM_WSUM);
    float* rowv = reinterpret_cast<float*>(sm + G::SM_ROW);
    // S: chain pair w (+ 8): local tile w >> 2, block n8 = w & 3 (P = 1: attn_kernel's tiles ta and ta + 2)
    const int n8 = warp & 3;
    const bool sw = warp < 4 * G::TPC;                           // this warp has S chains
    float sv[G::NPW][4];
    bool val[G::NPW][2];
    int tile_of[G::NPW];
#pragma unroll
    for (int j = 0; j < G::NPW; ++j) {
        tile_of[j] = G::TPC * k + (warp >> 2) + 2 * j;
        for (int e = 0; e < 4; ++e) sv[j][e] = 0.f;
    }
    if (sw) {
        const uint4* qf = reinterpret_cast<const uint4*>(sm + G::SM_Q);
        int lk[G::NPW];
#pragma unroll
        for (int j = 0; j < G::NPW; ++j) lk[j] = 32 * (tile_of[j] - G::TPC * k) + 8 * n8 + g;
#pragma unroll 4
        for (int s = 0; s < 32; ++s) {
            const uint4 af = qf[s * 32 + lane];
            const int k0 = klo<KW>(s, t), k1 = khi<KW>(s, t);
#pragma unroll
            for (int j = 0; j < G::NPW; ++j) {
                if (P == 1) {
                    mma(sv[j], af, spair_k<P>(sm, lk[j], lk[j], k0), spair_k<P>(sm, lk[j], lk[j], k1));
                } else {
                    mma(sv[j], af, sbf_k<KW, P>(sm, lk[j], k0), sbf_k<KW, P>(sm, lk[j], k1));
                }
            }
        }
#pragma unroll
        for (int j = 0; j < G::NPW; ++j) {
            const int tile = tile_of[j], kc = 32 * tile + 8 * n8 + 2 * t;
            val[j][0] = phys[kc] >= 0;
            val[j][1] = phys[kc + 1] >= 0;
            for (int e = 0; e < 4; ++e)
                sv[j][e] = val[j][e & 1] ? __fmul_rn(sv[j][e], __int_as_float(0x3D3504F3)) : NINF;
            float m0 = fmaxf(sv[j][0], sv[j][1]), m1 = fmaxf(sv[j][2], sv[j][3]);
            m0 = fmaxf(m0, __shfl_xor_sync(FULL, m0, 2));
            m0 = fmaxf(m0, __shfl_xor_sync(FULL, m0, 1));
            m1 = fmaxf(m1, __shfl_xor_sync(FULL, m1, 2));
            m1 = fmaxf(m1, __shfl_xor_sync(FULL, m1, 1));
            if (t == 0) {
                tmax[(tile * 4 + n8) * 16 + g] = m0;
                tmax[(tile * 4 + n8) * 16 + g + 8] = m1;
            }
        }
    }
    if (P > 1) {                                                 // this rank's tile maxima pushed to the others
        __syncthreads();
        for (int i = tid; i < 64 * G::TPC * (P - 1); i += NT) {
            const int e = 64 * G::TPC * k + i % (64 * G::TPC), q = (k + 1 + i / (64 * G::TPC)) % P;
            st_peer(tmax + e, q, __float_as_uint(tmax[e]));
        }
        cl_arrive();
        const int x0 = col2byte(G::W * k);                       // meanwhile: PV's B pairs, the fp8 slice -> bf16,
        uint32_t* vb = reinterpret_cast<uint32_t*>(sm + G::SM_VB);   // 4 columns a step (deq2 per column)
        for (int i = tid; i < (CH / 2) * (G::W / 4); i += NT) {
            const int kp = i / (G::W / 4), x = (i % (G::W / 4)) * 4, n = G::W * k + x, k0 = 2 * kp, k1 = k0 + 1;
            uint32_t o[4];
            if (n < NOPE) {
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(sm + G::SM_KP + swz16(k0, n - x0, G::SLB));
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(sm + G::SM_KP + swz16(k1, n - x0, G::SLB));
                const float s0 = pow2b(sm[G::SM_SC + k0 * SB + (n >> 6)]), s1 = pow2b(sm[G::SM_SC + k1 * SB + (n >> 6)]);
#pragma unroll
                for (int e = 0; e < 4; ++e) o[e] = deq2(((b0 >> (8 * e)) & 0xffu) | (((b1 >> (8 * e)) & 0xffu) << 8), s0, s1);
            } else {
                const uint2 a0 = *reinterpret_cast<const uint2*>(sm + G::SM_KP + swz16(k0, col2byte(n) - x0, G::SLB));
                const uint2 a1 = *reinterpret_cast<const uint2*>(sm + G::SM_KP + swz16(k1, col2byte(n) - x0, G::SLB));
                o[0] = (a0.x & 0xffffu) | (a1.x << 16);
                o[1] = (a0.x >> 16) | (a1.x & 0xffff0000u);
                o[2] = (a0.y & 0xffffu) | (a1.y << 16);
                o[3] = (a0.y >> 16) | (a1.y & 0xffff0000u);
            }
#pragma unroll
            for (int e = 0; e < 4; ++e) vb[kp * G::W + ((x + e) ^ vsw<KW>(kp))] = o[e];
        }
        cl_wait();
    }
    __syncthreads();
    if (tid < BMQ) {                                             // the running max over the tiles, a row a thread
        float m = NINF;
        for (int tile = 0; tile < 4; ++tile) {
            const float* tm4 = tmax + tile * 64 + tid;
            const float tm = fmaxf(fmaxf(tm4[0], tm4[16]), fmaxf(tm4[32], tm4[48]));
            const bool act = tm != NINF;
            const float nm = act ? fmaxf(m, tm) : m;
            const float al = act ? (m == NINF ? 0.f : texp(__fsub_rn(m, nm))) : 1.f;
            rowv[tile * 16 + tid] = nm;
            rowv[64 + tile * 16 + tid] = al;
            rowv[128 + tile * 16 + tid] = act ? 1.f : 0.f;
            m = nm;
        }
        reinterpret_cast<float*>(sm + G::SM_MISC)[tid] = m;
    }
    __syncthreads();
    uint32_t* pf = reinterpret_cast<uint32_t*>(sm + G::SM_P);    // [tile][step][lane][4]
    if (sw) {
#pragma unroll
        for (int j = 0; j < G::NPW; ++j) {                       // p, the quad sums, bf16(p) -> fragments
            const int tile = tile_of[j];
            float p[4];
            for (int e = 0; e < 4; ++e) {
                const int row = g + (e >> 1) * 8;
                const bool act = rowv[128 + tile * 16 + row] != 0.f;
                p[e] = (val[j][e & 1] && act) ? texp(__fsub_rn(sv[j][e], rowv[tile * 16 + row])) : 0.f;
            }
            float a0 = __fadd_rn(p[0], p[1]), a1 = __fadd_rn(p[2], p[3]);
            a0 = __fadd_rn(a0, __shfl_xor_sync(FULL, a0, 2));
            a0 = __fadd_rn(a0, __shfl_xor_sync(FULL, a0, 1));
            a1 = __fadd_rn(a1, __shfl_xor_sync(FULL, a1, 2));
            a1 = __fadd_rn(a1, __shfl_xor_sync(FULL, a1, 1));
            if (t == 0) {
                wsum[(tile * 4 + n8) * 16 + g] = a0;
                wsum[(tile * 4 + n8) * 16 + g + 8] = a1;
            }
            int step, tq, hi;
            frag_of<KW>(8 * n8 + 2 * t, step, tq, hi);
            const __nv_bfloat162 lo2 = __floats2bfloat162_rn(p[0], p[1]), hi2 = __floats2bfloat162_rn(p[2], p[3]);
            uint32_t* f = pf + ((tile * 2 + step) * 32 + g * 4 + tq) * 4;
            f[2 * hi] = *reinterpret_cast<const uint32_t*>(&lo2);
            f[2 * hi + 1] = *reinterpret_cast<const uint32_t*>(&hi2);
        }
    }
    if (P > 1) {                                                 // this rank's row sums and fragments pushed
        __syncthreads();
        constexpr int NW = G::TPC * (64 + 256);                  // words of this rank's tiles
        for (int i = tid; i < NW * (P - 1); i += NT) {
            const int e = i % NW, q = (k + 1 + i / NW) % P;
            uint32_t* w = e < 64 * G::TPC ? reinterpret_cast<uint32_t*>(wsum) + 64 * G::TPC * k + e
                                          : pf + 256 * G::TPC * k + (e - 64 * G::TPC);
            st_peer(w, q, *w);
        }
        cl_arrive();                                             // the last cluster access: no barrier before exit
        cl_wait();
    }
    __syncthreads();
    if (tid < BMQ) {                                             // l: fma(l, alpha, (W0 + W2) + (W1 + W3))
        float l = 0.f;
        for (int tile = 0; tile < 4; ++tile) {
            const float* w4 = wsum + tile * 64 + tid;
            const float sum = __fadd_rn(__fadd_rn(w4[0], w4[32]), __fadd_rn(w4[16], w4[48]));
            l = __fmaf_rn(l, rowv[64 + tile * 16 + tid], sum);
        }
        reinterpret_cast<float*>(sm + G::SM_MISC)[16 + tid] = l;
    }
    const int x0 = col2byte(G::W * k);
    float acc[G::J][4];
#pragma unroll
    for (int j = 0; j < G::J; ++j)
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.f;
    for (int tile = 0; tile < 4; ++tile) {                       // o = mma(P, kv, rn(o * alpha)), 2 k steps a tile
        const float al0 = rowv[64 + tile * 16 + g], al1 = rowv[64 + tile * 16 + g + 8];
#pragma unroll
        for (int j = 0; j < G::J; ++j) {
            acc[j][0] = __fmul_rn(acc[j][0], al0);
            acc[j][1] = __fmul_rn(acc[j][1], al0);
            acc[j][2] = __fmul_rn(acc[j][2], al1);
            acc[j][3] = __fmul_rn(acc[j][3], al1);
        }
        for (int step = 0; step < 2; ++step) {
            const uint4 af = reinterpret_cast<const uint4*>(pf)[(tile * 2 + step) * 32 + lane];
            const int k0 = 32 * tile + klo<KW>(step, t), k1 = 32 * tile + khi<KW>(step, t);
#pragma unroll
            for (int j = 0; j < G::J; ++j) {
                const int n = 64 * (G::J * k + j) + 8 * warp + g;
                if (P == 1) {
                    mma(acc[j], af, spair_n<P>(sm, k0, k0 + 1, n, x0), spair_n<P>(sm, k1, k1 + 1, n, x0));
                } else {
                    const uint32_t* vb = reinterpret_cast<const uint32_t*>(sm + G::SM_VB);
                    const int x = n - G::W * k;
                    mma(acc[j], af, vb[(k0 >> 1) * G::W + (x ^ vsw<KW>(k0 >> 1))],
                        vb[(k1 >> 1) * G::W + (x ^ vsw<KW>(k1 >> 1))]);
                }
            }
        }
    }
    const long long pbase = (static_cast<long long>(c) * R + r) * H + hb * BMQ;
#pragma unroll
    for (int j = 0; j < G::J; ++j) {
        const int n = 64 * (G::J * k + j) + 8 * warp + 2 * t;
        *reinterpret_cast<float2*>(a.po + (pbase + g) * D + n) = make_float2(acc[j][0], acc[j][1]);
        *reinterpret_cast<float2*>(a.po + (pbase + g + 8) * D + n) = make_float2(acc[j][2], acc[j][3]);
    }
    if (tid < BMQ) {                                             // every rank: the same (m, l), so each merge sees it
        a.pm[pbase + tid] = reinterpret_cast<float*>(sm + G::SM_MISC)[tid];
        a.pl[pbase + tid] = reinterpret_cast<float*>(sm + G::SM_MISC)[16 + tid];
    }
    int* flag = reinterpret_cast<int*>(sm + G::SM_MISC) + 32;
    int* tk = a.ticket + (r * (H / BMQ) + hb) * MAXP + k;
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        int need = 0;
        for (int cc = 0; cc < NCH; ++cc) need += chunk_works(a, r, cc) ? 1 : 0;
        *flag = atomicAdd(tk, 1) == need - 1;
    }
    __syncthreads();
    if (!*flag) return;
    __threadfence();
    if (tid == 0) atomicExch(tk, 0);                             // self-cleaning: the next launch / replay
    merge_cols<ROWS, G::W>(a, sm, G::SM_TMAX, r, hb, G::W * k);
}

template <int KW, bool ROWS, bool HAS_HI, int P>
cudaError_t launch_split(const Args& a, cudaStream_t stream) {
    using G = Split<P>;
    static bool init = false;
    if (!init) {
        const cudaError_t e = cudaFuncSetAttribute(split_kernel<KW, ROWS, HAS_HI, P>,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
        if (e != cudaSuccess) return e;
        init = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(NCH * P, a.H / BMQ, a.R);
    cfg.blockDim = dim3(NT);
    cfg.dynamicSmemBytes = G::SMEM;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = P;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    cfg.attrs = at;
    cfg.numAttrs = P > 1 ? 1 : 0;
    return cudaLaunchKernelEx(&cfg, split_kernel<KW, ROWS, HAS_HI, P>, a);
}

template <int P> inline cudaError_t dispatch_split(const Args& a, int kw, bool rows, bool has_hi, cudaStream_t s) {
    switch ((kw == 4 ? 4 : 0) + (rows ? 2 : 0) + (has_hi ? 1 : 0)) {
    case 0: return launch_split<2, false, false, P>(a, s);
    case 1: return launch_split<2, false, true, P>(a, s);
    case 2: return launch_split<2, true, false, P>(a, s);
    case 3: return launch_split<2, true, true, P>(a, s);
    case 4: return launch_split<4, false, false, P>(a, s);
    case 5: return launch_split<4, false, true, P>(a, s);
    case 6: return launch_split<4, true, false, P>(a, s);
    default: return launch_split<4, true, true, P>(a, s);
    }
}

// split: 0 attn_kernel (today's launch), 1 / 4 split_kernel with that many CTAs a chunk (the same bits)
inline cudaError_t dispatch(const Args& a, int kw, bool rows, bool has_hi, cudaStream_t stream, int split = 0) {
    if (a.R < 1 || a.H % BMQ || a.ring < WINDOW || (a.ring & (a.ring - 1))) return cudaErrorInvalidValue;
    if (split == 1) return dispatch_split<1>(a, kw, rows, has_hi, stream);
    if (split == 4) return dispatch_split<4>(a, kw, rows, has_hi, stream);
    if (split != 0) return cudaErrorInvalidValue;
    switch ((kw == 4 ? 4 : 0) + (rows ? 2 : 0) + (has_hi ? 1 : 0)) {
    case 0: return launch_one<2, false, false>(a, stream);
    case 1: return launch_one<2, false, true>(a, stream);
    case 2: return launch_one<2, true, false>(a, stream);
    case 3: return launch_one<2, true, true>(a, stream);
    case 4: return launch_one<4, false, false>(a, stream);
    case 5: return launch_one<4, false, true>(a, stream);
    case 6: return launch_one<4, true, false>(a, stream);
    default: return launch_one<4, true, true>(a, stream);
    }
}

}  // namespace dsv41_attn

#ifndef DSV41_ATTN_NO_TORCH      // the compile test builds the kernels alone (nvcc, no torch headers)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {
template <typename T> T* tp(const at::Tensor& t) { return t.numel() ? static_cast<T*>(t.data_ptr()) : nullptr; }
}

void dsv41_attn_run_cuda(const at::Tensor& q, const at::Tensor& cv, const at::Tensor& csc, const at::Tensor& tok,
                         const at::Tensor& cnt, const at::Tensor& sv, const at::Tensor& ssc, const at::Tensor& lo,
                         const at::Tensor& hi, const at::Tensor& pos, const at::Tensor& sl, const at::Tensor& pt,
                         const at::Tensor& sink, const at::Tensor& cs, at::Tensor& po, at::Tensor& pm,
                         at::Tensor& pl, at::Tensor& out, at::Tensor& ticket, int64_t ring, int64_t psh,
                         int64_t pts, int64_t kw, int64_t split, int64_t qrope) {
    using namespace dsv41_attn;
    Args a{};
    a.q = tp<const uint16_t>(q);
    a.has_comp = cv.numel() ? 1 : 0;
    a.cv = tp<const uint8_t>(cv); a.csc = tp<const uint8_t>(csc);
    a.cvs = cv.numel() ? cv.stride(0) : 0; a.css = csc.numel() ? csc.stride(0) : 0;
    a.tok = tp<const int>(tok); a.ts = tok.numel() ? tok.stride(0) : 0; a.cnt = tp<const int>(cnt);
    a.sv = tp<const uint8_t>(sv); a.ssc = tp<const uint8_t>(ssc); a.svs = sv.stride(0); a.sss = ssc.stride(0);
    a.lo = tp<const int>(lo); a.hi = tp<const int>(hi);
    a.pos = pos.data_ptr();
    a.sl = tp<const long long>(sl);
    a.pt = tp<const int>(pt); a.pts = pts; a.psh = static_cast<int>(psh);
    a.sink = tp<const float>(sink); a.cs = tp<const float>(cs); a.csst = cs.stride(0);
    a.po = tp<float>(po); a.pm = tp<float>(pm); a.pl = tp<float>(pl);
    a.out = tp<uint16_t>(out); a.ticket = tp<int>(ticket);
    a.R = static_cast<int>(q.size(0)); a.H = static_cast<int>(q.size(1)); a.ring = static_cast<int>(ring);
    a.qrope = qrope && split > 0 ? 1 : 0;
    TORCH_CHECK(!qrope || split > 0, "attn_cuda: q's RoPE folds into the split kernel only");
    C10_CUDA_CHECK(dispatch(a, static_cast<int>(kw), sl.numel() > 0, hi.numel() > 0,
                            at::cuda::getCurrentCUDAStream(), static_cast<int>(split)));
}

std::vector<int64_t> dsv41_attn_info_cuda(int64_t device) {   // GB10's limits for the two launches
    int sms = 0, optin = 0, cl = 0, blocks = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, static_cast<int>(device));
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, static_cast<int>(device));
    cudaDeviceGetAttribute(&cl, cudaDevAttrClusterLaunch, static_cast<int>(device));
    cudaFuncSetAttribute(dsv41_attn::attn_kernel<4, true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         dsv41_attn::SMEM);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, dsv41_attn::attn_kernel<4, true, false>, dsv41_attn::NT,
                                                  dsv41_attn::SMEM);
    return {sms, optin, cl, blocks, dsv41_attn::SMEM};
}
#endif

// The instantiations dispatch() launches: attn_kernel<KW 2 / 4, ROWS, HAS_HI>.
template __global__ void dsv41_attn::attn_kernel<2, false, false>(dsv41_attn::Args);
template __global__ void dsv41_attn::attn_kernel<2, false, true>(dsv41_attn::Args);
template __global__ void dsv41_attn::attn_kernel<2, true, false>(dsv41_attn::Args);
template __global__ void dsv41_attn::attn_kernel<2, true, true>(dsv41_attn::Args);
template __global__ void dsv41_attn::attn_kernel<4, false, false>(dsv41_attn::Args);
template __global__ void dsv41_attn::attn_kernel<4, false, true>(dsv41_attn::Args);
template __global__ void dsv41_attn::attn_kernel<4, true, false>(dsv41_attn::Args);
template __global__ void dsv41_attn::attn_kernel<4, true, true>(dsv41_attn::Args);
// R1: dispatch_split's split_kernel<KW, ROWS, HAS_HI, P> at P 1 and 4 (a cluster of 4)
template __global__ void dsv41_attn::split_kernel<2, false, false, 1>(dsv41_attn::Args);
template __global__ void dsv41_attn::split_kernel<2, false, true, 1>(dsv41_attn::Args);
template __global__ void dsv41_attn::split_kernel<2, true, false, 1>(dsv41_attn::Args);
template __global__ void dsv41_attn::split_kernel<2, true, true, 1>(dsv41_attn::Args);
template __global__ void dsv41_attn::split_kernel<4, false, false, 1>(dsv41_attn::Args);
template __global__ void dsv41_attn::split_kernel<4, false, true, 1>(dsv41_attn::Args);
template __global__ void dsv41_attn::split_kernel<4, true, false, 1>(dsv41_attn::Args);
template __global__ void dsv41_attn::split_kernel<4, true, true, 1>(dsv41_attn::Args);
template __global__ void dsv41_attn::split_kernel<2, false, false, 4>(dsv41_attn::Args);
template __global__ void dsv41_attn::split_kernel<2, false, true, 4>(dsv41_attn::Args);
template __global__ void dsv41_attn::split_kernel<2, true, false, 4>(dsv41_attn::Args);
template __global__ void dsv41_attn::split_kernel<2, true, true, 4>(dsv41_attn::Args);
template __global__ void dsv41_attn::split_kernel<4, false, false, 4>(dsv41_attn::Args);
template __global__ void dsv41_attn::split_kernel<4, false, true, 4>(dsv41_attn::Args);
template __global__ void dsv41_attn::split_kernel<4, true, false, 4>(dsv41_attn::Args);
template __global__ void dsv41_attn::split_kernel<4, true, true, 4>(dsv41_attn::Args);
