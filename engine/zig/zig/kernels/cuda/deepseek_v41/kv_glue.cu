// Locally owned SWA KV glue. Reference: tensorfold-decode1@78b703d,
// cuda/rmsnorm.py _rms and cuda/csa2/{compress.py _kv_store,rows.py}.
// K=512, bf16 input, fp32 RMS weight; one 128-thread CTA per row.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstddef>

namespace dsv41_kv_glue {
struct Args {
    uint64_t x, w, cs, v, s, pos, sl, norm_out;
    int64_t x_stride, cs_stride, v_stride, s_stride;
    float inv_k, eps;
    int32_t ring, rows;
};
static_assert(sizeof(Args) == 112 && offsetof(Args, x_stride) == 64 && offsetof(Args, inv_k) == 96,
              "Zig/CUDA KV glue ABI");

// cap16 d6a6a52... _rms: BK512, four warps, sizePerThread4. SASS0100
// multiplies TID by4; 0180 converts to a bf16 byte offset; 1cd0 loads
// four contiguous elements (LDG.64). K512 takes this loop's one-chunk tail.
// Its register sum is ((A0+A1)+A2)+A3, xor16..1, then(w0+w2)+(w1+w3).
// This differs from x3seg's _rms_row layout; use this KV producer's AOT tree.
// Runtime inv_k/eps are fp32 in AOT, widened before the fp64 operations.
__device__ __forceinline__ double row_rn(const __nv_bfloat16* xr, float inv_k, float eps) {
    __shared__ double ws[4];
    const int t = threadIdx.x;
    double a[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const double v = static_cast<double>(__bfloat162float(xr[4 * t + i]));
        a[i] = __dadd_rn(0.0, __dmul_rn(v, v));
    }
    double sum = __dadd_rn(__dadd_rn(__dadd_rn(a[0], a[1]), a[2]), a[3]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum = __dadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
    if ((t & 31) == 0) ws[t >> 5] = sum;
    __syncthreads();
    const double ss = __dadd_rn(__dadd_rn(ws[0], ws[2]), __dadd_rn(ws[1], ws[3]));
    const double var = __dadd_rn(__dmul_rn(ss, static_cast<double>(inv_k)), static_cast<double>(eps));
    double rn;
    asm("rsqrt.approx.f64 %0, %1;" : "=d"(rn) : "d"(var));
    return rn;
}

// rows.py:e4m3_rne, operation for operation. The explicit fp32 rounding
// precedes the hardware conversion, including its behavior on edge inputs.
__device__ __forceinline__ uint8_t e4m3_rne(float x) {
    const uint32_t bits = __float_as_uint(x);
    const float near = __uint_as_float((bits + 0x7ffffu + ((bits >> 20) & 1u)) & 0xfff00000u);
    const float abs_x = __uint_as_float(bits & 0x7fffffffu);
    const float small_abs = __fsub_rn(__fadd_rn(abs_x, 24576.0f), 24576.0f);
    const float small = __uint_as_float(__float_as_uint(small_abs) | (bits & 0x80000000u));
    return __nv_cvt_float_to_fp8(abs_x < 0.015625f ? small : near, __NV_SATFINITE, __NV_E4M3);
}

// cap16 cb414a38... _kv_store cubin: ordinary FMNMX, xor16..1,
// floor0x38d1b717; e=((bits>>23)&255)-135+(mantissa>0x600000).
// AOT folds the floor before its final xor1 max; FMNMX's NaN-as-number
// policy makes that equivalent to this final clamp, including all-NaN tiles.
__device__ __forceinline__ int tile_exponent(float even, float odd) {
    float peak = fmaxf(fabsf(even), fabsf(odd));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, o));
    const uint32_t bits = __float_as_uint(fmaxf(peak, __uint_as_float(0x38d1b717u)));
    return static_cast<int>((bits >> 23) & 255u) - 135 + ((bits & 0x7fffffu) > 0x600000u);
}

// AOT ring256 uses sign correction, mask0xffffff00, then subtraction:
// C signed remainder (not an unconditional q&255 for negative positions).
// The caller validates a positive power-of-two ring; avoid runtime division.
__device__ __forceinline__ int ring_offset(int q, int ring) {
    const uint32_t mask = static_cast<uint32_t>(ring - 1);
    if (q >= 0) return static_cast<int>(static_cast<uint32_t>(q) & mask);
    return -static_cast<int>((0u - static_cast<uint32_t>(q)) & mask);
}
}  // namespace dsv41_kv_glue

// Grid(R,1,1), block128, dynamic shared0. norm_out is optional debug output
// bf16[R,512]; its rows are written even when SL marks a padding/cache-skip row.
extern "C" __global__ void __launch_bounds__(128) kv_norm_store(const dsv41_kv_glue::Args a) {
    using namespace dsv41_kv_glue;
    __shared__ __align__(16) __nv_bfloat16 latent[512];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int64_t r = blockIdx.x;
    const auto* xr = reinterpret_cast<const __nv_bfloat16*>(a.x) + r * a.x_stride;
    const auto* w = reinterpret_cast<const float*>(a.w);
    const double rn = row_rn(xr, a.inv_k, a.eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int k = t + 128 * i;
        const double y = __dmul_rn(__dmul_rn(static_cast<double>(__bfloat162float(xr[k])), rn),
                                  static_cast<double>(w[k]));
        const __nv_bfloat16 norm = __float2bfloat16_rn(__double2float_rn(y));
        latent[k] = norm;
        if (a.norm_out) reinterpret_cast<__nv_bfloat16*>(a.norm_out)[r * 512 + k] = norm;
    }
    __syncthreads();
    const int64_t sl = a.rows ? reinterpret_cast<const int64_t*>(a.sl)[r] : 0;
    if (sl < 0) return;  // uniform across the CTA, after every optional norm write
    const int q = a.rows ? static_cast<int>(reinterpret_cast<const int64_t*>(a.pos)[r])
                        : static_cast<int>(static_cast<uint32_t>(*reinterpret_cast<const int32_t*>(a.pos))
                                           + static_cast<uint32_t>(r));
    const int64_t row = sl * a.ring + ring_offset(q, a.ring);
    auto* v = reinterpret_cast<uint8_t*>(a.v) + row * a.v_stride;
    auto* scales = reinterpret_cast<uint8_t*>(a.s) + row * a.s_stride;
    const auto* cs = reinterpret_cast<const float*>(a.cs) + static_cast<int64_t>(q) * a.cs_stride;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int pair = t + 128 * j;
        const float even = __bfloat162float(latent[2 * pair]);
        const float odd = __bfloat162float(latent[2 * pair + 1]);
        float ne, no;
        if (pair >= 224) {
            const float c = cs[pair - 224], s = cs[32 + pair - 224];
            // AOT SASS: round odd*s and odd*c, then fuse EVEN*c and EVEN*s.
            ne = __fmaf_rn(even, c, -__fmul_rn(odd, s));
            no = __fmaf_rn(even, s, __fmul_rn(odd, c));
        } else {
            // Preserve the AOT NoPE operands (including NaN and signed zero).
            const float neg_odd_zero = -__fmul_rn(0.0f, odd);
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(ne) : "f"(even), "f"(1.0f), "f"(neg_odd_zero));
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(no) : "f"(0.0f), "f"(even), "f"(odd));
        }
        const int tile = warp + 4 * j;
        if (tile < 7) {  // uniform within each warp
            const int e = tile_exponent(ne, no);
            const float inv_scale = __int_as_float((127 - e) << 23);
            v[2 * pair] = e4m3_rne(__fmul_rn(ne, inv_scale));
            v[2 * pair + 1] = e4m3_rne(__fmul_rn(no, inv_scale));
            if (lane == 0) scales[tile] = static_cast<uint8_t>(e + 127);
        } else {
            auto* rope = reinterpret_cast<__nv_bfloat16*>(v + 448);
            rope[2 * pair - 448] = __float2bfloat16_rn(ne);
            rope[2 * pair - 448 + 1] = __float2bfloat16_rn(no);
            if (lane == 0) scales[7] = 0;
        }
    }
}
