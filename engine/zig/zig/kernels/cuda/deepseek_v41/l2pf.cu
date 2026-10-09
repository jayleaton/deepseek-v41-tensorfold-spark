// Device code of src/tensorfold/families/glm5_next/spark/l2pf.cu (git blob ace57d9cab36 at dsv41-quant-e2; the whole file),
// generated and checked by zig/kernels/cuda/deepseek_v41/sync.py. Do not edit: change the Python source.
// L2 prefetch kernels for the decode rounds' idle windows (patches/0460, GLM53_TF_L2PF).
//
// Every kernel here only READS weight memory into the GPU's L2 and writes nothing the model reads, so it cannot change
// a bit of any result; it only changes where the next kernels find their weights (L2 instead of DRAM).
//
//   bulk   one cp.async.bulk.prefetch.L2.global per CHUNK bytes (sm_90+ PTX; GB10 is sm_121): the SM's bulk-copy
//          unit streams the range into L2 with no registers, no shared memory and no data returned to the SM, so
//          the kernel is a handful of instructions per thread and the side stream holds no SM for the transfer
//   lines  one prefetch.global.L2 per 128-byte line (any sm_70+ GPU): the fallback if the bulk form is slower here
//
// Two entry points:
//   prefetch_segments  a static per-site table of (address, bytes) (the plans of l2pf.py: the next kernels' weights)
//   prefetch_experts   device-indexed: the first ``bytes`` of every distinct routed expert the router picked this
//                      window (qmm.Group ids[0 .. count)), in each listed matrix (EXL3 trellis [E, ...] per expert),
//                      without a host round trip, so it is captured in the round's CUDA graph like the rest

#include <cuda_runtime.h>
#include <stdint.h>

namespace tfl2pf {

constexpr uint32_t LINE = 128;

__device__ __forceinline__ void bulk_prefetch(const void *p, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
#else   // the pre-sm_90 targets of TORCH_CUDA_ARCH_LIST (never run on GB10): a line prefetch per 128 bytes
    for (uint32_t o = 0; o < bytes; o += LINE) asm volatile("prefetch.global.L2 [%0];" ::"l"((const char *)p + o));
#endif
}

__device__ __forceinline__ void line_prefetch(const void *p) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}

// ``bytes`` of [p, p + bytes) in pieces of ``chunk`` bytes (a multiple of 16), piece i by thread i of ``stride``.
// Bulk prefetches need a 16-byte aligned address and a size that is a multiple of 16: the host rounds every segment
// to that (l2pf.py), and the last piece is the remainder (also a multiple of 16).
template <bool Bulk>
__device__ __forceinline__ void range(const uint8_t *p, uint64_t bytes, uint32_t chunk, uint64_t index,
                                      uint64_t stride) {
    if (Bulk) {
        const uint64_t pieces = (bytes + chunk - 1) / chunk;
        for (uint64_t i = index; i < pieces; i += stride) {
            const uint64_t off = i * chunk;
            const uint64_t left = bytes - off;
            bulk_prefetch(p + off, static_cast<uint32_t>(left < chunk ? left : chunk));
        }
    } else {
        const uint64_t lines = (bytes + LINE - 1) / LINE;
        for (uint64_t i = index; i < lines; i += stride) {
            line_prefetch(p + i * LINE);
        }
    }
}

// table: n pairs of int64 (device address, bytes)
template <bool Bulk>
__global__ void __launch_bounds__(256) segments_kernel(const int64_t *table, int n, uint32_t chunk) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (int s = 0; s < n; s++) {
        const uint8_t *p = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(table[2 * s]));
        const uint64_t bytes = static_cast<uint64_t>(table[2 * s + 1]);
        // segment s's first piece goes to thread s (mod the grid): a table of many one-piece heads (the chunk heads of
        // a 4-bit matrix) is spread over the threads instead of all landing on thread 0
        const uint64_t first = (index + stride - static_cast<uint64_t>(s) % stride) % stride;
        range<Bulk>(p, bytes, chunk, first, stride);
    }
}

// Block u (grid-stride over maxu) prefetches the first ``bytes`` of expert ids[u] in each of the ``nmat`` matrices
// (bases[m] + ids[u] * stride_bytes) when u < count[0]. ids / count are the router's device output (qmm.Group).
template <bool Bulk>
__global__ void __launch_bounds__(256) experts_kernel(const int32_t *ids, const int32_t *count, int maxu,
                                                      const int64_t *bases, int nmat, int64_t stride_bytes,
                                                      int64_t bytes, int64_t experts, uint32_t chunk) {
    const int n = min(static_cast<int>(*count), maxu);
    for (int u = blockIdx.x; u < n; u += gridDim.x) {
        const int64_t e = static_cast<int64_t>(ids[u]);
        if (e < 0 || e >= experts) {
            continue;                    // never read outside the expert tensors
        }
        for (int m = 0; m < nmat; m++) {
            const uint8_t *p = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(bases[m])) + e * stride_bytes;
            range<Bulk>(p, static_cast<uint64_t>(bytes), chunk, threadIdx.x, blockDim.x);
        }
    }
}

cudaError_t launch_segments(const int64_t *table, int n, int grid, int threads, uint32_t chunk, bool bulk,
                            cudaStream_t stream) {
    if (bulk) {
        segments_kernel<true><<<grid, threads, 0, stream>>>(table, n, chunk);
    } else {
        segments_kernel<false><<<grid, threads, 0, stream>>>(table, n, chunk);
    }
    return cudaGetLastError();
}

cudaError_t launch_experts(const int32_t *ids, const int32_t *count, int maxu, const int64_t *bases, int nmat,
                           int64_t stride_bytes, int64_t bytes, int64_t experts, int grid, int threads, uint32_t chunk,
                           bool bulk, cudaStream_t stream) {
    if (bulk) {
        experts_kernel<true><<<grid, threads, 0, stream>>>(ids, count, maxu, bases, nmat, stride_bytes, bytes, experts,
                                                           chunk);
    } else {
        experts_kernel<false><<<grid, threads, 0, stream>>>(ids, count, maxu, bases, nmat, stride_bytes, bytes, experts,
                                                            chunk);
    }
    return cudaGetLastError();
}

}  // namespace tfl2pf
