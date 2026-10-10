// Device code of src/tensorfold/families/deepseek_v41/cuda/l2pace.cu (git blob fcf123ce92e9 at dsv41-quant-e2; the whole file),
// generated and checked by zig/kernels/cuda/deepseek_v41/sync.py. Do not edit: change the Python source.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
//
// G14: the L2 prefetch of a site PACED (TF_DSV41_L2PF_PACE_GBPS), so it stops starving the RoCE exchange beside it.
//
// Why. ``l2pf.cu``'s segments_kernel hands every 32 KiB piece of a site (12 MiB at prod's TF_DSV41_L2PF_MB=12: 384
// pieces) to its own thread: all of it is requested at once. The memory system then holds ~12 MiB of prefetch in its
// queues, and anything else that needs DRAM or the C2C path in the next ~50 us waits behind it. The site is forked
// 0.1 us before the window's exchange (forward._layers: L2.mark then gather_partials), and the exchange is a chain of
// dependent round trips to pinned host memory (stage stores + release, the flag poll, the copy-out's system-scope
// loads). G14a measured it: an exchange beside the prefetch 26.3 us median in nsys against 9.0 without; copy-out p90
// 13 vs 4 us; 1-row window exchange kernels 1.36-1.42 ms a rank vs 0.96-1.06 with L2PF off (~0.2 ms on the critical
// path). Turning L2PF off costs 1.7 ms a window, so the prefetch stays and is paced instead.
//
// How. One thread per CTA (``ctas`` CTAs, 32 threads each, the rest idle) walks the site's table in order, the
// pieces dealt round-robin over the CTAs, and issues piece g no earlier than t0 + delay + g x ns_per_piece
// (%globaltimer). The prefetch then streams at ``rate`` = chunk / ns_per_piece, with only ~rate x latency bytes in
// flight (~0.2-0.5 MiB) instead of the whole site, so a latency-bound request beside it queues behind a few pieces,
// not 12 MiB. The order is the table's (the next launch's read order), as segments_kernel's first wave. ``delay``
// lets an exchange's stage pass first. Bandwidth-wise the site still lands in L2 before its consumer when rate x
// window >= the site (12 MiB in a 60 us window needs ~200 GB/s; the knob trades that against the exchange).
//
// Exactness: like l2pf.cu, it only READS weight memory into L2 (cp.async.bulk.prefetch.L2) and writes nothing.

#include <cuda_runtime.h>
#include <stdint.h>

namespace dsv41_l2pace {

__device__ __forceinline__ uint64_t now_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void bulk_prefetch(const void *p, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
#else
    for (uint32_t o = 0; o < bytes; o += 128) asm volatile("prefetch.global.L2 [%0];" ::"l"((const char *)p + o));
#endif
}

__device__ __forceinline__ void wait_until(uint64_t due) {
    while (true) {
        const uint64_t t = now_ns();
        if (t >= due) {
            return;
        }
        const uint64_t left = due - t;
        __nanosleep(static_cast<unsigned>(left > 2000u ? 1000u : (left > 64u ? left / 2u : 32u)));
    }
}

// table: n rows of int64 (address, bytes): 16-byte aligned addresses, byte counts multiples of 16 (l2pf.take)
__global__ void __launch_bounds__(32) paced_kernel(const int64_t *table, int n, uint32_t chunk, uint64_t ns_per_piece,
                                                   uint64_t delay_ns) {
    if (threadIdx.x != 0) {
        return;
    }
    const uint64_t c = blockIdx.x;
    const uint64_t g = gridDim.x;
    const uint64_t t0 = now_ns() + delay_ns;
    uint64_t k = 0;                                     // the piece's index over the whole table
    for (int s = 0; s < n; s++) {
        const uint8_t *p = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(table[2 * s]));
        const uint64_t bytes = static_cast<uint64_t>(table[2 * s + 1]);
        for (uint64_t off = 0; off < bytes; off += chunk, k++) {
            if (k % g != c) {
                continue;
            }
            wait_until(t0 + k * ns_per_piece);
            const uint64_t left = bytes - off;
            bulk_prefetch(p + off, static_cast<uint32_t>(left < chunk ? left : chunk));
        }
    }
}

cudaError_t launch_paced(const int64_t *table, int n, int ctas, uint32_t chunk, uint64_t ns_per_piece,
                         uint64_t delay_ns, cudaStream_t stream) {
    paced_kernel<<<ctas, 32, 0, stream>>>(table, n, chunk, ns_per_piece, delay_ns);
    return cudaGetLastError();
}

}  // namespace dsv41_l2pace
