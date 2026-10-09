#!/usr/bin/env bash
# Compile and run the fused/unfused oracle on a CUDA machine. All work takes a heavy slot.
set -euo pipefail
if [[ ${GLUE_ROUTE_HEAVY_HELD:-0} != 1 ]]; then
  exec env GLUE_ROUTE_HEAVY_HELD=1 "${HEAVY:?set HEAVY to a resource-limiting build wrapper}" bash "$0" "$@"
fi
cd "$(dirname "$0")/../../.."
mkdir -p .zig-cache/glue-route-gpu
trap 'rm -rf .zig-cache/glue-route-gpu' EXIT
"${NVCC:-nvcc}" -O3 -lineinfo -std=c++20 --expt-relaxed-constexpr \
  -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
  -gencode arch=compute_121,code=sm_121 -gencode arch=compute_120,code=sm_120 \
  -o .zig-cache/glue-route-gpu/check zig/kernels/cuda/deepseek_v41/proto/router_glue_check.cu
.zig-cache/glue-route-gpu/check "$@"
