#!/usr/bin/env bash
# Compile + production Triton byte oracle. --compile-only needs no GPU/Python CUDA.
set -euo pipefail
if [[ ${KV_GLUE_HEAVY_HELD:-0} != 1 ]]; then
  exec env KV_GLUE_HEAVY_HELD=1 "${HEAVY:?set HEAVY to a resource-limiting build wrapper}" bash "$0" "$@"
fi
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=$(cd "$HERE/../../.." && pwd)
cd "$TREE"
mkdir -p .zig-cache
SCRATCH=$(mktemp -d "$TREE/.zig-cache/kv-glue-gpu.XXXXXX")
trap 'rm -rf "$SCRATCH"' EXIT
COMPILER=${NVCC:-}
if [[ -z $COMPILER ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    COMPILER=$(command -v nvcc)
  else
    COMPILER="$TREE/zig/kernels/cuda/deepseek_v41/nvcc-docker.sh"
  fi
fi
# Default FMA enabled: the kernel must express required separate rounding itself.
"$COMPILER" -fatbin -O3 -lineinfo -std=c++20 --expt-relaxed-constexpr \
  -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
  -gencode arch=compute_121,code=sm_121 -gencode arch=compute_120,code=sm_120 \
  -o "$SCRATCH/kv_glue.fatbin" "$TREE/zig/kernels/cuda/deepseek_v41/kv_glue.cu"
if [[ ${1:-} == --compile-only ]]; then
  [[ $# == 1 ]] || { echo '--compile-only takes no further arguments' >&2; exit 2; }
  echo 'PASS: kv_glue.cu compiled for sm_121 + sm_120 with default FMA enabled'
else
  "${PYTHON:-python3}" -B "$HERE/kv_glue_gpu.py" --fatbin "$SCRATCH/kv_glue.fatbin" "$@"
fi
