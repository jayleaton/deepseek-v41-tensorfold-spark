#!/bin/bash
# Build and run tbo_bench (x3gm routed experts || pfdense dense GEMMs, serial vs concurrent) on a DGX Spark, inside the
# CUDA devel image already on the Sparks. Run from anywhere; arguments pass through to tbo_bench.
#
#   docker run --rm --gpus all -v $PWD:$PWD -w $PWD nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 \
#     nvcc -O3 -lineinfo -std=c++17 -gencode arch=compute_121a,code=sm_121a -o tbo_bench tbo_bench.cu -lcuda
#   docker run --rm --gpus all -v $PWD:$PWD -w $PWD -e CUDA_MODULE_LOADING=EAGER \
#     nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 ./tbo_bench [--rows 2048,4096] [--k2 6] [--reps 5] [--green] \
#     [--sms 48,44,40,36,32,28,24] [--zipf 0]
#
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"           # the proto dir: includes are relative (../x3gm.cu, ../pfdense.cuh)
IMAGE="${TBO_IMAGE:-nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04}"
# the parent dir is mounted too: the kernels include ../x3gm.cu etc.
ROOT="$(readlink -f ..)"
run() { docker run --rm --gpus all -v "$ROOT:$ROOT" -w "$PWD" -e CUDA_MODULE_LOADING=EAGER "$IMAGE" "$@"; }
echo "+ nvcc -O3 -lineinfo -std=c++17 -gencode arch=compute_121a,code=sm_121a -o tbo_bench tbo_bench.cu -lcuda"
run nvcc -O3 -lineinfo -std=c++17 -gencode arch=compute_121a,code=sm_121a -o tbo_bench tbo_bench.cu -lcuda
echo "+ ./tbo_bench $*"
run ./tbo_bench "$@"
