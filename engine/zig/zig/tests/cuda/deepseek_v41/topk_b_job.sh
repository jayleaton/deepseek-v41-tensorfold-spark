#!/bin/bash
# The row-bounded decode top-k's GPU micro-check on one GPU (DGX Spark GB10 sm_121, or a pod): tf-dsv41-test topk-b
# runs topk_cuda.cu's topk_kernel against topk_b.cu's twin (TF_DSV41_INDEX_BOUND: tf_dsv41_topk_b_v1.topk) and the
# Triton AOT `_dtopk` against `_dtopk_b` on the same synthetic index scores (a row window mixing short slots with a
# 154K slot, the same rows all long, a one-slot window; the CUDA select at ratio 2, select + blocks under 126,976 keys,
# blocks only, `_dtopk` select at 154K and blocks at 1,040,000), compares every output byte and times each. Needs a
# tf-dsv41-test built with the fatbins (-Dnvcc or -Dfatbins holding dsv41_topk_b.fatbin) and an AOT set holding
# both Triton kernels' variants (tools/zig/dsv41_m4/fill-ibound.env).
#
#   TF_DSV41_TEST=<tf-dsv41-test> AOT=<aot dir> [TOPK_B_REPS=20] bash topk_b_job.sh <out>
set -u
OUT=${1:?out dir}
T=${TF_DSV41_TEST:?tf-dsv41-test}
A=${AOT:?aot dir}
mkdir -p "$OUT"
rc=0
step() { echo "=== $(date -u +%H:%M:%S) $*"; }

step "device"
nvidia-smi --query-gpu=name,compute_cap,clocks.max.sm,memory.total --format=csv | tee "$OUT/device.txt"
grep -o '"fn": *"_dtopk[a-z0-9_]*"' "$A/aot.json" | sort | uniq -c | tee "$OUT/variants.txt"
# the gate compares the served kernel with its twin: the AOT set must hold both (the served set with the
# fill-ibound.env fill merged in: tools/zig/triton_fill.py merge <copy of the served aot dir> <the fill's aot dir>);
# a set holding one of them skips every compare, which is no verdict
for fn in _dtopk _dtopk_b; do
    if ! grep -q "\"fn\": *\"$fn\"" "$A/aot.json"; then
        echo "no $fn variant in $A/aot.json: merge the served set and the fill-ibound.env fill (triton_fill.py merge)" | tee "$OUT/no-variants.txt"
        exit 2
    fi
done

step "topk-b: topk_b_kernel / _dtopk_b against topk_kernel / _dtopk"
"$T" topk-b "$A" --reps "${TOPK_B_REPS:-20}" 2>&1 | tee "$OUT/topk_b.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=1

step "rc=$rc"
exit $rc
