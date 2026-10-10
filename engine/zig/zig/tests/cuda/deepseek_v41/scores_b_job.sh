#!/bin/bash
# The row-bounded index scores' GPU micro-check on one GPU (DGX Spark GB10 sm_121, or a pod): tf-dsv41-test scores-b
# runs the Triton AOT `_scores` and `_scores_b` (TF_DSV41_INDEX_BOUND) on the same synthetic inputs at prod's indexer
# shapes (ratio 2 / PSH 7 and ratio 1 / PSH 8; a row window mixing short slots with a 154K slot, the same rows all long,
# a one-slot window at 150K), compares the whole score output bit for bit and times each. Needs an AOT set holding both
# kernels' variants (tools/zig/dsv41_m4/fill-ibound.env).
#
#   TF_DSV41_TEST=<tf-dsv41-test> AOT=<aot dir> [SCORES_B_REPS=20] bash scores_b_job.sh <out>
set -u
OUT=${1:?out dir}
T=${TF_DSV41_TEST:?tf-dsv41-test}
A=${AOT:?aot dir}
mkdir -p "$OUT"
rc=0
step() { echo "=== $(date -u +%H:%M:%S) $*"; }

step "device"
nvidia-smi --query-gpu=name,compute_cap,clocks.max.sm,memory.total --format=csv | tee "$OUT/device.txt"
grep -o '"fn": *"_scores[a-z0-9_]*"' "$A/aot.json" | sort | uniq -c | tee "$OUT/variants.txt"
# the gate compares the served kernel with its twin: the AOT set must hold both (the served set with the
# fill-ibound.env fill merged in: tools/zig/triton_fill.py merge <copy of the served aot dir> <the fill's aot dir>);
# a set holding one of them skips every compare, which is no verdict
for fn in _scores _scores_b; do
    if ! grep -q "\"fn\": *\"$fn\"" "$A/aot.json"; then
        echo "no $fn variant in $A/aot.json: merge the served set and the fill-ibound.env fill (triton_fill.py merge)" | tee "$OUT/no-variants.txt"
        exit 2
    fi
done

step "scores-b: _scores_b against _scores"
"$T" scores-b "$A" --reps "${SCORES_B_REPS:-20}" 2>&1 | tee "$OUT/scores_b.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=1

step "rc=$rc"
exit $rc
