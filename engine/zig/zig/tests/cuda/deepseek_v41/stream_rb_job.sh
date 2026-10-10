#!/bin/bash
# The row-blocked stream top-k's GPU micro-check on one GPU (DGX Spark GB10 sm_121, or a pod): tf-dsv41-test
# stream-rb runs the Triton AOT `_stream` (MODE 0) and `_stream_rb2` / `_stream_rb4` on the same synthetic inputs at
# prod's indexer shapes (starts 32,768 and 129,024; 2,048 / 2,047 / 300 rows), compares every row's split buffers'
# first K keys slot for slot and times each. Needs an AOT set (aot.json + cubins/) holding the three kernels' variants.
#
#   TF_DSV41_TEST=<tf-dsv41-test> AOT=<aot dir> [STREAM_RB_REPS=5] [STREAM_RB_TWINS=1,2,4] bash stream_rb_job.sh <out>
# (STREAM_RB_TWINS: the twins the AOT set holds; 1 = _stream_pf, 2 / 4 = _stream_rb2 / _rb4)
set -u
OUT=${1:?out dir}
T=${TF_DSV41_TEST:?tf-dsv41-test}
A=${AOT:?aot dir}
mkdir -p "$OUT"
rc=0
step() { echo "=== $(date -u +%H:%M:%S) $*"; }

step "device"
nvidia-smi --query-gpu=name,compute_cap,clocks.max.sm,memory.total --format=csv | tee "$OUT/device.txt"
grep -o '"fn": *"_stream[a-z0-9_]*"' "$A/aot.json" | sort | uniq -c | tee "$OUT/variants.txt"

step "stream-rb: the twins (${STREAM_RB_TWINS:-1,2,4}) against _stream"
"$T" stream-rb "$A" --reps "${STREAM_RB_REPS:-5}" --twins "${STREAM_RB_TWINS:-1,2,4}" 2>&1 | tee "$OUT/stream_rb.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=1

step "rc=$rc"
exit $rc
