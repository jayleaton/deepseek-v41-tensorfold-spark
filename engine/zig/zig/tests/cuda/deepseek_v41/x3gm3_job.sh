#!/bin/bash
# x3gm v3's GPU gate on one RTX PRO 6000 (sm_120), in a CUDA-enabled PyTorch environment:
#   1. the oracle's x3gm fixtures only (the Python engine's gm_kernel outputs: the prefill chunks at 512 / 1,024 /
#      2,048 rows, prod's D 5,120 / I 1,152 geometry, a 1,337-row tail, and the small cases), then tf-dsv41-test all:
#      every v3 variant against them bit for bit (and v1 / v2 / the plan as before);
#   2. tf-dsv41-test x3gm3-bench: v2 and every v3 variant at prod shapes (384 experts, top-6, ragged 4-10), 512 /
#      1,024 / 2,048 rows, uniform and Zipf routing, timed and compared with v2 byte for byte.
# The fixtures are deleted at the end (GBs of random inputs); the logs keep the results.
#
#   TF_DSV41_PY_SRC=<Python twin> TF_DSV41_TEST=<tf-dsv41-test> bash x3gm3_job.sh <out>
set -u
OUT=${1:?out dir}
HERE="$(cd "$(dirname "$0")" && pwd)"
T=${TF_DSV41_TEST:?tf-dsv41-test}
PY=${TF_DSV41_PY_SRC:?Python twin}
mkdir -p "$OUT"
rc=0
step() { echo "=== $(date -u +%H:%M:%S) $*"; }

step "device"
nvidia-smi --query-gpu=name,compute_cap,clocks.max.sm,memory.total --format=csv | tee "$OUT/device.txt"
"$T" symbols 2>&1 | tee "$OUT/symbols.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=1

step "oracle: the x3gm cases"
rm -rf "$OUT/fixtures"
TF_DSV41_ORACLE_ONLY='x3gm_' PYTHONPATH="$PY/src" PYTHONDONTWRITEBYTECODE=1 python -B "$HERE/oracle.py" "$OUT" \
  > "$OUT/oracle.log" 2>&1 || { rc=1; echo "oracle failed"; tail -30 "$OUT/oracle.log"; }

step "bits: v1 / v2 / v3 against the fixtures"
"$T" all "$OUT/fixtures" 2>&1 | tee "$OUT/bits.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
rm -rf "$OUT/fixtures"

step "bench: v2 against v3 at prod shapes"
"$T" x3gm3-bench "${X3GM3_ROWS:-512,1024,2048}" "${X3GM3_REPS:-9}" 2>&1 | tee "$OUT/bench.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=1

step "rc=$rc"
exit $rc
