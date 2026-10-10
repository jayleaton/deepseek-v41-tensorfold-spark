#!/bin/bash
# TF_DSV41_X3LD_EPI's GPU micro-gate on one GPU (DGX Spark GB10 sm_121, or a pod): tf-dsv41-test x3ld-epi runs the
# decode MoE's expert chain on synthetic prod-shape data (backbone 384 + shared top-6 + shared with an fp32 and a bf16
# out, a DSpark block's 128 + shared top-3 + shared; 1 / 4 / 16 / 24 rows; pruned slots), prod's four launches
# (x3ld, gateup_epilogue, x3ld, down_combine) against the fused two (x3ld_epi gateup / down): Xd, y and out byte for
# byte, the tickets back at zero, and both timed (median of 50). PASS / FAIL in x3ld_epi.log; ~3.5 GB of device memory.
#
#   TF_DSV41_TEST=<tf-dsv41-test built with this branch's fatbins> [X3LD_EPI_ROWS=1,4,16,24] [X3LD_EPI_REPS=50] \
#     bash x3ld_epi_job.sh <out>
set -u
OUT=${1:?out dir}
T=${TF_DSV41_TEST:?tf-dsv41-test}
mkdir -p "$OUT"
rc=0
step() { echo "=== $(date -u +%H:%M:%S) $*"; }

step "device"
nvidia-smi --query-gpu=name,compute_cap,clocks.max.sm,memory.total --format=csv | tee "$OUT/device.txt"
"$T" symbols 2>&1 | tee "$OUT/symbols.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=1

step "x3ld-epi: fused against unfused"
"$T" x3ld-epi "${X3LD_EPI_ROWS:-1,4,16,24}" "${X3LD_EPI_REPS:-50}" 2>&1 | tee "$OUT/x3ld_epi.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=1

step "rc=$rc"
exit $rc
