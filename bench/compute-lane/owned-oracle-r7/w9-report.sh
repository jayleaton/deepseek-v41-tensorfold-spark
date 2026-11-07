#!/usr/bin/env bash
# w9-report.sh RUN BLOCKS OUT [r2|r5|r6|r7] [ONLY]: the W9 owned-oracle report on rank0-host/rank1-host only (never dev-host).
#   RUN  receipts (run-NN/<label>/<id>.json per block)   BLOCKS  T4's w9-block-NN-manifest.json directory
#   r2 (default) is the oracle T5 AGREED to; r5/r6/r7 are refused until T5 AGREEs (w9_report.py ORACLES).
# One verdict per block plus a pooled one, fields from T5-W8-DECISION.md: OUT/w9-report.md and .json, and per block
# OUT/block-NN/oracle/gate.json for w8_gate.py --oracle. Each block's gate.sh waits out QUIET, a held T4 window and an
# imr-measure at the head of ORDER; this script's own work runs in a 2 GiB no-swap scope at nice 19 / ionice c3.
# rc 0 PASS, 1 FAIL (BLOCK / investigate), 3 INCONCLUSIVE, 2 error.
set -uo pipefail
Q=${QUIET_FILE:-$HOME/work/queue/QUIET}
here=$(cd "$(dirname "$0")" && pwd)
run=$(realpath "$1"); blocks=$(realpath "$2"); out=$(realpath -m "$3"); oracle=${4:-r2}; only=${5:-}
while [ -e "$Q" ]; do sleep 30; done
systemd-run --user --scope --quiet -p MemoryMax=2G -p MemorySwapMax=0 nice -n 19 ionice -c3 \
  python3 -I "$here/w9_report.py" --run "$run" --blocks "$blocks" --out "$out" --oracle "$oracle" ${only:+--only "$only"}
