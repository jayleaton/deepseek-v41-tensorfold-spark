#!/usr/bin/env bash
# rank0-host, after a W9 turn's END (analyze done): that block's gate. T2's w9-report.sh with oracle r2 (advisory) and r7
# (adjudication, decides STOP), stale_live.py over the run, then w8_gate.py on the block's manifest.
# Run inside run-cpu-q.sh (2 GiB scope, nice 19, paused under QUIET); w9-report.sh takes its own 2 GiB scope.
# Usage: bash w9-block-gate.sh NN RUN   -> ~/work/imr-t4/w9gate/NN/{report/,stale.json,gate.md,gate.json}
set -uo pipefail
n=${1:?block NN}
run=${2:?run id}
M=<rank0-work>/imr-t4/measure
B=<rank0-work>/dsv41-bench/out/$run
g=<rank0-work>/imr-t4/w9gate/$n
test -d "$B" || { echo "no run $B"; exit 2; }
mkdir -p "$g/rec"
ln -sfn "$B" "$g/rec/run-$n"
bash <rank0-work>/owned-oracle/w9-report/w9-report.sh "$g/rec" "$M/w9" "$g/report" r2 "$((10#$n))" > "$g/report.log" 2>&1
echo "w9-report r2 rc=$?"
# r7 (T5 AGREE, live; the adjudication oracle) decides the STOP: r2 predates deadline-aware cuts, so a write the
# server deadline ended mid-value reads there as wrong-value (block 4: r2 wrong 1, r7 cut 1 at the native end)
bash <rank0-work>/owned-oracle/w9-report/w9-report.sh "$g/rec" "$M/w9" "$g/report-r7" r7 "$((10#$n))" > "$g/report-r7.log" 2>&1
echo "w9-report r7 rc=$?"
python3 -B "$M/stale_live.py" <rank0-work>/imr-t4/stale/scripts "$B" "$g/stale" > "$g/stale.json" 2> "$g/stale.err"
echo "stale rc=$?"
python3 -B "$M/w8_gate.py" --run "$B" --manifest "$M/w9/w9-block-$n-manifest.json" --tasks "$M/w9/w9-block-$n.jsonl" \
  --oracle "$g/report-r7/block-$n/oracle/gate.json" --stale "$g/stale.json" --json "$g/gate.json" > "$g/gate.md"
echo "w8_gate rc=$? (efficacy in gate.md/gate.json only: not posted before all 10 blocks, T5-W9-SPEC)"
# safety only on stdout; any wrong engine byte (or an owned FAIL) writes w9/STOP: window.sh leaves ORDER at the head
python3 -I - "$g" "$M/w9/STOP" "$n" <<'PY'
import json, sys
from pathlib import Path
g, stop, n = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rep = json.loads((g / 'report-r7/w9-report.json').read_text())['blocks'][0]
r2 = json.loads((g / 'report/w9-report.json').read_text())['blocks'][0]
gates = json.loads((g / 'gate.json').read_text())['gates']
eb, pv = rep['engine_bytes'], rep['provenance']
wrong = {k: eb[k] for k in ('wrong_landed_bytes', 'wrong-value', 'wrong-span', 'not-in-text', 'wrong-check')}
print('safety: r7', rep['verdict'], '(advisory r2', r2['verdict'], 'wrong', r2['engine_bytes']['wrong_landed_bytes'],
      'cut', r2['engine_bytes']['cut'], ')', '| owned values', eb['complete_verified_values'], 'wrong', wrong,
      '| unknown', eb['unknown'], 'cut', eb['cut'], '| receipts final', pv['receipts_final'], 'live-only',
      pv['receipts_live_only'], 'live contradictions', pv['live_contradictions'])
print('gates:', {k: gates[k]['verdict'] for k in ('integrity', 'owned', 'stale')})
print('cut (r7):', eb['cut'], json.dumps(rep['cut_gate']))
if any(wrong.values()) or gates['owned']['verdict'] == 'FAIL' or rep['verdict'] == 'FAIL':
    stop.write_text(f'block {n}: wrong engine bytes {wrong}, owned {gates["owned"]["verdict"]}, r7 {rep["verdict"]}\n')
    print('STOP written', stop)
PY
