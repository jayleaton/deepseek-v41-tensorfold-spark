#!/usr/bin/env bash
# window-report.sh ROOT MAP [OUT]: one report per experiment window, on rank0-host/rank1-host only (never dev-host).
#   ROOT  the window's receipts directory     MAP  arm=guide pairs (`r=none,rg=@d_mirror,rs=@trust`) or a JSON file
#   OUT   default ~/work/owned-oracle/out/window-<basename ROOT>
# Runs the owned-oracle gate (gate.sh) on ROOT, then window_report.py (recheck cost, T5 doubt detector, tokens, solve,
# cap hits) in a 2 GiB no-swap scope at nice 19 / ionice c3, after waiting out QUIET, a held T4 window and an imr-measure
# at the head of ORDER. Writes OUT/window-report.md and .json. rc: the gate's (0 pass, 1 a wrong engine write, 2 error, 3 inconclusive: live-only receipts).
set -uo pipefail
Q=${QUIET_FILE:-$HOME/work/queue/QUIET}  # the QUIET flag; window-hook.sh mirrors rank0-host's onto rank1-host
here=$(cd "$(dirname "$0")" && pwd)
root=$(realpath "$1"); map=$2; out=$(realpath -m "${3:-$HOME/work/owned-oracle/out/window-$(basename "$root")}")
mkdir -p "$out"
bash "$here/gate.sh" "$out/oracle" "$root" > "$out/gate.log" 2>&1; grc=$?
cat "$out/gate.log"
[ $grc -le 1 ] || [ $grc = 3 ] || { echo "window-report: the oracle gate did not run (rc=$grc)"; exit 2; }
held() { grep -E "HELD both locks|RELEASED" ~/work/imr-t4/measure/window.log 2>/dev/null | tail -1 | grep -q HELD; }
t4() { [ ~/work/imr-t4/measure/WINDOW-RUN -nt ~/work/imr-t4/measure/window-END ] 2>/dev/null || head -1 ~/work/queue/ORDER 2>/dev/null | grep -q imr-measure; }
while [ -e "$Q" ] || held || t4; do sleep 30; done
systemd-run --user --scope --quiet -p MemoryMax=2G -p MemorySwapMax=0 nice -n 19 ionice -c3 \
  python3 -I "$here/window_report.py" "$out" "$root" "$map" || { echo "window-report: report failed"; exit 2; }
exit $grc
