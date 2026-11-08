#!/usr/bin/env bash
# recheck.sh OUT ROOT...: recheck.py on rank0-host/rank1-host (never dev-host) in a 2 GiB no-swap scope at nice 19 / ionice c3,
# after waiting out QUIET, a held T4 window and an imr-measure at the head of ORDER (as run.sh)
set -uo pipefail
Q=${QUIET_FILE:-$HOME/work/queue/QUIET}  # the QUIET flag; window-hook.sh mirrors rank0-host's onto rank1-host
here=$(cd "$(dirname "$0")" && pwd)
held() { grep -E "HELD both locks|RELEASED" ~/work/imr-t4/measure/window.log 2>/dev/null | tail -1 | grep -q HELD; }
t4() { [ ~/work/imr-t4/measure/WINDOW-RUN -nt ~/work/imr-t4/measure/window-END ] 2>/dev/null || head -1 ~/work/queue/ORDER 2>/dev/null | grep -q imr-measure; }
while [ -e "$Q" ] || held || t4; do sleep 30; done
systemd-run --user --scope --quiet -p MemoryMax=2G -p MemorySwapMax=0 nice -n 19 ionice -c3 python3 -I "$here/recheck.py" "$@"
