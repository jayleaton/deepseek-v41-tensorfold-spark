#!/usr/bin/env bash
# owned-oracle on rank0-host/rank1-host (never dev-host): extract in a 2 GiB scope at nice 19, evaluate in a no-network,
# read-only container (2 GiB, 1 CPU, cpuset 0-4, nice 19) paused under QUIET, then report. Waits out QUIET, a held
# T4 window and an imr-measure at the head of ORDER before each step.
#   run.sh OUT ROOT...      e.g. run.sh ~/work/owned-oracle/out/today ~/work/dsv41-bench/out/imr-20261102T115935Z
set -uo pipefail
Q=${QUIET_FILE:-$HOME/work/queue/QUIET}  # the QUIET flag; window-hook.sh mirrors rank0-host's onto rank1-host
here=$(cd "$(dirname "$0")" && pwd)
out=$(realpath -m "$1"); shift
mkdir -p "$out"
img=${ORACLE_IMAGE:-tensorfold-dsv41-imr:r-6cdaf1bed8ae}
held() { grep -E "HELD both locks|RELEASED" ~/work/imr-t4/measure/window.log 2>/dev/null | tail -1 | grep -q HELD; }
t4() { [ ~/work/imr-t4/measure/WINDOW-RUN -nt ~/work/imr-t4/measure/window-END ] 2>/dev/null || head -1 ~/work/queue/ORDER 2>/dev/null | grep -q imr-measure; }
idle() { while [ -e "$Q" ] || held || t4; do sleep 30; done; }
scope() { systemd-run --user --scope --quiet -p MemoryMax=2G -p MemorySwapMax=0 nice -n 19 ionice -c3 "$@"; }

idle
start=$(date +%s)
scope python3 -I "$here/oracle.py" extract "$out" "$@" || exit 1
idle
name=owned-oracle-$$
docker run --rm --name "$name" --network none --read-only --tmpfs /tmp:rw,size=256m --memory 2g --memory-swap 2g \
  --cpus 1 --cpuset-cpus 0-4 --pids-limit 256 -u "$(id -u):$(id -g)" -v "$here:/tool:ro" -v "$out:/out" -e HOME=/tmp \
  --entrypoint /usr/bin/nice "$img" -n 19 python3 -I /tool/oracle.py evaluate /out &
pid=$!
paused=0
while kill -0 $pid 2>/dev/null; do
  if { [ -e "$Q" ] || held; } && [ $paused = 0 ]; then docker pause "$name" >/dev/null 2>&1 && paused=1; fi
  if [ ! -e "$Q" ] && ! held && [ $paused = 1 ]; then docker unpause "$name" >/dev/null 2>&1 && paused=0; fi
  sleep 5
done
wait $pid; rc=$?
docker kill "$name" >/dev/null 2>&1
[ $rc = 0 ] || { echo "evaluate failed rc=$rc"; exit 1; }
scope python3 -I "$here/oracle.py" report "$out"
echo "owned-oracle: $(( $(date +%s) - start )) s; report $out/report.md"
