#!/usr/bin/env bash
# window-hook.sh ROOT [MAP]: post-turn hook for experiment turns (T4 window.sh, S2 trace turns, S1 guide turns).
# The turn script starts it detached after G is restored, holding no GPU lock:
#   setsid nohup bash <rank0-work>/owned-oracle/tool/window-hook.sh "<receipts dir>" auto >/dev/null 2>&1 </dev/null &
# On rank0-host it waits out QUIET, copies the receipts and this tool to rank1-host (root, <root-home>/work/t2-window/<run>), runs
# window-report.sh there (2 GiB scopes, nice 19, ionice c3; rank0-host's QUIET is mirrored to rank1-host every 5 s, so the
# oracle container is paused and new steps wait), then puts window-report.md / .json / gate.json and RC into
# ROOT/window-report/ and removes the rank1-host copy. MAP as window-report.sh (default auto). Log: ROOT/window-report.log.
set -uo pipefail
root=$(realpath "$1"); map=${2:-auto}
N2=root@<rank1-link-ip>
tool=$(cd "$(dirname "$0")" && pwd)
name=$(basename "$root")
R=<root-home>/work/t2-window/$name
Q=${QUIET_FILE:-$HOME/work/queue/QUIET}
dst=$root/window-report
mkdir -p "$dst"
exec >> "$root/window-report.log" 2>&1
exec 9> "/tmp/t2-window-hook-$name.lock"
flock -n 9 || { echo "$(date -u +%FT%TZ) another hook runs for $name"; exit 0; }
log() { echo "$(date -u +%FT%TZ) $*"; }
n2() { ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$N2" "$@"; }
log "hook for $root (map $map)"
while [ -e "$Q" ]; do sleep 30; done
img=$(n2 "docker images --format '{{.Repository}}:{{.Tag}}' tensorfold-dsv41-imr | head -1")
[ -n "$img" ] || { log "no tensorfold-dsv41-imr image on rank1-host"; echo 2 > "$dst/RC"; exit 2; }
n2 "mkdir -p $R/receipts $R/tool" || { echo 2 > "$dst/RC"; exit 2; }
(cd "$root" && find . -path ./window-report -prune -o \( -name '*.json' -o -name 'grades.jsonl' \) -type f -print0 | tar --null -T - -cf -) \
  | ssh -o BatchMode=yes "$N2" "tar -xf - -C $R/receipts" || { log "receipt copy failed"; echo 2 > "$dst/RC"; exit 2; }
tar -C "$tool" -cf - oracle.py gate.sh run.sh recheck.py recheck.sh window_report.py window-report.sh detector.json \
  | ssh -o BatchMode=yes "$N2" "tar -xf - -C $R/tool" || { log "tool copy failed"; echo 2 > "$dst/RC"; exit 2; }
log "copied to rank1-host $R; oracle image $img"
ssh -o BatchMode=yes "$N2" "cd $R && env XDG_RUNTIME_DIR=/run/user/0 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/0/bus \
  QUIET_FILE=$R/QUIET ORACLE_IMAGE=$img bash $R/tool/window-report.sh $R/receipts '$map' $R/out > $R/report.log 2>&1; echo \$? > $R/RC" \
  < /dev/null &
job=$!
quiet=0
while kill -0 $job 2>/dev/null; do  # rank0-host's QUIET, mirrored onto rank1-host
  if [ -e "$Q" ] && [ $quiet = 0 ]; then n2 "touch $R/QUIET" && quiet=1 && log "QUIET: paused"; fi
  if [ ! -e "$Q" ] && [ $quiet = 1 ]; then n2 "rm -f $R/QUIET" && quiet=0 && log "QUIET clear: resumed"; fi
  sleep 5
done
wait $job
rc=$(n2 "cat $R/RC" 2>/dev/null || echo 2)
ssh -n -o BatchMode=yes "$N2" "tar -C $R -cf - out/window-report.md out/window-report.json out/oracle/gate.json report.log 2>/dev/null" \
  | tar -xf - -C "$dst" --strip-components=0
[ -f "$dst/out/window-report.md" ] && mv "$dst/out/window-report.md" "$dst/out/window-report.json" "$dst/out/oracle/gate.json" "$dst/" \
  && rm -r "$dst/out"
# the report names the receipts it read: the rank0-host run, not rank1-host's temporary copy
[ -f "$dst/window-report.md" ] && sed -i "s#$R/receipts#$root#g" "$dst/window-report.md" "$dst/window-report.json" "$dst/gate.json"
echo "${rc:-2}" > "$dst/RC"
n2 "rm -rf $R"
log "done rc=${rc:-2}: $(grep -m1 'owned-oracle gate' "$dst/window-report.md" 2>/dev/null)"
exit "${rc:-2}"
