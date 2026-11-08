#!/usr/bin/env bash
# On rank0-host, detached: the §14 measurement window as one command. TE's imr-turn.sh flow, unchanged in its holds and
# locks: queue as $me, wait (bounded 12 h) for ORDER's head, STOP the current efast2 serving run, load the image on
# rank1-host, take both locks (rank1-host first, read-only fd as root), run imr-measure.py over window-configs.json (boots
# off -> arms a,d; cr -> runner fence,repl; inl -> runner +inline; each: gates, placement/decode check, the shared
# prompt set, replay), release, hand ORDER to engine-Gr and run g-restore.sh. Then, with G back, the report:
# analyze.py at nice 19 / idle IO, paused under QUIET, inside README §5's 2 GiB (grading containers are 512 MiB each).
# Any failure after the G stop still restores G (TE's restore_g). rank1-host's docker stores the image under another ID.
# The image is read at the head, not at launch: TE's build turn writes imr-te/IMAGE2 (<sha> <rank0-host id> <rank1-host id>)
# after it loads rank1-host, and turn2.log 'imr-turn rc=0' after its re-smoke. Both must be there and pass (sha, smoke rc=0, both IDs and
# revision labels in docker, the imr base on the same IDs), or this window leaves ORDER before stopping G.
# Queues directly after $T4_AFTER (default think) if that entry is in ORDER, else at the end.
# Usage: IMR_SHA=<full engine sha> [T4_SAMPLE=all] [T4_CONFIGS=window3-configs.json] [T4_TASKS=<tasks file>]
#        bash window.sh   (IMR_IMAGE/IMR_IMAGE_R1 override the receipt)
# Log: $M/window.log; END in window-END; receipts in ~/work/dsv41-bench/out/<WINDOW-RUN>.
set -uo pipefail
M=$(dirname "$(readlink -f "$0")")  # this kit copy (each AB3 turn dir runs its own files)
TE=<rank0-work>/imr-te
q=<rank0-work>/queue/ORDER
B=<rank0-work>/dsv41-bench
ce=<rank0-work>/r5/cand-e
ecal=<rank0-work>/r5/ecal
me=${T4_QUEUE:-imr-measure}
exec >> "$M/window.log" 2>&1
log() { echo "$(date -u +%FT%TZ) $*"; }
unfence() { [[ -n ${fenced:-} ]] && { log "$(bash "$M/w9-fence.sh" off 8000 2>&1)"; fenced=; }; }
fin() { unfence; echo "rc=$1 $(date -u +%FT%TZ)" > "$M/window-END"; log "END rc=$1"; exit "$1"; }
free() {
  [[ -z $(fuser /tmp/rank1-host-gpu.lock 2>/dev/null) && -z $(ssh -n -o BatchMode=yes -o ConnectTimeout=10 root@<rank1-link-ip> 'fuser /tmp/rank1-host-gpu.lock 2>/dev/null') ]]
}
: "${IMR_SHA:?}"
after=${T4_AFTER:-think}
leave() {
  # out of ORDER before anything was stopped: G keeps serving
  sed -i "/^$me\$/d" $q
  log "left ORDER ($1): $(tr '\n' ' ' < $q)"
  fin "$2"
}
cd "$M" || fin 2
tasks=${T4_TASKS:-$M/tasks.jsonl}  # W9: a frozen block file from T5's manifest (the configs name the same file)
[[ -n ${T4_TASKS:-} ]] && { test -s "$tasks" || fin 2; }
test -s "$tasks" || python3 -I tasks.py build data/sanitized-mbpp.json "$tasks" || fin 2
log "tasks $(wc -l < "$tasks") sha256 $(sha256sum "$tasks" | cut -c1-16) sample ${T4_SAMPLE:-first}" \
  "sets ${T4_SETS:-all}"
# W9: the block before this one gets its safety gate (r7 decides STOP) before this turn queues, so a wrong engine byte
# stops the next turn (T2's gate.sh waits while an imr-measure window holds the head, so it cannot run mid-turn)
if [[ -n ${T4_PIN:-} && $me =~ ^w9-([0-9]+)$ ]] && ((10#${BASH_REMATCH[1]} > 1)); then
  prev=$(printf %02d $((10#${BASH_REMATCH[1]} - 1)))
  prun=$(sed -n "s/.*TURN-END $prev rc=0 run=\([^ ]*\).*/\1/p" "$M/w9/turns.log" | tail -1)
  pg=<rank0-work>/imr-t4/w9gate/$prev
  if [[ -n $prun && ! $pg/report-r7/w9-report.json -nt $B/out/$prun/status.json ]]; then
    mkdir -p "$pg"
    systemd-run --user --scope -q -p MemoryMax=2G nice -n 19 ionice -c3 \
      bash "$M/w9-block-gate.sh" "$prev" "$prun" > "$pg/auto.log" 2>&1 < /dev/null
    log "block $prev gate rc=$?: $(grep -E '^safety|^gates|STOP' "$pg/auto.log" | tr '\n' ' ' | cut -c1-400)"
  fi
  test -e "$M/w9/STOP" && leave "W9 STOP: $(head -1 "$M/w9/STOP")" 8
fi
if ! grep -qx "$me" $q; then
  if grep -qx "$after" $q; then sed -i "/^$after\$/a $me" $q; else echo "$me" >> $q; fi
fi
log "queued: $(tr '\n' ' ' < $q)"
deadline=$((SECONDS + 43200))
until [[ $(head -1 $q) == "$me" ]]; do ((SECONDS > deadline)) && leave 'not at the head in 12 h' 4; sleep 10; done
log "AT HEAD"
# W9: a block gate found a wrong engine byte (w9-block-gate.sh wrote it): no more W9 turns
test -e "$M/w9/STOP" && leave "W9 STOP: $(head -1 "$M/w9/STOP")" 8
# the image, read now: TE's receipts from the build turn just before this one
read -r rsha img loaded < "$TE/IMAGE2" 2>/dev/null
base=imr
# W9: the qualified image, pinned (T4_PIN="<sha> <rank0-host id> <rank1-host id> <arms base>"): TE's IMAGE2/arms-imr.json move on
# with its next build; this turn checks the pinned image's labels on both nodes and its own frozen arms file instead
[[ -n ${T4_PIN:-} ]] && read -r rsha img loaded base <<< "$T4_PIN"
IMR_IMAGE=${IMR_IMAGE:-${img:-}}
R1=${IMR_IMAGE_R1:-${loaded:-}}
# only a re-smoke rc written after this image's rank1-host line counts (an earlier turn's rc=0 says nothing about it)
smoke=$(sed -n "/image rank1-host ${R1:-none} /,\$p" "$TE/turn2.log" 2>/dev/null | grep -o 'imr-turn rc=[0-9]*' | tail -1)
log "IMAGE2: ${rsha:-none} ${img:-} ${loaded:-}; turn2.log: ${smoke:-no re-smoke}"
[[ ${rsha:-} == "$IMR_SHA" ]] || leave "IMAGE2 is not $IMR_SHA" 6
[[ $IMR_IMAGE == sha256:* && $R1 == sha256:* ]] || leave 'no image IDs' 6
# the re-smoke's rc, from the turn log: turn2-END comes only after G is back, maybe after this window is at the head
[[ -n ${T4_PIN:-} ]] || { [[ $TE/turn2.log -nt $TE/IMAGE2 && ${smoke:-} == 'imr-turn rc=0' ]] &&
  grep -q "image rank1-host $R1 " "$TE/turn2.log"; } || leave 'TE re-smoke did not end rc=0' 6
lab='{{.Id}} {{index .Config.Labels "org.tensorfold.revision"}}'
[[ $(docker image inspect -f "$lab" "$IMR_IMAGE" 2>/dev/null) == "$IMR_IMAGE $IMR_SHA" ]] || leave 'rank0-host image ID/revision' 6
[[ $(ssh -n root@<rank1-link-ip> "docker image inspect -f '$lab' $R1" 2>/dev/null) == "$R1 $IMR_SHA" ]] || leave 'rank1-host image ID/revision' 6
python3 -I -c 'import json, sys; d = json.load(open(sys.argv[1]))["arms"]["D"]
sys.exit(not (d["engine_sha"] == sys.argv[2] and [r["image_id"] for r in d["ranks"]] == sys.argv[3:]))' \
  "$B/arms-$base.json" "$IMR_SHA" "$IMR_IMAGE" "$R1" || leave "arms-$base.json is on another image" 6
grep -q '@BANK' "$M/${T4_CONFIGS:-window-configs.json}" && leave 'configs still hold the @BANK placeholder' 2
python3 -B make_window_driver.py "$TE/imr-turn.py" "$M/imr-measure.py" || leave 'window driver' 2
log "image ok: rank0-host $IMR_IMAGE rank1-host $R1 ($IMR_SHA, arms-$base${T4_PIN:+, pinned})"
report() {
  [[ -n ${run:-} ]] || return 0
  while test -e <rank0-work>/queue/QUIET; do sleep 10; done
  setsid nice -n 19 ionice -c3 timeout 7200 python3 -B "$M/analyze.py" --run "$B/out/$run" --tasks "$tasks" \
    > "$M/analyze-$run.log" 2>&1 < /dev/null &
  local apid=$!
  while kill -0 "$apid" 2>/dev/null; do
    if test -e <rank0-work>/queue/QUIET; then kill -STOP "$apid"; else kill -CONT "$apid"; fi
    sleep 5
  done
  wait "$apid"; log "analyze rc=$? -> $B/out/$run/report.md"
}
restore_g() {
  local rc=$1
  unfence
  if test -e <rank0-work>/imr-t4/bench-quiet.state; then
    bash "$M/bench-quiet.sh" restore > "$M/bench-quiet.log" 2>&1
    log "apt restore rc=$? $(tr '\n' ' ' < "$M/bench-quiet.log")"
  fi
  # G back: engine-Gr takes the head before this window pops itself, so nothing else boots in between
  if [[ $(head -1 $q) == "$me" ]]; then
    grep -qx engine-Gr $q || sed -i '1a engine-Gr' $q
    sed -i 1d $q
  fi
  log "ORDER: $(tr '\n' ' ' < $q)"
  local before
  before=$(stat -c %Y "$ecal/Gr-END" 2>/dev/null || echo 0)
  setsid bash "$ecal/g-restore.sh" > "$ecal/g-restore-measure.log" 2>&1 < /dev/null &
  for ((i=0; i<300; i++)); do (( $(stat -c %Y "$ecal/Gr-END" 2>/dev/null || echo 0) > before )) && break; sleep 15; done
  log "g-restore: $(cat "$ecal/Gr-END" 2>/dev/null || echo 'no END after 75 min')"
  log "/v1/models $(curl -s -o /dev/null -w '%{http_code}' --max-time 20 http://127.0.0.1:8000/v1/models)"
  report
  # T2's window report (Lead OK): detached, off the locks, on rank1-host, waits out QUIET; after analyze for grades.jsonl
  [[ -n ${run:-} && -d $B/out/$run ]] && setsid nohup bash <rank0-work>/owned-oracle/tool/window-hook.sh "$B/out/$run" auto > /dev/null 2>&1 < /dev/null &
  grep -q '^rc=0' "$ecal/Gr-END" 2>/dev/null || fin 7
  fin "$rc"
}
log "MY TURN"
fr=$(cat "$ce/EFAST2-RUN" 2>/dev/null)
if [[ -n $fr ]] && grep -q '"serving": true' "$B/out/$fr/status.json" 2>/dev/null; then
  touch "$B/out/$fr/STOP"; log "STOP serving ($fr)"
fi
for ((i=0; i<180; i++)); do free && break; sleep 10; done
free || { log 'locks not free 30 min after STOP'; restore_g 3; }
log "locks free"
coproc LEASE { exec ssh -o BatchMode=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 root@<rank1-link-ip> \
  'bash -c ": imr-measure-lease; exec 9</tmp/rank1-host-gpu.lock || exit 3; flock 9 || exit 3; echo LOCKED; cat >/dev/null"'; }
read -r -t 900 locked <&"${LEASE[0]}"
[[ ${locked:-} == LOCKED ]] || { log 'rank1-host lease failed'; exec {LEASE[1]}>&- 2>/dev/null; restore_g 3; }
exec 8>/tmp/rank1-host-gpu.lock
flock 8
log "HELD both locks"
if [[ -n ${T4_PIN:-} ]]; then
  # W9: the ranks serve on the host network at localhost:8000; only this driver's own scope may connect while the
  # window holds the locks (JJ's local client reached block 9's on-G1 arm), so :8000 is not G until RELEASED
  fenced=1; fo=$(bash "$M/w9-fence.sh" on "" 8000 2>&1); log "$fo"
  [[ $fo == *'fence on'* ]] || { log 'fence did not go on: no W9 requests unfenced'; restore_g 3; }
  # a requested preflight (w9/PREFLIGHT-NEXT, one shot) on this turn's first boot
  if test -e "$M/w9/PREFLIGHT-NEXT"; then export T4_PREFLIGHT_DEADLINE=1; mv "$M/w9/PREFLIGHT-NEXT" "$M/w9/PREFLIGHT-NEXT.used-$me"; fi
fi
# window rules: the hourly dashboard apt (both nodes) and rank1-host's apt-daily paused for the turn, restored after
bash "$M/bench-quiet.sh" pause > "$M/bench-quiet.log" 2>&1; log "apt pause rc=$? $(tr '\n' ' ' < "$M/bench-quiet.log")"
avail=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9); log "df ${avail}G"
rc=9
if ((avail > 100)); then
  run=imr-measure-$(date -u +%Y%m%dT%H%M%SZ)
  echo "$run" > "$M/WINDOW-RUN"
  cd "$B" && EF_NAME=$me EF_BASE=$base EF_SHA=$IMR_SHA python3 -B "$M/imr-measure.py" "$run" "$M/${T4_CONFIGS:-window-configs.json}" > "$M/imr-measure.log" 2>&1
  rc=$?
  log "imr-measure rc=$rc ($run)"
fi
flock -u 8; exec 8>&-
exec {LEASE[1]}>&-
wait "$LEASE_PID" 2>/dev/null
log "RELEASED"
unfence
restore_g "$rc"
