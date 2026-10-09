#!/usr/bin/env bash
# The single-pass window's exactness gate (quant repo results/zig-window-plan.md, step 2): ONE M2c-2 run on both ranks
# with every gated knob on, against the Python reference (prod's tree 8474f31) with the same knobs:
#   decode gate  RT=1: L2PF, ENGRAM_GATE, BRANCHES + PRIO=side, MHC_DEFER, GLM53_TF_ROCE_FAST, GREEDY_GPU (+ R1)
#   prefill gate PREFILL=1 PREFILL_MODE=replay (prompt tail verify) GM_V2=1, a two-segment prompt (2,600 tokens)
#   each: 512 tokens + logits equal on both ranks.
# On a FAIL it reruns ONCE with the failing stage's knobs dropped in a fixed order (newest / riskiest first) and writes
# OUT/gate-drops.env (KEY=0 / KEY= lines for `scripts/v2-rollback.sh zig <file>`) and OUT/gate.json. No bisection:
# the dropped set is the suspects, logged for an offline owner.
#   Decode: a `tp roce probe: ... DIFFERS` line drops GLM53_TF_ROCE_FAST alone; otherwise the first DROP_DECODE_N
#   (default all 6) of GREEDY_GPU, ROCE_FAST, ENGRAM_GATE, BRANCHES (+ PRIO), MHC_DEFER, L2PF. The rerun reuses the
#   reference (SKIP_REF=1).
#   Prefill: replay and GM_V2 both dropped (full prefill, x3gm v1); the rerun makes a new reference (full mode).
# Exit 0: PASS (drops possible: read gate-drops.env); 1: the rerun failed too (Zig cannot take prod this window).
# Run on rank 0 host from the kit dir, prod down, the caller holding the lease (zig-window-run.sh):
#   ./zig-gate.sh NAME          -> $KIT/out/NAME on both hosts
# STEPS (512), SEEDS (4101), DROP_DECODE_N (6).
set -u
NAME=${1:?out name}
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
W=${WORKER:?set WORKER to the SSH destination}; WKIT=${WKIT:?set WKIT to the remote kit directory}
OUT=$KIT/out/$NAME; WOUT=$WKIT/out/$NAME; mkdir -p "$OUT"
SEED=${SEEDS:-4101}
log() { echo "[zig-gate $(date +%T)] $*" | tee -a "$OUT/gate.log"; }
wssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$W" "$@"; }
# newest / riskiest first: the knob, its prod-zig.env drop line, RT_KNOBS words it removes
DECODE_ORDER=(GREEDY_GPU ROCE_FAST ENGRAM_GATE BRANCHES MHC_DEFER L2PF)
declare -A DROP_LINE=([GREEDY_GPU]="TF_DSV41_GREEDY_GPU=0" [ROCE_FAST]="GLM53_TF_ROCE_FAST=0" [ENGRAM_GATE]="TF_DSV41_ENGRAM_GATE=0"
    [BRANCHES]="TF_DSV41_BRANCHES=0" [MHC_DEFER]="TF_DSV41_MHC_DEFER=0" [L2PF]="TF_DSV41_L2PF=0")
declare -A RT_WORDS=([GREEDY_GPU]="TF_DSV41_GREEDY_GPU=1" [ROCE_FAST]="GLM53_TF_ROCE_FAST=1" [ENGRAM_GATE]="TF_DSV41_ENGRAM_GATE=1"
    [BRANCHES]="TF_DSV41_BRANCHES=1 TF_DSV41_BRANCHES_PRIO=side" [MHC_DEFER]="TF_DSV41_MHC_DEFER=1" [L2PF]="TF_DSV41_L2PF=1")
ALL_RT="TF_DSV41_L2PF=1 TF_DSV41_ENGRAM_GATE=1 TF_DSV41_BRANCHES=1 TF_DSV41_BRANCHES_PRIO=side TF_DSV41_MHC_DEFER=1 GLM53_TF_ROCE_FAST=1 TF_DSV41_GREEDY_GPU=1"

pair() { # $1 = the env words for run_host.sh on both ranks; both ranks at once, wait for both
    local envw="LEASE_FILE=/nonexistent STEPS=${STEPS:-512} SEEDS=$SEED PROMPT=${PROMPT:-2600} AOT_PROMPTS= $1"
    log "run_host.sh on both ranks: $envw"
    # shellcheck disable=SC2086
    ( cd "$KIT" && env $envw ./run_host.sh 0 "$OUT" > "$OUT/run_host-r0.out" 2>&1 ) &
    local p0=$!
    wssh "cd '$WKIT' && env $envw ./run_host.sh 1 '$WOUT'" > "$OUT/run_host-r1.out" 2>&1 &
    local p1=$!
    wait "$p0"; wait "$p1"
    mkdir -p "$OUT/r1"; wssh "cd '$WOUT/seed$SEED' && tar -cf - *.log 2>/dev/null" | tar -xf - -C "$OUT/r1" 2>/dev/null || log "could not fetch rank 1's logs"
}
logs() { echo "$OUT/seed$SEED/$1 $OUT/r1/$1"; }
decode_ok() { local f; for f in $(logs zig.log); do grep -q '^PASS M2b' "$f" 2>/dev/null || return 1; done; }
prefill_ok() { local f; for f in $(logs prefill.log); do grep -q '^PASS Zig prefill' "$f" 2>/dev/null && grep -q '^PASS M2b' "$f" || return 1; done; }
archive() { local f; for f in zig.log prefill.log; do [[ -f "$OUT/seed$SEED/$f" ]] && cp "$OUT/seed$SEED/$f" "$OUT/seed$SEED/$f.$1"; [[ -f "$OUT/r1/$f" ]] && cp "$OUT/r1/$f" "$OUT/r1/$f.$1"; done; }

t0=$(date +%s)
: > "$OUT/gate-drops.env"
pair "RT=1 PREFILL=1 PREFILL_MODE=replay GM_V2=1"
d1=pass; decode_ok || d1=fail; p1=pass; prefill_ok || p1=fail
log "run 1: decode $d1, prefill $p1 ($(( ($(date +%s) - t0) / 60 )) min)"
dropped=(); d2=$d1; p2=$p1
if [[ $d1 == fail || $p1 == fail ]]; then
    archive run1
    rt="$ALL_RT"; envw="RT=1 SKIP_REF=1"
    if [[ $d1 == fail ]]; then
        if cat $(logs zig.log) 2>/dev/null | grep -q 'tp roce probe: .*DIFFERS'; then sus=(ROCE_FAST)
        else sus=("${DECODE_ORDER[@]:0:${DROP_DECODE_N:-6}}"); fi
        for k in "${sus[@]}"; do
            for wd in ${RT_WORDS[$k]}; do rt=$(echo " $rt " | sed "s| $wd | |"); done
            dropped+=("$k"); echo "${DROP_LINE[$k]}" >> "$OUT/gate-drops.env"
        done
    else envw="$envw DECODE=0"; fi
    if [[ $p1 == fail ]]; then   # full prefill needs a full-mode reference: no SKIP_REF
        envw="${envw/SKIP_REF=1/SKIP_REF=0} PREFILL=1 PREFILL_MODE= GM_V2=0"
        dropped+=(PREFILL_REPLAY GM_V2); printf 'TF_DSV41_PREFILL=\nTF_DSV41_GM_V2=0\n' >> "$OUT/gate-drops.env"
    else envw="$envw PREFILL=0"; fi
    log "rerun once with ${dropped[*]} off"
    rt=$(echo $rt)
    pair "$envw RT_KNOBS=${rt// /,}"
    [[ $d1 == fail ]] && { d2=pass; decode_ok || d2=fail; }
    [[ $p1 == fail ]] && { p2=pass; prefill_ok || p2=fail; }
    log "rerun: decode $d2, prefill $p2"
fi
rc=0; [[ $d2 == pass && $p2 == pass ]] || rc=1
python3 - "$OUT/gate.json" "$d1" "$p1" "$d2" "$p2" "$rc" "$(( $(date +%s) - t0 ))" "${dropped[@]}" <<'PY'
import json, sys
o, d1, p1, d2, p2, rc, s, *drop = sys.argv[1:]
json.dump({"run1": {"decode": d1, "prefill": p1}, "final": {"decode": d2, "prefill": p2}, "pass": rc == "0",
           "dropped": drop, "seconds": int(s)}, open(o, "w"), indent=1)
PY
log "gate $([[ $rc == 0 ]] && echo PASS || echo FAIL): dropped [${dropped[*]}] -> $OUT/gate-drops.env ($(( ($(date +%s) - t0) / 60 )) min)"
exit $rc
