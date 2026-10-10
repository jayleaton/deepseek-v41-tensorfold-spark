#!/usr/bin/env bash
# The vision gate's served replay on this host's two GPUs: rank 1 follows, rank 0 replays the trace through the served
# engine (tf-dsv41-m1 generate: lanes over the GPU target, the image rows from TF_DSV41_VISION_HOLD); every rank gets
# the same environment (TF_DSV41_IMAGES=native, TF_DSV41_LAYERS, ...).
#   bash gen_ranks.sh BIN PACK ASSETS TRACE OUT
set -u
BIN=$1 PACK=$2 ASSETS=$3 TRACE=$4 OUT=$5
mkdir -p "$OUT"
# TF_DSV41_PROMPT_TAIL=prefill: the reference's replies follow Forward.prompt (vision_ref.py)
common=(TF_TP_WORLD=2 TF_TP_PORT=${VISION_PORT:-29790} TF_DSV41_FAILFAST=0 TF_DSV41_ASSETS="$ASSETS" TF_DSV41_DRAFTS=0
        TF_DSV41_PROMPT_TAIL=prefill)
env "${common[@]}" TF_TP_RANK=1 TF_TP_DEVICE=1 "$BIN" follow "$PACK" "$ASSETS" > "$OUT/rank1.log" 2>&1 &
follower=$!
env "${common[@]}" TF_TP_RANK=0 TF_TP_DEVICE=0 "$BIN" generate "$PACK" "$ASSETS" "$TRACE" > "$OUT/rank0.log" 2>&1
st=$?
# a rank 0 that stopped early leaves rank 1 in a collective for good (pod 30's vision-generate-bias timed out so):
# 30 s for it to see the leader go, then it is killed and the step fails with both logs' tails
if (( st != 0 )); then
    for _ in $(seq 30); do kill -0 "$follower" 2>/dev/null || break; sleep 1; done
    kill -9 "$follower" 2>/dev/null && echo "gen_ranks: rank 0 exited $st, rank 1 killed"
fi
wait "$follower" || st=1
grep -E "^(PASS|FAIL) M4|\"equal\"|images" "$OUT/rank0.log"
(( st == 0 )) || tail -25 "$OUT/rank0.log"
tail -${VISION_TAIL1:-2} "$OUT/rank1.log"
exit $st
