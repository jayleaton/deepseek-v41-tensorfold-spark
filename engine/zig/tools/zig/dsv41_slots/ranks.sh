#!/usr/bin/env bash
# The slots gate's two ranks on this host's two GPUs: rank 1 follows (tf-dsv41-m1 follow), rank 0 serves the trace
# (tf-dsv41-m1 slots: batched, then alone, each == the Python trace). Every rank gets the same environment.
#   bash ranks.sh BIN PACK ASSETS TRACE OUT     (env: TF_DSV41_SLOTS, TF_DSV41_GRAPHS, TF_DSV41_LAYERS, ...)
set -u
BIN=$1 PACK=$2 ASSETS=$3 TRACE=$4 OUT=$5
mkdir -p "$OUT"
common=(TF_TP_WORLD=2 TF_TP_PORT=${SLOTS_PORT:-29760} TF_DSV41_FAILFAST=0 TF_DSV41_ASSETS="$ASSETS" TF_DSV41_DRAFTS=0)
env "${common[@]}" TF_TP_RANK=1 TF_TP_DEVICE=1 "$BIN" follow "$PACK" "$ASSETS" > "$OUT/rank1.log" 2>&1 &
follower=$!
env "${common[@]}" TF_TP_RANK=0 TF_TP_DEVICE=0 "$BIN" slots "$PACK" "$ASSETS" "$TRACE" > "$OUT/rank0.log" 2>&1
st=$?
wait "$follower" || st=1
grep -E "^(PASS|FAIL) slots|slots timing|row windows|row graphs" "$OUT/rank0.log"
tail -2 "$OUT/rank1.log"
exit $st
