#!/usr/bin/env bash
# The several-slot drafting gate's two ranks on this host's two GPUs: rank 1 follows, rank 0 serves the traces with
# drafts (tf-dsv41-m1 mdraft: batched, then alone, each == the Python traces). Every rank gets the same environment.
#   bash ranks.sh BIN PACK ASSETS OUT TRACE...   (env: TF_DSV41_SLOTS, TF_DSV41_SLOT_DRAFTS, TF_DSV41_GRAPHS, ...)
set -u
BIN=$1 PACK=$2 ASSETS=$3 OUT=$4
shift 4
mkdir -p "$OUT"
common=(TF_TP_WORLD=2 TF_TP_PORT=${MDRAFT_PORT:-29770} TF_DSV41_FAILFAST=0 TF_DSV41_ASSETS="$ASSETS")
env "${common[@]}" TF_TP_RANK=1 TF_TP_DEVICE=1 "$BIN" follow "$PACK" "$ASSETS" > "$OUT/rank1.log" 2>&1 &
follower=$!
env "${common[@]}" TF_TP_RANK=0 TF_TP_DEVICE=0 "$BIN" mdraft "$PACK" "$ASSETS" "$@" > "$OUT/rank0.log" 2>&1
st=$?
wait "$follower" || st=1
grep -E "^(PASS|FAIL) mdraft|^FAIL tf-dsv41-m1|mdraft stats|mdraft gate|slot drafts|row windows|row graphs" "$OUT/rank0.log"
tail -2 "$OUT/rank1.log"
exit $st
