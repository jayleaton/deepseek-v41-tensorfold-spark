#!/usr/bin/env bash
# The sessions-over-slots gate's two ranks on this host's two GPUs: rank 1 follows (tf-dsv41-m1 follow), rank 0 serves
# the chats (tf-dsv41-m1 sess4). Every rank gets the same environment (TF_DSV41_SLOTS, TF_DSV41_SESSIONS, ...).
#   bash ranks.sh BIN PACK ASSETS OUT REF...
set -u
BIN=$1 PACK=$2 ASSETS=$3 OUT=$4
shift 4
mkdir -p "$OUT"
common=(TF_TP_WORLD=2 TF_TP_PORT=${SESS4_PORT:-29780} TF_DSV41_FAILFAST=0 TF_DSV41_ASSETS="$ASSETS")
env "${common[@]}" TF_TP_RANK=1 TF_TP_DEVICE=1 "$BIN" follow "$PACK" "$ASSETS" > "$OUT/rank1.log" 2>&1 &
follower=$!
env "${common[@]}" TF_TP_RANK=0 TF_TP_DEVICE=0 "$BIN" sess4 "$PACK" "$ASSETS" "$OUT" "$@" > "$OUT/rank0.log" 2>&1
st=$?
wait "$follower" || st=1
grep -E "^(PASS|FAIL) sess4|^sess4|slots: |row graphs" "$OUT/rank0.log"
tail -2 "$OUT/rank1.log"
exit $st
