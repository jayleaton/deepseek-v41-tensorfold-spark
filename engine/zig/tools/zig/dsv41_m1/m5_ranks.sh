#!/usr/bin/env bash
# M5's gate on this host's two GPUs (NCCL through the tp runtime): tf-dsv41-m1 m5 MODE a rank, in parallel.
#   bash m5_ranks.sh BIN MODE PACK REF OUT
set -u
BIN=$1 MODE=$2 PACK=$3 REF=$4 OUT=$5
mkdir -p "$OUT"
pids=()
for r in 0 1; do
    TF_TP_WORLD=2 TF_TP_RANK=$r TF_TP_DEVICE=$r TF_TP_PORT=${TF_TP_PORT:-29800} TF_DSV41_FAILFAST=0 \
        "$BIN" m5 "$MODE" "$PACK" "$REF" "$OUT" > "$OUT/rank$r.log" 2>&1 &
    pids+=($!)
done
st=0
for p in "${pids[@]}"; do wait "$p" || st=1; done
tail -3 "$OUT/rank0.log" "$OUT/rank1.log"
exit $st
