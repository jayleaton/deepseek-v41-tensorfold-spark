#!/usr/bin/env bash
# M2b's two ranks on this host's two GPUs (NCCL through the tp runtime): tf-dsv41-m1 m2b a rank, in parallel.
#   bash m2b_ranks.sh BIN PACK REF OUT
set -u
BIN=$1 PACK=$2 REF=$3 OUT=$4
mkdir -p "$OUT"
pids=()
for r in 0 1; do
    # M2B_PROF=1: each rank's eager windows timed into OUT/prof-rank<r>.json (profile.zig)
    prof=(); [[ "${M2B_PROF:-0}" == 1 ]] && prof=(TF_DSV41_PROFILE="$OUT/prof-rank$r.json")
    env "${prof[@]}" TF_TP_WORLD=2 TF_TP_RANK=$r TF_TP_DEVICE=$r TF_TP_PORT=${M2B_PORT:-29700} TF_DSV41_FAILFAST=0 \
        "$BIN" m2b "$PACK" "$REF" "$OUT" > "$OUT/rank$r.log" 2>&1 &
    pids+=($!)
done
st=0
for p in "${pids[@]}"; do wait "$p" || st=1; done
tail -3 "$OUT/rank0.log" "$OUT/rank1.log"
exit $st
