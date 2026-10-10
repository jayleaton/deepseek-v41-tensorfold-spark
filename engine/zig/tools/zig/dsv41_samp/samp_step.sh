#!/usr/bin/env bash
# The keyed sampling gate (gate) on TWO GPUs of one pod, inside M1_JOB=m2b once the pack is local and an
# M2b reference's assets exist (aot/, rope.bin, engram-host.bin):
#   1. samp-kernels: tf-dsv41-samp on GPU 0, the device steps (topk_keys + sampling.cu) against the host reference
#      ("PASS tf-dsv41-samp: 0 checks failed", under a minute);
#   2. samp-ref: the Python engine at TP=2, layers 0-HI, one sampled reply a prompt (tools/zig/dsv41_samp_ref.py:
#      Python's candidates, nucleus statistics, whole-row fallback and Batcher._choose, serial windows) into
#      LOCAL/samp-ref/trace.jsonl; skipped when SAMP_TRACE names a trace already written (dsv41_samp_ref.run_cases
#      from the M2b reference's own load);
#   3. samp-serial / samp-drafts: tf-dsv41-m1 generate on that trace (rank 1 following on GPU 1), drafts off then on:
#      every reply's tokens == Python's for the same request and seed ("PASS M4 CLI == HTTP: N/N replies equal" in
#      each log; the line's wording is generate's, here it is Zig == Python); SAMP_MODES token slots4: the same trace
#      with TF_DSV41_SLOTS=4 and 4 requests in flight (TF_DSV41_GEN_STREAMS), drafts off: batched sampling == Python.
#
#   bash samp_step.sh Z LP ASSETS HI RES LOCAL
# Env: SAMP_STEPS (64), SAMP_PROMPTS (37,50,66,83,128,292: with SAMP_OWN_PREFILL=1 each length needs its Triton
# variants in ASSETS/aot, dsv41_m2b_ref.py --aot-prompts), SAMP_OWN_PREFILL (1), SAMP_GRAPHS (1), SAMP_TRACE, SAMP_MIN
# (the minutes a step may take, 12), SAMP_KERNELS (1: the samp-kernels step), TF_DSV41_EXPERT_TOPP (as the reference was made; both sides read it).
set -u
Z=$1 LP=$2 A=$3 HI=$4 RES=$5 LOCAL=$6
mkdir -p "$RES" "$LOCAL"
rc=0
run() { # NAME CMD...: output in RES/NAME.log, rc in RES/steps.txt
    local name=$1 t0 r; shift
    t0=$(date +%s)
    timeout --kill-after=30 "$(( ${SAMP_MIN:-12} * 60 ))" "$@" > "$RES/$name.log" 2>&1; r=$?
    echo "$name rc=$r s=$(( $(date +%s) - t0 ))" | tee -a "$RES/steps.txt"
    grep -h "^PASS\|^FAIL\|candidates k\|stats T\|picks T\|first_differing\|sampled [0-9]" "$RES/$name.log" | tail -20
    (( r == 0 )) || rc=1
    return $r
}
[[ "${SAMP_KERNELS:-1}" == 1 ]] && CUDA_VISIBLE_DEVICES=0 run samp-kernels "$Z/bin/tf-dsv41-samp"
TRACE=${SAMP_TRACE:-}
if [[ -z "$TRACE" ]]; then
    TRACE=$LOCAL/samp-ref/trace.jsonl
    unset TF_DSV41_R1
    run samp-ref python -u -B "$Z/tools/zig/dsv41_samp_ref.py" --pack "$LP" --out "$LOCAL/samp-ref" --layers "0-$HI" \
        --steps "${SAMP_STEPS:-64}" --prompts "${SAMP_PROMPTS:-37,50,66,83,128,292}" --port 29641
fi
cp "$TRACE" "$RES/samp-trace.jsonl" 2>/dev/null
if [[ -s "$TRACE" ]]; then
    env=(TF_DSV41_SAMPLING=1 TF_TP_WORLD=2 TF_DSV41_ASSETS="$A" TF_DSV41_LAYERS="0-$HI" TF_DSV41_GRAPHS="${SAMP_GRAPHS:-1}"
         TF_DSV41_OWN_PREFILL="${SAMP_OWN_PREFILL:-1}" TF_DSV41_FAILFAST=0
         TF_DSV41_PROMPT_TAIL=prefill)   # the reference's replies follow Forward.prompt (dsv41_samp_ref.py)
    for mode in ${SAMP_MODES:-serial drafts}; do
        d=0; [[ $mode == drafts ]] && d=1
        port=$(( 29881 + d ))
        sl=()   # slots4: 4 live slots, 4 replies in flight (batched row windows, a sampler a slot's rows), drafts off
        [[ $mode == slots4 ]] && { sl=(TF_DSV41_SLOTS=4 TF_DSV41_GEN_STREAMS=4 TF_DSV41_POOL_TOKENS=$(( 4 * 4096 + 2048 ))); port=29885; }
        env "${env[@]}" "${sl[@]}" TF_DSV41_DRAFTS=$d TF_TP_RANK=1 TF_TP_DEVICE=1 TF_TP_PORT=$port \
            "$Z/bin/tf-dsv41-m1" follow "$LP" "$A" > "$RES/samp-follow-$mode.log" 2>&1 & fp=$!
        run "samp-$mode" env "${env[@]}" "${sl[@]}" TF_DSV41_DRAFTS=$d TF_TP_RANK=0 TF_TP_DEVICE=0 TF_TP_PORT=$port \
            "$Z/bin/tf-dsv41-m1" generate "$LP" "$A" "$TRACE"
        wait $fp
    done
else
    echo "samp: no reference trace" | tee -a "$RES/steps.txt"; rc=1
fi
exit $rc
