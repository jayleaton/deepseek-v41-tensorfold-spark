# The several-slots gate (gate), a fragment the combined pod job sources on a TWO-GPU pod after the pack's
# local copy (LP: the shards up to layer $hi). Needs: Z (staged tree), LP, LOCAL, RES, hi, the `step` / `left`
# helpers of tools/zig/dsv41_m1/job.sh, and a Python with rowgraphs + slots (SLOTS_PYSRC; the decode1 twin's src).
#   1. slots-ref: tools/zig/dsv41_slots/ref.py at TP=2: 4 slots in a replicated pool, prompts 45 / 173 / 9 / 640 (random
#      ids, seed 4101) prefilled alone, then SLOTS_STEPS greedy steps of every live slot in one row-mode run (row graphs:
#      serving's path), every row-mode bucket up to 16 warmed for the AOT set; Engram's host tables beside it
#   2. slots-s4-eager, slots-s4-graphs, slots-s2-graphs, slots-s1: our two ranks (ranks.sh) serving the trace through the
#      lanes engine with TF_DSV41_SLOTS 4 / 4 / 2 / 1 (1: today's one-slot path), graphs off / on / on / off. Each: every
#      reply batched == Python, alone == Python, batched == alone ("PASS slots ..." x 3). A run named *-rgraphs (e.g.
#      s4-rgraphs) is graphs on plus TF_DSV41_ROUND_GRAPH=1 (the window as head + tail graphs, the gate's event wait):
#      the same three PASS lines prove the split round == Python == the eager round.
# Knobs: SLOTS_PROMPTS (45,173,9,640), SLOTS_STEPS (64), SLOTS_RUNS ("s4-eager s4-graphs s2-graphs s1"), SLOTS_MIN (a
# run's minutes, 8).
slots_gate() {
    local ref=$LOCAL/slots-ref pysrc=${SLOTS_PYSRC:-/dsv41-tf/src} pool
    rm -rf "$ref"
    unset TF_DSV41_KV_SPLIT
    if ! { PYTHONPATH=$pysrc step slots-ref "${SLOTS_REF_MIN:-14}" python -u -B "$Z/tools/zig/dsv41_slots/ref.py" \
            --pack "$LP" --out "$ref" --layers "0-$hi" --slots 4 --prompts "${SLOTS_PROMPTS:-45,173,9,640}" \
            --steps "${SLOTS_STEPS:-64}" \
         && step slots-engram-host 3 python -B "$Z/tools/zig/dsv41_engram_host.py" --py-src "$pysrc" \
            --config "$LP/config.json" --tokenizer "$LP/tokenizer.json" --out "$ref/engram-host.bin"; }; then
        echo "slots gate: no reference" | tee -a "$RES/steps.txt"
        return 1
    fi
    mkdir -p "$RES/slots"
    cp "$ref/ref.json" "$ref/trace.jsonl" "$RES/slots/"
    pool=$(( 4 * 4096 + 2048 ))
    local run s g rg
    for run in ${SLOTS_RUNS:-s4-eager s4-graphs s2-graphs s1}; do
        s=${run%%-*}; s=${s#s}; g=0; [[ "$run" == *graphs ]] && g=1
        rg=0; [[ "$run" == *rgraphs ]] && rg=1
        (( $(left) >= 3 )) || { echo "slots-$run skipped: $(left) min left" | tee -a "$RES/steps.txt"; continue; }
        TF_DSV41_SLOTS=$s TF_DSV41_GRAPHS=$g TF_DSV41_ROUND_GRAPH=$rg TF_DSV41_POOL_TOKENS=$pool TF_DSV41_LAYERS=0-$hi \
            step "slots-$run" "${SLOTS_MIN:-8}" bash "$Z/tools/zig/dsv41_slots/ranks.sh" "$Z/bin/tf-dsv41-m1" "$LP" \
            "$ref" "$ref/trace.jsonl" "$RES/slots/$run"
    done
}
