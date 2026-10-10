# The multi-segment prefill gate (TF_DSV41_PIECE_RUNS=1, gate), a fragment tools/zig/dsv41_m1/job.sh sources
# inside M1_JOB=m2b on a TWO-GPU pod (M2B_PIECES=1), after the pack's local copy. Needs: Z (staged tree,
# bin/tf-dsv41-m1 built for this GPU), LP (the shards up to layer $hi, hi >= 20: the CED decoder starts at 20), LOCAL,
# RES, and job.sh's `step` / `left`. Python: prod's tree at /dsv41-tf/src (the image's; 8474f31) with its Triton.
#
# What it proves: several prompts admitted in one round prefill as prod's batcher prefills them (one CED encoder run
# over every prompt's piece: slots.prefill_runs, GLM 0560), each slot's reply equal to Python's, with slots at
# different positions and lengths (PIECES_PROMPTS, every piece past 16 rows, their rows within one 2,048-row run):
#   1. pieces-ref: dsv41_m2b_ref.py --batched at TP=2 on layers 0-$hi: Python's batched round (one encoder run over the
#      prompts' pieces), each slot's final and greedy reply, its state, the AOT set of the run's kernels;
#   2. pieces-alone: the same prompts one at a time (Python's own batched == alone, a line for the record);
#   3. pieces-run: our two ranks serving the prompts at once through the served engine (tf-dsv41-m1 sess4, one turn)
#      with the round planner (TF_DSV41_PREFILL_PIECES=1) and the multi-segment run (TF_DSV41_PIECE_RUNS=1): "PASS sess4
#      turn 1 == Python: n/n" against the batched reference, and the rank's log shows the run ("piece runs: ...");
#   4. pieces-one: the same with TF_DSV41_PIECE_RUNS=0 (a piece at a time): the same replies.
# Each verdict line goes to $RES/pieces-gate/summary.log (job.sh greps "PASS pieces" / "FAIL pieces" and "sess4").
# Knobs: PIECES_PROMPTS (160 300 500 700), PIECES_STEPS (24), PIECES_REF_MIN (12), PIECES_MIN (8 a Zig run).
pieces_gate() {
    local prompts=${PIECES_PROMPTS:-160 300 500 700} steps=${PIECES_STEPS:-24} ref=$LOCAL/pieces-ref
    local alone=$LOCAL/pieces-alone out=$RES/pieces-gate limit=4096 pool mb st=0 dirs=() p
    pool=$(( 4 * limit + 2048 ))
    rm -rf "$ref" "$alone"
    mkdir -p "$out"
    if (( hi < 20 )); then echo "FAIL pieces: layers 0-$hi do not reach the CED decoder (20)" | tee -a "$out/summary.log"; return 1; fi
    local gate_env=(TF_DSV41_SLOTS=4 TF_DSV41_POOL_TOKENS=$pool TF_DSV41_LAYERS=0-$hi TF_DSV41_CONTEXT=$limit
                    TF_DSV41_PREFILL=replay TF_DSV41_PROMPT_TAIL=verify TF_DSV41_OWN_PREFILL=1 TF_DSV41_DRAFTS=0
                    TF_DSV41_PREFILL_PIECES=1 TF_DSV41_SESSIONS=0 TF_DSV41_GRAPHS=1 SESS4_TURNS=1 SESS4_STEPS=$steps)
    local refargs=(--pack "$LP" --layers "0-$hi" --steps "$steps" --prompts "$(echo $prompts | tr ' ' ',')"
                   --prefill replay --prompt-tail verify --limit "$limit")
    mb=$(left); (( mb > ${PIECES_REF_MIN:-12} )) && mb=${PIECES_REF_MIN:-12}
    if ! { PYTHONPATH=/dsv41-tf/src step pieces-ref "$mb" python -u -B "$Z/tools/zig/dsv41_m2b_ref.py" "${refargs[@]}" \
               --batched --out "$ref" --port 29661 \
           && step pieces-engram-host 3 python -B "$Z/tools/zig/dsv41_engram_host.py" --py-src /dsv41-tf/src \
               --config "$LP/config.json" --tokenizer "$LP/tokenizer.json" --out "$ref/engram-host.bin"; }; then
        echo "FAIL pieces: no batched reference" | tee -a "$out/summary.log"
        return 1
    fi
    for p in $prompts; do dirs+=("$ref/p$p"); mkdir -p "$out/ref/p$p"; cp "$ref/p$p/ref.json" "$out/ref/p$p/"; done
    # Python's own batched == alone (the record: our runs are checked against the batched reference)
    if PYTHONPATH=/dsv41-tf/src step pieces-alone "$mb" python -u -B "$Z/tools/zig/dsv41_m2b_ref.py" "${refargs[@]}" \
           --out "$alone" --port 29671; then
        python3 - "$ref" "$alone" $prompts <<'EOF' | tee -a "$out/summary.log"
import json, sys
ref, alone, ps = sys.argv[1], sys.argv[2], sys.argv[3:]
same = sum(json.load(open(f"{ref}/p{p}/ref.json"))["tokens"] == json.load(open(f"{alone}/p{p}/ref.json"))["tokens"] for p in ps)
print(f"pieces: Python batched == alone, tokens: {same}/{len(ps)} (informational)")
EOF
    fi
    # the piece-at-a-time run's encoder shapes (one prompt a forward) are the alone reference's kernels
    [[ -f "$alone/aot/aot.json" ]] && python3 -B "$Z/tools/zig/dsv41_aot_merge.py" "$ref/aot" "$alone/aot" >> "$out/summary.log" 2>&1
    rm -rf "$alone"
    # the AOT fill for this gate's own config (sess4's way): every Triton variant our runs can launch (aot-needs with the
    # gate's env and both runs' knobs) compiled from prod's Triton source, the reference's set not compiled again
    local fill=$LOCAL/pieces-fill arch
    arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
    rm -rf "$fill"; mkdir -p "$fill"
    if ! (export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
          python -B "$Z/tools/zig/triton_fill.py" sigs --py /dsv41-tf/src --out "$fill/sigs.json" \
          && env "${gate_env[@]}" TF_DSV41_PIECE_RUNS=1 TF_DSV41_REPLAY_RUNS=1 "$Z/bin/tf-dsv41-m1" aot-needs "$fill/sigs.json" "$fill/needs.json" > "$out/needs.log" 2>&1 \
          && step pieces-fill "$(( $(left) > 10 ? 10 : $(left) ))" python -B "$Z/tools/zig/triton_fill.py" compile \
              --py /dsv41-tf/src --needs "$fill/needs.json" --options "$Z/tools/zig/dsv41_triton_options.json" \
              --arch "$arch" --have "$ref/aot" --out "$fill/aot" \
          && python -B "$Z/tools/zig/triton_fill.py" merge "$ref/aot" "$fill/aot") > "$out/fill.log" 2>&1; then
        echo "FAIL pieces: the AOT fill failed (fill.log)" | tee -a "$out/summary.log"
        return 1
    fi
    rm -rf "$fill"
    for run in run one replay; do
        (( $(left) >= 3 )) || { echo "FAIL pieces: $(left) min left before pieces-$run" | tee -a "$out/summary.log"; return 1; }
        local runs=1 replays=0
        [[ $run == one ]] && runs=0
        [[ $run == replay ]] && replays=1
        if ! (export "${gate_env[@]}" TF_DSV41_PIECE_RUNS=$runs TF_DSV41_REPLAY_RUNS=$replays SESS4_PORT=$(( 29791 + runs + 2 * replays ))
              step "pieces-$run" "${PIECES_MIN:-8}" bash "$Z/tools/zig/dsv41_sess4/ranks.sh" "$Z/bin/tf-dsv41-m1" "$LP" \
                  "$ref" "$out/$run" "${dirs[@]}"); then st=1; fi
        grep -h "PASS sess4\|FAIL sess4" "$out/$run/rank0.log" | sed "s/sess4/pieces/; s/^/$run: /" | tee -a "$out/summary.log"
        if [[ $run == replay ]] && ! grep -q "replay runs: " "$out/$run/rank0.log"; then
            echo "FAIL pieces: the replay run's rank 0 log shows no batched decoder replay (replay runs)" | tee -a "$out/summary.log"
            st=1
        fi
        if [[ $run != one ]] && ! grep -q "piece runs: " "$out/$run/rank0.log"; then
            echo "FAIL pieces: the run's rank 0 log shows no multi-segment run (piece runs)" | tee -a "$out/summary.log"
            st=1
        fi
    done
    return $st
}
