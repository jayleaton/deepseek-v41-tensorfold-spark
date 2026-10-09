# The sessions-over-slots gate (gate), a fragment tools/zig/dsv41_m1/job.sh sources inside M1_JOB=m2b on a
# TWO-GPU pod (M2B_SESS4=1), after the pack's local copy. Needs: Z (staged tree, bin/tf-dsv41-m1 built for this GPU),
# LP (the shards up to layer $hi, hi >= 20: the CED decoder starts at 20), LOCAL, RES, and job.sh's `step` / `left`.
# Python: prod's tree at /dsv41-tf/src (the image's; 8474f31) with its Triton.
#
# What it proves: a chat resumed from its prompt snapshot (RAM or NVMe) continues exactly as a fresh prefill of the same
# history, with 4 live slots, prod's replay prefill (TF_DSV41_PREFILL=replay) and served shape (own prefill, the last
# prompt token as the first verify row), while the other slots decode:
#   1. sess4-ref: dsv41_m2b_ref.py at TP=2 on layers 0-$hi, the served shape in replay mode (its contiguous slot: the
#      pool path runs Forward.prompt; paged == replicated is M5's gate), one prompt a chat
#      (SESS4_PROMPTS: 160 / 1100 / 1700 / 2300, the last crossing a 2,048-row segment): Python's fresh reply of each
#      chat's turn 1, rope.bin, the AOT set; Engram's host tables beside it.
#   2. sess4-fill (SESS4_FILL=1, default): every Triton variant the gate's config can launch (tf-dsv41-m1 aot-needs with
#      the gate's env: row mode's buckets, the replay prefill's encoder / decoder shapes) compiled from prod's Triton
#      source for this GPU (tools/zig/triton_fill.py), merged into the reference's set (turn 2 and 3 prompts have shapes
#      no reference ran).
#   3. sess4-fresh: our two ranks serving SESS4_TURNS turns of every chat (tf-dsv41-m1 sess4) with TF_DSV41_SESSIONS=0:
#      every prompt prefilled from an empty slot. "PASS sess4 turn 1 == Python: 4/4".
#   4. sess4-s4: the same with TF_DSV41_SESSIONS=1, the RAM tier one entry wide (SESS4_RAM_MIB 12: every other
#      snapshot parks), the NVMe tier in $LOCAL/sess4-disk taking entries of 128+ tokens: "PASS sess4 turn 1 ==
#      Python", "PASS sess4 resumed == fresh: 12/12", "PASS sess4 tiers: n RAM and m NVMe resumes".
#   5. sess4-s1 (in SESS4_RUNS): one slot with sessions, against the same fresh run (batched == alone, resumed == fresh).
# Each verdict line goes to $RES/sess4-gate/summary.log (job.sh greps "PASS sess4" / "FAIL sess4").
# Knobs: SESS4_RUNS ("s4"; add "s1"), SESS4_PROMPTS, SESS4_TURNS (3), SESS4_STEPS (24), SESS4_FILL (1), SESS4_GRAPHS
# (1), SESS4_REF_MIN (12), SESS4_MIN (8 a Zig run). Minutes (2x PRO 6000, layers 0-24): reference ~8, fill ~6, each run
# ~5 (load ~1.5).
sess4_gate() {
    local prompts=${SESS4_PROMPTS:-160 1100 1700 2300} steps=${SESS4_STEPS:-24} ref=$LOCAL/sess4-ref out=$RES/sess4-gate
    local limit=4096 pool run mb st=0 dirs=() p arch
    pool=$(( 4 * limit + 2048 ))
    rm -rf "$ref" "$LOCAL/sess4-disk"
    mkdir -p "$out"
    if (( hi < 20 )); then echo "FAIL sess4: layers 0-$hi do not reach the CED decoder (20)" | tee -a "$out/summary.log"; return 1; fi
    # the served config every run shares (the reference's: engine code defaults, replay, verify tail, own prefill)
    local gate_env=(TF_DSV41_SLOTS=4 TF_DSV41_POOL_TOKENS=$pool TF_DSV41_LAYERS=0-$hi TF_DSV41_CONTEXT=$limit
                    TF_DSV41_PREFILL=replay TF_DSV41_PROMPT_TAIL=verify TF_DSV41_OWN_PREFILL=1 TF_DSV41_DRAFTS=0
                    TF_DSV41_GRAPHS=${SESS4_GRAPHS:-1} SESS4_TURNS=${SESS4_TURNS:-3} SESS4_STEPS=$steps)
    mb=$(left); (( mb > ${SESS4_REF_MIN:-12} )) && mb=${SESS4_REF_MIN:-12}
    if ! { PYTHONPATH=/dsv41-tf/src step sess4-ref "$mb" python -u -B "$Z/tools/zig/dsv41_m2b_ref.py" --pack "$LP" \
               --out "$ref" --layers "0-$hi" --steps "$steps" --prompts "$(echo $prompts | tr ' ' ',')" \
               --prefill replay --prompt-tail verify --limit "$limit" --port 29651 \
           && step sess4-engram-host 3 python -B "$Z/tools/zig/dsv41_engram_host.py" --py-src /dsv41-tf/src \
               --config "$LP/config.json" --tokenizer "$LP/tokenizer.json" --out "$ref/engram-host.bin"; }; then
        echo "FAIL sess4: no reference" | tee -a "$out/summary.log"
        return 1
    fi
    for p in $prompts; do dirs+=("$ref/p$p"); mkdir -p "$out/ref/p$p"; cp "$ref/p$p/ref.json" "$out/ref/p$p/"; done
    if [[ "${SESS4_FILL:-1}" == 1 ]]; then
        arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
        local fill=$LOCAL/sess4-fill
        rm -rf "$fill"; mkdir -p "$fill"
        if ! (export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
              python -B "$Z/tools/zig/triton_fill.py" sigs --py /dsv41-tf/src --out "$fill/sigs.json" \
              && env "${gate_env[@]}" "$Z/bin/tf-dsv41-m1" aot-needs "$fill/sigs.json" "$fill/needs.json" > "$out/needs.log" 2>&1 \
              && step sess4-fill "$(( $(left) > 10 ? 10 : $(left) ))" python -B "$Z/tools/zig/triton_fill.py" compile \
                  --py /dsv41-tf/src --needs "$fill/needs.json" --options "$Z/tools/zig/dsv41_triton_options.json" \
                  --arch "$arch" --have "$ref/aot" --out "$fill/aot" \
              && python -B "$Z/tools/zig/triton_fill.py" merge "$ref/aot" "$fill/aot") > "$out/fill.log" 2>&1; then
            echo "sess4: the AOT fill failed (see fill.log); the runs use the reference's set" | tee -a "$out/summary.log"
        fi
    fi
    local fresh=$out/fresh
    (( $(left) >= 3 )) || { echo "FAIL sess4: $(left) min left before the fresh run" | tee -a "$out/summary.log"; return 1; }
    if ! (export "${gate_env[@]}" TF_DSV41_SESSIONS=0 SESS4_PORT=29781
          step sess4-fresh "${SESS4_MIN:-8}" bash "$Z/tools/zig/dsv41_sess4/ranks.sh" "$Z/bin/tf-dsv41-m1" "$LP" "$ref" \
              "$fresh" "${dirs[@]}"); then st=1; fi
    grep -h "PASS sess4\|FAIL sess4" "$fresh/rank0.log" | sed 's/^/fresh: /' | tee -a "$out/summary.log"
    [[ -f "$fresh/replies.jsonl" ]] || { echo "FAIL sess4: no fresh replies" | tee -a "$out/summary.log"; return 1; }
    local i=0 slots
    for run in ${SESS4_RUNS:-s4}; do
        slots=${run#s}
        (( $(left) >= 3 )) || { echo "sess4-$run skipped: $(left) min left" | tee -a "$out/summary.log"; st=1; continue; }
        rm -rf "$LOCAL/sess4-disk"; mkdir -p "$LOCAL/sess4-disk"
        if ! (export "${gate_env[@]}" TF_DSV41_SLOTS=$slots TF_DSV41_SESSIONS=1 TF_DSV41_SESSION_SLOTS=1 \
                  TF_DSV41_SESSION_RAM_MIB=${SESS4_RAM_MIB:-12} TF_DSV41_SESSION_DISK="$LOCAL/sess4-disk" \
                  TF_DSV41_SESSION_DISK_MIN=128 SESS4_FRESH="$fresh/replies.jsonl" SESS4_TIERS=1 SESS4_PORT=$(( 29782 + i ))
              step "sess4-$run" "${SESS4_MIN:-8}" bash "$Z/tools/zig/dsv41_sess4/ranks.sh" "$Z/bin/tf-dsv41-m1" "$LP" \
                  "$ref" "$out/$run" "${dirs[@]}"); then st=1; fi
        grep -h "PASS sess4\|FAIL sess4\|sess4 sessions" "$out/$run/rank0.log" | sed "s/^/$run: /" | tee -a "$out/summary.log"
        i=$(( i + 1 ))
    done
    rm -rf "$LOCAL/sess4-disk" "$LOCAL/sess4-fill"
    return $st
}
