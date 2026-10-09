# The structured-output gate (gate), a fragment the combined pod job sources inside M1_JOB=m2b on a
# TWO-GPU pod after the pack's local copy. Needs: Z (staged tree), LP (the shards up to layer $hi, with tokenizer.json),
# LOCAL, RES, hi, the `step` / `left` helpers of tools/zig/dsv41_m1/job.sh, an M2b reference's assets ($GRAMMAR_ASSETS,
# default ${R0:-$REF}: aot/, rope.bin, engram-host.bin) and Python with prod's tree (GRAMMAR_PYSRC, default
# /dsv41-tf/src: prod 8474f31 with the e1e2x extensions) and xgrammar==0.2.8 (installed --no-deps when missing).
#   1. grammar-ref: prod's grammar over the Python engine at TP=2, layers 0-$hi: 6 constrained replies (JSON schema,
#      tool_choice required, a named tool, strict tools, json_object; greedy and sampled), serial decoding with
#      prod's masks (tools/zig/dsv41_grammar/grammar_ref.py) into $LOCAL/grammar-ref/trace.jsonl. Skipped when the M2b
#      reference already wrote it (dsv41_m2b_ref.py --grammar: $REF/grammar/trace.jsonl, no second Python load).
#   2. grammar-drafts / grammar-serial: tf-dsv41-m1 generate on that trace with TF_DSV41_GRAMMAR=1 (rank 1 following),
#      DSpark drafts on then off: every constrained reply's tokens == Python's ("PASS M4 CLI == HTTP: 6/6 replies
#      equal" in each log: here Zig == Python).
#   3. grammar-http: tensorfold-dsv41 serve with TF_DSV41_GRAMMAR=1 answers requests.sh's chat requests over HTTP
#      (response_format json_schema / json_object, tool_choice required / named, strict tools; greedy and seeded), every
#      reply checked against its grammar ("PASS grammar http"), then generate replays the server's trace (with the
#      grammar each line recorded): HTTP == CLI ("PASS M4 CLI == HTTP").
# Prompt lengths 37, 50, 66, 83, 128, 292 (the sampled gate's): with the Zig prefill their Triton variants must be in
# the assets' AOT set (M2B_SAMP=1's --sampled puts them there; else add them to M2B_AOT_PROMPTS, or GRAMMAR_OWN_PREFILL=0).
# Sampled cases need the sampler (TF_DSV41_SAMPLING=1, dsv41_sampling.fatbin in the build, as M2B_SAMP).
# Knobs: GRAMMAR_MODES ("drafts serial http"), GRAMMAR_STEPS (48), GRAMMAR_MIN (a run's minutes, 7), GRAMMAR_OWN_PREFILL (1).
grammar_gate() {
    local pysrc=${GRAMMAR_PYSRC:-/dsv41-tf/src} A=${GRAMMAR_ASSETS:-${R0:-${REF:-}}} out=$RES/grammar
    local trace=${GRAMMAR_TRACE:-} ref=$LOCAL/grammar-ref mode d port r
    mkdir -p "$out"
    if [[ -z "$A" || ! -d "$A/aot" ]]; then
        echo "FAIL grammar: no M2b reference assets (GRAMMAR_ASSETS / R0 / REF)" | tee -a "$RES/steps.txt"
        return 1
    fi
    if ! PYTHONPATH=$pysrc python -c "import xgrammar" 2>/dev/null; then
        step grammar-xgrammar 3 python -m pip install --no-deps "xgrammar==0.2.8" || true
    fi
    [[ -z "$trace" && -s "${REF:-/nonexistent}/grammar/trace.jsonl" ]] && trace=$REF/grammar/trace.jsonl
    if [[ -z "$trace" ]]; then
        rm -rf "$ref"
        unset TF_DSV41_R1
        PYTHONPATH=$pysrc step grammar-ref "${GRAMMAR_REF_MIN:-12}" python -u -B "$Z/tools/zig/dsv41_grammar/grammar_ref.py" \
            --pack "$LP" --out "$ref" --layers "0-$hi" --steps "${GRAMMAR_STEPS:-48}" --port 29651
        trace=$ref/trace.jsonl
    fi
    if [[ ! -s "$trace" ]]; then
        echo "FAIL grammar: no reference trace" | tee -a "$RES/steps.txt"
        return 1
    fi
    cp "$trace" "$out/ref-trace.jsonl"
    local env=(TF_DSV41_GRAMMAR=1 TF_DSV41_SAMPLING=1 TF_TP_WORLD=2 TF_DSV41_ASSETS="$A" TF_DSV41_LAYERS="0-$hi"
               TF_DSV41_GRAPHS=1 TF_DSV41_OWN_PREFILL="${GRAMMAR_OWN_PREFILL:-1}" TF_DSV41_FAILFAST=0
               TF_DSV41_EXPERT_TOPP="${M2B_TOPP:-0.85}"
               TF_DSV41_PROMPT_TAIL=prefill)   # grammar_ref.py's replies follow Forward.prompt
    for mode in ${GRAMMAR_MODES:-drafts serial http}; do
        (( $(left) >= 4 )) || { echo "grammar-$mode skipped: $(left) min left" | tee -a "$RES/steps.txt"; continue; }
        d=1; [[ $mode == serial ]] && d=0
        port=$(( 29891 + d ))
        [[ $mode == http ]] && port=29895
        env "${env[@]}" TF_DSV41_DRAFTS=$d TF_TP_RANK=1 TF_TP_DEVICE=1 TF_TP_PORT=$port \
            "$Z/bin/tf-dsv41-m1" follow "$LP" "$A" > "$out/follow-$mode.log" 2>&1 &
        local fp=$!
        if [[ $mode != http ]]; then
            step "grammar-$mode" "${GRAMMAR_MIN:-7}" env "${env[@]}" TF_DSV41_DRAFTS=$d TF_DSV41_ENGRAM_PREFETCH=0 \
                TF_TP_RANK=0 TF_TP_DEVICE=0 TF_TP_PORT=$port "$Z/bin/tf-dsv41-m1" generate "$LP" "$A" "$trace"; r=$?
            wait $fp
            grep -h "^PASS\|^FAIL" "$RES/grammar-$mode.log" | sed "s/^\(PASS\|FAIL\) M4 CLI == HTTP/\1 grammar $mode: Zig == Python/" | tee -a "$RES/steps.txt"
            continue
        fi
        # http: the server over the same assets, the requests, then the CLI replay of the server's trace
        env "${env[@]}" TF_DSV41_DRAFTS=1 TF_TP_RANK=0 TF_TP_DEVICE=0 TF_TP_PORT=$port TF_DSV41_TRACE_TOKENS="$out/http-trace.jsonl" \
            "$Z/bin/tensorfold-dsv41" serve "$LP" --port 8093 --parallel 1 > "$out/server.log" 2>&1 &
        local sp=$! t=0
        until grep -q "serving model\|serving pack at" "$out/server.log" 2>/dev/null || (( t > 600 )) || ! kill -0 $sp 2>/dev/null; do sleep 5; t=$((t + 5)); done
        grep -h "structured output" "$out/server.log" | tee -a "$RES/steps.txt"
        step grammar-http 4 bash "$Z/tools/zig/dsv41_grammar/requests.sh" 8093 "$out/http"
        kill $sp 2>/dev/null; wait $sp 2>/dev/null; wait $fp 2>/dev/null
        grep -h "^PASS\|^FAIL" "$RES/grammar-http.log" | tee -a "$RES/steps.txt"
        if [[ -s "$out/http-trace.jsonl" ]] && (( $(left) >= 4 )); then
            env "${env[@]}" TF_DSV41_DRAFTS=1 TF_TP_RANK=1 TF_TP_DEVICE=1 TF_TP_PORT=29896 \
                "$Z/bin/tf-dsv41-m1" follow "$LP" "$A" > "$out/follow-http-replay.log" 2>&1 &
            fp=$!
            step grammar-http-replay "${GRAMMAR_MIN:-7}" env "${env[@]}" TF_DSV41_DRAFTS=1 TF_DSV41_ENGRAM_PREFETCH=0 \
                TF_TP_RANK=0 TF_TP_DEVICE=0 TF_TP_PORT=29896 "$Z/bin/tf-dsv41-m1" generate "$LP" "$A" "$out/http-trace.jsonl"
            wait $fp
            grep -h "^PASS\|^FAIL" "$RES/grammar-http-replay.log" | sed "s/^\(PASS\|FAIL\) M4 CLI == HTTP/\1 grammar http: CLI == HTTP/" | tee -a "$RES/steps.txt"
        fi
    done
}
