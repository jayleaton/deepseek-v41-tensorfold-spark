# The vision gate (gate, TF_DSV41_IMAGES=native), a fragment the combined pod job sources inside
# M1_JOB=m2b on a TWO-GPU pod after the pack's local copy. Needs: Z (staged tree), LP (the shards up to layer $hi; the
# vision tower and the 43 gate.bias_vl live in native.safetensors, which the copy takes), LOCAL, RES, hi, and job.sh's
# `step` / `left`. Python twin: /dsv41-tf/src (prod 8474f31, e1e2x extensions; VISION_PYSRC overrides).
#   1. vision-ref (~10 min): tools/zig/dsv41_vision/vision_ref.py at TP=2, one load: 3 Pillow-made images through
#      prod's vision_prep, the tower's stage dumps and span rows on rank 0 ($LOCAL/vision-ref/img), then 3 image prompts
#      (text around the images' spans; image rows from vision.STORE) through dsv41_m2b_ref.py's own path: p1..p3 with
#      the text bias, then p4..p6 the same prompts after vision.attach_bias_vl(fw.blocks, $LP) (gate.bias_vl from the
#      pack's native.safetensors: image rows in their own MoE call, prod's TF_DSV41_BIAS_VL); state dumps,
#      VISION_STEPS greedy tokens, the AOT set with the image paths' Triton / row-count variants
#   2. vision-vit (~1 min, GPU 0): tf-dsv41-m1 vision: the Zig tower's embed / block 0's ops (b0.norm1 .. b0.w2) /
#      block0 / 1 / 15 / 31 / vit / gelu stages and span rows == Python's, bit for bit ("PASS vision ViT + aligner ==
#      Python: 3/3 images"); on a FAIL, vision-vit-ops reruns it with TF_DSV41_VISION_RESYNC=1 (each stage on
#      Python's input: every differing op apart)
#   3. vision-prefill-p1..p6 (~2 min each): tf-dsv41-m1 m2b with M2B_PREFILL=1 on each prompt: the Zig prefill from an
#      empty slot (image rows into rank 0's embedding, Engram's keep; p4..p6 with TF_DSV41_BIAS_VL=$LP: the image
#      rows' own MoE call), every state role == Python's, then the tokens
#   4. vision-generate / vision-generate-bias (~3 min each, at most VISION_GEN_MIN = 8): the prompts through the served engine (tf-dsv41-m1
#      generate, drafts off): "PASS M4 CLI == HTTP: 3/3 replies equal" (here: Zig's replies == Python's)
# Each run prints "PASS vision ..." / "FAIL vision ..." into $RES/steps.txt. Knobs: VISION_STEPS (32), VISION_REF_MIN
# (14), VISION_RUNS ("vit prefill generate prefill-bias generate-bias"; about 22 min in all).
vision_gate() {
    local ref=$LOCAL/vision-ref vdir=$LOCAL/vision-ref/img pysrc=${VISION_PYSRC:-/dsv41-tf/src} d i=0 mb
    rm -rf "$ref"
    mkdir -p "$RES/vision"
    unset TF_DSV41_BIAS_VL TF_DSV41_KV_SPLIT
    if ! { PYTHONPATH=$pysrc step vision-ref "${VISION_REF_MIN:-14}" python -u -B "$Z/tools/zig/dsv41_vision/vision_ref.py" \
            --vdir "$vdir" --bias-vl "$LP" -- --pack "$LP" --out "$ref" --layers "0-$hi" --steps "${VISION_STEPS:-32}" \
            --prompts 1,2,3,4,5,6 \
            --port 29781 \
         && step vision-engram-host 3 python -B "$Z/tools/zig/dsv41_engram_host.py" --py-src "$pysrc" \
            --config "$LP/config.json" --tokenizer "$LP/tokenizer.json" --out "$ref/engram-host.bin"; }; then
        echo "FAIL vision: no reference" | tee -a "$RES/steps.txt"
        return 1
    fi
    for d in "$ref"/p*; do cp "$ref/engram-host.bin" "$d/"; done
    cp "$vdir/manifest.json" "$ref/trace.jsonl" "$ref/trace-bias.jsonl" "$RES/vision/" 2>/dev/null
    local runs=${VISION_RUNS:-vit prefill generate prefill-bias generate-bias}
    if [[ " $runs " == *" vit "* ]]; then
        if CUDA_VISIBLE_DEVICES=0 step vision-vit 4 "$Z/bin/tf-dsv41-m1" vision "$LP" "$vdir"; then
            echo "PASS vision vit" | tee -a "$RES/steps.txt"
        else
            echo "FAIL vision vit (first differing stage in vision-vit.log)" | tee -a "$RES/steps.txt"
            # each op on Python's input (TF_DSV41_VISION_RESYNC=1): every differing op of block 0 / each block, apart
            TF_DSV41_VISION_RESYNC=1 CUDA_VISIBLE_DEVICES=0 step vision-vit-ops 4 "$Z/bin/tf-dsv41-m1" vision "$LP" "$vdir"
            grep -E "differ|bytes, Python" "$RES/vision-vit-ops.log" | head -60
        fi
        cp "$RES/vision-vit.log" "$RES/vision-vit-ops.log" "$RES/vision/" 2>/dev/null
    fi
    export TF_DSV41_IMAGES=native TF_DSV41_VISION_HOLD=$vdir
    vprefill() { # PROMPT-DIRS...: the M2B_PREFILL gate on each
        local d
        for d in "$@"; do
            mb=$(left); (( mb > 4 )) && mb=4
            (( mb >= 3 )) || { echo "FAIL vision prefill $(basename "$d"): skipped, $mb min left" | tee -a "$RES/steps.txt"; continue; }
            if (export M2B_PREFILL=1 M2B_PORT=$(( 29783 + i ))
                step "vision-prefill-$(basename "$d")" "$mb" bash "$Z/tools/zig/dsv41_m1/m2b_ranks.sh" "$Z/bin/tf-dsv41-m1" \
                    "$LP" "$d" "$RES/vision/prefill-$(basename "$d")"); then
                echo "PASS vision prefill $(basename "$d")${TF_DSV41_BIAS_VL:+ bias_vl} (state roles, tokens)" | tee -a "$RES/steps.txt"
            else echo "FAIL vision prefill $(basename "$d")${TF_DSV41_BIAS_VL:+ bias_vl}" | tee -a "$RES/steps.txt"; fi
            i=$(( i + 1 ))
        done
    }
    vgenerate() { # NAME TRACE
        (( $(left) >= 4 )) || { echo "FAIL vision $1: skipped, $(left) min left" | tee -a "$RES/steps.txt"; return; }
        mb=$(left); (( mb > ${VISION_GEN_MIN:-8} )) && mb=${VISION_GEN_MIN:-8}
        if TF_DSV41_LAYERS=0-$hi VISION_PORT=$(( 29790 + i )) VISION_TAIL1=25 step "vision-$1" "$mb" bash "$Z/tools/zig/dsv41_vision/gen_ranks.sh" \
                "$Z/bin/tf-dsv41-m1" "$LP" "$ref/p1" "$2" "$RES/vision/$1"; then
            echo "PASS vision $1 (replies == Python's)" | tee -a "$RES/steps.txt"
        else echo "FAIL vision $1" | tee -a "$RES/steps.txt"; fi
        i=$(( i + 1 ))
    }
    [[ " $runs " == *" prefill "* ]] && vprefill "$ref"/p1 "$ref"/p2 "$ref"/p3
    [[ " $runs " == *" generate "* ]] && vgenerate generate "$ref/trace.jsonl"
    # prod's TF_DSV41_BIAS_VL: the pack's native.safetensors holds the 43 gate.bias_vl (as the reference attached them)
    if [[ -f "$ref/trace-bias.jsonl" ]]; then
        export TF_DSV41_BIAS_VL=$LP
        [[ " $runs " == *" prefill-bias "* ]] && vprefill "$ref"/p4 "$ref"/p5 "$ref"/p6
        [[ " $runs " == *" generate-bias "* ]] && vgenerate generate-bias "$ref/trace-bias.jsonl"
        unset TF_DSV41_BIAS_VL
    elif [[ " $runs " == *"-bias "* ]]; then echo "FAIL vision bias_vl: the reference wrote no p4..p6" | tee -a "$RES/steps.txt"; fi
    unset TF_DSV41_IMAGES TF_DSV41_VISION_HOLD
    rm -rf "$ref"/p*/rank*/state
}
