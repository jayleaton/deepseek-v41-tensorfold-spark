# The long-prompt gate (gate), a fragment tools/zig/dsv41_m1/job.sh sources inside M1_JOB=m2b on a
# TWO-GPU pod (M2B_LONGPF=1), after the pack's local copy. Needs: Z (staged tree), LP (the shards up to layer $hi),
# volume, unpacked to /opt/midprofile as M1_JOB=m5 does: the split pool is its code; its prefill is prod 8474f31's),
# and for the optional `prod` run prod's own twin (/dsv41-tf/src, the e1e2 extensions).
#
# What it checks: Zig's prefill past TF_DSV41_INDEX_STREAM_MIN (prod-knobs.env: 4,096 visible keys) at prompts of
# LONGPF_PROMPTS (8,192 / 32,768 / 131,072: stream top-k with 1 / 1 / 4 key splits a row), every state row == Python's
# after the prompt, then LONGPF_STEPS greedy steps' tokens and logits == Python's:
#   1. longpf-ref-pool, longpf-ref-split: dsv41_m2b_ref.py at TP=2 on layers 0-$hi, one load for every prompt, the slot
#      in the paged pool (--pool) / split over the ranks (--split) with TF_DSV41_KV_SPLIT_UNION_MIB=LONGPF_UNION_MIB (8:
#      the long prompts' unions pass the cap, so Python's prefill attends in row blocks, _attend_blocks)
#   2. longpf-paged, longpf-split: `tf-dsv41-m1 m5 paged|split` (m5_ranks.sh) on those references with
#      TF_DSV41_CONTEXT = the references' limit: the pool's rows in logical order (a split rank's owned rows), the SWA
#      rings and carries, position, Engram tail, first token, then every step ("PASS longpf <run> ...")
#   3. optional (LONGPF_RUNS has `prod`): the contiguous slot on prod's twin, `M2B_PREFILL=1` (m2b_ranks.sh) a prompt
# Each run's verdict: "PASS longpf ..." / "FAIL longpf ..." in $RES/longpf-gate/summary.log (job.sh greps it).
# Knobs: LONGPF_RUNS ("paged split"; add "prod"), LONGPF_PROMPTS ("8192 32768 131072"), LONGPF_STEPS (16),
# LONGPF_UNION_MIB (8), LONGPF_REF_MIN (10), LONGPF_MIN (8 a Zig run).
# Minutes (estimates, 2x PRO 6000, layers 0-24): a reference ~6 (load ~1.5 + prefill of 172K prompt tokens), a Zig run
# ~4 (load ~1.5); default runs ~20 min, ~11 with LONGPF_PROMPTS="8192 32768".
longpf_gate() {
    local prompts=${LONGPF_PROMPTS:-8192 32768 131072} steps=${LONGPF_STEPS:-16} runs=${LONGPF_RUNS:-paged split}
    local p max=0 limit pool twin="" mt="" run mode ref out mb ok=0
    for p in $prompts; do (( p > max )) && max=$p; done
    limit=$(( max + 2048 ))                      # the prompt, the steps, a segment's room (the rope tables add 2,048)
    pool=$(( (2 * limit + 4096 + 511) / 512 * 512 ))
    for mt in "$Z/midprofile.tar.gz" "${PACK%%/out/*}/m5/midprofile.tar.gz"; do [[ -f "$mt" ]] && break; mt=""; done
    if [[ -n "$mt" ]] && { [[ -d /opt/midprofile/src ]] || tar -xzf "$mt" -C /; }; then twin=/opt/midprofile; fi
    mkdir -p "$RES/longpf-gate"
    local eh=$LOCAL/m2b-ref/engram-host.bin
    if [[ ! -f "$eh" ]]; then
        eh=$LOCAL/longpf-engram-host.bin
        step longpf-engram-host 3 python -B "$Z/tools/zig/dsv41_engram_host.py" --py-src "${twin:-/dsv41-tf}/src" \
            --config "$LP/config.json" --tokenizer "$LP/tokenizer.json" --out "$eh" || { echo "FAIL longpf: no engram-host.bin" | tee -a "$RES/longpf-gate/summary.log"; return 1; }
    fi
    for run in $runs; do
        case $run in
            paged) mode=pool ;;
            split) mode=split ;;
            prod) mode=contig ;;
            *) echo "longpf: unknown run $run" | tee -a "$RES/longpf-gate/summary.log"; continue ;;
        esac
        if [[ "$mode" != contig && -z "$twin" ]]; then
            echo "FAIL longpf $run: no midprofile twin (set MIDPROFILE_TAR to a staged local archive)" | tee -a "$RES/longpf-gate/summary.log"
            ok=1; continue
        fi
        ref=$LOCAL/longpf-ref-$run
        rm -rf "$ref"
        mb=$(left); (( mb > ${LONGPF_REF_MIN:-10} )) && mb=${LONGPF_REF_MIN:-10}
        (( mb >= 4 )) || { echo "longpf-ref-$run skipped: $(left) min left" | tee -a "$RES/longpf-gate/summary.log"; ok=1; continue; }
        local args=(--limit "$limit")
        [[ $mode == pool ]] && args+=(--pool)
        [[ $mode == split ]] && args+=(--split --compact 0)
        if ! (set -a; source "$Z/tools/zig/dsv41_m1/prod-knobs.env"; set +a
              unset TF_DSV41_EXPERT_TOPP M2B_TOPP   # m5.zig turns M2B_TOPP (the m2b job's 0.85) into the expert top-p
              export TF_DSV41_KV_SPLIT_UNION_MIB=${LONGPF_UNION_MIB:-8} TF_DSV41_KV_SPLIT_LOG=1
              if [[ $mode != contig ]]; then export PYTHONPATH=$twin/src TORCH_EXTENSIONS_DIR=$twin/ext; fi
              step "longpf-ref-$run" "$mb" python -u -B "$Z/tools/zig/dsv41_m2b_ref.py" --pack "$LP" --out "$ref" \
                  --layers "0-$hi" --steps "$steps" --prompts "$(echo $prompts | tr ' ' ',')" --port 29641 "${args[@]}"); then
            echo "FAIL longpf $run: the reference did not finish" | tee -a "$RES/longpf-gate/summary.log"
            ok=1; continue
        fi
        for p in $prompts; do cp "$eh" "$ref/p$p/engram-host.bin"; mkdir -p "$RES/longpf-gate/ref-$run/p$p"; cp "$ref/p$p/ref.json" "$RES/longpf-gate/ref-$run/p$p/"; done
        if [[ $mode == contig ]]; then
            local i=0
            for p in $prompts; do
                out=$RES/longpf-$run-p$p
                mb=$(left); (( mb > ${LONGPF_MIN:-8} )) && mb=${LONGPF_MIN:-8}
                (( mb >= 3 )) || { echo "longpf-$run-p$p skipped: $(left) min left" | tee -a "$RES/longpf-gate/summary.log"; ok=1; continue; }
                if (export M2B_PREFILL=1 M2B_PORT=$(( 29741 + i ))
                    step "longpf-$run-p$p" "$mb" bash "$Z/tools/zig/dsv41_m1/m2b_ranks.sh" "$Z/bin/tf-dsv41-m1" "$LP" "$ref/p$p" "$out"); then
                    echo "PASS longpf $run p$p (contiguous, prod twin): state, first token, $steps steps" | tee -a "$RES/longpf-gate/summary.log"
                else
                    echo "FAIL longpf $run p$p (contiguous, prod twin): see $out" | tee -a "$RES/longpf-gate/summary.log"; ok=1
                fi
                i=$(( i + 1 ))
            done
            continue
        fi
        out=$RES/longpf-$run
        mb=$(left); (( mb > ${LONGPF_MIN:-8} )) && mb=${LONGPF_MIN:-8}
        (( mb >= 3 )) || { echo "longpf-$run skipped: $(left) min left" | tee -a "$RES/longpf-gate/summary.log"; ok=1; continue; }
        if (set -a; source "$Z/tools/zig/dsv41_m1/prod-knobs.env"; set +a
            unset TF_DSV41_EXPERT_TOPP M2B_TOPP   # m5.zig turns M2B_TOPP (the m2b job's 0.85) into the expert top-p
            export TF_TP_PORT=29761 TF_DSV41_CONTEXT=$limit M5_POOL_TOKENS=$pool \
                   TF_DSV41_KV_SPLIT_UNION_MIB=${LONGPF_UNION_MIB:-8}
            step "longpf-$run" "$mb" bash "$Z/tools/zig/dsv41_m1/m5_ranks.sh" "$Z/bin/tf-dsv41-m1" "$run" "$LP" "$ref" "$out"); then
            echo "PASS longpf $run: prompts $prompts, state rows, first token, $steps steps (limit $limit)" | tee -a "$RES/longpf-gate/summary.log"
        else
            echo "FAIL longpf $run: see $out (rank*.log, m5-rank*.jsonl)" | tee -a "$RES/longpf-gate/summary.log"; ok=1
        fi
        # the exchanges (split: unions and row blocks) and Python's union log, for the report
        grep -h "exchanges" "$out"/m5-rank*.jsonl 2>/dev/null | tee -a "$RES/longpf-gate/summary.log"
        grep -h "kv split: union" "$RES/longpf-ref-$run.log" 2>/dev/null | tail -3 >> "$RES/longpf-gate/summary.log"
    done
    rm -rf "$LOCAL"/longpf-ref-*
    return $ok
}
