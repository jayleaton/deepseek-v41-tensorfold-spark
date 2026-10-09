# The prod-knobs gate (gate), a fragment the combined pod job sources on a TWO-GPU pod inside M1_JOB=m2b,
# after the M2b reference (refs / R0) exists. Needs: Z, LP, RES, the `step` / `left` helpers of tools/zig/dsv41_m1/job.sh,
# and `refs` (the reference dirs; M2B_PROMPTS with a prompt over 2,048 rows, e.g. "2600": segments then cross).
# No Python load of its own: every run is the M2B_PREFILL gate (tf-dsv41-m1 m2b, Zig's prefill from an empty slot, every
# state role == the reference's dump, the same first token, then the decode gate) with one knob set on the Zig ranks:
#   pfd       TF_DSV41_PF_DENSE=fused (+ _TABLE=$KNOBS_PFD_TABLE when that file exists: prod's /cache/pfdense-table.json)
#   rows1024  TF_DSV41_PREFILL_CHUNK=1024 (segmentation invariance on Zig's emitter, and the AOT set at 1,024 rows)
#   rows512   TF_DSV41_PREFILL_CHUNK=512
#   budget64  TF_DSV41_INDEX_BUDGET_MIB=64 (prod's; the indexer in smaller row blocks)
#   ahead     TF_DSV41_PREFETCH_AHEAD=1 (needs M2B_ENGRAM=1's shards in TF_DSV41_ENGRAM_DIR; skipped without)
#   prod      all of them at prod's values together (rows 2,048)
# Expected in each rank0.log / rank1.log: the M2B_PREFILL gate's PASS lines (65/65 roles on layers 0-24, first token,
# 64/64 tokens and logits), and on rank 0 "prod knobs: pf_dense fused, ..." naming the run's settings. The prefill's
# seconds are in each log ("PASS Zig prefill rank 0: N rows in S s"): pfd vs the m2b-prefill run is the fused GEMM's speed-up.
# Knobs: KNOBS_RUNS (default "pfd rows1024 rows512 budget64 ahead prod"), KNOBS_REF (default: the last of refs, the
# longest prompt by job.sh's convention), KNOBS_MIN (a run's minutes, 6).
knobs_gate() {
    local ref=${KNOBS_REF:-${refs[${#refs[@]}-1]}} run i=0 kv table=() st=0
    [[ -f "${KNOBS_PFD_TABLE:-}" ]] && table=(TF_DSV41_PF_DENSE_TABLE="$KNOBS_PFD_TABLE")
    mkdir -p "$RES/knobs"
    for run in ${KNOBS_RUNS:-pfd rows1024 rows512 budget64 ahead prod}; do
        case $run in
            pfd) kv=(TF_DSV41_PF_DENSE=fused "${table[@]}") ;;
            rows1024) kv=(TF_DSV41_PREFILL_CHUNK=1024) ;;
            rows512) kv=(TF_DSV41_PREFILL_CHUNK=512) ;;
            budget64) kv=(TF_DSV41_INDEX_BUDGET_MIB=64) ;;
            ahead|prod)
                if [[ -z "${TF_DSV41_ENGRAM_DIR:-}" ]]; then
                    echo "knobs-$run skipped: no Engram shards (M2B_ENGRAM=1)" | tee -a "$RES/steps.txt"; continue
                fi
                kv=(TF_DSV41_PREFETCH_AHEAD=1)
                [[ $run == prod ]] && kv+=(TF_DSV41_PF_DENSE=fused "${table[@]}" TF_DSV41_PREFILL_CHUNK=2048 TF_DSV41_PREFILL_ROWS=2048 TF_DSV41_INDEX_BUDGET_MIB=64) ;;
            *) echo "knobs: unknown run $run" | tee -a "$RES/steps.txt"; st=1; continue ;;
        esac
        (( $(left) >= 3 )) || { echo "knobs-$run skipped: $(left) min left" | tee -a "$RES/steps.txt"; continue; }
        (export M2B_PREFILL=1 M2B_PORT=$(( 29760 + i )) "${kv[@]}"
         step "knobs-$run" "${KNOBS_MIN:-6}" bash "$Z/tools/zig/dsv41_m1/m2b_ranks.sh" "$Z/bin/tf-dsv41-m1" "$LP" "$ref" \
             "$RES/knobs/$run") || st=1
        grep -h "prod knobs:\|Zig prefill rank" "$RES/knobs/$run/rank0.log" 2>/dev/null | sed "s/^/knobs-$run: /" | tee -a "$RES/steps.txt"
        i=$(( i + 1 ))
    done
    return $st
}
