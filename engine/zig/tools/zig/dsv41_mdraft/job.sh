# DSpark drafting over several live slots (gate, TF_DSV41_SLOT_DRAFTS=1), a fragment
# tools/zig/dsv41_m1/job.sh sources (M2B_MDRAFT=1) on the TWO-GPU pod after slots_gate. It needs:
#   Z, LP, LOCAL, RES, hi, and job.sh's `step` / `left`;
#   $LOCAL/slots-ref (slots_gate's: trace.jsonl, aot/ with the row-mode buckets);
#   the R0 reference's assets (${R0:-$REF}: rope.bin, engram-host.bin, aot/ with DSpark's set P and the prompts' lengths);
#   optionally the sampled reference's trace ($REF/samp/trace.jsonl, M2B_SAMP).
# There is no Python load: drafted == serial (exact verification, keyed sampling), so the Python references are the
# slots gate's batched greedy replies and the sampled reference's keyed replies.
# Assets: a copy of R0's assets plus slots-ref's row-mode AOT set merged in. The 2- and 3-slot DSpark passes reuse set P's
# one-slot variants: dspark_emit.Layout keeps every launch's specialization (test-dsv41-mdraft).
# Runs (tf-dsv41-m1 mdraft through ranks.sh, 4 slots, every line with drafts, batched then alone):
#   mdraft-s4-eager  TF_DSV41_GRAPHS=0
#   mdraft-s4-graphs TF_DSV41_GRAPHS=1 (row graphs for the verify windows; the DSpark pass is eager, as in one slot)
# Each run's rank0.log has "PASS mdraft batched == Python", "... alone == Python", "... batched == alone" and
# "... several slots a pass", plus "mdraft stats batched / alone" (acceptance, tokens a round).
# Knobs: MDRAFT_RUNS ("s4-eager s4-graphs"), MDRAFT_MIN (7 a run), MDRAFT_DS_ASSETS (instead of R0 / REF),
# MDRAFT_SAMP_TRACE (default $REF/samp/trace.jsonl when present; "none" for greedy only), MDRAFT_GROUP
# (TF_DSV41_SLOT_DRAFT_GROUP: 1 = one pass a slot, the fallback if a variant were missing).
mdraft_gate() {
    local sref=$LOCAL/slots-ref ds=${MDRAFT_DS_ASSETS:-${R0:-${REF:-}}} assets=$LOCAL/mdraft-assets
    local strace=${MDRAFT_SAMP_TRACE:-${REF:+$REF/samp/trace.jsonl}} traces=() smp=() run g pool rc_=0
    if [[ ! -s "$sref/trace.jsonl" || ! -d "$sref/aot" ]]; then
        echo "FAIL mdraft: no slots reference ($sref/trace.jsonl, aot/): slots_gate runs first" | tee -a "$RES/steps.txt"
        return 1
    fi
    if [[ -z "$ds" || ! -d "$ds/aot" || ! -f "$ds/rope.bin" || ! -f "$ds/engram-host.bin" ]]; then
        echo "FAIL mdraft: no DSpark assets (R0 / REF / MDRAFT_DS_ASSETS: aot/, rope.bin, engram-host.bin)" | tee -a "$RES/steps.txt"
        return 1
    fi
    rm -rf "$assets"; mkdir -p "$assets" "$RES/mdraft"
    cp -r "$ds/aot" "$assets/aot"
    cp "$ds/rope.bin" "$ds/engram-host.bin" "$assets/"
    python -B "$Z/tools/zig/dsv41_aot_merge.py" "$assets/aot" "$sref/aot" | tee -a "$RES/steps.txt"
    traces=("$sref/trace.jsonl")
    # the sampled lines need the keyed sampler (its dsv41_sampling fatbin in the build); greedy lines alone do not
    if [[ "$strace" != none && -s "$strace" ]]; then traces+=("$strace"); smp=(TF_DSV41_SAMPLING=1); else echo "mdraft: greedy lines only (no sampled trace)" | tee -a "$RES/steps.txt"; fi
    cp "$sref/trace.jsonl" "$RES/mdraft/slots-trace.jsonl"
    [[ ${#traces[@]} == 2 ]] && cp "$strace" "$RES/mdraft/samp-trace.jsonl"
    pool=$(( 4 * 4096 + 2048 ))
    for run in ${MDRAFT_RUNS:-s4-eager s4-graphs}; do
        g=0; [[ "$run" == *graphs ]] && g=1
        (( $(left) >= 4 )) || { echo "mdraft-$run skipped: $(left) min left" | tee -a "$RES/steps.txt"; rc_=1; continue; }
        step "mdraft-$run" "${MDRAFT_MIN:-7}" env "${smp[@]}" TF_DSV41_SLOTS=4 TF_DSV41_SLOT_DRAFTS=1 TF_DSV41_DRAFTS=1 \
            TF_DSV41_GRAPHS=$g TF_DSV41_POOL_TOKENS=$pool TF_DSV41_LAYERS=0-$hi \
            ${MDRAFT_GROUP:+TF_DSV41_SLOT_DRAFT_GROUP=$MDRAFT_GROUP} \
            bash "$Z/tools/zig/dsv41_mdraft/ranks.sh" "$Z/bin/tf-dsv41-m1" "$LP" "$assets" "$RES/mdraft/$run" "${traces[@]}" || rc_=1
    done
    rm -rf "$assets"
    return $rc_
}
