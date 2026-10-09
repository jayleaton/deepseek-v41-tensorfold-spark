# The speculative DSpark pass's gate (gate; TF_DSV41_SPEC_DRAFT=1, prod's knob, and TF_DSV41_SPEC_NUCLEUS),
# a fragment tools/zig/dsv41_m1/job.sh sources inside M1_JOB=m2b (M2B_SPEC=1) on the TWO-GPU pod, after the M2b
# reference and the other gates. Needs: Z, LP, LOCAL, RES, hi, REF / R0 (the M2b reference: no extra Python load, no
# extra assets), job.sh's `step` / `left`; optional: $RES/m3-3 (M2B_LANES's mode 3, reused as the knob-off baseline),
# m4env + $RES/m4/trace.jsonl (M2B_M4), $REF/samp/trace.jsonl (M2B_SAMP's sampled reference, for SPEC_SAMP=1).
#   1. spec-m3-<k>: tf-dsv41-m1 m2b's lanes mode over the reference (m2b_ranks.sh), knob on. "3": DSpark's GPU pass
#      (begin / collect on rank 0, the whole pass on rank 1): tokens == the reference's serial greedy tokens (Python's)
#      and drafted / accepted == the knob-off run's (the same drafts, round for round), with speculations used.
#      "3t": with DSpark's trees (TF_DSV41_TREE=3; tree rounds are not speculated). "4": the tree oracle. Each must
#      keep its own PASS line ("PASS M3 lanes mode ...").
#   2. spec-m4 (with M2B_M4's trace): `generate` replays the server's replies (recorded knob off) with drafts and the
#      knob on: "PASS M4 CLI == HTTP" (tokens and drafted counts equal the knob-off server's; drafted == serial is
#      M2B_M4's own m4-serial).
#   3. spec-samp (SPEC_SAMP=1, with M2B_SAMP's reference and the sampling fatbin): the sampled replies with drafts,
#      TF_DSV41_SPEC_DRAFT=1 and TF_DSV41_SPEC_NUCLEUS=1 == Python's sampled replies (its top-p-without-top-k rows
#      speculate).
# Knobs: SPEC_RUNS ("3 3t 4"), SPEC_MIN (a run's minutes, 5), SPEC_M4 (1), SPEC_SAMP (0).
# Lines: "PASS spec <run>: ..." / "FAIL spec <run>: ..." in $RES/spec-<run>.log and $RES/steps.txt.

spec_verdict() { # RUN DIR [BASE_DIR]: the m3 run's json (and the knob-off baseline's) -> one PASS / FAIL line
    python3 - "$1" "$2/m3-rank0.json" "${3:-}" "$2/rank0.log" <<'EOF'
import json, re, sys
run, path, base, log = sys.argv[1:5]
try:
    r = json.load(open(path))
except Exception as e:
    print(f"FAIL spec {run}: no result ({e})"); sys.exit(1)
text = open(log, errors="replace").read() if log else ""
m = re.search(r"speculative DSpark passes (\d+) launched, (\d+) used", text)
launched, used = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
ok = r["pass"] and r["equal_prefix"] == r["want"]
why = [] if ok else ["its lanes gate failed"]
if run.startswith("3") and not (launched > 0 and used > 0):
    ok = False; why.append(f"no speculation used ({launched} launched, {used} used)")
if run == "3" and base:
    try:
        b = json.load(open(base))
        if (b["drafted"], b["accepted"]) != (r["drafted"], r["accepted"]):
            ok = False; why.append(f"drafted / accepted {r['drafted']} / {r['accepted']} vs knob off {b['drafted']} / {b['accepted']}")
    except Exception as e:
        ok = False; why.append(f"no knob-off baseline ({e})")
tpr = (r["tokens"] / r["rounds"]) if r.get("rounds") else 0.0
print(f"{'PASS' if ok else 'FAIL'} spec {run}: tokens {r['equal_prefix']}/{r['want']} equal (Python's serial), "
      f"drafted {r['drafted']}, accepted {r['accepted']}, speculations {launched} launched / {used} used"
      + (f", {tpr:.2f} tokens a round" if tpr else "") + ("" if ok else " -- " + "; ".join(why)))
sys.exit(0 if ok else 1)
EOF
}

spec_gate() {
    local A=${R0:-${REF:-}} rc=0 k m tree mb base port
    [[ -n "$A" && -f "$A/ref.json" ]] || { echo "FAIL spec: no M2b reference" | tee -a "$RES/steps.txt"; return 1; }
    # the knob-off baseline: M2B_LANES's mode 3 when it ran, else our own
    base=$RES/m3-3
    if [[ ! -f "$base/m3-rank0.json" ]]; then
        base=$RES/spec-off-3
        mb=$(left); (( mb > ${SPEC_MIN:-5} )) && mb=${SPEC_MIN:-5}
        if (( mb >= 3 )); then
            (export M2B_LANES=3 M2B_PORT=29781 TF_DSV41_TREE=0 TF_DSV41_SPEC_DRAFT=0
             step spec-off-3 "$mb" bash "$Z/tools/zig/dsv41_m1/m2b_ranks.sh" "$Z/bin/tf-dsv41-m1" "$LP" "$A" "$base") || rc=1
        fi
    fi
    for k in ${SPEC_RUNS:-3 3t 4}; do
        mb=$(left); (( mb > ${SPEC_MIN:-5} )) && mb=${SPEC_MIN:-5}
        (( mb >= 3 )) || { echo "FAIL spec $k: skipped, $mb min left" | tee -a "$RES/steps.txt"; rc=1; continue; }
        m=${k%t}; tree=0; [[ "$k" == *t ]] && tree=3
        port=$(( 29782 + m + (tree > 0 ? 10 : 0) ))
        (export M2B_LANES=$m M2B_PORT=$port TF_DSV41_TREE=$tree TF_DSV41_SPEC_DRAFT=1
         step "spec-m3-$k" "$mb" bash "$Z/tools/zig/dsv41_m1/m2b_ranks.sh" "$Z/bin/tf-dsv41-m1" "$LP" "$A" "$RES/spec-$k") || rc=1
        spec_verdict "$k" "$RES/spec-$k" "$base/m3-rank0.json" 2>&1 | tee "$RES/spec-$k.log" | tee -a "$RES/steps.txt"
        [[ "${PIPESTATUS[0]}" == 0 ]] || rc=1
        grep -h "dsv41 spec:" "$RES/spec-$k"/*.log 2>/dev/null | head -1 | sed "s/^/spec $k: /" | tee -a "$RES/steps.txt"
    done
    # 2. the served replies again with the knob on (M2B_M4's trace and env)
    if [[ "${SPEC_M4:-1}" == 1 ]]; then
        if [[ -s "$RES/m4/trace.jsonl" ]] && declare -p m4env > /dev/null 2>&1 && (( $(left) >= 6 )); then
            mb=$(left); (( mb > 8 )) && mb=8
            local A4 dir=$RES/spec-m4; A4=$(printf '%s\n' "${m4env[@]}" | sed -n 's/^TF_DSV41_ASSETS=//p')
            mkdir -p "$dir"
            (env "${m4env[@]}" TF_DSV41_DRAFTS=1 TF_DSV41_SPEC_DRAFT=1 TF_TP_RANK=1 TF_TP_DEVICE=1 TF_TP_PORT=29891 \
                 "$Z/bin/tf-dsv41-m1" follow "$LP" "$A4" > "$dir/follow.log" 2>&1 & fp=$!
             step spec-m4 "$mb" env "${m4env[@]}" TF_DSV41_DRAFTS=1 TF_DSV41_SPEC_DRAFT=1 TF_DSV41_ENGRAM_PREFETCH=0 \
                 TF_TP_RANK=0 TF_TP_DEVICE=0 TF_TP_PORT=29891 "$Z/bin/tf-dsv41-m1" generate "$LP" "$A4" "$RES/m4/trace.jsonl"
             r=$?; wait $fp; exit $r) || rc=1
            if grep -q "^PASS M4 CLI == HTTP" "$RES/spec-m4.log" && grep -q "speculative DSpark passes [1-9]" "$RES/spec-m4.log"; then
                echo "PASS spec m4: $(grep -h '^PASS M4' "$RES/spec-m4.log" | head -1); $(grep -ho 'speculative DSpark passes.*' "$RES/spec-m4.log" | head -1)"
            else
                echo "FAIL spec m4: $(grep -h '^PASS\|^FAIL' "$RES/spec-m4.log" | head -1)"; rc=1
            fi | tee -a "$RES/steps.txt"
        else echo "spec m4 skipped: no M2B_M4 trace / env or $(left) min left" | tee -a "$RES/steps.txt"; fi
    fi
    # 3. sampled replies with nucleus rows speculating (opt-in: needs M2B_SAMP's reference and the sampling fatbin)
    if [[ "${SPEC_SAMP:-0}" == 1 ]]; then
        if [[ -s "${REF:-}/samp/trace.jsonl" ]] && (( $(left) >= 7 )); then
            mb=$(left); (( mb > 8 )) && mb=8
            TF_DSV41_SPEC_DRAFT=1 TF_DSV41_SPEC_NUCLEUS=1 SAMP_KERNELS=0 SAMP_MODES=drafts SAMP_TRACE=$REF/samp/trace.jsonl \
                SAMP_MIN=$mb step spec-samp $(( mb + 1 )) bash "$Z/tools/zig/dsv41_samp/samp_step.sh" "$Z" "$LP" "$A" "$hi" \
                "$RES/spec-samp" "$LOCAL/spec-samp" || rc=1
            if grep -qh "^PASS M4 CLI == HTTP" "$RES"/spec-samp/samp-*.log 2>/dev/null; then
                echo "PASS spec samp: sampled replies == Python's with nucleus rows speculating"
            else echo "FAIL spec samp: $(grep -h '^PASS\|^FAIL' "$RES"/spec-samp/samp-*.log 2>/dev/null | head -1)"; rc=1; fi | tee -a "$RES/steps.txt"
        else echo "spec samp skipped: no sampled reference or $(left) min left" | tee -a "$RES/steps.txt"; fi
    fi
    return $rc
}
