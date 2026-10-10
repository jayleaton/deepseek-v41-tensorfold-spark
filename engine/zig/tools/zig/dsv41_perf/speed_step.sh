#!/usr/bin/env bash
# gate's pod step, after job.sh's M1_JOB=m2b run (docs/DSV41-ZIG-PORT.md 0a, "Pod steps"): the decode gap
# tables from what the R1 gates and M4 left in RES, and the pass / fail lines of the speed work.
#   bash speed_step.sh Z RES
# Reads: RES/prof-rank0.json (dsv41_m2b_ref.py --profile, M2B_PROFILE_ROWS), RES/m2b-r1/m2b-rank0.jsonl and
# RES/m2b-r1-graphs/m2b-rank0.jsonl (M2B_ZIG_PROF_ROWS, eager / graphed), RES/m2b-r1/prof-rank0.json (M2B_PROF=1),
# RES/m4/phases.json (TF_DSV41_PHASES_OUT), RES/m4/server.log. Writes RES/speed-gap.md; prints its summary.
set -u
Z=$1 RES=$2
out=$RES/speed-gap.md
{
    echo "# Decode gap, pod $(date -u +%F)"
    echo
    if [[ -f $RES/prof-rank0.json ]]; then
        zp=(); [[ -f $RES/m2b-r1/prof-rank0.json ]] && zp=(--zig-profile "$RES/m2b-r1/prof-rank0.json")
        zl=(); for d in m2b-r1 m2b-r1-graphs; do [[ -f $RES/$d/m2b-rank0.jsonl ]] && zl+=(--zig "$RES/$d/m2b-rank0.jsonl"); done
        echo "## Windows (rank 0, TP=2, layers 0-hi, R1): Zig vs Python"
        python3 -I "$Z/tools/zig/dsv41_perf/gap.py" windows --py "$RES/prof-rank0.json" "${zl[@]}" "${zp[@]}"
    else
        echo "no Python profile (M2B_PROFILE_ROWS unset or the R1 reference failed)"
    fi
    echo
    if [[ -f $RES/m4/phases.json ]]; then
        echo "## M4 rounds (Zig, served, TF_DSV41_PHASES=1)"
        echo '```'
        cat "$RES/m4/phases.json"
        echo '```'
    fi
} > "$out" 2>&1
cat "$out"
# the lines the integrator greps (graphs on by default in M4; Engram rows from the shards; R3's reads)
for f in "$RES"/m4/server.log "$RES"/m4/follow-server.log; do
    [[ -f $f ]] || continue
    grep -h "graphs: on\|graphs: .* captured\|engram: rows from\|engram: TF_DSV41_ENGRAM_DIR unset" "$f" | sed "s|^|$(basename "$f"): |"
done
