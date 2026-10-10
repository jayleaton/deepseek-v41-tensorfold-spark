# The x3gm v2 prefill gate (gate), a fragment the combined pod job sources on a TWO-GPU pod inside
# M1_JOB=m2b (M2B_GM2PF=1, ws_gates), after the M2b reference (refs / R0) exists. Needs: Z, LP, RES, LOCAL, the `step` /
# `left` helpers of tools/zig/dsv41_m1/job.sh, `refs`, and the midprofile twin (midprofile.tar.gz in $Z or on the volume
# are prod's 8474f31 byte for byte; /dsv41-tf and the R1 twin predate v2 and ignore TF_DSV41_GM_V2=1).
#   1. gm2pf-cap-v2 / -v1 (GPU 0): dsv41_m1_capture.py on the twin with prod-knobs.env + TF_DSV41_GM_V2=1 / 0, sets
#      GM2PF_SETS (B: layers 20-24), a GM2PF_ROWS-row prefill segment (2048: prod's chunk), the same prompt;
#      - v2: `tf-dsv41-m1 check` (host: block_prefill's v2 calls == the capture's launches, "options: x3gm v2 one",
#        0 mismatches), then `tf-dsv41-m1 both` (GPU 0: every captured launch replayed bit for bit, gateup2 / down2
#        included, then our emitter's calls on our buffers == Python v2's bits launch by launch);
#      - v1 vs v2: same.py: every routed partial and phase end the same bits (Python's v2 == v1);
#      small files to $RES/gm2pf-cap-v*/, the blobs deleted after each.
#   2. gm2pf-off / gm2pf-v2 (both GPUs): the M2B_PREFILL gate (Zig's prefill from an empty slot, every state role == the
#      reference's dump, the first token, then the decode gate) with TF_DSV41_GM_V2 unset, then =1 on the Zig ranks.
#      v2's state == the reference's == off's (the reference twin's x3gm, v1 or v2, is the same bits); rank 0 logs
#      "prod knobs: ..., x3gm v2 one". Timing: "Zig prefill rank 0: N rows in S s" of each, GM2PF_REPEAT times
#      alternating off / v2 (cold first prefills: compare like with like).
# Knobs: GM2PF_STEPS (default "cap prefill"), GM2PF_SETS (B), GM2PF_ROWS (2048), GM2PF_REF (default: the last of refs,
# the longest prompt), GM2PF_REPEAT (1), GM2PF_CAP_MIN (12), GM2PF_M1_MIN (12), GM2PF_MIN (a prefill run's minutes, 6).
gm2pf_gate() {
    local st=0 s
    mkdir -p "$RES/gm2pf"
    for s in ${GM2PF_STEPS:-cap prefill}; do
        case $s in
            cap) gm2pf_cap || st=1 ;;
            prefill) gm2pf_prefill || st=1 ;;
            *) echo "gm2pf: unknown step $s" | tee -a "$RES/steps.txt"; st=1 ;;
        esac
    done
    return $st
}

gm2pf_cap() {
    local mt="" c v dir name st=0 BIN=$Z/bin/tf-dsv41-m1
    for c in "${MIDPROFILE_TAR:-}" "$Z/midprofile.tar.gz" "${PACK%%/out/*}/m5/midprofile.tar.gz"; do [[ -f "$c" ]] && { mt=$c; break; }; done
    if [[ ! -d /opt/midprofile/src ]] && { [[ -z "$mt" ]] || ! tar -xzf "$mt" -C /; }; then
        echo "gm2pf-cap skipped: no midprofile.tar.gz (set MIDPROFILE_TAR to a staged local archive)" | tee -a "$RES/steps.txt"; return 1
    fi
    echo "gm2pf twin: $(head -1 /opt/midprofile/COMMIT 2>/dev/null)" | tee -a "$RES/steps.txt"
    for v in 1 0; do
        name=gm2pf-cap-v$((v ? 2 : 1)); dir=$LOCAL/$name
        (( $(left) >= 6 )) || { echo "$name skipped: $(left) min left" | tee -a "$RES/steps.txt"; st=1; continue; }
        rm -rf "$dir"
        if (set -a; source "$Z/tools/zig/dsv41_m1/prod-knobs.env"; set +a
            export PYTHONPATH=/opt/midprofile/src TORCH_EXTENSIONS_DIR=/opt/midprofile/ext TF_DSV41_GM_V2=$v \
                   CUDA_VISIBLE_DEVICES=0 M1_PREFILL_ROWS=${GM2PF_ROWS:-2048}
            unset M1_PAGED M1_SPLIT
            step "$name" "${GM2PF_CAP_MIN:-12}" python -B "$Z/tools/zig/dsv41_m1_capture.py" --pack "$LP" --out "$dir" \
                --sets "${GM2PF_SETS:-B}"); then
            du -sh "$dir" | tee "$RES/$name-size.txt"
            if (( v )); then
                "$BIN" check "$dir" > "$RES/$name-check.log" 2>&1 || st=1
                echo "$name check: $(grep -h 'options: x3gm' "$RES/$name-check.log") $(tail -1 "$RES/$name-check.log")" | tee -a "$RES/steps.txt"
                local mb; mb=$(left); (( mb > ${GM2PF_M1_MIN:-12} )) && mb=${GM2PF_M1_MIN:-12}
                if (( mb >= 3 )); then
                    CUDA_VISIBLE_DEVICES=0 step "m1-$name" "$mb" "$BIN" both "$LP" "$dir" "$RES/m1-$name" || st=1
                else echo "m1-$name skipped: $(left) min left" | tee -a "$RES/steps.txt"; st=1; fi
            fi
        else st=1; fi
        mkdir -p "$RES/$name"
        for c in meta.json weights.json manifest.json jit.json; do [[ -f "$dir/$c" ]] && cp "$dir/$c" "$RES/$name/"; done
        [[ -f "$dir/ops.jsonl" ]] && gzip -c "$dir/ops.jsonl" > "$RES/$name/ops.jsonl.gz"
        rm -rf "$dir/blobs"
    done
    if [[ -f "$RES/gm2pf-cap-v2/ops.jsonl.gz" && -f "$RES/gm2pf-cap-v1/ops.jsonl.gz" ]]; then
        python -I "$Z/tools/zig/dsv41_gm2pf/same.py" "$RES/gm2pf-cap-v2" "$RES/gm2pf-cap-v1" > "$RES/gm2pf-same.log" 2>&1 || st=1
        tail -1 "$RES/gm2pf-same.log" | tee -a "$RES/steps.txt"
    else echo "FAIL gm2pf-same: a capture is missing" | tee -a "$RES/steps.txt"; st=1; fi
    return $st
}

gm2pf_prefill() {
    local ref=${GM2PF_REF:-${refs[${#refs[@]}-1]}} r k i=0 st=0 kv
    for r in $(seq 1 "${GM2PF_REPEAT:-1}"); do
        for k in off v2; do
            (( $(left) >= 3 )) || { echo "gm2pf-$k-$r skipped: $(left) min left" | tee -a "$RES/steps.txt"; continue; }
            kv=(TF_DSV41_GM_V2=0); [[ $k == v2 ]] && kv=(TF_DSV41_GM_V2=1)
            (export M2B_PREFILL=1 M2B_PORT=$(( 29790 + i )) "${kv[@]}"
             step "gm2pf-$k-$r" "${GM2PF_MIN:-6}" bash "$Z/tools/zig/dsv41_m1/m2b_ranks.sh" "$Z/bin/tf-dsv41-m1" "$LP" \
                 "$ref" "$RES/gm2pf/$k-$r") || st=1
            grep -h "prod knobs:\|Zig prefill rank" "$RES/gm2pf/$k-$r/rank0.log" 2>/dev/null | sed "s/^/gm2pf-$k-$r: /" \
                | tee -a "$RES/steps.txt"
            i=$(( i + 1 ))
        done
    done
    return $st
}
