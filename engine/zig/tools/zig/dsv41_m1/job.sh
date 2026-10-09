#!/usr/bin/env bash
# DeepSeek-V4.1's M1 GPU job on ONE GPU of the Python twin's architecture (RTX PRO 6000, sm_120), in the e1e2 runtime
#   (order: the capture and our gates first, then the kernel job in the budget left, then the optional r1c step)
#   1. kernels: the kernel exactness job (zig/tests/cuda/deepseek_v41/job.sh: oracle fixtures + the twin's own GPU
#      suites recorded and replayed), on the decode1 twin when r1c.tar.gz is staged
#   2. capture: tools/zig/dsv41_m1_capture.py, the Python engine's blocks 1-3 and 20-24 on our pack, every launch's
#      buffers before / after (local NVMe)
#   3. m1: tf-dsv41-m1 both, our loader + every launch re-issued and compared bit for bit (the replay), then block.zig's
#      calls for every captured decode window on our own buffers, structure on the host then bits on the GPU (the
#      block forward; M1_MODE=replay|forward runs one)
#   4. r1c (optional, M1_R1C_MIN > 0 with r1c.tar.gz staged): a second capture with prod + R1c knobs (r1c-knobs.env) on
#      the decode1 twin, set B by default, replayed bit for bit, when M1_R1C_MIN minutes of the budget remain
# Each step has its own timeout and log in $RES; a failing step does not stop the next. Exit 0 = every step passed.
# Budget: M1_BUDGET_MIN (default 52) of a 60-minute hard max, the rest for the runtime unpack and packing: capture 18
# + m1 14 min at most, the kernel job gets what is left.
#
#   Z=<staged tree: bin/ fatbin/ tools/ zig/> PACK=<pack dir> RES=<results dir> LOCAL=<scratch> bash job.sh
set -u
Z=${Z:?staged tree}; PACK=${PACK:?pack}; RES=${RES:?results}; LOCAL=${LOCAL:?scratch}
mkdir -p "$RES" "$LOCAL"
export PYTHONPATH=/dsv41-tf/src TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/opt/e1e2/ext} \
    TRITON_CACHE_DIR=$LOCAL/triton CUDA_CACHE_PATH=$LOCAL/nv PYTHONDONTWRITEBYTECODE=1
rc=0
T0=$(date +%s)
# wrote (pod 12 lost a whole M2b run: the container disk goes with the pod, the tarball comes only at the end)
VOLRES=${PACK%%/out/*}/results/live/$(basename "$(dirname "$RES")")
( while sleep 30; do mkdir -p "$VOLRES" && cp -ru "$RES/." "$VOLRES/" 2>/dev/null; done ) &
LIVE=$!
trap 'kill $LIVE 2>/dev/null; mkdir -p "$VOLRES" && cp -ru "$RES/." "$VOLRES/" 2>/dev/null' EXIT
step() { # NAME MINUTES CMD...: output in $RES/NAME.log, rc and seconds in $RES/steps.txt
    local name=$1 min=$2 t0 r; shift 2
    t0=$(date +%s); echo "=== $(date -u +%H:%M:%S) $name: $*"
    timeout --kill-after=30 "$(( min * 60 ))" "$@" > "$RES/$name.log" 2>&1; r=$?
    echo "$name rc=$r s=$(( $(date +%s) - t0 ))" | tee -a "$RES/steps.txt"
    tail -5 "$RES/$name.log"
    (( r == 0 )) || rc=1
    return $r
}
# M1_KNOBS=prod: the prod config's kernel knobs (prod-knobs.env) for the capture; default: the engine's code defaults
if [[ "${M1_KNOBS:-}" == prod ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$Z/tools/zig/dsv41_m1/prod-knobs.env"
    set +a
    echo "knobs: prod ($(grep -c '^TF_' "$Z/tools/zig/dsv41_m1/prod-knobs.env") TF_DSV41_* settings)"
fi
{ cat "$Z/COMMIT" 2>/dev/null; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
  python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)"
  free -g | head -2; df -h "$LOCAL" | tail -1; } > "$RES/env.txt" 2>&1
cat "$RES/env.txt"

# the decode1 twin (r1c.tar.gz: /opt/r1c/{src,tests,ext}, its extensions prebuilt for sm_120 at these absolute paths, so
# torch's JIT loads them without building): the kernel job's Python twin, and the optional R1c capture's
PYTWIN=/dsv41-tf PYEXT=$TORCH_EXTENSIONS_DIR
if [[ -f "$Z/r1c.tar.gz" ]] && tar -xzf "$Z/r1c.tar.gz" -C /; then PYTWIN=/opt/r1c PYEXT=/opt/r1c/ext; fi
echo "kernel job twin: $PYTWIN" | tee -a "$RES/env.txt"
left() { echo $(( ${M1_BUDGET_MIN:-52} - ($(date +%s) - T0) / 60 )); }

# M1_JOB=x3gm3: the kernel x3gm v3 gate alone (zig/tests/cuda/deepseek_v41/x3gm3_job.sh: the x3gm fixtures,
# every v3 variant bit for bit, the prod-shape timing bench), in the budget
if [[ "${M1_JOB:-m1}" == x3gm3 ]]; then
    (for v in $(compgen -e | grep '^TF_DSV41_'); do unset "$v"; done
     export TF_DSV41_PY_SRC=$PYTWIN TF_DSV41_TEST=$Z/bin/tf-dsv41-test TORCH_EXTENSIONS_DIR=$PYEXT
     step x3gm3 "$(left)" bash "$Z/zig/tests/cuda/deepseek_v41/x3gm3_job.sh" "$RES/x3gm3") || rc=1
    exit $rc
fi

# M1_JOB=m5 (TWO GPUs): M5's Python side on the midprofile twin (tensorfold-decode1 dsv41-midprofile: split KV +
# sessions; midprofile.tar.gz = /opt/midprofile/{src,tests,ext}, its extensions prebuilt for sm_120, from
# in what the budget leaves (a capture starts only while M5_REF_RESERVE minutes stay for the references and gates):
#   1. the pack's shards up to layer M5_HI (24) copied to local NVMe (as M2b);
#   2. captures on GPU 0 (prod-knobs.env + x3gm v2, as M2B_CAPTURE): M5_CAPTURES, each MODE:SETS:PREFILL_ROWS with
#      MODE paged (M1_PAGED=1), split (M1_SPLIT=1) or compact (split + TF_DSV41_KV_SPLIT_COMPACT=force); then
#      `tf-dsv41-m1 check` on each (host only), `tf-dsv41-m1 replay` on the ones named in M5_REPLAY (GPU 0, bit for bit);
#      small files to $RES/m5-cap-<mode>-<sets>-p<rows>/, the blobs deleted after each;
#   3. the references at TP=2 (dsv41_m2b_ref.py on both GPUs, layers 0-M5_HI, prompts M5_PROMPTS, M2B_STEPS steps,
#      expert top-p off, TF_DSV41_INDEX_KV=M5_INDEX_KV): M5_REFS of pool (--pool), split (--split), compact (--split
#      --compact force), park (--park on M5_PARK_KV's pool: save / park / restore == the uninterrupted run); each into
#      $LOCAL/m5-ref-<mode>/p<N>/ with engram-host.bin; then m5-agree: every mode's tokens and logits digests == pool's
#      (and the park round trip's == its own uninterrupted run);
#   4. the Zig gates: `tf-dsv41-m1 m5 MODE $LP $REF $RES/m5-MODE` on both ranks (m5_ranks.sh) for M5_GATES (split
#      compact park; REF = $LOCAL/m5-ref-MODE, holding p<N>/ a prompt), skipped with a message when the binary's
#      usage has no m5 command.
# Knobs: M5_HI M5_PROMPTS M2B_STEPS M5_CAPTURES M5_REPLAY M5_REFS M5_GATES M5_INDEX_KV M5_PARK_KV M5_REF_RESERVE
# M5_CAPTURE_MIN M5_REF_MIN M5_GATE_MIN M2B_COPY_MIN (defaults below), M5_REFS_FROM (an earlier pod's references:
# no reference runs, no m5-agree); M1_BUDGET_MIN as every job.
if [[ "${M1_JOB:-m1}" == m5 ]]; then
    hi=${M5_HI:-24}
    LP=$LOCAL/pack
    mkdir -p "$LP"
    : > "$LOCAL/pack-files"
    for f in "$PACK"/*; do
        b=$(basename "$f")
        if [[ "$b" =~ ^layer-([0-9]+)\.safetensors$ ]]; then
            L=$(( 10#${BASH_REMATCH[1]} ))
            (( L > hi )) && continue
        fi
        echo "$b" >> "$LOCAL/pack-files"
    done
    ( while sleep 30; do echo "$(date -u +%H:%M:%S) $(du -sm "$LP" | cut -f1) MB"; done ) > "$RES/pack-copy-progress.log" 2>&1 &
    prog=$!
    if ! (cd "$PACK" && step pack-copy "${M2B_COPY_MIN:-10}" xargs -a "$LOCAL/pack-files" -P 16 -I{} cp {} "$LP/"); then
        echo "pack copy did not finish: loading from the volume" | tee -a "$RES/steps.txt"
        LP=$PACK
    fi
    kill $prog 2>/dev/null
    du -sh "$LP" | tee -a "$RES/steps.txt"
    # the midprofile twin: nothing below runs without it (the split pool, Exchange and sessions are its code)
    MT=""
    for c in "${MIDPROFILE_TAR:-}" "$Z/midprofile.tar.gz" "${PACK%%/out/*}/m5/midprofile.tar.gz"; do [[ -f "$c" ]] && { MT=$c; break; }; done
    if [[ -z "$MT" ]] || ! tar -xzf "$MT" -C /; then
        echo "m5: no midprofile.tar.gz (set MIDPROFILE_TAR to a staged local archive) or it did not unpack: nothing to run" \
            | tee -a "$RES/steps.txt"
        echo "=== rc=1"; exit 1
    fi
    TWIN=/opt/midprofile
    { echo "m5 twin: $MT -> $TWIN ($(cat "$TWIN/COMMIT" 2>/dev/null | head -1))"; ls "$TWIN/ext" | tr '\n' ' '; echo; } \
        | tee -a "$RES/env.txt"
    BIN=$Z/bin/tf-dsv41-m1
    reserve=${M5_REF_RESERVE:-14}
    # 2. the captures (GPU 0)
    for spec in ${M5_CAPTURES:-paged:B,D:2048 split:B,D:2048 paged:B,D:12 split:B,D:12 compact:B:12}; do
        IFS=: read -r mode sets rows <<< "$spec"
        name=m5-cap-$mode-${sets//,/}-p$rows
        dir=$LOCAL/$name
        l=$(left); mb=$(( l - reserve )); (( mb > ${M5_CAPTURE_MIN:-10} )) && mb=${M5_CAPTURE_MIN:-10}
        (( mb >= 3 )) || { echo "$name skipped: $l min left, $reserve kept for the references" | tee -a "$RES/steps.txt"; rc=1; continue; }
        rm -rf "$dir"
        if (set -a; source "$Z/tools/zig/dsv41_m1/prod-knobs.env"; set +a
            export PYTHONPATH=$TWIN/src TORCH_EXTENSIONS_DIR=$TWIN/ext TF_DSV41_GM_V2=1 CUDA_VISIBLE_DEVICES=0 \
                   M1_PREFILL_ROWS=$rows
            case $mode in
                paged) export M1_PAGED=1 ;;
                split) export M1_SPLIT=1 TF_DSV41_KV_SPLIT_COMPACT=0 ;;
                compact) export M1_SPLIT=1 TF_DSV41_KV_SPLIT_COMPACT=force ;;
                *) echo "unknown capture mode $mode"; exit 2 ;;
            esac
            step "$name" "$mb" python -B "$Z/tools/zig/dsv41_m1_capture.py" --pack "$LP" --out "$dir" --sets "$sets"); then
            du -sh "$dir" | tee "$RES/$name-size.txt"
            "$BIN" check "$dir" > "$RES/$name-check.log" 2>&1; echo "$name check: $(tail -1 "$RES/$name-check.log")" | tee -a "$RES/steps.txt"
            if [[ " ${M5_REPLAY:-paged-p2048 split-p2048} " == *" $mode-p$rows "* ]]; then
                l=$(left); mb=$(( l - reserve )); (( mb > 10 )) && mb=10
                if (( mb >= 3 )); then
                    CUDA_VISIBLE_DEVICES=0 step "replay-$name" "$mb" "$BIN" replay "$LP" "$dir" "$RES/replay-$name" || rc=1
                else echo "replay-$name skipped: $l min left" | tee -a "$RES/steps.txt"; fi
            fi
        else rc=1; fi
        mkdir -p "$RES/$name"
        for f in meta.json weights.json manifest.json jit.json; do [[ -f "$dir/$f" ]] && cp "$dir/$f" "$RES/$name/"; done
        [[ -d "$dir/aot" ]] && tar -czf "$RES/$name/aot.tar.gz" -C "$dir" aot
        [[ -f "$dir/ops.jsonl" ]] && gzip -c "$dir/ops.jsonl" > "$RES/$name/ops.jsonl.gz"
        [[ -f "$dir/launches.json" ]] && gzip -c "$dir/launches.json" > "$RES/$name/launches.json.gz"
        rm -rf "$dir/blobs"
    done
    # 3. the references (both GPUs), engine defaults but the index keys' format, expert top-p off
    unset TF_DSV41_EXPERT_TOPP
    export TF_DSV41_INDEX_KV=${M5_INDEX_KV:-fp8}
    PROMPTS=${M5_PROMPTS:-2060 9}
    port=29811
    okrefs=()
    for mode in ${M5_REFS:-pool split compact park}; do
        REF=$LOCAL/m5-ref-$mode
        rm -rf "$REF"
        # M5_REFS_FROM=<dir>: an earlier pod's references (m5-ref-<mode>/, engram-host.bin in each prompt's dir)
        if [[ -n "${M5_REFS_FROM:-}" && -d "$M5_REFS_FROM/m5-ref-$mode" ]]; then
            cp -r "$M5_REFS_FROM/m5-ref-$mode" "$REF" && okrefs+=("$mode") && echo "m5-ref-$mode from $M5_REFS_FROM" | tee -a "$RES/steps.txt"
            continue
        fi
        case $mode in
            pool) args=(--pool) ;;
            split) args=(--split --compact 0) ;;
            compact) args=(--split --compact force) ;;
            park) args=(--park); [[ "${M5_PARK_KV:-pool}" == split ]] && args+=(--split --compact 0) ;;
            *) echo "unknown reference mode $mode" | tee -a "$RES/steps.txt"; rc=1; continue ;;
        esac
        l=$(left); mb=$l; (( mb > ${M5_REF_MIN:-8} )) && mb=${M5_REF_MIN:-8}
        (( mb >= 3 )) || { echo "m5-ref-$mode skipped: $l min left" | tee -a "$RES/steps.txt"; rc=1; continue; }
        # prod's kernel knobs (the path the Zig emitters emit), as every M2b reference runs
        if (set -a; source "$Z/tools/zig/dsv41_m1/prod-knobs.env"; set +a; unset TF_DSV41_EXPERT_TOPP
            export PYTHONPATH=$TWIN/src TORCH_EXTENSIONS_DIR=$TWIN/ext TF_DSV41_INDEX_KV=${M5_INDEX_KV:-fp8}
            step "m5-ref-$mode" "$mb" python -u -B "$Z/tools/zig/dsv41_m2b_ref.py" --pack "$LP" --out "$REF" \
                --layers "0-$hi" --steps "${M2B_STEPS:-64}" --prompts "$(echo $PROMPTS | tr ' ' ',')" --port $port "${args[@]}"); then
            okrefs+=("$mode")
        else rc=1; fi
        port=$(( port + 10 ))
        o=$RES/m5-ref-$mode
        for p in $PROMPTS; do
            d=$REF/p$p; mkdir -p "$o/p$p"
            for f in ref.json rank0.json rank1.json manifest.json; do [[ -f "$d/$f" ]] && cp "$d/$f" "$o/p$p/"; done
            for r in 0 1; do [[ -f "$d/rank$r/state/state.json" ]] && cp "$d/rank$r/state/state.json" "$o/p$p/state-rank$r.json"; done
        done
        [[ -d "$REF/aot" ]] && tar -czf "$o/aot.tar.gz" -C "$REF" aot
    done
    if (( ${#okrefs[@]} )) && [[ -z "${M5_REFS_FROM:-}" ]]; then
        if step m5-engram-host 3 python -B "$Z/tools/zig/dsv41_engram_host.py" --py-src "$TWIN/src" \
                --config "$LP/config.json" --tokenizer "$LP/tokenizer.json" --out "$LOCAL/m5-engram-host.bin"; then
            for mode in "${okrefs[@]}"; do for p in $PROMPTS; do cp "$LOCAL/m5-engram-host.bin" "$LOCAL/m5-ref-$mode/p$p/engram-host.bin"; done; done
        else rc=1; fi
        # every mode's tokens and per-step logits digests == the replicated pool's (split == replicated bit for bit;
        # the park reference's uninterrupted run too, and its restored run == that run: ref.json park_equal)
        step m5-agree 1 python -B - "$LOCAL" "$PROMPTS" "${okrefs[@]}" <<'PY' || rc=1
import json, sys
from pathlib import Path
local, prompts, modes = Path(sys.argv[1]), sys.argv[2].split(), sys.argv[3:]
bad = 0
for p in prompts:
    refs = {m: json.loads((local / f"m5-ref-{m}" / f"p{p}" / "ref.json").read_text()) for m in modes
            if (local / f"m5-ref-{m}" / f"p{p}" / "ref.json").exists()}
    base = refs.get("pool")
    for m, r in refs.items():
        same = base is None or (r["tokens"] == base["tokens"] and
                                all(a["logits_sha256"] == b["logits_sha256"] for a, b in zip(r["ranks"], base["ranks"])))
        park = r.get("park_equal")
        ok = same and park is not False
        bad += not ok
        print(f"p{p} {m}: tokens {len(r['tokens'])}, == pool {same if base else '(no pool reference)'}"
              + (f", park round trip {'==' if park else '!='} uninterrupted" if park is not None else "")
              + ("" if ok else "  MISMATCH"))
print(f"m5-agree: {bad} mismatches")
sys.exit(1 if bad else 0)
PY
    fi
    # 4. the Zig gates (tf-dsv41-m1 m5 MODE PACK REF OUT): placeholders until the binary has the command
    for mode in ${M5_GATES:-split compact park}; do
        REF=$LOCAL/m5-ref-$mode
        if ! "$BIN" 2>&1 | grep -q 'tf-dsv41-m1 m5'; then
            echo "m5-$mode skipped: $BIN has no m5 command yet (its usage lists none)" | tee -a "$RES/steps.txt"; continue
        fi
        [[ " ${okrefs[*]} " == *" $mode "* ]] || { echo "m5-$mode skipped: no m5-ref-$mode" | tee -a "$RES/steps.txt"; rc=1; continue; }
        l=$(left); mb=$l; (( mb > ${M5_GATE_MIN:-6} )) && mb=${M5_GATE_MIN:-6}
        (( mb >= 3 )) || { echo "m5-$mode skipped: $l min left" | tee -a "$RES/steps.txt"; rc=1; continue; }
        (export TF_TP_PORT=$port; step "m5-$mode" "$mb" bash "$Z/tools/zig/dsv41_m1/m5_ranks.sh" "$BIN" "$mode" "$LP" "$REF" "$RES/m5-$mode") || rc=1
        port=$(( port + 10 ))
    done
    echo "=== rc=$rc"
    exit $rc
fi

# M1_JOB=m2a: M2a instead of M1 (the Python reference over layers M2A_LAYERS (0-8) with its slot state, rope tables
# and AOT set; the Engram host tables; tf-dsv41-m1 m2a; then the kernel job in what is left)
# M1_JOB=m2b (TWO GPUs): M2b. The pack's shards up to layer M2B_HI (27: ~69 GB a rank; the whole backbone does not
# fit a 96 GB GPU a rank) copied to local NVMe in parallel; the Python engine at TP=2 (tools/zig/dsv41_m2b_ref.py,
# prebuilt sm_120 extensions from e1e2x.tar.gz, expert top-p unset) prompts and decodes greedily; then our two ranks
# (tf-dsv41-m1 m2b, NCCL through the tp runtime) from its state; then the kernel job in what is left.
if [[ "${M1_JOB:-m1}" == m2b ]]; then
    hi=${M2B_HI:-24}
    LANES_MODES=${M2B_LANES:-}; unset M2B_LANES   # the gate's mode list; tf-dsv41-m1 reads M2B_LANES as one mode
    LP=$LOCAL/pack
    mkdir -p "$LP"
    : > "$LOCAL/pack-files"
    for f in "$PACK"/*; do
        b=$(basename "$f")
        if [[ "$b" =~ ^layer-([0-9]+)\.safetensors$ ]]; then
            L=$(( 10#${BASH_REMATCH[1]} ))
            # M2B_CAPTURE with set P: DSpark's target layers 36-39 come too (the capture's backbone blocks)
            (( L > hi )) && ! { [[ ",${M2B_CAPTURE:-},${M2B_CAPTURE_R1:-}," == *,P,* ]] && (( L >= 36 && L <= 39 )); } && continue
        fi
        echo "$b" >> "$LOCAL/pack-files"
    done
    ( while sleep 30; do echo "$(date -u +%H:%M:%S) $(du -sm "$LP" | cut -f1) MB"; done ) > "$RES/pack-copy-progress.log" 2>&1 &
    prog=$!
    # the local copy within 10 minutes, else every load reads the volume
    if ! (cd "$PACK" && step pack-copy "${M2B_COPY_MIN:-10}" xargs -a "$LOCAL/pack-files" -P 16 -I{} cp {} "$LP/"); then
        echo "pack copy did not finish: loading from the volume" | tee -a "$RES/steps.txt"
        LP=$PACK
    fi
    kill $prog 2>/dev/null
    du -sh "$LP" | tee -a "$RES/steps.txt"
    if [[ -f "$Z/e1e2x.tar.gz" ]]; then tar -xzf "$Z/e1e2x.tar.gz" -C / && export TORCH_EXTENSIONS_DIR=/opt/e1e2-120; fi
    # M2B_CAPTURE (sets, e.g. P,B): a capture on GPU 0 first, on prod's knobs + x3gm v2 (TF_DSV41_GM_V2=1, as prod
    # 8474f31) with an M1_PREFILL_ROWS-row prefill segment (2048: prod's chunk), replayed bit for bit by our loader;
    # its small files (ops, meta, weights, manifest, AOT set) come back for the offline emitters (DSpark pass, prefill).
    # M2B_CAPTURE_R1 (sets): a second capture with r1c-knobs.env's R1 on top, on the R1 twin (r1c.tar.gz: decode1
    # 78b703d), into capture-r1 / m1-r1: block.zig's R1 path (Options.r1, from the capture's env) checked the same way
    capture_set() { # NAME SETS R1(0/1): capture, M1_MODE over it, small files to $RES/NAME
        local name=$1 sets=$2 r1=$3 dir=$LOCAL/$1
        rm -rf "$dir"
        if (set -a; source "$Z/tools/zig/dsv41_m1/prod-knobs.env"
            (( r1 )) && source "$Z/tools/zig/dsv41_m1/r1c-knobs.env"; set +a
            (( r1 )) && export PYTHONPATH=/opt/r1c/src TORCH_EXTENSIONS_DIR=/opt/r1c/ext
            export TF_DSV41_GM_V2=1 CUDA_VISIBLE_DEVICES=0 M1_PREFILL_ROWS=${M1_PREFILL_ROWS:-2048}
            step "$name" "${M1_CAPTURE_MIN:-12}" python -B "$Z/tools/zig/dsv41_m1_capture.py" --pack "$LP" --out "$dir" \
                --sets "$sets"); then
            du -sh "$dir" | tee "$RES/$name-size.txt"
            # M1_MODE=both: also our emitters' launches on our buffers (block.zig, block_prefill.zig, dspark_emit.zig)
            "$Z/bin/tf-dsv41-m1" check "$dir" > "$RES/$name-check.log" 2>&1; tail -1 "$RES/$name-check.log" | tee -a "$RES/steps.txt"
            [[ ",$sets," == *,P,* ]] && { "$Z/bin/tf-dsv41-m1" dcheck "$dir" > "$RES/$name-dcheck.log" 2>&1; tail -1 "$RES/$name-dcheck.log" | tee -a "$RES/steps.txt"; }
            [[ "${M1_MODE:-replay}" == skip ]] || CUDA_VISIBLE_DEVICES=0 step "m1-$name" "${M1_REPLAY_MIN:-10}" "$Z/bin/tf-dsv41-m1" "${M1_MODE:-replay}" "$LP" "$dir" "$RES/m1-$name" || rc=1
        else rc=1; fi
        mkdir -p "$RES/$name"
        for f in meta.json weights.json manifest.json jit.json; do [[ -f "$dir/$f" ]] && cp "$dir/$f" "$RES/$name/"; done
        [[ -d "$dir/aot" ]] && tar -czf "$RES/$name/aot.tar.gz" -C "$dir" aot
        [[ -f "$dir/ops.jsonl" ]] && gzip -c "$dir/ops.jsonl" > "$RES/$name/ops.jsonl.gz"
        [[ -f "$dir/launches.json" ]] && gzip -c "$dir/launches.json" > "$RES/$name/launches.json.gz"
        rm -rf "$dir/blobs"
    }
    [[ -n "${M2B_CAPTURE_R1:-}" ]] && capture_set capture-r1 "$M2B_CAPTURE_R1" 1
    if [[ -n "${M2B_CAPTURE:-}" ]]; then capture_set capture "$M2B_CAPTURE" 0; CAP=$LOCAL/capture; fi
    # M2B_TOPP unset: top-p off (M2b); set (prod: 0.85): the same gate with prod's expert prune (M2c's first stage)
    unset TF_DSV41_EXPERT_TOPP
    [[ -n "${M2B_TOPP:-}" ]] && export TF_DSV41_EXPERT_TOPP=$M2B_TOPP
    REF=$LOCAL/m2b-ref
    rm -rf "$REF"
    # M2B_PROMPTS ("2060 2049 ..."): one reference load for several prompts, each from a fresh slot into $REF/p<N>; the
    # M2b gate runs on the first's state, the prefill gate (M2B_PREFILL_GATE=1) on every one's
    PROMPTS=${M2B_PROMPTS:-}
    refargs=(--prompt "${M2B_PROMPT:-1536}")
    [[ -n "$PROMPTS" ]] && refargs=(--prompts "$(echo $PROMPTS | tr ' ' ',')")
    # M2B_AOT_PROMPTS (1,9,45,...): more prompt lengths for their Triton variants (every prefill path; M4's assets);
    # M2B_PROFILE_ROWS ("1 4 8"): the Python side of the decode gap (dsv41_m2b_ref.py --profile) on the R1 reference
    extra=(); [[ -n "${M2B_AOT_PROMPTS:-}" ]] && extra=(--aot-prompts "$M2B_AOT_PROMPTS")
    # M2B_SAMP=1 (gate): the reference also writes the sampled replies (--sampled: $REF/samp/trace.jsonl, its
    # prompt lengths in the AOT set); tools/zig/dsv41_samp/samp_step.sh runs on it after M4
    [[ "${M2B_SAMP:-0}" == 1 ]] && extra+=(--sampled)
    # M2B_GRAMMAR=1 (gate): the reference also writes the grammar-constrained replies (--grammar, xgrammar
    # 0.2.8 on the prod twin, installed --no-deps when missing); grammar_gate runs on them in the workstream loop
    if [[ "${M2B_GRAMMAR:-0}" == 1 ]]; then
        PYTHONPATH=/dsv41-tf/src python -c "import xgrammar" 2>/dev/null \
            || step grammar-xgrammar 3 python -m pip install --no-deps "xgrammar==0.2.8" || true
        extra+=(--grammar)
    fi
    # M2B_ENGRAM=1 (gate, R3): the references read Engram's rows from packed shards they write first (sparse,
    # records under the prompts' rows), the Zig ranks from the same files (TF_DSV41_ENGRAM_DIR): the real-row path
    # under every gate. R0's and R1's references each get their own directory.
    eng0=(); eng1=()
    if [[ "${M2B_ENGRAM:-0}" == 1 ]]; then
        eng0=(--engram-dir "$LOCAL/engram-r0" --engram-make); eng1=(--engram-dir "$LOCAL/engram-r1" --engram-make)
        export TF_DSV41_ENGRAM_DIR=$LOCAL/engram-r0
    fi
    refs=("$REF")
    if step m2b-ref "${M2B_REF_MIN:-18}" python -u -B "$Z/tools/zig/dsv41_m2b_ref.py" --pack "$LP" --out "$REF" \
            --layers "0-$hi" --steps "${M2B_STEPS:-64}" "${refargs[@]}" "${extra[@]}" "${eng0[@]}" \
       && step engram-host 3 python -B "$Z/tools/zig/dsv41_engram_host.py" --py-src /dsv41-tf/src \
            --config "$LP/config.json" --tokenizer "$LP/tokenizer.json" --out "$REF/engram-host.bin"; then
        if [[ -n "$PROMPTS" ]]; then
            refs=(); for p in $PROMPTS; do cp "$REF/engram-host.bin" "$REF/p$p/"; refs+=("$REF/p$p"); done
        fi
        R0=${refs[0]}
    ws_gates() {
        # the gap-closing workstreams (each tools/zig/dsv41_<name>/job.sh with a <name>_gate, same env as slots_gate):
        # M2B_LONGPF (long prompts past 4,096 keys), M2B_SPEC (SPEC_DRAFT early pass), M2B_GRAMMAR (structured output),
        # M2B_VISION (image input), M2B_KNOBS (prod engine knobs on the Zig side), M2B_SESS4 (sessions with several slots),
        # M2B_GM2PF (x3gm v2 on Zig's prefill: a v2 capture checked + replayed, v2 == v1, the prefill gate off / v2 timed)
        for name in ${*:-longpf spec grammar vision knobs sess4 gm2pf pieces}; do
            knob=M2B_${name^^}
            [[ "${!knob:-0}" == 1 ]] || continue
            if [[ -f "$Z/tools/zig/dsv41_$name/job.sh" ]]; then
                (source "$Z/tools/zig/dsv41_$name/job.sh"; "${name}_gate") || rc=1
                grep -h "PASS $name\|FAIL $name" "$RES"/$name*.log "$RES"/$name-*/*.log 2>/dev/null | tee -a "$RES/steps.txt"
            else echo "$name skipped: no tools/zig/dsv41_$name/job.sh" | tee -a "$RES/steps.txt"; rc=1; fi
        done
    }
        # DSpark's Triton variants (the reference never drafts): capture P's AOT set merged into the reference's
        if [[ -d "${CAP:-}/aot" ]]; then for d in "${refs[@]}"; do python -B "$Z/tools/zig/dsv41_aot_merge.py" "$d/aot" "$CAP/aot" | tee -a "$RES/steps.txt"; done; fi
        mb=$(left); (( mb > 16 )) && mb=16
        step m2b "$mb" bash "$Z/tools/zig/dsv41_m1/m2b_ranks.sh" "$Z/bin/tf-dsv41-m1" "$LP" "$R0" "$RES/m2b" \
            || rc=1
        # M2B_GATES_FIRST=1: the workstream gates (never run on a GPU) before R1 / M4 / sampling / slots (gated before)
        [[ "${M2B_GATES_FIRST:-0}" == 1 ]] && ws_gates longpf spec grammar vision knobs sess4
        # M2B_PREFILL_GATE=1: Zig's own prefill from an empty slot (M2B_PREFILL=1): every state role == the
        # reference's dump and its first token, then the decode gate (M2B_PROMPT 2600: a 2,048-row chunk + a ragged one)
        if [[ "${M2B_PREFILL_GATE:-0}" == 1 ]]; then
            i=0
            for d in "${refs[@]}"; do
                name=m2b-prefill; [[ -n "$PROMPTS" ]] && name=m2b-prefill-$(basename "$d")
                mb=$(left); (( mb > 10 )) && mb=10
                (( mb >= 3 )) || { echo "$name skipped: $mb min left" | tee -a "$RES/steps.txt"; rc=1; continue; }
                (export M2B_PREFILL=1 M2B_PORT=$(( 29717 + i ))
                 step "$name" "$mb" bash "$Z/tools/zig/dsv41_m1/m2b_ranks.sh" "$Z/bin/tf-dsv41-m1" "$LP" "$d" "$RES/$name") \
                    || rc=1
                i=$(( i + 1 ))
            done
        fi
        # M2B_GRAPHS=1: the same gate again with graphed decode windows (TF_DSV41_GRAPHS=1, R4): == the reference
        if [[ "${M2B_GRAPHS:-0}" == 1 ]]; then
            mb=$(left); (( mb > 8 )) && mb=8
            (export TF_DSV41_GRAPHS=1 M2B_PORT=29719
             step m2b-graphs "$mb" bash "$Z/tools/zig/dsv41_m1/m2b_ranks.sh" "$Z/bin/tf-dsv41-m1" "$LP" "$R0" "$RES/m2b-graphs") \
                || rc=1
        fi
        # M2B_LANES ("1 2"): M3's gates, the same reference through lanes on rank 0 (1: drafts off; 2: oracle drafts;
        # 3: DSpark's GPU pass; 4: draft trees with the tree oracle; 3t: mode 3 with DSpark's trees, TF_DSV41_TREE=3)
        for k in $LANES_MODES; do
            mb=$(left); (( mb > 8 )) && mb=8
            (( mb >= 3 )) || { echo "m3-$k skipped: $mb min left" | tee -a "$RES/steps.txt"; rc=1; continue; }
            m=${k%t}; tree=0; [[ "$k" == *t ]] && tree=3
            (export M2B_LANES=$m M2B_PORT=$(( 29720 + m + (tree > 0 ? 10 : 0) )) TF_DSV41_TREE=$tree
             step "m3-$k" "$mb" bash "$Z/tools/zig/dsv41_m1/m2b_ranks.sh" "$Z/bin/tf-dsv41-m1" "$LP" "$R0" "$RES/m3-$k") \
                || rc=1
        done
        # M2B_R1=1: the M2b gate on R1 (the reference with r1c-knobs.env on the R1 twin; ours follows its env)
        if [[ "${M2B_R1:-0}" == 1 ]]; then
            REF1=$LOCAL/m2b-ref-r1
            rm -rf "$REF1"
            mb=$(left); (( mb > 8 )) && mb=8
            prof=(); [[ -n "${M2B_PROFILE_ROWS:-}" ]] && prof=(--profile "$M2B_PROFILE_ROWS")
            if (( mb >= 5 )) && (set -a; source "$Z/tools/zig/dsv41_m1/r1c-knobs.env"; set +a
                export PYTHONPATH=/opt/r1c/src TORCH_EXTENSIONS_DIR=/opt/r1c/ext
                step m2b-ref-r1 "$mb" python -u -B "$Z/tools/zig/dsv41_m2b_ref.py" --pack "$LP" --out "$REF1" \
                    --layers "0-$hi" --steps "${M2B_STEPS:-64}" --port 29631 "${extra[@]}" "${prof[@]}" "${eng1[@]}"); then
                cp "$REF/engram-host.bin" "$REF1/"
                cp "$REF1"/prof-rank*.json "$RES/" 2>/dev/null
                # DSpark's variants under R1 (M2B_CAPTURE_R1=P) for M4 on these assets
                [[ -d "$LOCAL/capture-r1/aot" ]] && python -B "$Z/tools/zig/dsv41_aot_merge.py" "$REF1/aot" "$LOCAL/capture-r1/aot" | tee -a "$RES/steps.txt"
                mb=$(left); (( mb > 6 )) && mb=6
                # the decode gap (M2B_PROF=1): this gate's eager windows profiled, then M2B_PROF_ROWS windows timed
                (export M2B_PORT=29741 M2B_PROF=1 M2B_PROF_ROWS="${M2B_ZIG_PROF_ROWS:-}"
                 (( ${#eng1[@]} )) && export TF_DSV41_ENGRAM_DIR=$LOCAL/engram-r1
                 step m2b-r1 "$mb" bash "$Z/tools/zig/dsv41_m1/m2b_ranks.sh" "$Z/bin/tf-dsv41-m1" "$LP" "$REF1" "$RES/m2b-r1") \
                    || rc=1
                if [[ "${M2B_R1_GRAPHS:-0}" == 1 ]]; then
                    mb=$(left); (( mb > 6 )) && mb=6
                    (export TF_DSV41_GRAPHS=1 M2B_PROF=0 M2B_PROF_ROWS="${M2B_ZIG_PROF_ROWS:-}" M2B_PORT=29751
                     (( ${#eng1[@]} )) && export TF_DSV41_ENGRAM_DIR=$LOCAL/engram-r1
                     step m2b-r1-graphs "$mb" bash "$Z/tools/zig/dsv41_m1/m2b_ranks.sh" "$Z/bin/tf-dsv41-m1" "$LP" "$REF1" "$RES/m2b-r1-graphs") \
                        || rc=1
                fi
                # M2B_R1_DIAG=1 (pod 29: rank 1's illegal address in the profiled 4-row R1 windows): eager, no event
                # profile, every call synced (TF_DSV41_SYNC_EACH: the faulting call named), timed rows 1 / 2 / 4
                if [[ "${M2B_R1_DIAG:-0}" == 1 ]]; then
                    mb=$(left); (( mb > 7 )) && mb=7
                    (export M2B_PROF=0 TF_DSV41_SYNC_EACH=1 M2B_PROF_ROWS="${M2B_R1_DIAG_ROWS:-1 2 4}" M2B_PORT=29761
                     (( ${#eng1[@]} )) && export TF_DSV41_ENGRAM_DIR=$LOCAL/engram-r1
                     step m2b-r1-diag "$mb" bash "$Z/tools/zig/dsv41_m1/m2b_ranks.sh" "$Z/bin/tf-dsv41-m1" "$LP" "$REF1" "$RES/m2b-r1-diag") \
                        || rc=1
                    grep -h "sync-each\|rows windows\|-row windows\|^PASS\|^FAIL" "$RES"/m2b-r1-diag/rank*.log 2>/dev/null | tee -a "$RES/steps.txt"
                fi
                mkdir -p "$RES/m2b-ref-r1"
                for f in ref.json rank0.json rank1.json; do [[ -f "$REF1/$f" ]] && cp "$REF1/$f" "$RES/m2b-ref-r1/"; done
            else echo "m2b-r1 skipped or its reference failed" | tee -a "$RES/steps.txt"; rc=1; fi
        fi
    else rc=1; fi
    # M2B_M4=1: M4 on the pod (the backbone prefix, R1 with the R1 reference's assets when it ran): the server on rank 0
    # (drafts, graphs, Zig prefill) answers prompts.txt over HTTP, then the CLI replays its trace (HTTP == CLI with the
    # drafts' counts), then again with drafts off (drafted == serial); each line reports its first differing token
    if [[ "${M2B_M4:-0}" == 1 ]]; then
        # the R0 reference's assets (its AOT set holds set P's DSpark variants: M2B_CAPTURE=P); M2B_M4_R1=1: the R1
        # reference's (then M2B_CAPTURE_R1=P for DSpark's R1 variants)
        A=${R0:-$REF}; eng=$LOCAL/engram-r0
        if [[ "${M2B_M4_R1:-0}" == 1 && -d "${REF1:-}/aot" ]]; then A=$REF1; eng=$LOCAL/engram-r1; fi
        m4env=(TF_TP_WORLD=2 TF_DSV41_ASSETS="$A" TF_DSV41_LAYERS="0-$hi" TF_DSV41_GRAPHS=1 TF_DSV41_OWN_PREFILL=1
               TF_DSV41_FAILFAST=0 TF_DSV41_EXPERT_TOPP=${M2B_TOPP:-0.85} TF_DSV41_PHASES=1 TF_DSV41_PHASES_OUT=$RES/m4/phases.json)
        [[ "$A" == "${REF1:-}" ]] && m4env+=(TF_DSV41_R1=1)
        [[ "${M2B_ENGRAM:-0}" == 1 ]] && m4env+=(TF_DSV41_ENGRAM_DIR=$eng)
        # M2B_M4_ENV ("K=V ..."): more knobs on every M4 process, e.g. TF_DSV41_CALIB_FILE=<prod's calib-*.json> (the
        # served drafts priced by prod's table) or TF_DSV41_TREE=3 TF_DSV41_POOL_TOKENS=... TF_DSV41_KV_SPLIT=1
        [[ -n "${M2B_M4_ENV:-}" ]] && read -ra extra_m4 <<< "$M2B_M4_ENV" && m4env+=("${extra_m4[@]}")
        m4dir=$RES/m4; mkdir -p "$m4dir"
        follow() { env "${m4env[@]}" TF_TP_RANK=1 TF_TP_DEVICE=1 TF_TP_PORT=$1 "$Z/bin/tf-dsv41-m1" follow "$LP" "$A" > "$m4dir/follow-$2.log" 2>&1; }
        mb=$(left)
        if (( mb >= 8 )); then
            follow 29861 server & fpid=$!
            env "${m4env[@]}" TF_TP_RANK=0 TF_TP_DEVICE=0 TF_TP_PORT=29861 TF_DSV41_TRACE_TOKENS="$m4dir/trace.jsonl" \
                "$Z/bin/tensorfold-dsv41" serve "$LP" --port 8091 --parallel 1 > "$m4dir/server.log" 2>&1 & spid=$!
            t=0; until grep -q "serving model\|serving pack at" "$m4dir/server.log" 2>/dev/null || (( t > 600 )) || ! kill -0 $spid 2>/dev/null; do sleep 5; t=$((t + 5)); done
            step m4-requests 6 bash "$Z/tools/zig/dsv41_m4/requests.sh" 8091 "${M2B_M4_TOKENS:-128}" || rc=1
            kill $spid 2>/dev/null; wait $spid 2>/dev/null; wait $fpid 2>/dev/null
            for mode in drafts serial; do
                mb=$(left); (( mb >= 3 )) || { echo "m4-$mode skipped: $mb min left" | tee -a "$RES/steps.txt"; rc=1; continue; }
                d=1; [[ "$mode" == serial ]] && d=0
                port=$(( 29871 + d ))
                (m4env+=(TF_DSV41_DRAFTS=$d); follow $port "$mode" & fp=$!
                 step "m4-$mode" "$mb" env "${m4env[@]}" TF_DSV41_DRAFTS=$d TF_DSV41_ENGRAM_PREFETCH=0 TF_TP_RANK=0 TF_TP_DEVICE=0 TF_TP_PORT=$port \
                     "$Z/bin/tf-dsv41-m1" generate "$LP" "$A" "$m4dir/trace.jsonl"; r=$?; wait $fp; exit $r) || rc=1
                grep -h "^PASS\|^FAIL\|DSpark" "$RES/m4-$mode.log" | tee -a "$RES/steps.txt"
            done
        else echo "m4 skipped: $mb min left" | tee -a "$RES/steps.txt"; rc=1; fi
    fi
    # M2B_SAMP=1: the keyed sampling gate (2 GPUs): tf-dsv41-samp, then tf-dsv41-m1 generate on the reference's sampled
    # replies with drafts off and on (Zig == Python token for token: "PASS M4 CLI == HTTP: 6/6 replies equal" each)
    if [[ "${M2B_SAMP:-0}" == 1 ]]; then
        mb=$(left); (( mb > 12 )) && mb=12
        if (( mb >= 4 )) && [[ -s "$REF/samp/trace.jsonl" ]]; then
            SAMP_TRACE=$REF/samp/trace.jsonl SAMP_MIN=$mb step samp $(( 3 * mb )) bash "$Z/tools/zig/dsv41_samp/samp_step.sh" \
                "$Z" "$LP" "${R0:-$REF}" "$hi" "$RES/samp" "$LOCAL/samp" || rc=1
            grep -h "^samp\|^PASS\|^FAIL" "$RES/samp/steps.txt" "$RES"/samp/samp-*.log 2>/dev/null | tee -a "$RES/steps.txt"
        else echo "samp skipped: $mb min left or no sampled reference" | tee -a "$RES/steps.txt"; rc=1; fi
    fi
    # gate's tables (speed-gap.md: Zig eager / graphed vs Python R1 a window, GPU ms by family, M4's phases)
    [[ -f "$Z/tools/zig/dsv41_perf/speed_step.sh" ]] && bash "$Z/tools/zig/dsv41_perf/speed_step.sh" "$Z" "$RES" 2>&1 | tail -20 | tee -a "$RES/steps.txt"
    # M2B_SLOTS=1 (gate): several live slots against Python's batched row mode (tools/zig/dsv41_slots/job.sh;
    # SLOTS_PYSRC: a twin with rowgraphs / slots AND the split pool (ref.py passes split=): the midprofile twin, unpacked
    # here when M1_JOB=m5 did not run first; pod 29's /opt/r1c twin predates split), SLOTS_RUNS of its runs
    if [[ "${M2B_SLOTS:-0}" == 1 || "${M2B_LONGPF:-0}" == 1 ]] && [[ ! -d /opt/midprofile/src ]]; then
        for c in "${MIDPROFILE_TAR:-}" "$Z/midprofile.tar.gz" "${PACK%%/out/*}/m5/midprofile.tar.gz"; do
            [[ -f "$c" ]] && { tar -xzf "$c" -C / && echo "midprofile twin from $c" | tee -a "$RES/steps.txt"; break; }
        done
    fi
    if [[ "${M2B_SLOTS:-0}" == 1 ]]; then
        (unset TF_DSV41_ENGRAM_DIR
         export SLOTS_PYSRC=${SLOTS_PYSRC:-/opt/midprofile/src} TORCH_EXTENSIONS_DIR=${SLOTS_EXT:-/opt/midprofile/ext}
         source "$Z/tools/zig/dsv41_slots/job.sh"; slots_gate) || rc=1
        grep -h "^slots\|PASS slots\|FAIL slots" "$RES"/slots*.log "$RES"/slots-*/rank0.log 2>/dev/null | tee -a "$RES/steps.txt"
        # with M2B_SAMP: the sampled trace again at TF_DSV41_SLOTS=4, 4 replies in flight (samp_step.sh's slots4: a
        # sampler a slot's rows == Python's serial replies), on the sampled reference's assets plus the slots
        # reference's row-mode AOT variants
        if [[ "${M2B_SAMP:-0}" == 1 && -s "$REF/samp/trace.jsonl" && -d "$LOCAL/slots-ref/aot" ]] && (( $(left) >= 5 )); then
            sa=$LOCAL/samp-slots-assets
            rm -rf "$sa"; cp -r "${R0:-$REF}" "$sa"
            python -B "$Z/tools/zig/dsv41_aot_merge.py" "$sa/aot" "$LOCAL/slots-ref/aot" | tee -a "$RES/steps.txt"
            mb=$(left); (( mb > 10 )) && mb=10
            SAMP_KERNELS=0 SAMP_MODES=slots4 SAMP_TRACE=$REF/samp/trace.jsonl SAMP_MIN=$mb step samp-slots $(( mb + 1 )) \
                bash "$Z/tools/zig/dsv41_samp/samp_step.sh" "$Z" "$LP" "$sa" "$hi" "$RES/samp" "$LOCAL/samp" || rc=1
            grep -h "^PASS\|^FAIL" "$RES/samp/samp-slots4.log" 2>/dev/null | sed 's/^/samp-slots4: /' | tee -a "$RES/steps.txt"
        fi
    fi
    # M2B_MDRAFT=1 (gate): DSpark drafting with several live slots (TF_DSV41_SLOT_DRAFTS) against Python's
    # batcher, after the slots gate (its $LOCAL/slots-ref assets): tools/zig/dsv41_mdraft/job.sh's mdraft_gate
    if [[ "${M2B_MDRAFT:-0}" == 1 ]]; then
        if [[ -f "$Z/tools/zig/dsv41_mdraft/job.sh" ]]; then
            (unset TF_DSV41_ENGRAM_DIR
             export SLOTS_PYSRC=${SLOTS_PYSRC:-/opt/midprofile/src} TORCH_EXTENSIONS_DIR=${SLOTS_EXT:-/opt/midprofile/ext}
             source "$Z/tools/zig/dsv41_mdraft/job.sh"; mdraft_gate) || rc=1
            grep -h "^mdraft\|PASS mdraft\|FAIL mdraft" "$RES"/mdraft*.log "$RES"/mdraft-*/rank0.log 2>/dev/null | tee -a "$RES/steps.txt"
        else echo "mdraft skipped: no tools/zig/dsv41_mdraft/job.sh" | tee -a "$RES/steps.txt"; rc=1; fi
    fi
    # the gap-closing workstreams (ws_gates, defined after the M2b reference): here, unless M2B_GATES_FIRST=1 ran them
    # right after the M2b decode gate
    # (with M2B_GATES_FIRST=1, gm2pf still here: its captures and timing come after the correctness gates)
    if declare -F ws_gates > /dev/null; then
        if [[ "${M2B_GATES_FIRST:-0}" == 1 ]]; then ws_gates gm2pf; else ws_gates; fi
    fi
    for d in "${refs[@]}"; do
        o=$RES/m2b-ref; [[ -n "$PROMPTS" ]] && o=$RES/m2b-ref/$(basename "$d")
        mkdir -p "$o"
        for f in ref.json rank0.json rank1.json manifest.json; do [[ -f "$d/$f" ]] && cp "$d/$f" "$o/"; done
    done
    [[ -d "$REF/aot" ]] && tar -czf "$RES/m2b-ref/aot.tar.gz" -C "$REF" aot
    km=$(left); (( km > 15 )) && km=15
    if (( km >= 6 )); then
        (for v in $(compgen -e | grep '^TF_DSV41_'); do unset "$v"; done
         export TF_DSV41_PY_SRC=$PYTWIN TF_DSV41_TEST=$Z/bin/tf-dsv41-test TF_DSV41_FATBINS=$Z/fatbin \
                TORCH_EXTENSIONS_DIR=$PYEXT TF_DSV41_CAPTURE_MIN=$(( km - 3 ))
         step kernels "$km" bash "$Z/zig/tests/cuda/deepseek_v41/job.sh" "$RES/kernels") || rc=1
    else
        echo "kernels skipped: $km min left" | tee -a "$RES/steps.txt"; rc=1
    fi
    echo "=== rc=$rc"
    exit $rc
fi

if [[ "${M1_JOB:-m1}" == m2a ]]; then
    REF=$LOCAL/m2a-ref
    rm -rf "$REF"
    # pod 10: layers 0-24 did not finish loading + prefill in 22 min (Python loads ~40 s a block from the volume);
    # 0-8 holds SWA + Engram (1), a ratio-2 KV + index source with its carry (2, 8) and the reuse layers between
    if step m2a-ref "${M2A_REF_MIN:-20}" python -u -B "$Z/tools/zig/dsv41_m2a_digests.py" --pack "$PACK" --out "$REF" \
            --layers "${M2A_LAYERS:-0-8}" \
       && step engram-host 3 python -B "$Z/tools/zig/dsv41_engram_host.py" --py-src /dsv41-tf/src \
            --config "$PACK/config.json" --tokenizer "$PACK/tokenizer.json" --out "$REF/engram-host.bin"; then
        step m2a 14 "$Z/bin/tf-dsv41-m1" m2a "$PACK" "$REF" "$RES/m2a"
    else rc=1; fi
    mkdir -p "$RES/m2a-ref"
    for f in digests.json manifest.json; do [[ -f "$REF/$f" ]] && cp "$REF/$f" "$RES/m2a-ref/"; done
    [[ -f "$REF/state/state.json" ]] && cp "$REF/state/state.json" "$RES/m2a-ref/"
    [[ -d "$REF/aot" ]] && tar -czf "$RES/m2a-ref/aot.tar.gz" -C "$REF" aot
    # the kernel job last, in what is left (up to 15 min): pod 10 ran it first on a cold Triton cache and its twin
    # suites (Triton JIT) did not fit 6 minutes; after the reference, the shared kernels are compiled
    km=$(left); (( km > 15 )) && km=15
    if (( km >= 6 )); then
        (for v in $(compgen -e | grep '^TF_DSV41_'); do unset "$v"; done
         export TF_DSV41_PY_SRC=$PYTWIN TF_DSV41_TEST=$Z/bin/tf-dsv41-test TF_DSV41_FATBINS=$Z/fatbin \
                TORCH_EXTENSIONS_DIR=$PYEXT TF_DSV41_CAPTURE_MIN=$(( km - 3 ))
         step kernels "$km" bash "$Z/zig/tests/cuda/deepseek_v41/job.sh" "$RES/kernels") || rc=1
    else
        echo "kernels skipped: $km min left" | tee -a "$RES/steps.txt"; rc=1
    fi
    echo "=== rc=$rc"
    exit $rc
fi

# 1-2. the prod capture and our gates first (the replay + block.zig's forward on one load)
CAP=$LOCAL/capture
rm -rf "$CAP"
if step capture "${M1_CAPTURE_MIN:-18}" python -B "$Z/tools/zig/dsv41_m1_capture.py" --pack "$PACK" --out "$CAP" \
        --sets "${M1_SETS:-A,B,C,D,P}"; then
    du -sh "$CAP" "$CAP/blobs" | tee "$RES/capture-size.txt"
    step m1 "${M1_REPLAY_MIN:-14}" "$Z/bin/tf-dsv41-m1" "${M1_MODE:-both}" "$PACK" "$CAP" "$RES/m1"
fi

# 3. the kernel exactness job in what is left (its pytest capture gets 4 minutes less), every TF_DSV41_ knob
# unset: the twin's suites pick their own parametrizations
kmin=$(( $(left) - ${M1_R1C_MIN:-0} ))
if (( kmin >= 6 )); then
    (for v in $(compgen -e | grep '^TF_DSV41_'); do unset "$v"; done
     export TF_DSV41_PY_SRC=$PYTWIN TF_DSV41_TEST=$Z/bin/tf-dsv41-test TF_DSV41_FATBINS=$Z/fatbin \
            TORCH_EXTENSIONS_DIR=$PYEXT TF_DSV41_CAPTURE_MIN=$(( kmin - 4 ))
     step kernels "$kmin" bash "$Z/zig/tests/cuda/deepseek_v41/job.sh" "$RES/kernels") || rc=1
else
    echo "kernels skipped: $(left) min of the budget left" | tee -a "$RES/steps.txt"; rc=1
fi

# 4. optional (M1_R1C_MIN > 0): a capture with prod + R1c knobs on the decode1 twin, replayed bit for bit
if (( ${M1_R1C_MIN:-0} > 0 )) && [[ "$PYTWIN" == /opt/r1c ]]; then
    l=$(left)
    if (( l >= M1_R1C_MIN )); then
        CAP2=$LOCAL/capture-r1c
        rm -rf "$CAP2"
        half=$(( (l - 1) / 2 ))
        if (set -a; source "$Z/tools/zig/dsv41_m1/prod-knobs.env"; source "$Z/tools/zig/dsv41_m1/r1c-knobs.env"; set +a
            export PYTHONPATH=/opt/r1c/src TORCH_EXTENSIONS_DIR=/opt/r1c/ext
            step r1c-capture "$half" python -B "$Z/tools/zig/dsv41_m1_capture.py" --pack "$PACK" --out "$CAP2" \
                --sets "${M1_R1C_SETS:-B}"); then
            step r1c-replay "$half" "$Z/bin/tf-dsv41-m1" replay "$PACK" "$CAP2" "$RES/m1-r1c"
        else rc=1; fi
        mkdir -p "$RES/capture-r1c"
        for f in meta.json manifest.json; do [[ -f "$CAP2/$f" ]] && cp "$CAP2/$f" "$RES/capture-r1c/"; done
        [[ -f "$CAP2/ops.jsonl" ]] && gzip -c "$CAP2/ops.jsonl" > "$RES/capture-r1c/ops.jsonl.gz"
    else
        echo "r1c skipped: $l min of the budget left" | tee -a "$RES/steps.txt"
    fi
fi

# the capture's small files come back (the blobs stay on the pod: GBs); ops.jsonl compressed
mkdir -p "$RES/capture"
for f in meta.json weights.json manifest.json jit.json; do [[ -f "$CAP/$f" ]] && cp "$CAP/$f" "$RES/capture/"; done
# the AOT set whole (aot.json + cubins, a few MB): M2 launches Triton through it (zig/src/cuda/aot.zig)
[[ -d "$CAP/aot" ]] && tar -czf "$RES/capture/aot.tar.gz" -C "$CAP" aot
[[ -f "$CAP/ops.jsonl" ]] && gzip -c "$CAP/ops.jsonl" > "$RES/capture/ops.jsonl.gz"
[[ -f "$CAP/launches.json" ]] && gzip -c "$CAP/launches.json" > "$RES/capture/launches.json.gz"
echo "=== rc=$rc"
exit $rc
