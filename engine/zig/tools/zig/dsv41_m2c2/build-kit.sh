#!/usr/bin/env bash
# Builds the Spark kit (build host, no Spark): tf-dsv41-m1 and tensorfold-dsv41 for aarch64 with the sm_121 fatbins
# embedded, the TP mailbox / RoCE fatbin, the Python tools, run_host.sh (M2c-2), run_m4.sh + requests.sh + prompts.txt
# (M4), bench_http.py, the slots reference, diverge.py (Zig vs Python replies), gap.py and SHA256SUMS. Build
# from a clean archive of the tree (KIT_COMMIT=<hash>): the shared worktree may hold uncommitted files.
#   FATBINS=<dir of sm_121 dsv41_*.fatbin> [TP_FATBIN=<prebuilt tp_mailbox.fatbin>] bash build-kit.sh KITDIR
# kit-fatbins.sh first makes KITDIR/fatbins: FATBINS' files + tp_mailbox.fatbin built from this tree's mailbox.cu (nvcc
# 13.3, tp.zig's flags; TP_FATBIN only when it passes the same checks), the glue fatbin checked against glue.cu. The
# binaries embed that dir, and it is the prod image's: ZIG_FATBINS=KITDIR/fatbins scripts/zig-image.sh stage (zig-serve).
# TRITON_FILL_FROM=<a kept fill dir (its aot/, compile.json, needs.json, the env)>: copied to KITDIR/triton-fill instead of
# compiling one (TRITON_FILL=1): the fill of the same served env (tools/zig/dsv41_m4/fill-prod.env) and emitter.
# The single-pass window's scripts (tools/zig/dsv41_window: zig-gate.sh, zig-bench.sh) go to the kit's root.
set -euo pipefail
# the build host's shared build lock (the HEAVY wrapper: one heavy job at a time, 8 GiB, -j4), when it exists
H=${HEAVY:-}
[[ -n "${HEAVY_HELD:-}" || ! -x "$H" ]] || exec env HEAVY_HELD=1 "$H" bash "$0" "$@"
KIT=${1:?kit dir}; : "${FATBINS:?sm_121 fatbin dir}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); TREE=$(cd "$HERE/../../.." && pwd)
export PATH=$HOME/.local/opt/zig:$PATH
rm -rf "$KIT" && mkdir -p "$KIT/tools/zig/dsv41_m1" && KIT=$(cd "$KIT" && pwd)
cd "$TREE"
bash "$HERE/kit-fatbins.sh" "$KIT/fatbins"
zig build -Dtarget=aarch64-linux-gnu -Dfatbins="$KIT/fatbins" -Dsm=121 -Doptimize=ReleaseSafe -j4 --prefix "$KIT/.build" install
cp "$KIT/.build/bin/tf-dsv41-m1" "$KIT/.build/bin/tensorfold-dsv41" "$KIT/"
# the kernel exactness runner (zig/tests/cuda/deepseek_v41/job.sh: TF_DSV41_TEST) for bit gates on the Sparks
if [[ -x "$KIT/.build/bin/tf-dsv41-test" ]]; then cp "$KIT/.build/bin/tf-dsv41-test" "$KIT/"; fi
rm -rf "$KIT/.build"
cp "$KIT/fatbins/tp_mailbox.fatbin" "$KIT/tp_mailbox.fatbin"      # TF_TP_KERNEL_IMAGE=/kit/tp_mailbox.fatbin
cp tools/zig/dsv41_m2b_ref.py tools/zig/dsv41_m1_capture.py tools/zig/triton_aot_manifest.py tools/zig/dsv41_engram_host.py \
    tools/zig/dsv41_aot_merge.py tools/zig/dsv41_m5_kv.py "$KIT/tools/zig/"
cp tools/zig/dsv41_m1/prod-knobs.env tools/zig/dsv41_m1/r1c-knobs.env "$KIT/tools/zig/dsv41_m1/"
cp "$HERE/run_host.sh" "$KIT/"
mkdir -p "$KIT/tools/zig/dsv41_m2c2" && cp "$HERE/peer.py" "$HERE/pieces_check.py" "$KIT/tools/zig/dsv41_m2c2/"
# the engine's knob names (pieces_check.py: a run's every TF_ / GLM53_ / SESS4_ key is one it reads)
grep -rhoE '"(TF|GLM53|SESS4)_[A-Z0-9_]+"' --include=*.zig zig/src | tr -d '"' | sort -u > "$KIT/tools/zig/dsv41_m2c2/knob-names.txt"
# M4: the Zig server on rank 0, the CLI replay, DSpark's acceptance (tools/zig/dsv41_m4)
cp tools/zig/dsv41_m4/run_host.sh "$KIT/run_m4.sh"
cp tools/zig/dsv41_m4/requests.sh tools/zig/dsv41_m4/prompts.txt "$KIT/"
# The speed client and slots reference.
cp "$HERE/bench_http.py" "$KIT/"
mkdir -p "$KIT/tools/zig/dsv41_slots" && cp tools/zig/dsv41_slots/ref.py "$KIT/tools/zig/dsv41_slots/"
# Zig vs Python replies (prompt tail, CED: `diverge.py ask / compare / fromtrace / trace`) and the phases / prefill gaps
mkdir -p "$KIT/tools/zig/dsv41_diverge" "$KIT/tools/zig/dsv41_perf"
cp tools/zig/dsv41_diverge/diverge.py "$KIT/tools/zig/dsv41_diverge/"
cp tools/zig/dsv41_perf/gap.py "$KIT/tools/zig/dsv41_perf/"
# The two-host vision gate and its Python reference.
mkdir -p "$KIT/tools/zig/dsv41_vision" && cp tools/zig/dsv41_vision/vision_ref.py tools/zig/dsv41_vision/spark.sh "$KIT/tools/zig/dsv41_vision/"
cp tools/zig/dsv41_rope_tables.py "$KIT/tools/zig/"
mkdir -p "$KIT/tools/zig/dsv41_ops" && cp tools/zig/dsv41_ops/wire_probe.py "$KIT/tools/zig/dsv41_ops/"
# the AOT fill (port doc "AOT fill"): every Triton variant the served config can launch (tf-dsv41-m1 aot-needs, a host
# build), compiled from the served Python tree's Triton source for sm_121 (tools/zig/triton_fill.py) into
# $KIT/triton-fill/aot, minus the variants the sets in TRITON_FILL_HAVE already hold. TRITON_FILL=1 with:
#   TRITON_FILL_PY      the served tree's src/ (prod's TF_COMMIT: git archive 8474f31 of tensorfold-decode1)
#   TRITON_FILL_PYTHON  a python with triton==3.7.1 and torch (CPU) - the image's Triton
#   TRITON_FILL_PTXAS   ptxas V13.3.73 (the image's /usr/local/cuda/bin/ptxas; PyPI nvidia-cuda-nvcc==13.3.73)
#   TRITON_FILL_ENV     the served engine's env file (TF_DSV41_CONTEXT, _SLOTS, _ROWS_CAP, R1's knobs, ...)
#   TRITON_FILL_HAVE    aot dirs already served (space separated; optional)
#   TRITON_FILL_STAMPS  stamps.json of the image's files (optional: byte-identical cubins; without, the same SASS)
if [[ "${TRITON_FILL:-0}" == 1 ]]; then
    PYF=${TRITON_FILL_PYTHON:?python with triton 3.7.1}; FILL=$KIT/triton-fill; mkdir -p "$FILL"
    export TRITON_PTXAS_PATH=${TRITON_FILL_PTXAS:?ptxas 13.3.73} TRITON_PTXAS_BLACKWELL_PATH=$TRITON_FILL_PTXAS
    zig build -Doptimize=ReleaseFast -j4 --prefix "$KIT/.host" install
    "$PYF" tools/zig/triton_fill.py sigs --py "${TRITON_FILL_PY:?served src}" --out "$FILL/sigs.json"
    (set -a; source "${TRITON_FILL_ENV:?served env}"; set +a
     "$KIT/.host/bin/tf-dsv41-m1" aot-needs "$FILL/sigs.json" "$FILL/needs.json") 2>&1 | tee "$FILL/needs.log"
    rm -rf "$KIT/.host"
    # shellcheck disable=SC2086
    "$PYF" tools/zig/triton_fill.py compile --py "$TRITON_FILL_PY" --needs "$FILL/needs.json" \
        --options tools/zig/dsv41_triton_options.json --arch 121 --out "$FILL/aot" \
        ${TRITON_FILL_HAVE:+--have $TRITON_FILL_HAVE} ${TRITON_FILL_STAMPS:+--stamps "$TRITON_FILL_STAMPS"} | tee "$FILL/compile.json"
    cp tools/zig/triton_fill.py tools/zig/dsv41_triton_options.json "$KIT/tools/zig/"
fi
if [[ -n "${TRITON_FILL_FROM:-}" ]]; then
    [[ "${TRITON_FILL:-0}" != 1 ]] || { echo "TRITON_FILL=1 and TRITON_FILL_FROM: one of them" >&2; exit 2; }
    grep -q '"failed": \[\]' "$TRITON_FILL_FROM/compile.json" || { echo "$TRITON_FILL_FROM/compile.json: not failed: []" >&2; exit 2; }
    mkdir -p "$KIT/triton-fill" && cp -R "$TRITON_FILL_FROM/aot" "$KIT/triton-fill/"
    cp "$TRITON_FILL_FROM"/*.json "$TRITON_FILL_FROM"/*.env "$TRITON_FILL_FROM"/*.log "$KIT/triton-fill/" 2>/dev/null || true
    cp tools/zig/triton_fill.py tools/zig/dsv41_triton_options.json "$KIT/tools/zig/"
fi
cp tools/zig/dsv41_window/zig-gate.sh tools/zig/dsv41_window/zig-bench.sh "$KIT/"
chmod +x "$KIT/zig-gate.sh" "$KIT/zig-bench.sh" "$KIT/run_host.sh" "$KIT/run_m4.sh" "$KIT/requests.sh" "$KIT/tf-dsv41-m1" "$KIT/tensorfold-dsv41"
git rev-parse HEAD > "$KIT/COMMIT" 2>/dev/null || echo "${KIT_COMMIT:?KIT_COMMIT: the commit of an archived tree (no .git)}" > "$KIT/COMMIT"
(cd "$KIT" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
file "$KIT/tf-dsv41-m1"; du -sh "$KIT"; echo "kit at $KIT (tree $(head -c 12 "$KIT/COMMIT"))"
