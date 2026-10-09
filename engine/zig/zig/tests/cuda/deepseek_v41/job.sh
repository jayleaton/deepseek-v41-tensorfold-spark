#!/bin/bash
# The DeepSeek-V4.1 kernel exactness job on one GPU: the Python engine's kernels write fixtures (oracle.py, seeded
# random inputs, no checkpoint), then tf-dsv41-test runs our fatbins on the same inputs and compares every output bit
# for bit; finally the SASS of every kernel of ours against the Python extension's (.so) of the same GPU.
#
# usage: bash job.sh <out dir>
#   TF_DSV41_PY_SRC   Python TensorFold checkout whose src/ the oracle imports   (required)
#   TF_DSV41_TEST     the tf-dsv41-test binary (built with the fatbins for this GPU's SM)            (required)
#   TF_DSV41_FATBINS  directory of the dsv41_*.fatbin the binary embeds (for the SASS step; optional)
# Needs python with torch (a CUDA-enabled PyTorch environment), cuobjdump for the SASS step. Exit 0 = all pass.
set -u
OUT="${1:?out dir}"
PY="${TF_DSV41_PY_SRC:?set TF_DSV41_PY_SRC}"
T="${TF_DSV41_TEST:?set TF_DSV41_TEST}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$OUT/torch_ext}"
rc=0
step() { echo "=== $(date -u +%H:%M:%S) $*"; }

step "device"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | tee "$OUT/device.txt"
"$T" symbols 2>&1 | tee "$OUT/symbols.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
"$T" kvsplit 2>&1 | tee "$OUT/kvsplit.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=1

step "oracle (the Python engine's extensions, JIT-built for this GPU unless prebuilt in TORCH_EXTENSIONS_DIR)"
rm -rf "$OUT/fixtures"
PYTHONPATH="$PY/src" PYTHONDONTWRITEBYTECODE=1 python -B "$HERE/oracle.py" "$OUT" > "$OUT/oracle.log" 2>&1 \
  || { rc=1; echo "oracle failed"; tail -30 "$OUT/oracle.log"; }

step "capture: the Python engine's own GPU suites, their extension calls recorded (dsv41_capture.py)"
# Every allowed extension call the suites make (attention + top-k, mHC decode / prefill, router GEMV incl. R1c's prune,
# pfdense, dense3 / x3seg / upstream linear, incl. R1 / R1c variants when the twin has them) becomes a replay case.
# Timing, emulator and real-weight tests are left out. A failing Python test does not fail the job (its calls still
# replay); the log keeps pytest's summary.
CAPTURE_TESTS=""
for t in attn_cuda mhc_cuda mhc_pf router_gemv router_gemv_narrow pfdense dense3 fused_proj; do
  f="$PY/tests/cuda/test_dsv41_${t}_gpu.py"
  [ -f "$f" ] && CAPTURE_TESTS="$CAPTURE_TESTS $f"
done
SKIP_K="not time and not us_ and not sweep and not cold and not real_layer and not emulator and not device_info and not launch_ms and not chunk_budget and not end_to_end and not kwidth"
if [ -n "$CAPTURE_TESTS" ]; then
  (cd "$PY" && PYTHONPATH="$HERE:$PY/src:$PY/tests" PYTHONDONTWRITEBYTECODE=1 TF_DSV41_CAPTURE="$OUT/fixtures" \
     timeout "${TF_DSV41_CAPTURE_MIN:-25}m" python -m pytest -q -p dsv41_capture -p no:cacheprovider $CAPTURE_TESTS -k "$SKIP_K") \
    > "$OUT/capture.log" 2>&1
  echo "capture: pytest rc $? ($(tail -1 "$OUT/capture.log"))"
  # the plugin writes capture.json at session end, calls or none: its absence means it never loaded
  if cp "$OUT/fixtures/capture.json" "$OUT/capture.json" 2>/dev/null; then
    calls=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["calls"])' "$OUT/capture.json")
    echo "capture: $calls calls recorded"
    [ "$calls" -gt 0 ] || { rc=1; echo "capture: FAIL no extension call recorded"; }
  else
    rc=1; echo "capture: FAIL no capture.json (the plugin did not load)"
  fi
else
  echo "capture: no GPU suites under $PY/tests/cuda"
fi

step "bit checks"
"$T" all "$OUT/fixtures" 2>&1 | tee "$OUT/bits.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=1

if [ -n "${TF_DSV41_FATBINS:-}" ] && command -v cuobjdump >/dev/null; then
  step "SASS: the Python extensions against our fatbins"
  sm=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
  for pair in "tensorfold_exl3_experts_v1:exl3_experts" "tf_dsv41_x3ld_v1:x3ld" "tf_dsv41_x3pf_v1:x3pf" \
              "tf_dsv41_x3gm_v1:x3gm"; do
    ext=${pair%%:*}; name=${pair##*:}
    so=$(ls "$TORCH_EXTENSIONS_DIR"/$ext/$ext*.so 2>/dev/null | head -1)
    [ -n "$so" ] || { echo "no $ext build (not loaded by the oracle)"; continue; }
    cuobjdump -sass "$so" > "$OUT/sass-py-$name.txt"
    cuobjdump -sass "$TF_DSV41_FATBINS/dsv41_$name.fatbin" > "$OUT/sass-zig-$name.txt"
    python3 "$HERE/../../../kernels/cuda/deepseek_v41/sass_check.py" "$OUT/sass-py-$name.txt" "$OUT/sass-zig-$name.txt" \
      --sm "$sm" | tail -3 | tee -a "$OUT/sass.log" || rc=1
    rm -f "$OUT/sass-py-$name.txt" "$OUT/sass-zig-$name.txt"
  done
fi

rm -rf "$OUT/fixtures"   # tens of MB of random inputs; the logs keep the results
step "rc=$rc"
exit $rc
