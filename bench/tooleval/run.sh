#!/usr/bin/env bash
# Tool-calling benchmarks against a DeepSeek-V4.1 endpoint: tool-eval-bench (SeraphimSerapis, MIT) and spark-bench
# TrueScore (Weschera, README says MIT) at pinned commits, installed under TOOLEVAL_CACHE (default
# ~/.cache/tooleval), plus `own`: bench/tooleval/chains.py, a stdlib multi-step chain eval (no install).
#
#   bench/tooleval/run.sh setup                         # clone the pinned commits + venvs (no server needed)
#   bench/tooleval/run.sh teb  off|low|high [label]     # tool-eval-bench, 69 scenarios
#   bench/tooleval/run.sh sb   off|low|high [label]     # spark-bench TrueScore, 76 scenarios x 2, temp 0.3
#   bench/tooleval/run.sh both off|low|high [label]
#   bench/tooleval/run.sh chains off|low|high [label]   # quick: tool-eval-bench C (multi-step) + spark-bench agentic
#   bench/tooleval/run.sh own off|low|high [label]      # chains.py: 6 multi-step scenarios, simulated tools
#
# Modes: off  = chat_template_kwargs {"enable_thinking": false}
#        low  = thinking on, effort low (50 in DeepSeek's encoding)
#        high = thinking on, effort high (75; TF_DSV41_DEFAULT_EFFORT). The server reads chat_template_kwargs
#               enable_thinking / reasoning_effort and the top-level reasoning_effort.
# Env: BASE_URL (http://127.0.0.1:8000/v1), MODEL (DeepSeek-V4.1-Flash-TF), API_KEY, OUT (results/tooleval),
#      TEB_ARGS / SB_ARGS (extra flags), TEB_TEMP (0.0, the bench default), SB_TEMP (0.3), SB_REPEATS (2),
#      TEB_TIMEOUT (off: 300, thinking: 900 s between streamed chunks).
# Each run writes <OUT>/<date>-<bench>-<mode>[-label]/ with the bench's own reports, the server's /health and
# /v1/models before and after, and the exact command line. API_KEY never reaches a saved file or the terminal: the
# command lines, snapshots and logs are sanitized before writing. Third-party evaluators need an audited
# adapter for their direct report/dump writes and are refused until that boundary is available.
{ set +x; } 2>/dev/null
safe_exec="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/safe_exec.py"
if [[ "${TF_SAFE_EXEC_PARENT:-}" != "$PPID" ]]; then
    exec python3 "$safe_exec" bash "${BASH_SOURCE[0]}" "$@"
fi
unset TF_SAFE_EXEC_PARENT
set -euo pipefail
umask 077

CACHE=${TOOLEVAL_CACHE:-$HOME/.cache/tooleval}
TEB_REPO=https://github.com/SeraphimSerapis/tool-eval-bench.git
TEB_REV=c7b5b9550                      # v2.7.0+8 (2026-09-29), the commit this doc's reading is from
SB_REPO=https://github.com/Weschera/spark-bench.git
SB_REV=125ba161                        # v6.8.0 (2026-08-29)
BASE_URL=${BASE_URL:-http://127.0.0.1:8000/v1}
MODEL=${MODEL:-DeepSeek-V4.1-Flash-TF}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-$ROOT/results/tooleval}

die() { echo "run.sh: $*" >&2; exit 1; }
redact() { python3 -u "$ROOT/scripts/redact.py" "$@"; }
redact_snapshot() {
    python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); from safe_output import clean; sys.stdout.write(clean(sys.stdin.read()))' "$ROOT/scripts"
}

# curl with the key as a header read from a file descriptor, never in argv (ps) or a saved command
api() {
    if [[ -n ${API_KEY:-} ]]; then curl -H @<(printf 'Authorization: Bearer %s\n' "$API_KEY") "$@"; else curl "$@"; fi
}

setup() {
    mkdir -p "$CACHE"
    if [[ ! -d $CACHE/tool-eval-bench/.git ]]; then git clone -q "$TEB_REPO" "$CACHE/tool-eval-bench"; fi
    git -C "$CACHE/tool-eval-bench" fetch -q origin && git -C "$CACHE/tool-eval-bench" checkout -q "$TEB_REV"
    [[ -x $CACHE/tool-eval-bench/.venv/bin/python ]] || python3 -m venv "$CACHE/tool-eval-bench/.venv"
    "$CACHE/tool-eval-bench/.venv/bin/pip" install -q -e "$CACHE/tool-eval-bench"
    if [[ ! -d $CACHE/spark-bench/.git ]]; then git clone -q "$SB_REPO" "$CACHE/spark-bench"; fi
    git -C "$CACHE/spark-bench" fetch -q origin && git -C "$CACHE/spark-bench" checkout -q "$SB_REV"
    [[ -x $CACHE/spark-bench/.venv/bin/python ]] || python3 -m venv "$CACHE/spark-bench/.venv"
    # the golden gate renders pages (VIS scenarios): playwright + pillow and a chromium
    "$CACHE/spark-bench/.venv/bin/pip" install -q playwright pillow
    "$CACHE/tool-eval-bench/.venv/bin/tool-eval-bench" --help >/dev/null
    (cd "$CACHE/spark-bench" && .venv/bin/python spark_bench.py eval --help >/dev/null)
    echo "installed: tool-eval-bench $(git -C "$CACHE/tool-eval-bench" rev-parse --short HEAD)," \
         "spark-bench $(git -C "$CACHE/spark-bench" rev-parse --short HEAD) in $CACHE"
}

# thinking mode -> request fields. Both benches merge extra body fields into every request; ours reads
# chat_template_kwargs (enable_thinking, reasoning_effort) and the top-level reasoning_effort.
kwargs_for() {
    case $1 in
        off)  echo '{"enable_thinking": false}' ;;
        low)  echo '{"enable_thinking": true, "reasoning_effort": "low"}' ;;
        high) echo '{"enable_thinking": true, "reasoning_effort": "high"}' ;;
        *) die "mode must be off, low or high (got '$1')" ;;
    esac
}

snapshot() {       # $1 = dir, $2 = tag: what the server was (patch knobs show in the log, not here)
    local host=${BASE_URL%/v1}
    api -fsS -m 10 "$host/health" 2> >(redact text >&2) | redact_snapshot > "$1/health-$2.json" || echo '{"unreachable": true}' > "$1/health-$2.json"
    api -fsS -m 10 "$BASE_URL/models" 2> >(redact text >&2) | redact_snapshot > "$1/models-$2.json" || true
}

preflight() {
    api -fsS -m 10 "$BASE_URL/models" >/dev/null || die "no server at $BASE_URL (scripts/serve.sh start)"
    api -fsS -m 10 "$BASE_URL/models" | grep -q "\"$MODEL\"" || die "$BASE_URL/models does not list $MODEL"
}

# These pinned tools own additional direct-to-disk reports, request dumps and native/browser outputs.
# An stdout pipe or an EXIT scrub cannot guarantee that a credential never reaches a file.
# Keep setup available, but refuse execution until an audited pre-write adapter covers those paths.
third_party_unavailable() {
    echo "run.sh: third-party report/dump pre-write sanitization is unavailable; use 'own' or add an audited adapter" >&2
    exit 78
}

cmd=${1:-}; mode=${2:-}; LABEL=${3:-}
case $cmd in
    setup) setup; exit 0 ;;
    teb|sb|both|chains) third_party_unavailable ;;
    own) kwargs_for "$mode" >/dev/null ;;
    *) sed -n '2,21p' "$0"; exit 2 ;;
esac
preflight
dir=$OUT/$(date +%Y%m%d-%H%M)-$cmd-$mode${LABEL:+-$LABEL}
mkdir -p "$dir"
snapshot "$dir" before
case $cmd in
    own)    python3 "$ROOT/bench/tooleval/chains.py" --base "${BASE_URL%/v1}" --model "$MODEL" --mode "$mode" \
                --out "$dir/chains.json" 2>&1 | redact text | tee "$dir/chains.log" ;;
esac
snapshot "$dir" after
echo "results: $dir"
