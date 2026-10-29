#!/usr/bin/env bash
# Guard watchdog for the native dsv41-serve TP=2 pair; run on rank 0 with explicit lifecycle integration.
#
#   scripts/guard-watch-zig.sh --once    one tick (exit 0 healthy or standing down, 1 unhealthy)
#   scripts/guard-watch-zig.sh           a tick every WATCH_INTERVAL s
#
# Stands down while WATCH_LEASE (a file you touch while you benchmark or maintain the pair) is younger than
# WATCH_LEASE_MIN minutes, while WATCH_STATE_DIR/stopped exists (touch it when you stop the pair on purpose), while a
# heal holds the lock, and while the guard's lifecycle check (GUARD_CONFIG, scripts/guard) holds the pair.
# A tick is bad when a rank is not running or /health fails; WATCH_FAILS bad ticks in a row heal (WATCH_HEAL=1 runs
# WATCH_RESTART_ARGV synchronously) at most once every WATCH_MIN_HEAL s. Fail-fast (exit 70), the engine's own
# "stalled" verdict and WATCH_PROBE_FAILS failed probes in a row heal on that tick, at most every WATCH_FF_MIN_HEAL s.
# Never trace configured credentials, including when sourced by a traced caller.
{ set +x; } 2>/dev/null
safe_dir=${BASH_SOURCE[0]%/*}
[[ "$safe_dir" != "${BASH_SOURCE[0]}" ]] || safe_dir=.
source "$safe_dir/safe_shell.sh"
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then safe_main "${BASH_SOURCE[0]}" "$@"; fi
set -uo pipefail
unset BASH_ENV ENV
umask 077
cd "${safe_exec%/*}/.." 2>/dev/null || { printf '%s\n' "watchdog checkout unavailable" >&2; return 78 2>/dev/null || exit 78; }
config=${CONFIG:-config/prod-zig.env}
watch_redact() { safe_filter; }
watch_run() { safe_command "$@"; }
watch_capture() { safe_capture "$@"; }
watch_filter() { safe_filter --control "$1"; }
# Accept direct raw JSON as well as our canonical projection; parser input is
# always stdin, never a body or credential in Python process arguments.
watch_predicate() { [[ "$(printf '%s' "$2" | watch_filter "$1")" == 1 ]]; }
wssh_control() {
    local mode=$1
    shift
    watch_capture "$mode" ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_SSH" "$@"
}
# Config is trusted shell code; suppress diagnostics until its keys are known.
if [[ -f "$config" ]]; then
    set -a
    source "$config" >/dev/null 2>&1
    { set +x; } 2>/dev/null
    unset BASH_ENV ENV
    set +a
fi
# Include credential shell variables set by a sourcing caller without putting any
# values in argv or printing an environment dump. Python sees variable names only.
while IFS= read -r watch_key_name; do
    export "$watch_key_name"
done < <(compgen -v | safe_capture names python3 -c '
import sys
sys.path.insert(0, "scripts")
from redact import secret_name
for name in sys.stdin:
    name = name.strip()
    if secret_name(name): print(name)
')
CONTAINER_PREFIX="${CONTAINER_PREFIX:-tensorfold-zig}"
NAME="$CONTAINER_PREFIX"
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "invalid CONTAINER_PREFIX" >&2; return 78 2>/dev/null || exit 78; }
WORKER_SSH="${WORKER_SSH:-}"
BASE="${WATCH_BASE:-http://127.0.0.1:${WATCH_PORT:-8000}}"
WATCH_MODEL="${WATCH_MODEL:-DeepSeek-V4.1-Flash-TF}"
WATCH_STATE_DIR="${WATCH_STATE_DIR:-${XDG_STATE_HOME:-${HOME:-/tmp}/.local/state}/tf-zig-watch}"
GUARD_CONFIG="${GUARD_CONFIG:-}"
WATCH_GRACE="${WATCH_GRACE:-1800}"; WATCH_FAILS="${WATCH_FAILS:-3}"; WATCH_HEAL="${WATCH_HEAL:-0}"
WATCH_MIN_HEAL="${WATCH_MIN_HEAL:-1800}"; WATCH_FF_MIN_HEAL="${WATCH_FF_MIN_HEAL:-120}"; WATCH_ALERT="${WATCH_ALERT:-}"
WATCH_LEASE="${WATCH_LEASE:-}"; WATCH_LEASE_MIN="${WATCH_LEASE_MIN:-20}"; WATCH_RESTART_ARGV="${WATCH_RESTART_ARGV:-}"
# The probe: /health stays 200 while a wedged engine answers nothing. Every WATCH_PROBE_EVERY s one tiny completion
# (max_tokens 1, no reasoning, header X-TF-Probe: watchdog) must answer within WATCH_PROBE_TIMEOUT s. It runs only when
# the pair is idle: skipped while every slot is busy, and the guard defers it (429 watchdog_deferred) behind user work,
# queued requests and admission holds, so load is never counted as an engine failure.
WATCH_PROBE="${WATCH_PROBE:-1}"; WATCH_PROBE_EVERY="${WATCH_PROBE_EVERY:-180}"; WATCH_PROBE_TIMEOUT="${WATCH_PROBE_TIMEOUT:-30}"
export WATCH_RESTART_ARGV
WATCH_PROBE_FAILS="${WATCH_PROBE_FAILS:-3}"
WATCH_INTERVAL="${WATCH_INTERVAL:-60}"

_watch_log() { printf '[tf-zig-watch %s] %s\n' "$(watch_capture uint date +%s)" "$*" | watch_redact >&2; }
_watch_alert() {
    local message
    message=$(printf '%s\n' "$*" | watch_redact) || return 1
    log "ALERT: $message"
    [[ -z "$WATCH_ALERT" ]] || watch_run "$WATCH_ALERT" "$message" || true
}
wssh() { watch_run ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_SSH" "$@"; }
_watch_lease_fresh() {
    watch_validate_numbers || return $?
    [[ -n "$WATCH_LEASE" && -f "$WATCH_LEASE" ]] || return 1
    local now modified
    now=$(watch_capture uint date +%s) || return 1
    modified=$(watch_capture uint stat -c %Y "$WATCH_LEASE") || return 1
    (( now - modified < WATCH_LEASE_MIN * 60 ))
}
_watch_guard_permit() { # the guard's lifecycle decision: 0 permit, 75 hold, 78 durable STOP (never heal through a STOP)
    # The native supervisor's durable operator STOP also holds this optional watchdog.
    local native_stop="${STOP_FILE:-${STATE_DIR:-}/native-ops/STOP}"
    [[ ! -e "$native_stop" ]] || return 78
    [[ -n "$GUARD_CONFIG" ]] || return 78
    watch_run python3 scripts/guard/lifecycle.py --config "$GUARD_CONFIG" check --operation "$1" > /dev/null
}
_watch_running() { # $1 = rank: true / false / absent (/ unreachable for rank 1)
    local out
    if [[ "$1" == 0 ]]; then out=$(watch_capture running docker inspect -f '{{.State.Running}}' "$NAME-r0" 2>/dev/null) || { echo absent; return; }
    else
        [[ -n "$WORKER_SSH" ]] || { echo unreachable; return; }
        out=$(wssh_control running "docker inspect -f '{{.State.Running}}' $NAME-r1 2>/dev/null || echo absent") || { echo unreachable; return; }
    fi
    printf '%s\n' "$out"
}

_watch_probe_tick() { # $1 = now, $2 = the /health body. 1: WATCH_PROBE_FAILS tiny completions in a row got nothing; else 0
    watch_validate_numbers || return $?
    [[ "$WATCH_PROBE" == 1 ]] || return 0
    local last_f="$WATCH_STATE_DIR/probe_last" pf_f="$WATCH_STATE_DIR/probe_fails" last pf out code busy
    last=$(watch_capture uint cat "$last_f" 2>/dev/null || echo 0); (( $1 - last >= WATCH_PROBE_EVERY )) || return 0
    busy=$(printf '%s' "$2" | watch_filter busy)
    if [[ "$busy" == 1 ]]; then log "watch: every request slot busy: probe skipped"; return 0; fi
    echo "$1" > "$last_f"
    out=$(watch_capture http_control curl -s -m "$WATCH_PROBE_TIMEOUT" -H 'Content-Type: application/json' -H 'X-TF-Probe: watchdog' -w '\n%{http_code}' \
        -d '{"model":"'"$WATCH_MODEL"'","messages":[{"role":"user","content":"ok"}],"max_tokens":1,"temperature":0,"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false}}' \
        "$BASE/v1/chat/completions" || true)
    code=${out%%$'\n'*}; out=${out#*$'\n'}
    if [[ "$code" == 429 ]] && watch_predicate deferred "$out"; then
        log "watch: guard deferred the idle-only probe (user work or admission hold)"; return 0
    fi
    if [[ "$code" == 200 ]] && watch_predicate choices "$out"; then echo 0 > "$pf_f"; return 0; fi
    pf=$(( $(watch_capture uint cat "$pf_f" 2>/dev/null || echo 0) + 1 )); echo "$pf" > "$pf_f"
    log "watch: probe $pf of $WATCH_PROBE_FAILS failed (http ${code:-none} within ${WATCH_PROBE_TIMEOUT} s)"
    (( pf >= WATCH_PROBE_FAILS )) || return 0
    echo 0 > "$pf_f"; return 1
}

# The native idle round clock can be old when newly active work starts prefill (docs/KNOWN-ISSUES.md). Only this
# exact young-request verdict gets a bounded recheck; all other failures keep their policy.
_watch_startup_stale_idle() {
    [[ "$1" == 503 ]] || return 1
    watch_predicate stale "$2"
}

_watch_startup_recheck_healthy() {
    [[ "$1" == 200 ]] || return 1
    watch_predicate healthy "$2"
}

_watch_startup_retry() { # $1 = code, $2 = body; prints "code<newline>body" after at most 3 s of rechecks
    local code=$1 body retry_deadline retry_left retry_timeout
    [[ "$code" =~ ^[0-9]{3}$ ]] || code=000
    body=$(printf '%s' "$2" | watch_filter project) || return 1
    retry_deadline=$(( ${EPOCHREALTIME/./} + 3000000 ))
    log "watch: young prefill with stale idle round clock; rechecking health for at most 3 s"
    while startup_stale_idle "$code" "$body"; do
        retry_left=$(( retry_deadline - ${EPOCHREALTIME/./} ))
        (( retry_left > 0 )) || break
        # Curl itself is bounded by the remaining window, including a hung health endpoint.
        printf -v retry_timeout '%d.%06d' "$((retry_left / 1000000))" "$((retry_left % 1000000))"
        body=$(watch_capture http_control curl -s -m "$retry_timeout" -w '\n%{http_code}' "$BASE/health" || true)
        code=${body%%$'\n'*}; body=${body#*$'\n'}
        startup_stale_idle "$code" "$body" || break
        retry_left=$(( retry_deadline - ${EPOCHREALTIME/./} ))
        if (( retry_left > 200000 )); then watch_run sleep 0.2; fi
    done
    if startup_recheck_healthy "$code" "$body" && (( ${EPOCHREALTIME/./} <= retry_deadline )); then
        log "watch: startup false-stall 503 resolved by actual healthy 200 within 3 s"; return 0
    fi
    printf '%s\n%s' "$code" "$body"; return 1
}

_watch_evidence_dump() { # $1 = why. Before a heal: WATCH_STATE_DIR/heal-evidence/<UTC stamp>/ gets health, logs and state
    local d file
    d=$(watch_capture stamp date -u +%Y%m%d-%H%M%S) || return 1
    d="$WATCH_STATE_DIR/heal-evidence/$d"
    watch_run mkdir -p -m 700 "$WATCH_STATE_DIR" "$WATCH_STATE_DIR/heal-evidence" "$d" || return 1
    watch_run chmod 700 "$WATCH_STATE_DIR" "$WATCH_STATE_DIR/heal-evidence" "$d" || return 1
    # Tighten existing files too (a previous run may have used a permissive umask).
    for file in why.txt health.json r0.log r0-state.json r1.log r1-state.json; do
        (umask 077; : >> "$d/$file") || return 1
        watch_run chmod 600 "$d/$file" || return 1
    done
    printf '%s\n' "$1" | watch_redact > "$d/why.txt" || return 1
    watch_capture json curl -s -m 10 "$BASE/health" > "$d/health.json" 2>&1 || true
    watch_run docker logs --timestamps --tail 400 "$NAME-r0" > "$d/r0.log" 2>&1 || true
    watch_capture state docker inspect -f '{{json .State}}' "$NAME-r0" > "$d/r0-state.json" 2>&1 || true
    if [[ -n "$WORKER_SSH" ]]; then
        wssh "docker logs --timestamps --tail 400 $NAME-r1 2>&1" > "$d/r1.log" 2>&1 || true
        wssh_control state "docker inspect -f '{{json .State}}' $NAME-r1" > "$d/r1-state.json" 2>&1 || true
    fi
    log "watch: heal evidence in $d"
}

_watch_watch_tick() {
    watch_validate_numbers || return $?
    watch_run mkdir -p -m 700 "$WATCH_STATE_DIR" || return 1
    watch_run chmod 700 "$WATCH_STATE_DIR" || return 1
    local fails_f="$WATCH_STATE_DIR/fails" heal_f="$WATCH_STATE_DIR/last_heal" fails r0 r1 age started code body now bad="" last
    local probe_bad=0 retried
    guard_permit watch || { log "watch: guard lifecycle holds; standing down"; return 0; }
    fails=$(watch_capture uint cat "$fails_f" 2>/dev/null || echo 0)
    now=$(watch_capture uint date +%s) || return 1
    if lease_fresh; then echo 0 > "$fails_f"; log "watch: $WATCH_LEASE is fresh; standing down"; return 0; fi
    if [[ -e "$WATCH_STATE_DIR/stopped" ]]; then echo 0 > "$fails_f"; log "watch: stopped on purpose; standing down"; return 0; fi
    if ! watch_run flock -n "$WATCH_STATE_DIR/lock" true; then log "watch: a heal is in progress; standing down"; return 0; fi
    r0=$(running 0); r1=$(running 1)
    if [[ "$r0" != true ]]; then bad="rank 0 $r0"
    elif [[ "$r1" == unreachable ]]; then log "watch: worker unreachable over ssh (not counted)"
    elif [[ "$r1" != true ]]; then bad="rank 1 $r1"
    else
        if started=$(watch_capture started docker inspect -f '{{.State.StartedAt}}' "$NAME-r0"); then
            age=$(( now - started ))
        else
            age=$WATCH_GRACE # An invalid timestamp cannot grant loading grace.
        fi
        body=$(watch_capture http_control curl -s -m 10 -w '\n%{http_code}' "$BASE/health" || true)
        code=${body%%$'\n'*}; body=${body#*$'\n'}
        if startup_stale_idle "$code" "$body"; then
            if retried=$(startup_retry "$code" "$body"); then code=200; body='{"ok":true}'
            else
                code=${retried%%$'\n'*}; body=${retried#*$'\n'}
                bad="stalled: startup health recheck unresolved: /health $code ${body:0:300}"; probe_bad=1
            fi
        fi
        if [[ -n "$bad" ]]; then :
        elif [[ "$code" == 200 ]]; then
            probe_tick "$now" "$body" || { bad="wedged: $WATCH_PROBE_FAILS tiny completions in a row got nothing within ${WATCH_PROBE_TIMEOUT} s (/health 200)"; probe_bad=1; }
        elif [[ "$code" == 000 && $age -lt $WATCH_GRACE ]]; then log "watch: loading (${age}s)"; echo 0 > "$fails_f"; return 0
        elif [[ "$code" == 503 ]] && watch_predicate stalled "$body"; then
            # the engine's own stall verdict: heal now, like a fail-fast exit
            bad="stalled: ${body:0:300}"; probe_bad=1
        else bad="/health $code ${body:0:300}"; fi
    fi
    if [[ -z "$bad" ]]; then echo 0 > "$fails_f"; return 0; fi
    local need=$WATCH_FAILS min=$WATCH_MIN_HEAL e0 e1=""
    e0=$(watch_capture uint docker inspect -f '{{.State.ExitCode}}' "$NAME-r0" 2>/dev/null || true)
    [[ -z "$WORKER_SSH" ]] || e1=$(wssh_control uint "docker inspect -f '{{.State.ExitCode}}' $NAME-r1" 2>/dev/null || true)
    if [[ "$r0" == false && "$e0" == 70 ]] || [[ "$r1" == false && "$e1" == 70 ]]; then
        need=1; min=$WATCH_FF_MIN_HEAL; bad="$bad (fail-fast exit 70)"
    fi
    # the probe's failures were counted by probe_tick already
    if [[ "$probe_bad" == 1 ]]; then need=1; min=$WATCH_FF_MIN_HEAL; fi
    fails=$((fails + 1)); echo "$fails" > "$fails_f"
    log "watch: bad tick $fails of $need: $bad"
    (( fails >= need )) || return 0
    last=$(watch_capture uint cat "$heal_f" 2>/dev/null || echo 0)
    if [[ "$WATCH_HEAL" != 1 || -z "$WATCH_RESTART_ARGV" ]]; then alert "unhealthy ($bad); not restarting (WATCH_HEAL=1 and WATCH_RESTART_ARGV JSON argv)"; return 1; fi
    if (( now - last < min )); then alert "unhealthy ($bad); healed $((now - last))s ago, waiting"; return 1; fi
    guard_permit start || { log "watch: guard lifecycle holds the heal; standing down"; return 0; }
    echo "$now" > "$heal_f"; echo 0 > "$fails_f"
    evidence_dump "$bad" || return 1
    alert "unhealthy ($bad); restarting both ranks"
    : >> "$WATCH_STATE_DIR/heal.log" || return 1
    watch_run chmod 600 "$WATCH_STATE_DIR/heal.log" || return 1
    # flock and the selected launcher share the checked supervisor. Restart data
    # never enters a shell interpreter; only the reviewed recovery argv is admitted.
    watch_run flock "$WATCH_STATE_DIR/lock" python3 "$safe_exec" --restart >> "$WATCH_STATE_DIR/heal.log" 2>&1 || return $?
    return 1
}

watch_validate_numbers() {
    [[ -z "${WATCH_RESTART_CMD:-}" ]] || { printf '%s\n' 'WATCH_RESTART_CMD is removed; use WATCH_RESTART_ARGV JSON argv' >&2; return 78; }
    local key value
    for key in WATCH_GRACE WATCH_FAILS WATCH_HEAL WATCH_MIN_HEAL WATCH_FF_MIN_HEAL WATCH_LEASE_MIN WATCH_PROBE WATCH_PROBE_EVERY WATCH_PROBE_TIMEOUT WATCH_PROBE_FAILS WATCH_INTERVAL; do
        value=${!key}
        [[ "$value" =~ ^[0-9]{1,9}$ ]] || { printf '%s\n' 'invalid watchdog numeric setting' >&2; return 78; }
        while [[ ${#value} -gt 1 && "$value" == 0* ]]; do value=${value#0}; done
        printf -v "$key" '%s' "$value"
    done
    [[ "$WATCH_HEAL" =~ ^[01]$ && "$WATCH_PROBE" =~ ^[01]$ ]] || { printf '%s\n' 'invalid watchdog boolean setting' >&2; return 78; }
    (( WATCH_FAILS > 0 && WATCH_PROBE_FAILS > 0 && WATCH_PROBE_TIMEOUT > 0 && WATCH_INTERVAL > 0 )) || { printf '%s\n' 'invalid watchdog positive setting' >&2; return 78; }
}

log() { safe_shell text _watch_log "$@"; }
alert() { safe_shell text _watch_alert "$@"; }
lease_fresh() { safe_shell text _watch_lease_fresh "$@"; }
guard_permit() { safe_shell text _watch_guard_permit "$@"; }
running() { safe_shell running _watch_running "$@"; }
probe_tick() { safe_shell text _watch_probe_tick "$@"; }
startup_stale_idle() { safe_shell text _watch_startup_stale_idle "$@"; }
startup_recheck_healthy() { safe_shell text _watch_startup_recheck_healthy "$@"; }
startup_retry() { safe_shell health_result _watch_startup_retry "$@"; }
evidence_dump() { safe_shell text _watch_evidence_dump "$@"; }
watch_tick() { safe_shell text _watch_watch_tick "$@"; }

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    watch_validate_numbers || exit $?
    if [[ "${1:-}" == --once ]]; then watch_tick; exit $?; fi
    while :; do
        watch_tick || true
        watch_run sleep "$WATCH_INTERVAL" || exit $?
    done
fi
