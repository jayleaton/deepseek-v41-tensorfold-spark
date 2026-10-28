#!/usr/bin/env bash
# Shared shell boundary. Source before reading configuration or running bodies.
{ set +x; } 2>/dev/null
safe_library_dir=${BASH_SOURCE[0]%/*}
[[ "$safe_library_dir" != "${BASH_SOURCE[0]}" ]] || safe_library_dir=.
safe_candidate="$(builtin cd -- "$safe_library_dir" && builtin pwd)/safe_exec.py"
if [[ "${safe_exec+x}" == x && "$safe_exec" != "$safe_candidate" ]]; then
    printf '%s\n' 'conflicting subprocess boundary' >&2
    return 78
fi
if [[ "${safe_exec+x}" != x ]]; then safe_exec=$safe_candidate; fi
readonly safe_exec

# $$ remains the sourcing Bash PID inside pipeline subshells. Pair it with the
# kernel start time so supervisors cannot mistake PID reuse for a live owner.
IFS= read -r safe_owner_stat < "/proc/$$/stat"
IFS=' ' read -r -a safe_owner_fields <<< "${safe_owner_stat##*) }"
export TF_SAFE_EXEC_OWNER_PID=$$
export TF_SAFE_EXEC_OWNER_START=${safe_owner_fields[19]}

safe_quote_argv() { python3 "$safe_exec" --quote-argv "$@"; }

safe_check_argv() { python3 "$safe_exec" --check-argv "$@"; }

safe_command() { python3 "$safe_exec" --command "$@"; }
safe_capture() {
    local mode=$1
    shift
    python3 "$safe_exec" --control "$mode" --command "$@"
}
safe_filter() { python3 "$safe_exec" --filter "$@"; }

safe_shell() (
    { set +x; } 2>/dev/null
    local mode=$1 function=$2
    shift 2
    if ! declare -F -- "$function" >/dev/null; then
        printf '%s\n' 'shell boundary requires a function' >&2
        return 1
    fi
    set -o pipefail
    # Both filters are pipeline members and are waited for. The extra fd carries
    # stdout past the stderr pipeline; raw arguments remain in this Bash process.
    (
        ( "$function" "$@" ) 2>&1 1>&3 3>&- | safe_filter >&2
    ) 3>&1 | {
        if [[ "$mode" == text ]]; then
            safe_filter
        else
            safe_filter --control "$mode"
        fi
    }
)

safe_main() {
    if [[ "${TF_SAFE_EXEC_PARENT:-}" != "$PPID" ]]; then
        exec python3 "$safe_exec" --command bash "$@"
    fi
    unset TF_SAFE_EXEC_PARENT
}
