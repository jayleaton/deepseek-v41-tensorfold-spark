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

safe_known_values=()
if [[ -n "${TF_SAFE_REGISTRY_FD:-}" ]]; then
    while IFS= read -r -d '' -u "$TF_SAFE_REGISTRY_FD" safe_known_value; do
        safe_known_values+=("$safe_known_value")
    done
    # The bootstrap descriptor number is data, never shell syntax.
    # Bash cannot dynamically close arbitrary fds without eval in Bash 3.2.
    # The bootstrap maps every inherited registry to fd 9 before shell exec.
    exec 9<&-
    unset TF_SAFE_REGISTRY_FD safe_known_value
fi
safe_known_names=()
safe_registry() {
    local name value
    for value in "${safe_known_values[@]}"; do
        printf '%s\0' "$value"
    done
    for name in "${safe_known_names[@]}"; do
        printf '%s\0' "${!name}"
    done
}
safe_quote_argv() { python3 "$safe_exec" --known-fd 9 --quote-argv "$@" 9< <(safe_registry); }

safe_check_argv() { python3 "$safe_exec" --known-fd 9 --check-argv "$@" 9< <(safe_registry); }

safe_command() { python3 "$safe_exec" --known-fd 9 --command "$@" 9< <(safe_registry); }
safe_capture() {
    local mode=$1
    shift
    python3 "$safe_exec" --known-fd 9 --control "$mode" --command "$@" 9< <(safe_registry)
}
safe_filter() { python3 "$safe_exec" --known-fd 9 --filter "$@" 9< <(safe_registry); }

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
        exec python3 "$safe_exec" --known-fd 9 --command bash "$@" 9< <(safe_registry)
    fi
    unset TF_SAFE_EXEC_PARENT
}

# Keep known values in unexported shell storage solely for the checked boundary.
# The registry uses a private pipe, never an environment variable or argv value.
safe_credentials() {
    local name names
    names=$(compgen -v | python3 "$safe_exec" --credential-names) || return $?
    safe_known_names=()
    while IFS= read -r name; do
        [[ -z "$name" ]] || safe_known_names+=("$name")
    done <<< "$names"
    names=$(python3 "$safe_exec" --known-fd 9 --removed-environment-names 9< <(safe_registry)) || return $?
    while IFS= read -r name; do
        [[ -z "$name" ]] || export -n "$name"
    done <<< "$names"
}
