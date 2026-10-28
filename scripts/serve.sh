#!/usr/bin/env bash
# Pair lifecycle wrapper. The rank launcher also supports standalone diagnostics.
{ set +x; } 2>/dev/null
safe_exec="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/safe_exec.py"
if [[ "${TF_SAFE_EXEC_PARENT:-}" != "$PPID" ]]; then
    export CONFIG=${CONFIG:-config/prod-zig.env}
    [[ "$CONFIG" == /* ]] || CONFIG="$(dirname "$safe_exec")/../$CONFIG"
    exec python3 "$safe_exec" bash "${BASH_SOURCE[0]}" "$@"
fi
unset TF_SAFE_EXEC_PARENT
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
operation=${1:-status}
shift "$(( $# > 0 ? 1 : 0 ))"
case "$operation" in
    build) exec bash scripts/build-native.sh "$@" ;;
    rank) exec bash scripts/run-zig.sh "$@" ;;
esac
export CONFIG=${CONFIG:-config/prod-zig.env}
[[ -f "$CONFIG" ]] || { echo 'configuration missing; copy a native profile to CONFIG' >&2; exit 2; }
set -a
source "$CONFIG" >/dev/null 2>&1
{ set +x; } 2>/dev/null
set +a
exec python3 scripts/native_ops.py "$operation" "$@"
