#!/usr/bin/env bash
# Pair lifecycle wrapper. The rank launcher also supports standalone diagnostics.
{ set +x; } 2>/dev/null
safe_dir=${BASH_SOURCE[0]%/*}
[[ "$safe_dir" != "${BASH_SOURCE[0]}" ]] || safe_dir=.
source "$safe_dir/safe_shell.sh"
export CONFIG=${CONFIG:-config/prod-zig.env}
[[ "$CONFIG" == /* ]] || CONFIG="${safe_exec%/*}/../$CONFIG"
[[ -z "${TENSORFOLD_API_KEY:-}" ]] || export TF_NATIVE_REQUIRE_FILE=1
safe_main "${BASH_SOURCE[0]}" "$@"
set -euo pipefail
cd "${safe_exec%/*}/.."
operation=${1:-status}
shift "$(( $# > 0 ? 1 : 0 ))"
case "$operation" in
    build) safe_command bash scripts/build-native.sh "$@"; exit $? ;;
    rank) safe_command bash scripts/run-zig.sh "$@"; exit $? ;;
esac
export CONFIG=${CONFIG:-config/prod-zig.env}
[[ -f "$CONFIG" ]] || { echo 'configuration missing; copy a native profile to CONFIG' >&2; exit 2; }
set -a
source "$CONFIG" >/dev/null 2>&1
{ set +x; } 2>/dev/null
set +a
safe_credentials || exit $?
[[ -z "${TENSORFOLD_API_KEY:-}" ]] || export TF_NATIVE_REQUIRE_FILE=1
safe_command python3 scripts/native_ops.py "$operation" "$@"
