#!/usr/bin/env bash
# Pair lifecycle wrapper. The rank launcher also supports standalone diagnostics.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
operation=${1:-status}
shift "$(( $# > 0 ? 1 : 0 ))"
case "$operation" in
    build) exec bash scripts/build-native.sh "$@" ;;
    rank) exec bash scripts/run-zig.sh "$@" ;;
esac
export CONFIG=${CONFIG:-config/prod-zig.env}
[[ -f "$CONFIG" ]] || { echo "missing $CONFIG; copy a prod-zig example" >&2; exit 2; }
set -a
source "$CONFIG"
set +a
exec python3 scripts/native_ops.py "$operation" "$@"
