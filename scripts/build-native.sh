#!/usr/bin/env bash
# Build the native pair from one reviewed upstream pin; no local engine export or patches.
{ set +x; } 2>/dev/null
safe_exec="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/safe_exec.py"
if [[ "${TF_SAFE_EXEC_PARENT:-}" != "$PPID" ]]; then
    exec python3 "$safe_exec" bash "${BASH_SOURCE[0]}" "$@"
fi
unset TF_SAFE_EXEC_PARENT
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rev=$(cat config/tensorfold.rev)
[[ "$rev" =~ ^[0-9a-f]{40}$ ]] || { echo 'invalid config/tensorfold.rev' >&2; exit 2; }
exec python3 "$safe_exec" --command docker build -f docker/zig/Dockerfile --build-arg "TENSORFOLD_REV=$rev" "$@" .
