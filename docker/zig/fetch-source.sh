#!/usr/bin/env bash
# Resolve the exact recipe pin, with no network fallback for a supplied bundle.
{ set +x; } 2>/dev/null
safe_dir=${BASH_SOURCE[0]%/*}
[[ "$safe_dir" != "${BASH_SOURCE[0]}" ]] || safe_dir=.
source "$safe_dir/../../scripts/safe_shell.sh"
safe_main "${BASH_SOURCE[0]}" "$@"
set -euo pipefail
rev=${1:?revision required}
destination=${2:?destination required}
bundle=${3:?bundle path required}
repo=${4:-https://github.com/ashhart/TensorFold.git}
[[ "$rev" =~ ^[0-9a-f]{40}$ ]] || { echo 'invalid engine revision' >&2; exit 2; }
[[ "$repo" =~ ^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git$ ]] || { echo 'invalid engine repository' >&2; exit 2; }
safe_command git init "$destination" >/dev/null
safe_command git -C "$destination" remote add origin "$repo"
if [[ -f "$bundle" ]]; then
    safe_command git -C "$destination" bundle verify "$bundle"
    safe_command git -C "$destination" fetch --no-tags "$bundle" "$rev"
else
    safe_command git -C "$destination" fetch --depth=1 origin "$rev"
fi
safe_command git -C "$destination" checkout --detach FETCH_HEAD
test "$(safe_capture revision git -C "$destination" rev-parse HEAD)" = "$rev"
