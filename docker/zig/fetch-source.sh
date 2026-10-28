#!/usr/bin/env bash
# Resolve the exact recipe pin, with no network fallback for a supplied bundle.
set -euo pipefail
rev=${1:?revision required}
destination=${2:?destination required}
bundle=${3:?bundle path required}
[[ "$rev" =~ ^[0-9a-f]{40}$ ]] || { echo 'invalid engine revision' >&2; exit 2; }
git init "$destination" >/dev/null
git -C "$destination" remote add origin https://github.com/ashhart/TensorFold.git
if [[ -f "$bundle" ]]; then
    git -C "$destination" bundle verify "$bundle"
    git -C "$destination" fetch --no-tags "$bundle" "$rev"
else
    git -C "$destination" fetch --depth=1 origin "$rev"
fi
git -C "$destination" checkout --detach FETCH_HEAD
test "$(git -C "$destination" rev-parse HEAD)" = "$rev"
