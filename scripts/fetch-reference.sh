#!/usr/bin/env bash
# Rebuild the build-only Python reference tree that the RoPE and Triton AOT generators import.
# Usage: scripts/fetch-reference.sh DEST [UPSTREAM_BUNDLE]   (DEST/src is the generators' PY_SOURCE)
# TensorFold 0.6.0 at a fixed revision, plus the recipe's three historical patches read from a fixed recipe
# commit and checked against their SHA-256 values. A supplied bundle is used without network fallback.
{ set +x; } 2>/dev/null
safe_dir=${BASH_SOURCE[0]%/*}
[[ "$safe_dir" != "${BASH_SOURCE[0]}" ]] || safe_dir=.
source "$safe_dir/safe_shell.sh"
safe_main "${BASH_SOURCE[0]}" "$@"
set -euo pipefail
root=$(cd "$safe_dir/.." && pwd -P)
destination=${1:?destination required}
bundle=${2:-/nonexistent/upstream.bundle}
upstream_repo=https://github.com/ashhart/TensorFold.git
upstream_rev=c4646171139ee8a3c38103eaa1699dad226ec12b
recipe_repo=https://github.com/jayleaton/deepseek-v41-tensorfold-spark.git
recipe_rev=391d677b300d9b53e64f4265a261c0b329ea58e0
patch_sums='e6d83093bd29fb41d406487ce4abe48a0c0a9bee23998fd4758b195d4c9ce625  patches/0001-spark-stack-060.patch
f03c5095e529d88ac778caba79d8f6d396ed4ad76c744469fc7940643059088d  patches/0002-deepseek-v41-family.patch
76f147c74337aab4e419753842b7116d7040d958ade733972dace4dc4b69e553  patches/0003-dsv41-literal-added-tokens.patch'

if [[ -e "$destination" ]]; then
    echo 'destination exists; choose a new directory' >&2
    exit 2
fi
safe_command mkdir -p "$destination"
destination=$(cd "$destination" && pwd -P)
[[ "$bundle" == /* ]] || bundle=$PWD/$bundle
cd "$root"
safe_command bash docker/zig/fetch-source.sh "$upstream_rev" "$destination" "$bundle" "$upstream_repo"

# The patches left the tree with the native switch; any full clone still holds them, else fetch that one commit.
history=$root
if ! safe_command git -C "$root" cat-file -e "$recipe_rev^{commit}" 2>/dev/null; then
    history=$destination/.reference-recipe
    safe_command git init -q "$history"
    safe_command git -C "$history" fetch -q --no-tags --depth=1 "$recipe_repo" "$recipe_rev"
fi
work=$destination/.reference-patches
safe_command mkdir "$work"
safe_command git -C "$history" archive --output="$work/patches.tar" "$recipe_rev" patches
safe_command tar -x -C "$work" -f "$work/patches.tar"
printf '%s\n' "$patch_sums" > "$work/SHA256SUMS"
(cd "$work" && safe_command sha256sum --quiet --strict -c SHA256SUMS)
while read -r _ name; do
    safe_command git -C "$destination" apply --whitespace=nowarn "$work/$name"
done <<< "$patch_sums"
printf 'upstream %s %s\nrecipe %s %s\n%s\n' "$upstream_repo" "$upstream_rev" "$recipe_repo" "$recipe_rev" \
    "$patch_sums" > "$destination/REFERENCE-SOURCE.txt"
safe_command rm -rf -- "$work" "$destination/.reference-recipe"
test -f "$destination/src/tensorfold/families/deepseek_v41/cuda/rope.py"
echo "reference source: $destination/src"
