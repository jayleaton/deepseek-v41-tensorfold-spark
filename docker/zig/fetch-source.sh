#!/usr/bin/env bash
# Resolve the exact recipe pin, with no network fallback for a supplied bundle, then apply the recipe's patch
# series 0001..LAST in order and require the resulting engine tree.
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
patches=${5:-}
last=${6:-}
tree=${7:-}
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
# All three patch arguments, or none (the plain pin, as reference fetching uses it); never a partial set, which would
# skip the series and the tree check while a count or tree was asked for.
if [[ -z "$patches" ]]; then
    [[ -z "$last" && -z "$tree" ]] || { echo 'patch series, count and engine tree must be given together' >&2; exit 2; }
    exit 0
fi
[[ "$last" =~ ^[0-9]{4}$ && "$tree" =~ ^[0-9a-f]{40}$ ]] || { echo 'invalid patch series or engine tree' >&2; exit 2; }
series=()
n=0
for p in "$patches"/[0-9][0-9][0-9][0-9]-*.patch; do
    [[ -f "$p" ]] || break
    name=${p##*/}
    [[ "${name:0:4}" > "$last" ]] && break
    n=$((n + 1))
    printf -v number '%04d' "$n"
    [[ "${name:0:4}" == "$number" ]] || { echo "patch $number missing or not unique" >&2; exit 2; }
    series+=("$p")
done
printf -v number '%04d' "$n"
[[ "$number" == "$last" ]] || { echo "patch $last missing" >&2; exit 2; }
# Fixed committer and dates, so the same series gives the same commits on any builder.
[[ ${#series[@]} == 0 ]] || safe_command git -C "$destination" -c user.name='TensorFold recipe' -c user.email=recipe@invalid -c core.autocrlf=false \
    am --quiet --committer-date-is-author-date --whitespace=nowarn "${series[@]}"
test "$(safe_capture revision git -C "$destination" rev-parse 'HEAD^{tree}')" = "$tree" \
    || { echo 'patched engine tree differs from config/tensorfold.tree' >&2; exit 2; }
