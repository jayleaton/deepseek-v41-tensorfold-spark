#!/usr/bin/env bash
# Build one exact upstream pin, from upstream or an optional local checkout/bundle, with the recipe's patch series.
{ set +x; } 2>/dev/null
safe_dir=${BASH_SOURCE[0]%/*}
[[ "$safe_dir" != "${BASH_SOURCE[0]}" ]] || safe_dir=.
source "$safe_dir/safe_shell.sh"
safe_main "${BASH_SOURCE[0]}" "$@"
set -euo pipefail
cd "${safe_exec%/*}/.."
rev=$(safe_capture revision cat config/tensorfold.rev)
[[ "$rev" =~ ^[0-9a-f]{40}$ ]] || { echo 'invalid config/tensorfold.rev' >&2; exit 2; }
# The repository that publishes the pin; a supplied checkout or bundle replaces the network fetch.
repo=https://github.com/ashhart/TensorFold.git
[[ ! -f config/tensorfold.repo ]] || { read -r repo || [[ -n "$repo" ]]; } < config/tensorfold.repo
[[ "$repo" =~ ^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git$ ]] || { echo 'invalid config/tensorfold.repo' >&2; exit 2; }
# The patch series patches/0001..LAST and the engine tree it must give (config/tensorfold.patches, .tree).
series=()
if [[ -f config/tensorfold.tree ]]; then
    tree=$(safe_capture revision cat config/tensorfold.tree)
    { read -r last || [[ -n "$last" ]]; } < config/tensorfold.patches
    [[ "$tree" =~ ^[0-9a-f]{40}$ && "$last" =~ ^[0-9]{4}$ ]] || { echo 'invalid config/tensorfold.tree or .patches' >&2; exit 2; }
    n=0
    for p in patches/[0-9][0-9][0-9][0-9]-*.patch; do
        [[ -f "$p" ]] || break
        name=${p##*/}
        [[ "${name:0:4}" > "$last" ]] && break
        n=$((n + 1))
        printf -v number '%04d' "$n"
        [[ "${name:0:4}" == "$number" ]] || { echo "patches/$number missing or not unique" >&2; exit 2; }
        series+=("$p")
    done
    printf -v number '%04d' "$n"
    [[ "$number" == "$last" ]] || { echo "patches/$last missing" >&2; exit 2; }
fi
scratch=$(safe_capture path mktemp -d)
trap 'safe_command python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1])" "$scratch"' EXIT
context="$scratch/context"
safe_command mkdir -p "$context/docker/zig/source"
safe_command cp docker/zig/Dockerfile docker/zig/fetch-source.sh "$context/docker/zig/"
safe_command mkdir -p "$context/scripts"
safe_command cp scripts/safe_exec.py scripts/safe_quote.py scripts/safe_output.py scripts/redact.py scripts/safe_shell.sh scripts/watchdog_io.py "$context/scripts/"
safe_command touch "$context/docker/zig/source/.keep"
safe_command mkdir -p "$context/patches"
safe_command touch "$context/patches/.keep"
[[ ${#series[@]} == 0 ]] || safe_command cp "${series[@]}" "$context/patches/"
patch_args=()
[[ ${#series[@]} == 0 ]] || patch_args=("$PWD/patches" "$last" "$tree")
if [[ -n "${TENSORFOLD_SOURCE:-}" ]]; then
    if [[ -d "$TENSORFOLD_SOURCE" ]]; then
        # Fetch into task-owned scratch; never change the supplied engine checkout.
        safe_command git init --bare "$scratch/source.git" >/dev/null
        safe_command git -C "$scratch/source.git" fetch --no-tags "$TENSORFOLD_SOURCE" "$rev:refs/heads/recipe-pin"
        safe_command git -C "$scratch/source.git" bundle create "$context/docker/zig/source/engine.bundle" refs/heads/recipe-pin
    elif [[ -f "$TENSORFOLD_SOURCE" ]]; then
        safe_command cp "$TENSORFOLD_SOURCE" "$context/docker/zig/source/engine.bundle"
    else
        echo 'TENSORFOLD_SOURCE must be a Git checkout or bundle' >&2; exit 2
    fi
    # Validate before handing the context to Docker, including missing prerequisites.
    safe_command bash docker/zig/fetch-source.sh "$rev" "$scratch/verified" "$context/docker/zig/source/engine.bundle" "$repo" ${patch_args[@]+"${patch_args[@]}"}
fi
build_args=(--build-arg "TENSORFOLD_REV=$rev" --build-arg "TENSORFOLD_REPO=$repo")
[[ ${#series[@]} == 0 ]] || build_args+=(--build-arg "TENSORFOLD_PATCHES=$last" --build-arg "TENSORFOLD_TREE=$tree")
safe_command docker build -f docker/zig/Dockerfile "${build_args[@]}" "$@" "$context"
