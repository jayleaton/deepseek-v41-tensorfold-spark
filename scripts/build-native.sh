#!/usr/bin/env bash
# Build one exact pin from upstream or an optional local checkout/bundle.
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
scratch=$(mktemp -d)
trap 'python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1])" "$scratch"' EXIT
context="$scratch/context"
mkdir -p "$context/docker/zig/source"
cp docker/zig/Dockerfile docker/zig/fetch-source.sh "$context/docker/zig/"
touch "$context/docker/zig/source/.keep"
if [[ -n "${TENSORFOLD_SOURCE:-}" ]]; then
    if [[ -d "$TENSORFOLD_SOURCE" ]]; then
        # Fetch into task-owned scratch; never change the supplied engine checkout.
        git init --bare "$scratch/source.git" >/dev/null
        python3 "$safe_exec" --command git -C "$scratch/source.git" fetch --no-tags "$TENSORFOLD_SOURCE" "$rev:refs/heads/recipe-pin"
        git -C "$scratch/source.git" bundle create "$context/docker/zig/source/engine.bundle" refs/heads/recipe-pin
    elif [[ -f "$TENSORFOLD_SOURCE" ]]; then
        cp "$TENSORFOLD_SOURCE" "$context/docker/zig/source/engine.bundle"
    else
        echo 'TENSORFOLD_SOURCE must be a Git checkout or bundle' >&2; exit 2
    fi
    # Validate before handing the context to Docker, including missing prerequisites.
    bash docker/zig/fetch-source.sh "$rev" "$scratch/verified" "$context/docker/zig/source/engine.bundle"
fi
python3 "$safe_exec" --command docker build -f docker/zig/Dockerfile --build-arg "TENSORFOLD_REV=$rev" "$@" "$context"
