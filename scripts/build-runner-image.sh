#!/usr/bin/env bash
# Layer the in-model runner (circuit frontend, circuit, graph bank) over an image scripts/build-native.sh built.
# Without RUNNER_ASSETS_DIR the assets are downloaded from config/runner-assets.release (scripts/fetch_runner_assets.py);
# either way they are checked against config/runner-assets.sha256 in task-owned scratch before the build.
{ set +x; } 2>/dev/null
safe_dir=${BASH_SOURCE[0]%/*}
[[ "$safe_dir" != "${BASH_SOURCE[0]}" ]] || safe_dir=.
source "$safe_dir/safe_shell.sh"
safe_main "${BASH_SOURCE[0]}" "$@"
set -euo pipefail
usage='usage: build-runner-image.sh BASE_IMAGE [RUNNER_ASSETS_DIR] [docker build arguments, e.g. -t TAG]'
[[ $# -ge 1 ]] || { echo "$usage" >&2; exit 2; }
base=$1 assets=
shift
if [[ $# -gt 0 && $1 != -* ]]; then assets=$1; shift; fi
[[ -z "$assets" || -d "$assets" ]] || { echo 'RUNNER_ASSETS_DIR must be a directory' >&2; exit 2; }
[[ -z "$assets" || "$assets" == /* ]] || assets=$PWD/$assets
cd "${safe_exec%/*}/.."
scratch=$(safe_capture path mktemp -d)
trap 'safe_command python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1])" "$scratch"' EXIT
context="$scratch/context"
safe_command mkdir -p "$context/docker/zig"
safe_command cp docker/zig/runner.Dockerfile "$context/docker/zig/"
if [[ -n "$assets" ]]; then
    safe_command cp -R "$assets" "$context/runner"
    safe_command python3 scripts/check_runner_assets.py config/runner-assets.sha256 "$context/runner"
else
    safe_command python3 scripts/fetch_runner_assets.py "$context/runner"
fi
safe_command docker build -f "$context/docker/zig/runner.Dockerfile" --build-arg "BASE_IMAGE=$base" "$@" "$context"
