#!/usr/bin/env bash
# Layer the in-model runner (circuit frontend, circuit, graph bank) over an image scripts/build-native.sh built.
# The assets are checked against config/runner-assets.sha256 after they are copied into task-owned scratch.
{ set +x; } 2>/dev/null
safe_dir=${BASH_SOURCE[0]%/*}
[[ "$safe_dir" != "${BASH_SOURCE[0]}" ]] || safe_dir=.
source "$safe_dir/safe_shell.sh"
safe_main "${BASH_SOURCE[0]}" "$@"
set -euo pipefail
usage='usage: build-runner-image.sh BASE_IMAGE RUNNER_ASSETS_DIR [docker build arguments, e.g. -t TAG]'
[[ $# -ge 2 ]] || { echo "$usage" >&2; exit 2; }
base=$1 assets=$2
shift 2
[[ -d "$assets" ]] || { echo 'RUNNER_ASSETS_DIR must be a directory' >&2; exit 2; }
cd "${safe_exec%/*}/.."
scratch=$(safe_capture path mktemp -d)
trap 'safe_command python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1])" "$scratch"' EXIT
context="$scratch/context"
safe_command mkdir -p "$context/docker/zig"
safe_command cp docker/zig/runner.Dockerfile "$context/docker/zig/"
safe_command cp -R "$assets" "$context/runner"
safe_command python3 scripts/check_runner_assets.py config/runner-assets.sha256 "$context/runner"
safe_command docker build -f "$context/docker/zig/runner.Dockerfile" --build-arg "BASE_IMAGE=$base" "$@" "$context"
