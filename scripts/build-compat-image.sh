#!/usr/bin/env bash
# Layer the CUDA forward-compatibility driver (docker/zig/compat.Dockerfile) over an image scripts/build-runner-image.sh
# or scripts/build-native.sh built. scripts/cuda_compat.py takes the driver out of the image config/cuda-compat.release
# pins and checks it against config/cuda-compat.sha256 and config/cuda-compat.links in task-owned scratch before the
# build; tf-compat-entry runs ahead of the base entrypoint with the base CMD kept. Then, with no GPU: TF_CUDA_COMPAT=0
# and an image without a device both log the host driver, and arguments reach the command unchanged. Prints a JSON
# receipt.
{ set +x; } 2>/dev/null
safe_dir=${BASH_SOURCE[0]%/*}
[[ "$safe_dir" != "${BASH_SOURCE[0]}" ]] || safe_dir=.
source "$safe_dir/safe_shell.sh"
safe_main "${BASH_SOURCE[0]}" "$@"
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: build-compat-image.sh BASE_IMAGE TAG' >&2; exit 2; }
base=$1 tag=$2
cd "${safe_exec%/*}/.."
scratch=$(safe_capture path mktemp -d)
trap 'safe_command python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1])" "$scratch"' EXIT
base_id=$(safe_capture image_id docker image inspect -f '{{.Id}}' "$base")
safe_command python3 scripts/cuda_compat.py prepare "$base_id" "$scratch/context" >&2
# The classic builder takes the base by its local image ID, so the layer goes over exactly the inspected image.
safe_command env DOCKER_BUILDKIT=0 docker build --build-arg "BASE_IMAGE=$base_id" -t "$tag" "$scratch/context" >&2
safe_command python3 scripts/cuda_compat.py verify "$base_id" "$tag"
