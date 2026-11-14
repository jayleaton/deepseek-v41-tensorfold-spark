#!/usr/bin/env bash
# Layer the CUDA 13.3 forward-compatibility driver (docker/compat) over an image scripts/build-runner-image.sh built.
# The driver files are copied out of COMPAT_IMAGE (default NVIDIA's NGC PyTorch 26.07 container, by digest; pulled if
# missing) from COMPAT_PATH (default /usr/local/cuda-13.3/compat/lib.real) and must match docker/compat/compat.sha256
# and compat.links exactly, so any image holding the same files gives the same layer (COMPAT_PATH=/opt/tensorfold/compat
# reads them back out of an image this script built). The finished image starts its command through tf-compat-entry;
# with no GPU visible the entry must report the host driver, which the build checks.
#   scripts/build-compat-image.sh RUNNER_IMAGE [docker build arguments, e.g. -t TAG]
{ set +x; } 2>/dev/null
safe_dir=${BASH_SOURCE[0]%/*}
[[ "$safe_dir" != "${BASH_SOURCE[0]}" ]] || safe_dir=.
source "$safe_dir/safe_shell.sh"
safe_main "${BASH_SOURCE[0]}" "$@"
set -euo pipefail
usage='usage: build-compat-image.sh RUNNER_IMAGE [docker build arguments, e.g. -t TAG]'
[[ $# -ge 1 && $1 != -* ]] || { echo "$usage" >&2; exit 2; }
base=$1
shift
compat_image=${COMPAT_IMAGE:-nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c}
compat_path=${COMPAT_PATH:-/usr/local/cuda-13.3/compat/lib.real}
[[ "$compat_path" == /* ]] || { echo 'COMPAT_PATH must be absolute' >&2; exit 2; }
cd "${safe_exec%/*}/.."
scratch=$(safe_capture path mktemp -d)
trap 'safe_command python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1])" "$scratch"' EXIT
context="$scratch/context"
source_container="tf-compat-source-$$"
safe_command mkdir -p "$context"
safe_command docker image inspect "$compat_image" > /dev/null 2>&1 || safe_command docker pull "$compat_image"
safe_command docker create --name "$source_container" "$compat_image" /bin/true > /dev/null
copied=0
safe_command docker cp "$source_container:$compat_path" "$context/compat" > /dev/null && copied=1
safe_command docker rm "$source_container" > /dev/null
[[ "$copied" == 1 ]] || { echo "no $compat_path in $compat_image" >&2; exit 1; }
# exactly the pinned files and links, nothing else
safe_command python3 scripts/check_compat_files.py files "$context/compat" docker/compat/compat.sha256 docker/compat/compat.links
safe_command cp docker/compat/Dockerfile docker/compat/tf-compat-entry "$context/"
safe_command chmod 755 "$context/tf-compat-entry"
safe_command docker build -f "$context/Dockerfile" --build-arg "BASE_IMAGE=$base" --iidfile "$scratch/iid" "$@" "$context"
# docker writes the id with no trailing newline, which read reports as end of file
read -r iid < "$scratch/iid" || [[ -n "${iid:-}" ]]
[[ "$iid" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo 'docker build wrote no image id' >&2; exit 1; }
# with no GPU the entry falls back to the host driver and says so
safe_command docker run --rm --network none --runtime=runc -e NVIDIA_VISIBLE_DEVICES=void \
    --memory 1g --memory-swap 1g "$iid" /bin/true 2> "$scratch/entry.txt"
safe_command python3 scripts/check_compat_files.py entry "$scratch/entry.txt"
echo "compat image $iid"
