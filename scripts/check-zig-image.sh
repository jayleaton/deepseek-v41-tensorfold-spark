#!/usr/bin/env bash
# Run the image's shared-library proof through the same host credential boundary.
{ set +x; } 2>/dev/null
safe_dir=${BASH_SOURCE[0]%/*}
[[ "$safe_dir" != "${BASH_SOURCE[0]}" ]] || safe_dir=.
source "$safe_dir/safe_shell.sh"
safe_main "${BASH_SOURCE[0]}" "$@"
set -euo pipefail
image=${1:?usage: check-zig-image.sh IMAGE}
[[ "$image" != --inside ]] || { echo 'run the image check from the host with IMAGE' >&2; exit 78; }
platform=$(safe_command docker image inspect -f '{{.Os}}/{{.Architecture}}' "$image")
safe_command docker run --rm --platform "$platform" --entrypoint bash     -e "CUBLAS_VERSION=${CUBLAS_VERSION:-13.6.0.2}" "$image" /opt/tensorfold/share/libproof.sh
