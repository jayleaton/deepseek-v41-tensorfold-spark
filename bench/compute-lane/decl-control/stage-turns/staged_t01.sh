#!/usr/bin/env bash
# P -> D -> L. Scheduling and queue append remain disabled until sealing.
# preflight-requires: <rank0-work>/stdbench/decl-control/T5-DECL-CONTROL-SPEC.md
set -euo pipefail
[[ $# == 0 ]] || exit 2
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec bash "$base/staged_launch.sh" t01
