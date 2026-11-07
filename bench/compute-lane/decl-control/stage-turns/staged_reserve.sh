#!/usr/bin/env bash
# Fourth turn only: fixed unmeasured units after interruption; no fresh cohort.
# Parent/future sealed controller must enforce attempt count and no replacement.
# preflight-requires: <rank0-work>/stdbench/decl-control/T5-DECL-CONTROL-SPEC.md
set -euo pipefail
[[ $# == 0 ]] || exit 2
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec bash "$base/staged_launch.sh" reserve
