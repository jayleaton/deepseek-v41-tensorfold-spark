#!/usr/bin/env bash
# preflight-requires: <rank0-work>/stdbench/compute-dl/a1-prep/turn.sh
# preflight-requires: <rank0-work>/stdbench/compute-dl/a1-prep/stage_control.py
set -euo pipefail
# Prospective launcher; deliberately has no queue/GPU mutation before A1 freeze.
base=<rank0-work>/stdbench/compute-dl/a1-prep
if [[ ${1:-} == --preflight-only && $# == 1 ]]; then
  exec python3 -B "$base/stage_control.py" --preflight-only
fi
# Do not create a unit, append ORDER, or delegate to the old D/L-only controller.
exec python3 -B "$base/stage_control.py"
# When the frozen A1 collector is bound, its service must set IgnoreSIGPIPE=no
# and TimeoutStopSec>=7200; never overwrite this script while it is running.
