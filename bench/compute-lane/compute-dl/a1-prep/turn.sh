#!/usr/bin/env bash
# preflight-requires: <rank0-work>/stdbench/compute-dl/a1-prep/stage_control.py
set -euo pipefail
base=<rank0-work>/stdbench/compute-dl/a1-prep
# Both direct entry and launcher entry fail closed before any GPU operation.
if [[ ${1:-} == --preflight-only && $# == 1 ]]; then
  exec python3 -B "$base/stage_control.py" --preflight-only
fi
exec python3 -B "$base/stage_control.py"
