#!/usr/bin/env bash
# preflight-requires: <rank0-work>/stdbench/compute-dl/a2-continuation/turn.sh
# preflight-requires: <rank0-work>/stdbench/compute-dl/a2-continuation/cpu_gate.sh
# preflight-requires: <rank0-work>/stdbench/compute-dl/a2-continuation/protocol.py
set -euo pipefail
base=<rank0-work>/stdbench/compute-dl/a2-continuation
campaign=${1:?campaign.json}
turn=${2:?A2-plain}
[[ $# == 2 ]] || exit 2
# Command substitution returns only after the CPU guard AND producer have settled.
gate=$(bash "$base/cpu_gate.sh" python3 -B "$base/controller.py" "$campaign" "$turn" --protocol-check)
read -r unit seconds <<< "$gate"
[[ $unit == stdbench-compute-a2-20261106-v1-A2-plain ]] || exit 3
[[ $seconds =~ ^[0-9]+$ ]] && (( seconds >= 7200 )) || exit 3
exec systemd-run --user --unit="$unit" --service-type=exec --wait --pipe \
  -p IgnoreSIGPIPE=no -p MemoryMax=2G -p MemorySwapMax=0 -p KillMode=process -p TimeoutStopSec=7200 \
  bash "$base/turn.sh" "$campaign" "$turn"
