#!/usr/bin/env bash
# preflight-requires: <rank0-work>/stdbench/compute-dl/turn.sh
# preflight-requires: <rank0-work>/stdbench/compute-dl/protocol.py
# preflight-requires: <rank0-work>/stdbench/phase2/cpu-rank0-host.sh
set -euo pipefail
base=<rank0-work>/stdbench/compute-dl
campaign=${1:?campaign.json}
turn=${2:?frozen turn name}
[[ $# == 2 ]] || exit 2
# The existing CPU admission guard supplies the 2G/noswap/nice/ionice scope.
# Refusal happens before the turn service or any queue/GPU mutation.
gate=$(bash <rank0-work>/stdbench/phase2/cpu-rank0-host.sh \
  python3 -B "$base/controller.py" "$campaign" "$turn" --protocol-check)
read -r unit seconds <<< "$gate"
[[ $unit =~ ^stdbench-compute-dl-[A-Za-z0-9_.-]+$ && $seconds =~ ^[0-9]+$ ]] || exit 3
(( seconds >= 7200 )) || exit 3
exec systemd-run --user --unit="$unit" --service-type=exec --wait --pipe \
  -p MemoryMax=2G -p MemorySwapMax=0 -p KillMode=process -p TimeoutStopSec=7200 \
  bash "$base/turn.sh" "$campaign" "$turn"
