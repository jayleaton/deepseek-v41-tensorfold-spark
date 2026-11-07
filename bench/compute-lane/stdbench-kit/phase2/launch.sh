#!/usr/bin/env bash
# One foreground service per turn. G's intentional detached serving controller survives the completed turn.
# preflight-requires: <rank0-work>/stdbench/phase2/turn.sh
set -euo pipefail
campaign=${1:?campaign.json}
turn=${2:?pilot|t01..t06}
[[ $turn =~ ^(pilot|t0[1-6])$ ]] || exit 2
unit=stdbench-20261105-v2-$turn
exec systemd-run --user --unit="$unit" --service-type=exec --wait --pipe \
  -p MemoryMax=2G -p MemorySwapMax=0 -p KillMode=process -p TimeoutStopSec=7200 \
  bash <rank0-work>/stdbench/phase2/turn.sh "$campaign" "$turn"
