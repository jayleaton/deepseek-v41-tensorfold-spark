#!/usr/bin/env bash
# preflight-requires: <rank0-work>/stdbench/decl-control/turn3.sh
set -euo pipefail
campaign=${1:?campaign.json}
turn=${2:?t01..t03}
[[ $turn =~ ^t0[1-3]$ ]] || exit 2
unit=stdbench-decl-a1-20261105-v1-$turn
exec systemd-run --user --unit="$unit" --service-type=exec --wait --pipe \
  -p MemoryMax=2G -p MemorySwapMax=0 -p KillMode=process -p TimeoutStopSec=7200 \
  bash <rank0-work>/stdbench/decl-control/turn3.sh "$campaign" "$turn"
