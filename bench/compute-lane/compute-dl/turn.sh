#!/usr/bin/env bash
# preflight-requires: <rank0-work>/stdbench/compute-dl/controller.py
# preflight-requires: <rank0-work>/stdbench/compute-dl/protocol.py
# preflight-requires: <rank0-work>/stdbench/compute-dl/request_adapter.py
# preflight-requires: <rank0-work>/stdbench/compute-dl/rank_ops.py
# preflight-requires: <rank0-work>/stdbench/compute-dl/fence.sh
# preflight-requires: <rank0-work>/stdbench/decl-control/client3.py
# preflight-requires: <rank0-work>/stdbench/decl-control/reconciliation.py
# preflight-requires: <rank0-work>/stdbench/decl-control/generate_envs3.py
# preflight-requires: <rank0-work>/stdbench/phase2/cpu-rank0-host.sh
set -euo pipefail
base=<rank0-work>/stdbench/compute-dl
campaign=${1:?campaign.json}
turn=${2:?frozen turn name}
[[ $# == 2 ]] || exit 2
gate=$(bash <rank0-work>/stdbench/phase2/cpu-rank0-host.sh \
  python3 -B "$base/controller.py" "$campaign" "$turn" --protocol-check)
read -r unit seconds <<< "$gate"
[[ $unit =~ ^stdbench-compute-dl-[A-Za-z0-9_.-]+$ && $seconds =~ ^[0-9]+$ ]] || exit 3
(( seconds >= 7200 )) || exit 3
source /etc/spark/thermal.env
export GUARD_PAUSE_C GUARD_RESUME_C GUARD_KILL_C GUARD_KILL_HOLD_S
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAKEFLAGS=-j1
exec nice -n 19 ionice -c3 timeout "$seconds" python3 -B "$base/controller.py" "$campaign" "$turn"
