#!/usr/bin/env bash
# preflight-requires: <rank0-work>/stdbench/phase2/controller.py
# preflight-requires: <rank0-work>/stdbench/phase2/rank_ops.py
# preflight-requires: <rank0-work>/stdbench/phase2/client.py
# preflight-requires: <rank0-work>/stdbench/phase2/jail.py
# preflight-requires: <rank0-work>/stdbench/phase2/fence.sh
set -euo pipefail
base=<rank0-work>/stdbench/phase2
source /etc/spark/thermal.env
export GUARD_PAUSE_C GUARD_RESUME_C GUARD_KILL_C GUARD_KILL_HOLD_S
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAKEFLAGS=-j1
exec nice -n 19 ionice -c3 timeout 43200 python3 -B "$base/controller.py" "$@"
