#!/usr/bin/env bash
# preflight-requires: <rank0-work>/stdbench/decl-control/controller3.py
# preflight-requires: <rank0-work>/stdbench/decl-control/client3.py
# preflight-requires: <rank0-work>/stdbench/decl-control/reconciliation.py
# preflight-requires: <rank0-work>/stdbench/decl-control/rank_ops3.py
# preflight-requires: <rank0-work>/stdbench/decl-control/fence3.sh
set -euo pipefail
base=<rank0-work>/stdbench/decl-control
source /etc/spark/thermal.env
export GUARD_PAUSE_C GUARD_RESUME_C GUARD_KILL_C GUARD_KILL_HOLD_S
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAKEFLAGS=-j1
exec nice -n 19 ionice -c3 timeout 43200 python3 -B "$base/controller3.py" "$@"
