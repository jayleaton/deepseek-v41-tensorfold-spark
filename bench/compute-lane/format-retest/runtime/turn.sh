#!/usr/bin/env bash
# preflight-requires: <rank0-work>/stdbench/format-retest/runtime/controller.py
# preflight-requires: <rank0-work>/stdbench/format-retest/runtime/protocol.py
# preflight-requires: <rank0-work>/stdbench/format-retest/runtime/request_adapter.py
# preflight-requires: <rank0-work>/stdbench/format-retest/runtime/rank_ops.py
# preflight-requires: <rank0-work>/stdbench/format-retest/runtime/clipping.py
# preflight-requires: <rank0-work>/stdbench/format-retest/runtime/budget.py
# preflight-requires: <rank0-work>/stdbench/format-retest/runtime/cpu_gate.sh
# preflight-requires: <rank0-work>/stdbench/format-retest/runtime/fence.sh
# preflight-requires: <rank0-work>/stdbench/compute-dl/a1-prep/reference_policy.py
# preflight-requires: <rank0-work>/stdbench/decl-control/client3.py
# preflight-requires: <rank0-work>/stdbench/decl-control/reconciliation.py
# preflight-requires: <rank0-work>/stdbench/decl-control/generate_envs3.py
# preflight-requires: <rank0-work>/stdbench/format-retest/runtime/coordination.py
set -euo pipefail
base=<rank0-work>/stdbench/format-retest/runtime
campaign=${1:?campaign.json}
turn=${2:?A1 turn}
[[ $# == 2 ]] || exit 2
gate=$(bash "$base/cpu_gate.sh" python3 -B "$base/controller.py" "$campaign" "$turn" --protocol-check)
read -r unit seconds <<< "$gate"
[[ $unit == stdbench-format-20261107-v1-format ]] || exit 3
[[ $seconds =~ ^[0-9]+$ ]] && (( seconds >= 7200 )) || exit 3
source /etc/spark/thermal.env
export GUARD_PAUSE_C GUARD_RESUME_C GUARD_KILL_C GUARD_KILL_HOLD_S
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAKEFLAGS=-j1
exec nice -n 19 ionice -c3 timeout "$seconds" python3 -B "$base/controller.py" "$campaign" "$turn"
# preflight-requires: <rank0-work>/stdbench/format-retest/runtime/manifest.py
# preflight-requires: <rank0-work>/stdbench/format-retest/runtime/qualification.py
# preflight-requires: <rank0-work>/stdbench/format-retest/runtime/prep_bridge.py
