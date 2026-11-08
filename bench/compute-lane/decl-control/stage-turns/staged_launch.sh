#!/usr/bin/env bash
# Staging only: no queue, lock, rank, engine or SSH operations in this wrapper.
# Existing literal rank0-host references for the parent's preflight; not executed here.
# preflight-requires: <rank0-work>/stdbench/phase2/generate_envs.py
# preflight-requires: <rank0-work>/stdbench/phase2/rank_ops.py
# preflight-requires: <rank0-work>/stdbench/phase2/references/base-arms.json
# preflight-requires: <rank0-work>/stdbench/phase2/references/base-r0.env
# preflight-requires: <rank0-work>/stdbench/phase2/references/base-r1.env
# preflight-requires: <rank0-work>/stdbench/decl-control/T5-DECL-CONTROL-SPEC.md
# preflight-requires: /etc/spark/thermal.env
set -euo pipefail
[[ $# == 1 && $1 =~ ^(t01|t02|t03|reserve)$ ]] || exit 2
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAKEFLAGS=-j1
# Gate runs before systemd-run or any dispatch. Missing seals return 78.
exec python3 -B "$base/staged_gate.py" "$1"
