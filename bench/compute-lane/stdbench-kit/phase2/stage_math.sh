#!/usr/bin/env bash
# CPU/file staging only, after G restore; no queue, lock or endpoint operations.
# preflight-requires: <rank0-work>/stdbench/phase2/seal_math_early.py
# preflight-requires: <rank0-work>/stdbench/phase2/cpu-rootrank1_host.sh
set -euo pipefail
base=<rank0-work>/stdbench/phase2
peer=root@<rank1-link-ip>
remote=<rank1-work>/stdbench/phase2
ssh "$peer" "test ! -e '$remote/math-data-v1'; mkdir '$remote/math-data-v1'"
for turn in pilot t02 t05; do
  for arm in plain lane; do
    version=v2
    [[ $turn != pilot || $arm != plain ]] || version=v1
    source=$base/runs/stdbench-20261105-$version-$turn/$turn-$arm/responses.jsonl
    target=$remote/math-data-v1/$turn-$arm
    ssh "$peer" "mkdir '$target'"
    scp "$source" "$peer:$target/responses.jsonl"
  done
done
scp "$base/seal_math_early.py" "$base/postcollect.py" "$peer:$remote/"
bash "$base/cpu-rootrank1_host.sh" python3 "$remote/seal_math_early.py"
