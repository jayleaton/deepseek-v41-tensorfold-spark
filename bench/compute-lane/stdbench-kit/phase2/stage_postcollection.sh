#!/usr/bin/env bash
# Offline data/label staging after all GPU turns settle; no endpoint/queue/lock.
# preflight-requires: <rank0-work>/stdbench/phase2/review_disputes.py
# preflight-requires: <rank0-work>/stdbench/phase2/postcollect.py
# preflight-requires: <rank0-work>/stdbench/phase2/cpu-rootrank1_host.sh
set -euo pipefail
base=<rank0-work>/stdbench/phase2
peer=root@<rank1-link-ip>
remote=<rank1-work>/stdbench/phase2
ssh "$peer" "test ! -e '$remote/post-data-v1'; mkdir '$remote/post-data-v1'"
for turn in pilot t01 t02 t03 t04 t05 t06; do
  for arm in plain lane; do
    version=v2
    [[ $turn != pilot || $arm != plain ]] || version=v1
    run=stdbench-20261105-$version-$turn
    target=$remote/post-data-v1/$run/$turn-$arm
    ssh "$peer" "mkdir -p '$target'"
    ssh "$peer" "mkdir -p '$remote/post-proof-v1/rank0'"
    scp -r "$base/stdbench-20261105-$version/proof/$turn-$arm" "$peer:$remote/post-proof-v1/rank0/"
    scp "$base/runs/$run/steps.jsonl" "$peer:$remote/post-data-v1/$run/steps.jsonl"
    if test -f "$base/runs/$run/G-restore.log"; then scp "$base/runs/$run/G-restore.log" "$peer:$remote/post-data-v1/$run/G-restore.log"; fi
    for name in responses.jsonl diagnostics.jsonl safety.jsonl rank0.log; do
      scp "$base/runs/$run/$turn-$arm/$name" "$peer:$target/$name"
    done
  done
done
scp "$base/review_disputes.py" "$base/postcollect.py" "$peer:$remote/"
scp "$base/math-review-v1/reviewer-a.json" "$base/math-review-v1/reviewer-b.json" "$base/math-review-v1/labels-premerge.sha256" "$peer:$remote/math-review-v1/"
bash "$base/cpu-rootrank1_host.sh" bash -c "python3 '$remote/review_disputes.py' && python3 '$remote/postcollect.py' --data-root '$remote/post-data-v1' --out '$remote/post-v1'"
