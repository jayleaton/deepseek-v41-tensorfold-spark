#!/usr/bin/env bash
# preflight-requires: <rank0-work>/stdbench/phase2/fence.sh
# preflight-requires: <rank0-work>/r5/ecal/g-restore.sh
set -euo pipefail
queue=<rank0-work>/queue/ORDER
[[ $(head -1 "$queue") == stdbench-20261105-v1-pilot ]] || exit 3
for pid in 2096514 2098888; do
  if test -e /proc/$pid; then echo "owned PID still pending: $pid"; exit 3; fi
done
for rank in 0 1; do
  name=stdbench-20261105-v1-pilot-plain-r$rank
  if [[ $rank == 0 ]]; then
    if docker inspect "$name" >/dev/null 2>/tmp/stdbench-recovery-r0.err; then exit 3; fi
    grep -qi 'no such object\|no such container' /tmp/stdbench-recovery-r0.err || exit 3
  else
    ssh root@<rank1-link-ip> "if docker inspect '$name' >/dev/null 2>/tmp/stdbench-recovery-r1.err; then exit 3; fi; grep -qi 'no such object\|no such container' /tmp/stdbench-recovery-r1.err"
  fi
done
bash <rank0-work>/stdbench/phase2/fence.sh off
! grep -qx 'engine-Gr' "$queue" || exit 3
sed -i '1a engine-Gr' "$queue"
sed -i '1d' "$queue"
echo 'OWNED_RANKS_ABSENT_G_RESTORE_AT_HEAD'
bash <rank0-work>/r5/ecal/g-restore.sh
