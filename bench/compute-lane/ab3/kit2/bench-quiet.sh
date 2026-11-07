#!/bin/bash
# On rank0-host: pause (or restore) the background apt sources for a bench window: <vendor-dashboard>.service on both
# nodes (hourly apt at :13) and rank1-host's apt-daily.timer. pause records each unit's prior active state in
# <rank0-work>/imr-t4/bench-quiet.state; restore starts again only what was active before.
# Usage: bench-quiet.sh pause|restore|status
set -euo pipefail
state=<rank0-work>/imr-t4/bench-quiet.state
n2() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 root@<rank1-link-ip> "$@"; }
case $1 in
pause)
  [[ ! -e $state ]] || { echo "already paused (state file present)"; exit 2; }
  {
    echo "rank0-host <vendor-dashboard>.service $(systemctl is-active <vendor-dashboard>.service || true)"
    echo "rank1-host <vendor-dashboard>.service $(n2 systemctl is-active <vendor-dashboard>.service || true)"
    echo "rank1-host apt-daily.timer $(n2 systemctl is-active apt-daily.timer || true)"
  } > "$state"
  sudo -n systemctl stop <vendor-dashboard>.service
  n2 systemctl stop <vendor-dashboard>.service apt-daily.timer
  echo "paused $(date -u +%FT%TZ)"; cat "$state" ;;
restore)
  while read -r node unit was; do
    [[ $was == active ]] || continue
    if [[ $node == rank0-host ]]; then sudo -n systemctl start "$unit"; else n2 systemctl start "$unit"; fi
  done < "$state"
  mv "$state" "$state.restored-$(date -u +%Y%m%dT%H%M%SZ)"
  echo "restored $(date -u +%FT%TZ)" ;;
status) ;;
*) echo "usage: $0 pause|restore|status"; exit 2 ;;
esac
echo "rank0-host <vendor-dashboard>-admin: $(systemctl is-active <vendor-dashboard>.service || true)"
echo "rank1-host <vendor-dashboard>-admin: $(n2 systemctl is-active <vendor-dashboard>.service || true), apt-daily.timer: $(n2 systemctl is-active apt-daily.timer || true)"
