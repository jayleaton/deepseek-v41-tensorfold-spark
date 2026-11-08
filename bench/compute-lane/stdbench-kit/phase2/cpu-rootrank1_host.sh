#!/usr/bin/env bash
# Read-only rank0-host queue monitor -> root rank1-host constrained CPU scope; never queues GPU work.
set -u
job=stdbench-cpu-$(date +%s)
produce_state() {
  while :; do
    if test -e <rank0-work>/queue/QUIET; then echo PAUSE
    elif test -s <rank0-work>/queue/ORDER; then echo BOOT
    elif ! awk '/MemAvailable:/ {exit !($2 >= 6291456)}' /proc/meminfo; then echo LOWMEM
    else echo RUN; fi
    sleep 1
  done
}
printf -v command '%q ' "$@"
produce_state | ssh -o BatchMode=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 root@<rank1-link-ip> "systemd-run --user --scope --unit=$job -p MemoryMax=2G -p MemorySwapMax=0 env STDBENCH_CPU_JOB=$job PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAKEFLAGS=-j1 nice -n 19 ionice -c3 timeout 7200 python3 <rank1-work>/stdbench/phase2/cpu_guard.py -- $command"
exit "${PIPESTATUS[1]}"
