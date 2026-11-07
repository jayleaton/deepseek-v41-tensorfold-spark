#!/usr/bin/env bash
# rank0-host's small verification only; baseline init.scope is not a workload scope.
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
produce_state | systemd-run --user --scope --unit="$job" -p MemoryMax=2G -p MemorySwapMax=0 \
  env STDBENCH_CPU_JOB="$job" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAKEFLAGS=-j1 \
  nice -n 19 ionice -c3 timeout 7200 python3 <rank0-work>/stdbench/phase2/cpu_guard.py -- "$@"
exit "${PIPESTATUS[1]}"
