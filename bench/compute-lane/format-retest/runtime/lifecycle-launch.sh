#!/usr/bin/env bash
# preflight-requires: <rank0-work>/stdbench/format-retest/runtime/cpu_gate.sh
# preflight-requires: <rank0-work>/stdbench/format-retest/runtime/lifecycle_probe.py
set -u
mode=${1:?success or failure}
[[ $mode == success || $mode == failure ]] || exit 2
base=<rank0-work>/stdbench/format-retest/runtime
# Intentionally inherit SIGPIPE ignore to reproduce the old producer-stall trigger.
# Actual test payload still runs in the real bounded --scope admission wrapper.
exec systemd-run --user --unit="stdbench-cpu-format-v1-lifecycle-$mode" --service-type=exec --wait --pipe \
  -p IgnoreSIGPIPE=yes -p MemoryMax=2G -p MemorySwapMax=0 -p TimeoutStopSec=7200 \
  nice -n 19 ionice -c3 timeout 7200 bash "$base/cpu_gate.sh" python3 -B "$base/lifecycle_probe.py" "$mode"
