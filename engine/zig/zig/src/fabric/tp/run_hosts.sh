#!/usr/bin/env bash
# One rank of the two-host TP=2 gate (two DGX Sparks over CX7). Start it on both hosts at about the same time
# (each step's bootstrap waits up to TF_DSV41_PLAN_TIMEOUT for the other host):
#   head:   run_hosts.sh 0 <rank-0-link-address> ./tf-tp-test OUTDIR
#   worker: run_hosts.sh 1 <rank-0-link-address> ./tf-tp-test OUTDIR
# Steps (same order on both hosts, one port each): roce-host (RoCE wire + proxy, no GPU), suite over RoCE, suite over
# NCCL, then fail-fast: each backend x (own failure | SIGKILL) x victim rank. docs/DSV41-ZIG-TP.md has the expected
# lines. Knobs from the environment pass through (TF_TP_ROCE_HCA, TF_TP_ROCE_CPU, NCCL_SOCKET_IFNAME, ...).
set -u
RANK=${1:?rank}; MASTER=${2:?master address}; BIN=${3:?tf-tp-test}; OUT=${4:?outdir}
mkdir -p "$OUT"
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1}
ulimit -l unlimited 2> /dev/null || echo "warning: memlock is $(ulimit -l) KiB: ibv_reg_mr of the RoCE region may fail" >&2
{ hostname; nvidia-smi -L; ibv_devices 2>&1; ulimit -l; } > "$OUT/host.txt" 2>&1
PORT=29700
step() { # NAME BACKEND ARGS...: this host's rank; log OUT/NAME.log ending with an EXIT line
    local name=$1 backend=$2; shift 2
    PORT=$((PORT + 10))
    ( TF_TP_RANK=$RANK TF_TP_MASTER=$MASTER TF_TP_PORT=$PORT TF_COMM_BACKEND=$backend TF_TP_ROCE_FALLBACK=error \
        timeout 900 "$BIN" "$@"; echo "EXIT rank $RANK rc $? at $(date +%s.%N)" ) > "$OUT/$name.log" 2>&1
    echo "$name: $(grep -h '^SUMMARY\|^PASS roce-host\|^INJECT\|^EXIT' "$OUT/$name.log" | tr '\n' ' ')"
}
step roce-host roce roce-host
step suite-roce roce suite
step suite-nccl nccl suite
for backend in roce nccl; do
    for mode in fatal kill; do
        for victim in 0 1; do step "ff-$backend-$mode-$victim" "$backend" failfast "$victim" "$mode"; done
    done
done
