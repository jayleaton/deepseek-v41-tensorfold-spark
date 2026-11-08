#!/usr/bin/env bash
# Exact private chain, only while this owned GPU turn holds both locks.
set -euo pipefail
chain=STDBENCH_COMPUTE_DL
case ${1:?on|off} in
on)
  group=$(sed -n 's/^0:://p' /proc/self/cgroup)
  [[ $group != / && $group != *init.scope* ]] || { echo 'dedicated turn cgroup required'; exit 3; }
  for table in iptables ip6tables; do
    if sudo -n "$table" -S "$chain" >/dev/null 2>&1; then echo 'private chain already exists; refuse'; exit 3; fi
  done
  created=()
  cleanup_partial() {
    for owned_table in "${created[@]}"; do
      sudo -n "$owned_table" -D OUTPUT -o lo -p tcp --dport 8000 --syn -j "$chain" 2>/dev/null || true
      sudo -n "$owned_table" -F "$chain" 2>/dev/null || true
      sudo -n "$owned_table" -X "$chain" 2>/dev/null || true
    done
  }
  trap cleanup_partial ERR
  for table in iptables ip6tables; do
    sudo -n "$table" -N "$chain"
    created+=("$table")
    sudo -n "$table" -A "$chain" -m cgroup --path "${group#/}" -j RETURN
    sudo -n "$table" -A "$chain" -p tcp -j REJECT --reject-with tcp-reset
    sudo -n "$table" -I OUTPUT -o lo -p tcp --dport 8000 --syn -j "$chain"
  done
  trap - ERR
  echo 'FENCE_ON'
  ;;
off)
  for table in iptables ip6tables; do
    if sudo -n "$table" -S "$chain" >/dev/null 2>&1; then
      sudo -n "$table" -D OUTPUT -o lo -p tcp --dport 8000 --syn -j "$chain" || true
      sudo -n "$table" -F "$chain"
      sudo -n "$table" -X "$chain"
    fi
  done
  echo 'FENCE_OFF'
  ;;
*) exit 2;;
esac
