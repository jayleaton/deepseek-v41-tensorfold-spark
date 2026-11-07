#!/usr/bin/env bash
# rank0-host: keep a W9 window's server to its driver. The ranks serve on the host network at localhost:PORT, so any local
# client (an ssh tunnel's sshd, a local agent) reaches it; this fence lets only sockets of the driver's own cgroup
# (systemd scope) connect to PORT on loopback and resets everyone else, IPv4 and IPv6. Off restores open access (G).
# Usage: w9-fence.sh on [CGROUP_PATH] [PORT] | off [PORT] | status   (CGROUP_PATH default: the caller's own scope)
set -uo pipefail
cmd=${1:?on|off|status}
ipt() { sudo -n iptables "$@" && sudo -n ip6tables "$@"; }
case $cmd in
on)
  path=${2:-$(sed -n 's/^0:://p' /proc/self/cgroup)}; port=${3:-8000}
  path=${path#/}
  "$0" off "$port" > /dev/null 2>&1
  ipt -N T4W9 && ipt -A T4W9 -m cgroup --path "$path" -j RETURN && ipt -A T4W9 -p tcp -j REJECT --reject-with tcp-reset &&
    ipt -I OUTPUT -o lo -p tcp --dport "$port" --syn -j T4W9 || { echo "fence FAILED"; "$0" off "$port"; exit 1; }
  echo "fence on: port $port only for cgroup $path" ;;
off)
  port=${2:-8000}
  for t in iptables ip6tables; do
    while sudo -n $t -D OUTPUT -o lo -p tcp --dport "$port" --syn -j T4W9 2> /dev/null; do :; done
    sudo -n $t -F T4W9 2> /dev/null; sudo -n $t -X T4W9 2> /dev/null
  done
  echo "fence off: port $port" ;;
status)
  sudo -n iptables -S T4W9 2> /dev/null; sudo -n ip6tables -S T4W9 2> /dev/null
  sudo -n iptables -S OUTPUT | grep T4W9 ;;
esac
