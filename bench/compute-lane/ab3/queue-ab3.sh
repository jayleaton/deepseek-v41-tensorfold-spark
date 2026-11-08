#!/usr/bin/env bash
# Queues the six frozen AB3 turns on rank0-host, each behind the previous (T4_AFTER), pinned to r-3bf584af1c54.
set -euo pipefail
B=<rank0-work>/ab3-20261104
P="3bf584af1c5407006f154e0c70d05c6f495d0487 sha256:26c0b4107869671656476006743be5288dc48c8ee1f5c5b3879bd606fba59535 sha256:1cfb03ebae4f58de4ef61be0bd8527e7d88dd4c046bfce98844bc53305d37292 imr-3bf584af1c54"
prev=think
for n in 1 2 3 4 5 6; do
  t=t0$n
  d=$B/$t
  b=$(( (n + 1) / 2 ))
  cd "$d"
  T4_QUEUE=ab3-$t T4_AFTER=$prev T4_CONFIGS=ab3-$t-configs.json T4_TASKS=$d/ab3-b$b-units.jsonl \
    T4_PIN="$P" IMR_SHA=3bf584af1c5407006f154e0c70d05c6f495d0487 setsid -f bash window.sh >> window.log 2>&1 < /dev/null
  sleep 15
  prev=ab3-$t
done
cat <rank0-work>/queue/ORDER
for n in 1 2 3 4 5 6; do echo "== t0$n"; tail -2 $B/t0$n/window.log; done
