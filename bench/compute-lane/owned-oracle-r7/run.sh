#!/usr/bin/env bash
# CPU only. The caller supplies the bounded scope and queue/QUIET admission policy.
# Usage: run.sh OUT RECEIPT...; ORACLE_IMAGE selects a public CPython 3.12 image.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/cpu-scope.sh"
oracle_require_scope
[[ $# -ge 2 ]] || { echo 'usage: run.sh OUT RECEIPT...' >&2; exit 2; }
out=$(realpath -m "$1"); shift
mkdir -p "$out"
img=${ORACLE_IMAGE:-python:3.12.3-slim-bookworm@sha256:afc139a0a640942491ec481ad8dda10f2c5b753f5c969393b12480155fe15a63}
quiet=${QUIET_FILE:-}
job=${STDBENCH_CPU_JOB:-oracle-$$}
name=owned-oracle-$$-$RANDOM
pid=
cleanup() {
  docker rm -f "$name" >/dev/null 2>&1 || true
  if [[ -n "$pid" ]]; then wait "$pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
idle() { while [[ -n "$quiet" && -e "$quiet" ]]; do sleep 1; done; }
idle
start=$(date +%s)
python3 -I "$here/oracle.py" extract "$out" "$@"
idle
# No implicit pull: fetch/verify the selected public image under the caller's CPU protocol first.
docker image inspect "$img" >/dev/null 2>&1 || { echo 'ORACLE_IMAGE is not loaded; pull the documented public image first' >&2; exit 2; }
timeout 7200 docker run --rm --pull=never --runtime=runc --name "$name" --label "stdbench.cpu_job=$job" \
  --network none --read-only --security-opt no-new-privileges --cap-drop ALL \
  --tmpfs /tmp:rw,nosuid,nodev,size=256m --memory 2g --memory-swap 2g --cpus 1 --pids-limit 256 \
  -u "$(id -u):$(id -g)" -v "$here:/tool:ro" -v "$out:/out" -e HOME=/tmp -e NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint /usr/bin/nice "$img" -n 19 python3 -I /tool/oracle.py evaluate /out &
pid=$!
paused=0
while kill -0 "$pid" 2>/dev/null; do
  if [[ -n "$quiet" && -e "$quiet" && "$paused" == 0 ]]; then
    docker pause "$name" >/dev/null 2>&1 && paused=1 || true
  elif [[ ( -z "$quiet" || ! -e "$quiet" ) && "$paused" == 1 ]]; then
    docker unpause "$name" >/dev/null 2>&1 && paused=0 || true
  fi
  sleep 0.1
done
rc=0
wait "$pid" || rc=$?
pid=
[[ "$rc" == 0 ]] || { echo "evaluate failed rc=$rc" >&2; exit 1; }
idle
python3 -I "$here/oracle.py" report "$out"
echo "owned-oracle: $(( $(date +%s) - start )) s; report $out/report.md"
