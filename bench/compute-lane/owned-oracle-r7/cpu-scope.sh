#!/usr/bin/env bash
# Source from host entry points: the caller owns one bounded, QUIET-aware CPU scope.
oracle_require_scope() {
  local hierarchy controllers path memory swap
  if [[ -n "${QUIET_FILE:-}" && "$QUIET_FILE" != /* ]]; then
    echo 'QUIET_FILE must be an absolute path when set' >&2
    return 2
  fi
  while IFS=: read -r hierarchy controllers path; do
    [[ "$hierarchy" == 0 && -z "$controllers" ]] || continue
    while [[ -n "$path" ]]; do
      if [[ -r "/sys/fs/cgroup$path/memory.max" && -r "/sys/fs/cgroup$path/memory.swap.max" ]]; then
        read -r memory < "/sys/fs/cgroup$path/memory.max"
        read -r swap < "/sys/fs/cgroup$path/memory.swap.max"
        if [[ "$memory" =~ ^[0-9]+$ && "$memory" -le 2147483648 && "$swap" == 0 ]]; then
          return 0
        fi
      fi
      [[ "$path" != / ]] || break
      path=${path%/*}
      [[ -n "$path" ]] || path=/
    done
  done < /proc/self/cgroup
  echo 'oracle requires a caller-owned cgroup-v2 CPU scope: MemoryMax=2G MemorySwapMax=0 (see README)' >&2
  return 2
}
