#!/usr/bin/env bash
# Optional four-Spark keeper (cron on the head): start after a reboot, restart after 3 failed /health checks in a row.
#   @reboot sleep 240 && bash <repo>/scripts/keeper4.sh boot
#   */2 * * * * bash <repo>/scripts/keeper4.sh
# Stands down while $STATE_DIR/maintenance was touched in the last hour (benchmarks, upgrades), while another start
# runs, and while a container named in KEEPER_CONFLICTS runs (another model on the same GPUs).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STATE_DIR="${STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/dsv41-tf4}"; mkdir -p "$STATE_DIR"
PORT="${PORT:-$(grep -hE '^PORT=' config/prod.env config/tp4.env 2>/dev/null | tail -1 | cut -d= -f2)}"; PORT="${PORT:-8000}"
LOG=$STATE_DIR/keeper.log; F=$STATE_DIR/keeper.fails
log() { echo "$(date -Is) $*" >> "$LOG"; }
[[ -f $STATE_DIR/maintenance && $(( $(date +%s) - $(stat -c %Y "$STATE_DIR/maintenance") )) -lt 3600 ]] && exit 0
exec 9>"$STATE_DIR/keeper.lock"; flock -n 9 || exit 0
for c in ${KEEPER_CONFLICTS:-}; do
    [[ "$(docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" == true ]] && { log "$c running: standing down"; exit 0; }
done
if curl -sf -m 15 "localhost:$PORT/health" >/dev/null; then echo 0 > "$F"; exit 0; fi
n=$(( $(cat "$F" 2>/dev/null || echo 0) + 1 )); echo $n > "$F"
[[ "${1:-}" == boot ]] && n=3
log "health check failed ($n)"
(( n >= 3 )) || exit 0
log "restarting"; echo 0 > "$F"
bash scripts/serve4.sh stop >> "$LOG" 2>&1; bash scripts/serve4.sh start >> "$LOG" 2>&1 && log "started" || log "start FAILED"
