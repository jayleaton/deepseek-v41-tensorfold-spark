#!/usr/bin/env bash
# Four-Spark (TP=4) launcher for the deepseek_v41 TensorFold family (patches/0003). Run on the head (rank 0).
#
#   serve4.sh ship          docker save the image here | docker load on the three workers (in parallel)
#   serve4.sh prebuild      the CUDA extensions into CACHE_VOL on all four nodes (nothing else on the GPUs)
#   serve4.sh start         drop caches, memory gate, ranks 3, 2, 1 then 0, wait for /v1/models, slot check
#   serve4.sh stop | status | logs [R] | args [R]
#   serve4.sh run MODULE [ARGS...]   an engine module on all four ranks instead of the server (benchmarks)
#
# Config: config/prod.env (the 2-Spark knobs: every TF_DSV41_* lever, names, ports) and then config/tp4.env (the
# four-Spark overrides: image, workers, link, paths; cp config/tp4.env.example config/tp4.env). Every node uses the
# same host paths (MODEL, ENGRAM, PREPARED, STATE). See docs/FOUR_SPARKS.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONFIG="${CONFIG:-config/prod.env}"
CONFIG4="${CONFIG4:-config/tp4.env}"
[[ -f "$CONFIG" && -f "$CONFIG4" ]] || { echo "[tf4] need $CONFIG and $CONFIG4 (docs/FOUR_SPARKS.md)" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$CONFIG"
# shellcheck disable=SC1090
source "$CONFIG4"
set +a
WORLD=4
read -r -a WORKERS <<< "${WORKERS:?set WORKERS (ssh targets of ranks 1, 2, 3) in $CONFIG4}"
NAME="${NAME:-dsv41-tf4}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
BASE="http://127.0.0.1:$PORT"
PARALLEL="${PARALLEL:-4}"
MASTER_PORT="${MASTER_PORT:-29571}"
CACHE_VOL="${CACHE_VOL:-dsv41-tf-cache}"
MEM_GATE_GIB="${MEM_GATE_GIB:-104}"
MEM_GATE_TIMEOUT="${MEM_GATE_TIMEOUT:-600}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
MODEL="${MODEL:?}" ENGRAM="${ENGRAM:?}" PREPARED="${PREPARED:?}" STATE="${STATE:?}"
log() { echo "[tf4] $(date +%T) $*"; }
on() { # on RANK CMD...: run on that rank's node
    local r=$1; shift
    if [[ "$r" == 0 ]]; then bash -c "$*"; else ssh -n -o BatchMode=yes -o ConnectTimeout=10 "${WORKERS[$((r-1))]}" "$*"; fi
}

container_args() {
    local a="--gpus all --ipc=host --network host --device /dev/infiniband --ulimit memlock=-1"
    a+=" --cap-add IPC_LOCK --log-opt max-size=200m --log-opt max-file=3"
    a+=" -v $MODEL:/model:ro -v $ENGRAM:/engram:ro -v $PREPARED:/prepared"
    a+=" -v $STATE:/state -v $STATE/sessions:/sessions -v $CACHE_VOL:/cache -e PYTHONDONTWRITEBYTECODE=1"
    a+=" -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton"
    a+=" -e CUDA_CACHE_PATH=/cache/nv/ComputeCache -e TF_DSV41_IMAGE_ID=${IMAGE_ID:-$IMAGE}"
    a+=" $(env | grep -E '^(TF_DSV41_[A-Z0-9_]+|GLM53_TF_[A-Z0-9_]+|(MI)?MALLOC_[A-Z_]+|NCCL_[A-Z_]+)=' \
          | grep -vE '^TF_DSV41_(LAUNCH_T0|IMAGE_ID)=' | sort | sed 's/^/-e /' | tr '\n' ' ')"
    a+=" -e GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    echo "$a"
}

run_args() {
    local r=$1 a
    a="--name $NAME-r$r -d $(container_args) --entrypoint python $IMAGE -m tensorfold serve /model --tp $WORLD"
    a+=" --rank $r --master $HEAD_IP --master-port $MASTER_PORT --context ${CONTEXT:-300000} --parallel $PARALLEL"
    a+=" --kv-dtype ${KV_DTYPE:-fp8} --no-update-check"
    if [[ "$r" == 0 ]]; then
        a+=" --host $HOST --port $PORT${SERVED_NAME:+ --name $SERVED_NAME}${MAX_TOKENS:+ --max-tokens $MAX_TOKENS}"
        local al; for al in ${ALIASES:-}; do a+=" --alias $al"; done
    fi
    echo "$a"
}

all() { # all CMD: on every rank in parallel; non-zero if any failed
    local pids=() r rc=0
    for r in 0 1 2 3; do on "$r" "$*" & pids+=($!); done
    for r in "${pids[@]}"; do wait "$r" || rc=1; done
    return $rc
}
stop_all() { local r; for r in 0 1 2 3; do on "$r" "docker rm -f $NAME-r$r >/dev/null 2>&1 || true" & done; wait; }
gpu_busy() { local r; for r in 0 1 2 3; do on "$r" "nvidia-smi --query-compute-apps=pid --format=csv,noheader" 2>/dev/null; done | grep -q '[0-9]'; }
memfree() { on "$1" "awk '/^MemFree:/ {printf \"%d\", \$2/1048576}' /proc/meminfo"; }
drop_caches() { all "sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'" || log "drop_caches refused somewhere"; }
mem_gate() {
    local t0 r m ok
    t0=$(date +%s)
    while :; do
        ok=1; m=""
        for r in 0 1 2 3; do v=$(memfree "$r"); m+="$v "; [[ "${v:-0}" -ge "$MEM_GATE_GIB" ]] || ok=0; done
        if [[ $ok == 1 ]]; then log "MemFree $m GiB >= $MEM_GATE_GIB"; return 0; fi
        (( $(date +%s) - t0 < MEM_GATE_TIMEOUT )) || { log "MemFree $m GiB under $MEM_GATE_GIB; not starting"; return 1; }
        log "waiting for memory: MemFree $m"; drop_caches; sleep 10
    done
}
running() { on "$1" "docker inspect -f '{{.State.Running}}' $NAME-r$1 2>/dev/null" || echo absent; }

cmd_start() {
    if gpu_busy; then log "a CUDA process is running on a node; stop it first"; exit 1; fi
    for r in 0 1 2 3; do on "$r" "test -f $ENGRAM/engram-l1-r${r}of4.bin && test -f $ENGRAM/engram-l14-r${r}of4.bin && test -f $MODEL/config.json" \
        || { log "rank $r: model or Engram shards missing"; exit 1; }; done
    stop_all; drop_caches; mem_gate || exit 1
    IMAGE_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE")
    export IMAGE_ID TF_DSV41_LAUNCH_T0
    TF_DSV41_LAUNCH_T0=$(date +%s.%N)
    for r in 3 2 1; do
        # shellcheck disable=SC2046
        on "$r" "mkdir -p $STATE/sessions $PREPARED && docker run $(run_args "$r")" >/dev/null &
    done
    mkdir -p "$STATE/sessions" "$PREPARED"
    # shellcheck disable=SC2046
    docker run $(run_args 0) >/dev/null
    wait
    local t0; t0=$(date +%s)
    log "waiting for rank 0 on :$PORT"
    until curl -sf -m 5 "$BASE/v1/models" >/dev/null; do
        for r in 0 1 2 3; do
            [[ "$(running "$r")" == true ]] || { log "rank $r exited"; on "$r" "docker logs --tail 60 $NAME-r$r 2>&1"; exit 1; }
        done
        (( $(date +%s) - t0 < READY_TIMEOUT )) || { log "not ready after ${READY_TIMEOUT}s"; exit 1; }
        sleep 3
    done
    log "ready after $(( $(date +%s) - t0 )) s"
    docker logs "$NAME-r0" 2>&1 | grep -E 'serving: [0-9]+ slot' | tail -1 || true
    drop_caches
}

cmd_ship() {
    local w pids=()
    for w in "${WORKERS[@]}"; do (docker save "$IMAGE" | ssh -o BatchMode=yes "$w" docker load) & pids+=($!); done
    for w in "${pids[@]}"; do wait "$w"; done
    log "image on every node"
}

cmd_prebuild() {
    if gpu_busy; then log "a CUDA process is running on a node; stop it first"; exit 1; fi
    local run="docker run --rm -i --name $NAME-prebuild --gpus all --ipc=host -v $CACHE_VOL:/cache"
    run+=" -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton"
    run+=" -e CUDA_CACHE_PATH=/cache/nv/ComputeCache -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python $IMAGE -"
    local r pids=()
    all "docker run --rm -v $CACHE_VOL:/cache --entrypoint find $IMAGE /cache/torch_extensions -name lock -delete 2>/dev/null; true"
    for r in 1 2 3; do ssh -o BatchMode=yes "${WORKERS[$((r-1))]}" "$run" < "$PREBUILD_PY" > "/tmp/tf4-prebuild-r$r.log" 2>&1 & pids+=($!); done
    bash -c "$run" < "$PREBUILD_PY" > /tmp/tf4-prebuild-r0.log 2>&1 || { log "prebuild failed on rank 0"; tail /tmp/tf4-prebuild-r0.log; }
    for r in "${pids[@]}"; do wait "$r" || log "a worker prebuild failed (/tmp/tf4-prebuild-r*.log)"; done
    grep -hE '^(FAIL|prebuild:)' /tmp/tf4-prebuild-r*.log || true
}

cmd_run() {
    local mod="${1:?usage: serve4.sh run MODULE [ARGS...]}"; shift
    local out="${OUT:-$PWD/results/run-$(date +%Y%m%d-%H%M%S)}" r
    if gpu_busy; then log "a CUDA process is running on a node; stop it first"; exit 1; fi
    mkdir -p "$out"; drop_caches; mem_gate || exit 1
    for r in 3 2 1; do
        # shellcheck disable=SC2046
        on "$r" "mkdir -p /tmp/tf4-run $STATE/sessions && docker run --rm --name $NAME-run-r$r $(container_args) -v /tmp/tf4-run:/out \
            --entrypoint python $IMAGE -m $mod --model /model --rank $r --world $WORLD --master $HEAD_IP --port $MASTER_PORT $*" \
            > "$out/r$r.log" 2>&1 &
    done
    sleep 2
    # shellcheck disable=SC2046
    docker run --rm --name "$NAME-run-r0" $(container_args) -v "$out:/out" --entrypoint python "$IMAGE" -m "$mod" \
        --model /model --rank 0 --world $WORLD --master "$HEAD_IP" --port "$MASTER_PORT" "$@" 2>&1 | tee "$out/r0.log"
    wait || true
    log "logs in $out"
}

PREBUILD_PY="${PREBUILD_PY:-$PWD/scripts/prebuild_ext.py}"
case "${1:-}" in
ship) cmd_ship ;;
prebuild) cmd_prebuild ;;
start) cmd_start ;;
stop) stop_all; log "stopped" ;;
status) for r in 0 1 2 3; do echo "rank $r: $(running "$r")"; done; curl -s -m 10 "$BASE/health" || true; echo ;;
logs) r="${2:-0}"; on "$r" "docker logs --tail ${TAIL:-80} $NAME-r$r" ;;
args) run_args "${2:-0}"; echo ;;
run) shift; cmd_run "$@" ;;
*) sed -n '2,12p' "$0"; exit 2 ;;
esac
