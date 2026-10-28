#!/usr/bin/env bash
# Run one native rank with the reviewed engine, pack, profile and rank assets.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rank=${1:?usage: run-zig.sh 0|1 [--dry-run|--identity|--check-aot|--detach]}
[[ "$rank" == 0 || "$rank" == 1 ]] || { echo 'rank must be 0 or 1' >&2; exit 2; }
mode=${2:-}
case "$mode" in ''|--dry-run|--identity|--check-aot|--detach) ;; *) echo "unknown option: $mode" >&2; exit 2 ;; esac
config=${CONFIG:-config/prod-zig.env}
[[ -f "$config" ]] || { echo "missing $config; copy config/prod-zig-flash17.env.example" >&2; exit 2; }
set -a
source "$config"
set +a
for key in IMAGE TF_TP_MASTER NCCL_SOCKET_IFNAME NCCL_IB_HCA MODEL_DIR ENGRAM_DIR ASSETS_DIR PREPARED_DIR STATE_DIR CACHE_DIR; do
    [[ -n "${!key:-}" && "${!key}" != *'<'* ]] || { echo "set $key in $config" >&2; exit 2; }
done
rev=$(cat config/tensorfold.rev)
export TF_EXPERIMENTAL=deepseek
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export NCCL_CUMEM_HOST_ENABLE=${NCCL_CUMEM_HOST_ENABLE:-0}
envs=()
while IFS='=' read -r key value; do
    # TF_TP_ROCE_HCA forwarding contributed by rootsbymenda in #63.
    case "$key" in TF_DSV41_*|GLM53_TF_*|NCCL_*|TF_TP_ROCE_HCA|TF_EXPERIMENTAL|TF_CUDA_*) envs+=(-e "$key");; esac
done < <(env)
prefix=${CONTAINER_PREFIX:-tensorfold-zig}
[[ "$prefix" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo 'invalid CONTAINER_PREFIX' >&2; exit 2; }
args=(docker run --rm --init --name "$prefix-r$rank" --gpus all --ipc=host --network host
      --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK
      --label "org.tensorfold.revision=$rev" --label "org.tensorfold.rank=$rank"
      -v "$MODEL_DIR:/model:ro" -v "$ENGRAM_DIR:/engram:ro" -v "$ASSETS_DIR:/assets:ro"
      -v "$PREPARED_DIR:/prepared" -v "$STATE_DIR:/state" -v "$STATE_DIR/sessions:/sessions"
      -v "$CACHE_DIR:/cache" "${envs[@]}"
      -e TF_TP_WORLD=2 -e "TF_TP_RANK=$rank" -e TF_TP_DEVICE=0
      -e "TF_TP_MASTER=$TF_TP_MASTER" -e "TF_TP_PORT=${TF_TP_PORT:-29571}"
      -e "TF_COMM_BACKEND=${TF_COMM_BACKEND:-roce}"
      -e TF_DSV41_ASSETS=/assets -e TF_DSV41_ENGRAM_DIR=/engram
      -e TF_TP_KERNEL_IMAGE=/opt/tensorfold/share/tp_mailbox.fatbin "$IMAGE")
if [[ "$rank" == 0 ]]; then
    args+=(tensorfold-dsv41 serve /model --host "${HOST:-localhost}" --port "${PORT:-8000}"
           --parallel "${TF_DSV41_SLOTS:-1}" --context "${TF_DSV41_CONTEXT:-4096}"
           --name DeepSeek-V4.1-Flash-TF --max-tokens 32768)
    [[ "${TF_DSV41_THINKING:-1}" != 0 ]] || args+=(--no-thinking)
else
    args+=(tf-dsv41 follow /model /assets)
fi
if [[ "$mode" == --dry-run ]]; then printf '%q ' "${args[@]}"; printf '\n'; exit; fi
image_rev=$(docker image inspect -f '{{index .Config.Labels "org.tensorfold.revision"}}' "$IMAGE")
[[ "$image_rev" == "$rev" ]] || { echo "image engine revision differs from config/tensorfold.rev" >&2; exit 2; }
image_id=$(docker image inspect -f '{{.Id}}' "$IMAGE")
if [[ "$mode" == --check-aot ]]; then
    [[ -f "$ASSETS_DIR/sigs.json" ]] || { echo 'missing assets/sigs.json; see docs/ZIG-SERVE.md' >&2; exit 2; }
    mkdir -p "$STATE_DIR"
    # No GPU device is exposed for host-only enumeration.
    docker run --rm --init --runtime=runc -e NVIDIA_VISIBLE_DEVICES=void \
        -v "$ASSETS_DIR:/assets:ro" -v "$STATE_DIR:/state" "${envs[@]}" -e TF_TP_WORLD=2 \
        "$IMAGE" tf-dsv41 aot-needs /assets/sigs.json "/state/aot-needs-r$rank.json"
    exec python3 scripts/check_native_assets.py --needs "$STATE_DIR/aot-needs-r$rank.json" --aot "$ASSETS_DIR/aot"
fi
for dir in "$MODEL_DIR" "$ENGRAM_DIR" "$ASSETS_DIR" "$CACHE_DIR"; do
    [[ -d "$dir" ]] || { echo "missing directory $dir" >&2; exit 2; }
done
identity=$(python3 scripts/check_native_assets.py --rank "$rank" --model "$MODEL_DIR" \
    --engram "$ENGRAM_DIR" --assets "$ASSETS_DIR" --context "${TF_DSV41_CONTEXT:-4096}" \
    --revision "$rev" --image-id "$image_id")
if [[ "$mode" == --identity ]]; then printf '%s\n' "$identity"; exit; fi
if [[ "$mode" == --detach ]]; then args=(docker run --detach "${args[@]:2}"); fi
# PREFLIGHT=1 opts into the read-only local Engram/memory checks from #61.
if [[ "${PREFLIGHT:-0}" == 1 ]]; then
    rc=0
    read -r -a pf_args <<< "${PREFLIGHT_ARGS:-}"
    python3 scripts/preflight.py --config "$config" --rank "$rank" "${pf_args[@]}" || rc=$?
    if (( rc >= 2 )); then echo "rank $rank: preflight failed (exit $rc)" >&2; exit 2; fi
fi
mkdir -p "$PREPARED_DIR" "$STATE_DIR/sessions"
exec "${args[@]}"
