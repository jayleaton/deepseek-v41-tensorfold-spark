#!/usr/bin/env bash
# Run one native rank with the reviewed engine, pack, profile and rank assets.
{ set +x; } 2>/dev/null
safe_dir=${BASH_SOURCE[0]%/*}
[[ "$safe_dir" != "${BASH_SOURCE[0]}" ]] || safe_dir=.
source "$safe_dir/safe_shell.sh"
export CONFIG=${CONFIG:-config/prod-zig.env}
[[ "$CONFIG" == /* ]] || CONFIG="${safe_exec%/*}/../$CONFIG"
# Carry intent, never the credential, through sanitized nested launchers.
[[ -z "${TENSORFOLD_API_KEY:-}" ]] || export TF_NATIVE_REQUIRE_FILE=1
safe_main "${BASH_SOURCE[0]}" "$@"
set -euo pipefail
cd "${safe_exec%/*}/.."
rank=${1:?usage: run-zig.sh 0|1 [--dry-run|--identity|--check-aot|--detach]}
[[ "$rank" == 0 || "$rank" == 1 ]] || { echo 'rank must be 0 or 1' >&2; exit 2; }
mode=${2:-}
case "$mode" in ''|--dry-run|--identity|--check-aot|--detach) ;; *) echo 'unknown launch option' >&2; exit 2 ;; esac
config=${CONFIG:-config/prod-zig.env}
[[ -f "$config" ]] || { echo 'configuration missing; copy a native profile to CONFIG' >&2; exit 2; }
set -a
source "$config" >/dev/null 2>&1
{ set +x; } 2>/dev/null
set +a
safe_credentials || exit $?
serving=0
if [[ "$rank" == 0 && "$mode" != --check-aot && "$mode" != --identity ]]; then serving=1; fi
if [[ "$serving" == 1 && ( -n "${TENSORFOLD_API_KEY:-}" || "${TF_NATIVE_REQUIRE_FILE:-0}" == 1 ) && -z "${NATIVE_AUTH_FILE:-}" ]]; then
    echo 'native authentication requires NATIVE_AUTH_FILE; move TENSORFOLD_API_KEY out of CONFIG into an owned mode-0600 file' >&2
    exit 2
fi
for key in IMAGE TF_TP_MASTER NCCL_SOCKET_IFNAME NCCL_IB_HCA MODEL_DIR ENGRAM_DIR ASSETS_DIR PREPARED_DIR STATE_DIR CACHE_DIR; do
    [[ -n "${!key:-}" && "${!key}" != *'<'* ]] || { echo "set required native setting $key" >&2; exit 2; }
done
rev=$(safe_capture revision cat config/tensorfold.rev)
export TF_EXPERIMENTAL=deepseek
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export NCCL_CUMEM_HOST_ENABLE=${NCCL_CUMEM_HOST_ENABLE:-0}
envs=()
# Enumerate names only; never pipe credential values through an environment dump.
forward_names=$(compgen -v | safe_capture names python3 -c '
import sys
sys.path.insert(0, "scripts")
from redact import secret_name
for name in sys.stdin:
    name = name.strip()
    selected = name.startswith(("TF_DSV41_", "GLM53_TF_", "NCCL_", "TF_CUDA_")) or name in ("TF_TP_ROCE_HCA", "TF_EXPERIMENTAL")
    if selected and not secret_name(name):
        print(name)
') || exit 2
while IFS= read -r key; do
    # Check shell values before the supervisor removes credential aliases from
    # child environments. A tainted control must fail, not silently disappear.
    safe_check_argv "${!key}" || exit 2
    export "$key"
    # TF_TP_ROCE_HCA forwarding contributed by rootsbymenda in #63.
    case "$key" in TF_DSV41_*|GLM53_TF_*|NCCL_*|TF_TP_ROCE_HCA|TF_EXPERIMENTAL|TF_CUDA_*) envs+=(-e "$key");; esac
done <<< "$forward_names"
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
      -e TF_TP_KERNEL_IMAGE=/opt/tensorfold/share/tp_mailbox.fatbin)
[[ -z "${CPUSET_CPUS:-}" ]] || args+=(--cpuset-cpus "$CPUSET_CPUS")
# Authentication belongs to the head server alone. The native engine reads the
# restricted file itself; neither Docker environment nor argv carries its keys.
if [[ "$serving" == 1 && -n "${NATIVE_AUTH_FILE:-}" ]]; then
    safe_command python3 -c '
import os, stat, sys
path, shared = sys.argv[1], [os.path.realpath(d) for d in sys.argv[2:] if d]
try:
    if (not os.path.isabs(path) or any(c in path for c in ":\r\n")
            or os.path.realpath(path) != os.path.normpath(path)):
        raise ValueError
    # Owned by the operator, or by root so that only the serving container can read it.
    info = os.lstat(path)
    if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid not in (os.geteuid(), 0) or not 0 < info.st_size <= 1048576):
        raise ValueError
    # Never inside a directory mounted into other containers or read by diagnostics.
    if any(os.path.commonpath([path, d]) == d for d in shared):
        raise ValueError
except (OSError, ValueError):
    raise SystemExit("NATIVE_AUTH_FILE must be a nonempty regular file with mode 0600, owned by you or root, at an absolute symlink-free path outside the checkout and the model, engram, assets, prepared, state and cache directories")
' "$NATIVE_AUTH_FILE" "$PWD" "$MODEL_DIR" "$ENGRAM_DIR" "$ASSETS_DIR" "$PREPARED_DIR" "$STATE_DIR" "$CACHE_DIR" || exit 2
    args+=(-v "$NATIVE_AUTH_FILE:/run/tensorfold/auth-keys:ro")
fi
args+=("$IMAGE")
if [[ "$rank" == 0 ]]; then
    args+=(tensorfold-dsv41 serve /model --host "${HOST:-localhost}" --port "${PORT:-8000}"
           --parallel "${TF_DSV41_SLOTS:-1}" --context "${TF_DSV41_CONTEXT:-4096}"
           --name DeepSeek-V4.1-Flash-TF --max-tokens 32768)
    [[ "${TF_DSV41_THINKING:-1}" != 0 ]] || args+=(--no-thinking)
    [[ "$serving" != 1 || -z "${NATIVE_AUTH_FILE:-}" ]] || args+=(--api-key-file /run/tensorfold/auth-keys)
else
    args+=(tf-dsv41 follow /model /assets)
fi
if [[ "$mode" == --dry-run ]]; then
    safe_quote_argv "${args[@]}" || exit $?
    exit
fi
image_rev=$(safe_capture revision docker image inspect -f '{{index .Config.Labels "org.tensorfold.revision"}}' "$IMAGE")
[[ "$image_rev" == "$rev" ]] || { echo "image engine revision differs from config/tensorfold.rev" >&2; exit 2; }
image_id=$(safe_capture image_id docker image inspect -f '{{.Id}}' "$IMAGE")
if [[ "$mode" == --check-aot ]]; then
    [[ -f "$ASSETS_DIR/sigs.json" ]] || { echo 'missing assets/sigs.json; see docs/ZIG-SERVE.md' >&2; exit 2; }
    safe_command mkdir -p "$STATE_DIR"
    # No GPU device is exposed for host-only enumeration.
    safe_command docker run --rm --init --runtime=runc -e NVIDIA_VISIBLE_DEVICES=void \
        -v "$ASSETS_DIR:/assets:ro" -v "$STATE_DIR:/state" "${envs[@]}" \
        -e TF_TP_WORLD=2 -e "TF_TP_RANK=$rank" -e TF_TP_DEVICE=0 \
        -e "TF_TP_MASTER=$TF_TP_MASTER" -e "TF_TP_PORT=${TF_TP_PORT:-29571}" \
        -e "TF_COMM_BACKEND=${TF_COMM_BACKEND:-roce}" \
        -e TF_DSV41_ASSETS=/assets -e TF_DSV41_ENGRAM_DIR=/engram \
        -e TF_TP_KERNEL_IMAGE=/opt/tensorfold/share/tp_mailbox.fatbin \
        "$IMAGE" tf-dsv41 aot-needs /assets/sigs.json "/state/aot-needs-r$rank.json"
    safe_command python3 scripts/check_native_assets.py --needs "$STATE_DIR/aot-needs-r$rank.json" --aot "$ASSETS_DIR/aot"
    exit $?
fi
for key in MODEL_DIR ENGRAM_DIR ASSETS_DIR CACHE_DIR; do
    [[ -d "${!key}" ]] || { echo "required native directory missing: $key" >&2; exit 2; }
done
identity=$(safe_capture json python3 scripts/check_native_assets.py --rank "$rank" --model "$MODEL_DIR" \
    --engram "$ENGRAM_DIR" --assets "$ASSETS_DIR" --context "${TF_DSV41_CONTEXT:-4096}" \
    --revision "$rev" --image-id "$image_id")
if [[ "$mode" == --identity ]]; then printf '%s\n' "$identity"; exit; fi
stop_file=${STOP_FILE:-$STATE_DIR/native-ops/STOP}
[[ ! -e "$stop_file" ]] || { echo 'durable STOP is set; use serve.sh start explicitly' >&2; exit 78; }
if [[ "$mode" == --detach ]]; then args=(docker run --detach "${args[@]:2}"); fi
if [[ -n "${GUARD_CONFIG:-}" ]]; then
    safe_command python3 scripts/guard/lifecycle.py --config "$GUARD_CONFIG" check --operation start
fi
# PREFLIGHT=1 opts into the read-only local Engram/memory checks from #61.
if [[ "${PREFLIGHT:-0}" == 1 ]]; then
    rc=0
    read -r -a pf_args <<< "${PREFLIGHT_ARGS:-}"
    safe_command python3 scripts/preflight.py --config "$config" --rank "$rank" "${pf_args[@]}" || rc=$?
    if (( rc >= 2 )); then echo "rank $rank: preflight failed (exit $rc)" >&2; exit 2; fi
fi
safe_command mkdir -p "$PREPARED_DIR" "$STATE_DIR/sessions"
safe_command "${args[@]}"
