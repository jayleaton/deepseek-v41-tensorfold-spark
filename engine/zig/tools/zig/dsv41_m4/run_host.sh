#!/usr/bin/env bash
# following (tf-dsv41-m1 follow), then the CLI replay of the server's recorded replies (tf-dsv41-m1 generate) with
# rank 1 following again. Prod must be stopped first; start both hosts within about a minute for each phase.
#   rank 0 host:  ./run_host.sh 0 OUTDIR server     then (after the requests)  ./run_host.sh 0 OUTDIR generate
#   rank 1 host:  ./run_host.sh 1 OUTDIR follow     then                        ./run_host.sh 1 OUTDIR follow
#   both hosts once before the server (drafts): ./run_host.sh R OUTDIR dspark-aot (ASSETS/aot gains DSpark's variants;
#   ASSETS a writable copy: cp -r OUTDIR/seedN/ref/{rope.bin,engram-host.bin,aot} OUTDIR/m4-assets/)
# Knobs: ASSETS (a dir with rope.bin, engram-host.bin, aot/: an M2c-2 seed's ref/ dir has all three; DSpark's Triton
# variants come from a set P capture merged in, else run DRAFTS=0), PORT (8091), DRAFTS (1), BACKEND (roce | nccl),
# OWN_PREFILL (1: prompts through the Zig prefill; 0: 16-row decode windows; both ranks alike), GRAPHS (1: decode
# windows through CUDA graphs, both ranks alike; 0: eager), DRAFT_TIMED (0: drafts sized from the cost tables, the
# same for a request whatever the load, so HTTP == CLI; 1: the measured round times), PROMPT_TAIL (verify: the prompt's
# last token as the first decode row, as prod's Python server; prefill: every prompt row prefilled, Forward.prompt).
#   SAMPLING (1: keyed sampling, T > 0 served), SLOTS (1; N > 1: N live slots in row mode, the server's --parallel N;
#   needs POOL_TOKENS and the rows-aot phase's variants; drafts off unless SLOT_DRAFTS=1: DSpark per slot with a
#   batched verify, as prod's batcher drafts), POOL_TOKENS (the paged pool;
#   unset: the contiguous slot), KV_SPLIT (1: comp rows sharded over the ranks, needs POOL_TOKENS), INDEX_KV (prod: fp8),
#   ENGRAM (1: R3, the node's packed Engram shards as prod reads them: ENGRAM_DIR, prod-perf1.env's HEAD_ENGRAM /
#   WORKER_ENGRAM), CALIB (real: the drafts priced by prod's stored table, CACHE_VOL read-only; "": the built-in costs),
#   TREE (DSpark draft trees, e.g. 3; "": chains), PHASES (1: per-round phase times into OUTDIR/phases.json),
#   GRAMMAR (1: structured output), IMAGES (native | placeholder), BIAS_VL (1: prod's /cache/dsv41-bias-vl, with
#   IMAGES=native), GM_V2 (1: x3gm v2 prefill), PF_DENSE (fused: prod's fused dense prefill + its table),
#   PREFETCH_AHEAD (1: the next segment's Engram reads early; needs ENGRAM=1); each on only once its pod gate passed.
#   SPEC (1: the speculative DSpark pass, prod's TF_DSV41_SPEC_DRAFT=1; 0: off), CONTEXT (the served slot limit, prod
#   1048576; unset: 4,096; needs the long-aot phase's rope.bin and _stream variants), PREFILL_MODE (replay: CED replay
#   prefill, prod's TF_DSV41_PREFILL=replay, needs PROMPT_TAIL=verify; unset: full).
# dspark-aot's capture: CAPTURE_SETS (P), CAPTURE_ENV (more capture settings, space separated KEY=value, e.g.
#   "M1_WINDOWS=1,3,16,20,24 M1_DRAFT_SLOTS=4": windows past 16 rows and the 4-slot pass), CAPTURE_OUT (capture-p: the
#   dir under OUTDIR), CAPTURE_MERGE (1: its aot/ merged into ASSETS/aot; 0: kept apart, e.g. until `dcheck` passed).
# long-aot (both hosts, once, before CONTEXT past 4,096): Python's prefill of one 8,448-token prompt on layers 0-24 at TP=2
# over the two hosts (the stream top-k's _stream variants past 4,096 visible keys) merged into ASSETS/aot, and ASSETS'
# rope.bin rebuilt with LONG_ROWS rows (1,050,624 = 1M + 2,048) (about 10 min).
# rows-aot (both hosts, once, before SLOTS > 1): the slots reference (tools/zig/dsv41_slots/ref.py) at TP=2 over the
# two hosts on prod's tree, its row-mode Triton variants merged into ASSETS/aot (about 8 min).
set -u
RANK=${1:?rank}; OUT=${2:?outdir}; PHASE=${3:?server|generate|follow}
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# rank 0 host's test-window lease (Spark-tool-kit's campaign.sh windows): held = the Sparks are not ours, nothing starts
LEASE=${LEASE_FILE:-$KIT/.test-window-lease}
if [[ "$RANK" == 0 && -f "$LEASE" ]]; then echo "the test-window lease is held ($LEASE): another window owns the Sparks"; exit 3; fi
MASTER=${MASTER:?set MASTER to rank 0 rendezvous address}
IMAGE=${IMAGE:?set IMAGE to the CUDA Python reference image}
MODEL=${PACK:?set PACK to the model pack directory}; ASSETS=${ASSETS:?ASSETS: rope.bin, engram-host.bin, aot/ (writable: dspark-aot merges into it)}; PORT=${PORT:-8091}
DRAFTS=${DRAFTS:-1}; BACKEND=${BACKEND:-roce}; SLOTS=${SLOTS:-1}; CACHE_VOL=${CACHE_VOL:?set CACHE_VOL to the Python reference cache volume}
# several live slots: drafts only with SLOT_DRAFTS=1 (TF_DSV41_SLOT_DRAFTS: per-slot DSpark, batched verify); else off
(( SLOTS > 1 )) && [[ "${SLOT_DRAFTS:-0}" != 1 ]] && DRAFTS=0
if [[ "${ENGRAM:-1}" == 1 ]]; then : "${ENGRAM_DIR:?set ENGRAM_DIR to the local Engram shards}"; fi
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
ARGS=(--rm --gpus all --ipc=host --network host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK
      -v "$MODEL:/model:ro" -v "$KIT:/kit:ro" -v "$ASSETS:/assets:ro" -v "$OUT:/out"
      -e TF_TP_WORLD=2 -e TF_TP_RANK="$RANK" -e TF_TP_DEVICE=0 -e TF_TP_MASTER="$MASTER" -e TF_TP_PORT=29851
      -e TF_COMM_BACKEND="$BACKEND" -e TF_TP_ROCE_FALLBACK=error -e TF_TP_KERNEL_IMAGE=/kit/tp_mailbox.fatbin
      -e TF_DSV41_ASSETS=/assets -e TF_DSV41_EXPERT_TOPP=0.85 -e TF_DSV41_R1="${R1:-1}" -e TF_DSV41_DRAFTS="$DRAFTS" -e TF_DSV41_OWN_PREFILL="${OWN_PREFILL:-1}"
      -e TF_DSV41_GRAPHS="${GRAPHS:-1}" -e TF_DSV41_DRAFT_TIMED="${DRAFT_TIMED:-0}" -e TF_DSV41_PROMPT_TAIL="${PROMPT_TAIL:-verify}"
      -e TF_DSV41_PLAN_TIMEOUT=1800 -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1}
      -e NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1}
      -e TF_DSV41_SAMPLING="${SAMPLING:-1}" -e TF_DSV41_SLOTS="$SLOTS"
      -e GLM53_TF_ROCE_MAX_KB="${ROCE_MAX_KB:-1024}" -e TF_DSV41_INDEX_BUDGET_MIB=64 -e TF_DSV41_PREFILL_CHUNK=2048
      -e TF_DSV41_PREFILL_ROWS=2048)   # prod's: 4-slot x 16-row windows stay on RoCE (Zig's default 256 KiB sends them to NCCL)
[[ -n "${POOL_TOKENS:-}" ]] && ARGS+=(-e TF_DSV41_POOL_TOKENS="$POOL_TOKENS")
[[ "${KV_SPLIT:-0}" == 1 ]] && ARGS+=(-e TF_DSV41_KV_SPLIT=1)
[[ -n "${INDEX_KV:-}" ]] && ARGS+=(-e TF_DSV41_INDEX_KV="$INDEX_KV")
[[ "${ENGRAM:-1}" == 1 ]] && ARGS+=(-v "$ENGRAM_DIR:/engram:ro" -e TF_DSV41_ENGRAM_DIR=/engram)
# prod's cache volume (read-only): the calibration tables, bias_vl, the pfdense table
if [[ "${CALIB-real}" == real || "${BIAS_VL:-0}" == 1 || -n "${PF_DENSE:-}" ]]; then ARGS+=(-v "$CACHE_VOL:/cache:ro"); fi
[[ "${CALIB-real}" == real ]] && ARGS+=(-e TF_DSV41_CALIB=real -e TF_DSV41_CALIB_DIR=/cache/dsv41-calib)
# the features gated on pods 29 / 30 (each off unless set; prod-perf1.env's values when on)
[[ "${GRAMMAR:-0}" == 1 ]] && ARGS+=(-e TF_DSV41_GRAMMAR=1 -e TF_DSV41_TOOL_GRAMMAR=0)
[[ -n "${IMAGES:-}" ]] && ARGS+=(-e TF_DSV41_IMAGES="$IMAGES")
[[ "${BIAS_VL:-0}" == 1 ]] && ARGS+=(-e TF_DSV41_BIAS_VL=/cache/dsv41-bias-vl)
[[ "${GM_V2:-0}" == 1 ]] && ARGS+=(-e TF_DSV41_GM_V2=1)
[[ -n "${PF_DENSE:-}" ]] && ARGS+=(-e TF_DSV41_PF_DENSE="$PF_DENSE" -e TF_DSV41_PF_DENSE_TABLE=/cache/pfdense-table.json)
[[ "${PREFETCH_AHEAD:-0}" == 1 ]] && ARGS+=(-e TF_DSV41_PREFETCH_AHEAD=1)
[[ -n "${TREE:-}" ]] && ARGS+=(-e TF_DSV41_TREE="$TREE")
[[ "${SLOT_DRAFTS:-0}" == 1 ]] && ARGS+=(-e TF_DSV41_SLOT_DRAFTS=1)
[[ "${SPEC:-1}" == 1 ]] && ARGS+=(-e TF_DSV41_SPEC_DRAFT=1)    # prod's SPEC_DRAFT=1: the next round's pass early
[[ -n "${CONTEXT:-}" ]] && ARGS+=(-e TF_DSV41_CONTEXT="$CONTEXT")  # the served slot limit (rope.bin: CONTEXT + 2,048 rows)
[[ "${PHASES:-0}" == 1 ]] && ARGS+=(-e TF_DSV41_PHASES=1 -e TF_DSV41_PHASES_OUT=/out/phases.json)
[[ -n "${PREFILL_MODE:-}" ]] && ARGS+=(-e TF_DSV41_PREFILL="$PREFILL_MODE")
NCCL='export TF_NCCL_LIB=$(python -c "import nvidia.nccl, os; print(os.path.join(os.path.dirname(nvidia.nccl.__file__), \"lib\", \"libnccl.so.2\"))" 2>/dev/null || echo libnccl.so.2)'
case "$PHASE" in
    server)
        nd=(); [[ "$DRAFTS" == 0 ]] && nd=(--no-drafts)
        docker run "${ARGS[@]}" -e TF_DSV41_TRACE_TOKENS=/out/trace.jsonl --name dsv41-zig-m4-r0 --entrypoint bash "$IMAGE" \
            -c "$NCCL; exec /kit/tensorfold-dsv41 serve /model --port $PORT --parallel $SLOTS ${nd[*]}" 2>&1 | tee "$OUT/server.log" ;;
    generate)
        # GEN_STREAMS=N: N replies in flight (several slots: batched rows, as served)
        docker run "${ARGS[@]}" -e TF_DSV41_GEN_STREAMS="${GEN_STREAMS:-1}" --name dsv41-zig-m4-gen --entrypoint bash "$IMAGE" \
            -c "$NCCL; exec /kit/tf-dsv41-m1 generate /model /assets /out/trace.jsonl" 2>&1 | tee "$OUT/generate.log" ;;
    dspark-aot)
        # DSpark's Triton variants on this GPU (the M2c-2 reference never drafts): a set P capture (one GPU, ~3 min) on
        # prod's tree and knobs (+ R1), its AOT set merged into ASSETS/aot
        CO=${CAPTURE_OUT:-capture-p}; CAP=$OUT/$CO; rm -rf "$CAP"
        CE=(); for kv in ${CAPTURE_ENV:-}; do CE+=(-e "$kv"); done
        KN=(); for f in prod-knobs.env r1c-knobs.env; do while read -r kv; do [[ "$kv" =~ ^TF_ ]] && KN+=(-e "$kv"); done < "$KIT/tools/zig/dsv41_m1/$f"; done
        C=8474f3164ea60cb9c9f8e9bd5a4493fdf13eaa47
        TF=${TF_TREE:?set TF_TREE to the Python reference checkout}
        docker run --rm --gpus all --ipc=host -v "$TF:/dsv41-tf:ro" -v "$MODEL:/model:ro" -v "$KIT:/kit:ro" -v "$OUT:/out" \
            -v "${CACHE_VOL:-dsv41-perf1-cache}:/cache" "${KN[@]}" -e TF_DSV41_GM_V2=1 -e M1_PREFILL_ROWS=0 "${CE[@]}" \
            -e PYTHONPATH=/dsv41-tf/src:/cache/pylib -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton \
            --name "dsv41-zig-capp-r$RANK" --entrypoint python "$IMAGE" -u -B /kit/tools/zig/dsv41_m1_capture.py --pack /model --out "/out/$CO" --sets "${CAPTURE_SETS:-P}" \
            > "$OUT/$CO.log" 2>&1 || { echo "capture ${CAPTURE_SETS:-P} failed: $OUT/$CO.log"; exit 1; }
        # the blobs are the container's (root-owned): removed as root in the image; the merge does not depend on it
        docker run --rm -v "$OUT:/out" --entrypoint rm "$IMAGE" -rf "/out/$CO/blobs" || echo "warning: $CAP/blobs left"
        if [[ "${CAPTURE_MERGE:-1}" == 1 ]]; then python3 "$KIT/tools/zig/dsv41_aot_merge.py" "$ASSETS/aot" "$CAP/aot" | tee -a "$OUT/$CO.log"
        else echo "$CAP/aot kept apart (CAPTURE_MERGE=0)" | tee -a "$OUT/$CO.log"; fi ;;
    rows-aot)
        # the row-mode Triton variants (ROWS, int64 POS, SL / PTS) of every bucket up to 16 on this GPU: Python's slots
        # engine (prod's tree: rowtab / rowmode / rowgraphs) at TP=2 over the two hosts, 4 slots, layers 0-24 (every
        # layer kind), then this host's AOT set merged into ASSETS/aot
        C=8474f3164ea60cb9c9f8e9bd5a4493fdf13eaa47
        TF=${TF_TREE:?set TF_TREE to the Python reference checkout}
        KN=(); for f in prod-knobs.env r1c-knobs.env; do while read -r kv; do [[ "$kv" =~ ^TF_ ]] && KN+=(-e "$kv"); done < "$KIT/tools/zig/dsv41_m1/$f"; done
        RA=$OUT/rows-aot; rm -rf "$RA"
        docker run --rm --gpus all --ipc=host --network host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK \
            -v "$TF:/dsv41-tf:ro" -v "$MODEL:/model:ro" -v "$KIT:/kit:ro" -v "$OUT:/out" -v "$CACHE_VOL:/cache" "${KN[@]}" \
            -e TF_DSV41_GM_V2=1 -e TF_DSV41_EXPERT_TOPP=0.85 -e PYTHONPATH=/dsv41-tf/src:/cache/pylib \
            -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton \
            -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1} -e NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1} \
            -e GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1} --name "dsv41-zig-rows-r$RANK" \
            --entrypoint python "$IMAGE" -u -B /kit/tools/zig/dsv41_slots/ref.py --pack /model --out /out/rows-aot \
            --layers 0-24 --slots 4 --steps "${ROWS_STEPS:-8}" --rank "$RANK" --master "$MASTER" --port 29821 \
            > "$OUT/rows-aot.log" 2>&1 \
            && python3 "$KIT/tools/zig/dsv41_aot_merge.py" "$ASSETS/aot" "$RA/aot" | tee -a "$OUT/rows-aot.log" ;;
    long-aot)
        C=8474f3164ea60cb9c9f8e9bd5a4493fdf13eaa47
        TF=${TF_TREE:?set TF_TREE to the Python reference checkout}
        KN=(); for f in prod-knobs.env r1c-knobs.env; do while read -r kv; do [[ "$kv" =~ ^TF_ ]] && KN+=(-e "$kv"); done < "$KIT/tools/zig/dsv41_m1/$f"; done
        LA=$OUT/long-aot; rm -rf "$LA"
        RUN=(docker run --rm --gpus all --ipc=host --network host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK
             -v "$TF:/dsv41-tf:ro" -v "$MODEL:/model:ro" -v "$KIT:/kit:ro" -v "$OUT:/out" -v "$CACHE_VOL:/cache" "${KN[@]}"
             -e TF_DSV41_GM_V2=1 -e TF_DSV41_EXPERT_TOPP=0.85 -e PYTHONPATH=/dsv41-tf/src:/cache/pylib
             -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton
             -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1} -e NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1}
             -e GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1} --entrypoint python)
        "${RUN[@]}" --name "dsv41-zig-long-r$RANK" "$IMAGE" -u -B /kit/tools/zig/dsv41_m2b_ref.py --pack /model --out /out/long-aot --layers 0-24 --prompt 8448 \
            --steps 1 --limit 10496 --rank "$RANK" --master "$MASTER" --port 29831 > "$OUT/long-aot.log" 2>&1 \
            || { echo "long-aot reference failed: $OUT/long-aot.log"; exit 1; }
        python3 "$KIT/tools/zig/dsv41_aot_merge.py" "$ASSETS/aot" "$LA/aot" | tee -a "$OUT/long-aot.log"
        "${RUN[@]}" --name "dsv41-zig-rope-r$RANK" "$IMAGE" -B /kit/tools/zig/dsv41_rope_tables.py --config /model/config.json --rows "${LONG_ROWS:-1050624}" \
            --out /out/rope-long.bin >> "$OUT/long-aot.log" 2>&1 && cp "$OUT/rope-long.bin" "$ASSETS/rope.bin" \
            && echo "rope.bin: ${LONG_ROWS:-1050624} rows" | tee -a "$OUT/long-aot.log" ;;
    taps-aot)
        # a drafting server's prefill keeps DSpark's taps (the tap layers' `_site` with TAP_ON): Python's prefill of
        # the AOT prompt lengths with taps kept, at TP=2 over the two hosts on the whole model, merged into ASSETS/aot
        C=8474f3164ea60cb9c9f8e9bd5a4493fdf13eaa47
        TF=${TF_TREE:?set TF_TREE to the Python reference checkout}
        KN=(); for f in prod-knobs.env r1c-knobs.env; do while read -r kv; do [[ "$kv" =~ ^TF_ ]] && KN+=(-e "$kv"); done < "$KIT/tools/zig/dsv41_m1/$f"; done
        TA=$OUT/taps-aot; rm -rf "$TA"
        docker run --rm --gpus all --ipc=host --network host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK \
            -v "$TF:/dsv41-tf:ro" -v "$MODEL:/model:ro" -v "$KIT:/kit:ro" -v "$OUT:/out" -v "$CACHE_VOL:/cache" "${KN[@]}" \
            -e TF_DSV41_EXPERT_TOPP=0.85 -e PYTHONPATH=/dsv41-tf/src:/cache/pylib \
            -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton \
            -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1} -e NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1} \
            -e GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1} --name "dsv41-zig-taps-r$RANK" \
            --entrypoint python "$IMAGE" -u -B /kit/tools/zig/dsv41_m2b_ref.py --pack /model --out /out/taps-aot --layers 0-39 \
            --prompt 64 --steps 1 --aot-taps --aot-prompts "${AOT_PROMPTS-1,2,9,16,17,33,45,54,64,65,96,173,300,1025,2049,2600,4090}" \
            --rank "$RANK" --master "$MASTER" --port 29841 > "$OUT/taps-aot.log" 2>&1 \
            || { echo "taps-aot reference failed: $OUT/taps-aot.log"; exit 1; }
        python3 "$KIT/tools/zig/dsv41_aot_merge.py" "$ASSETS/aot" "$TA/aot" | tee -a "$OUT/taps-aot.log" ;;
    pool-aot)
        # prompts prefilled into the paged pool (the `_fused` variants with PT, every row count's % 16 / == 1, every
        # layer kind): SLOTS > 1 or POOL_TOKENS serves every prompt there. dsv41_m2b_ref.py --pool (POOL_AOT_MODE=split:
        # --split, a tree whose Pool takes split=) at TP=2 over the two hosts on layers 0-24, merged into ASSETS/aot
        # 4,090: a last segment whose key count is not % 16 while its candidate blocks (cdiv(n, 8)) are: layer 20's
        # _block_keys (n = 113..127 mod 128); a served prompt's last segment ends anywhere
        # long prompts (CONTEXT past 4,096; their paged _scores / _stream variants): AOT_PROMPTS=8448 POOL_AOT_LIMIT=10496
        # decode windows at every key count's % 16 and past the top-k (the paged decode _scores): POOL_AOT_STEPS=600 decodes
        # that many steps from the POOL_AOT_PROMPT-token prompt (POOL_AOT_PROMPT=8448 POOL_AOT_STEPS=40: long positions)
        C=8474f3164ea60cb9c9f8e9bd5a4493fdf13eaa47
        TF=${TF_TREE:?set TF_TREE to the Python reference checkout}
        KN=(); for f in prod-knobs.env r1c-knobs.env; do while read -r kv; do [[ "$kv" =~ ^TF_ ]] && KN+=(-e "$kv"); done < "$KIT/tools/zig/dsv41_m1/$f"; done
        PA=$OUT/pool-aot; rm -rf "$PA"
        docker run --rm --gpus all --ipc=host --network host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK \
            -v "$TF:/dsv41-tf:ro" -v "$MODEL:/model:ro" -v "$KIT:/kit:ro" -v "$OUT:/out" -v "$CACHE_VOL:/cache" "${KN[@]}" \
            -e TF_DSV41_EXPERT_TOPP=0.85 -e PYTHONPATH=/dsv41-tf/src:/cache/pylib \
            -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton \
            -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1} -e NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1} \
            -e GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1} --name "dsv41-zig-pool-r$RANK" \
            --entrypoint python "$IMAGE" -u -B /kit/tools/zig/dsv41_m2b_ref.py --pack /model --out /out/pool-aot --layers 0-24 \
            --"${POOL_AOT_MODE:-pool}" --prompt "${POOL_AOT_PROMPT:-64}" --steps "${POOL_AOT_STEPS:-1}" --aot-taps --limit "${POOL_AOT_LIMIT:-4096}" \
            --aot-prompts "${AOT_PROMPTS-1,2,9,16,17,33,45,48,54,64,65,96,173,300,1025,2049,2600,4090}" \
            --rank "$RANK" --master "$MASTER" --port 29861 > "$OUT/pool-aot.log" 2>&1 \
            || { echo "pool-aot reference failed: $OUT/pool-aot.log"; exit 1; }
        python3 "$KIT/tools/zig/dsv41_aot_merge.py" "$ASSETS/aot" "$PA/aot" | tee -a "$OUT/pool-aot.log" ;;
    follow)
        docker run "${ARGS[@]}" --name dsv41-zig-m4-r$RANK --entrypoint bash "$IMAGE" \
            -c "$NCCL; exec /kit/tf-dsv41-m1 follow /model /assets" 2>&1 | tee -a "$OUT/follow.log" ;;
    *) echo "phase: server | generate | follow | dspark-aot | rows-aot | long-aot | taps-aot | pool-aot"; exit 2 ;;
esac
