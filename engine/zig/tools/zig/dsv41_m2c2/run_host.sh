#!/usr/bin/env bash
# M2c-2 on the two DGX Sparks, one host's rank: the Python engine's prod tree (8474f31, prod's image and cache volume)
# at TP=2 over the whole backbone + head as the reference (a 1,536-token prompt, then greedy decoding at expert top-p
# 0.85 with M1's prod kernel knobs: the launches block.zig emits), then our forward at TP=2 from its state over the
# hybrid collective (RoCE small exchanges, NCCL large), every token / logits digest / first-window block compared.
# Start it on both hosts within about a minute (each phase's rendezvous waits up to 30 min for the other host):
#   rank 0 host:  ./run_host.sh 0 OUTDIR
#   rank 1 host:  ./run_host.sh 1 OUTDIR
# Single-pass window knobs: RT_KNOBS (RT=1's decode knobs, KEY=value separated by spaces or commas; default the prod set below),
# PF_KNOBS (PREFILL=1: extra KEY=value knobs for the Zig prefill gate's ranks, e.g. TF_DSV41_PF_4K=1 ...),
# SKIP_REF (1: reuse OUTDIR's reference of the same seed, for a rerun), DECODE (0: skip the decode gate, e.g. a rerun
# of the prefill gate alone). LEASE_FILE=/nonexistent when the caller holds the lease itself (zig-gate.sh).
set -u
RANK=${1:?rank}; OUT=${2:?outdir}
RT_KNOBS=${RT_KNOBS-TF_DSV41_L2PF=1 TF_DSV41_ENGRAM_GATE=1 TF_DSV41_BRANCHES=1 TF_DSV41_BRANCHES_PRIO=side TF_DSV41_MHC_DEFER=1 GLM53_TF_ROCE_FAST=1 TF_DSV41_GREEDY_GPU=1}
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# rank 0 host's test-window lease (Spark-tool-kit's campaign.sh windows): held = the Sparks are not ours, nothing starts
LEASE=${LEASE_FILE:-$KIT/.tensorfold-lease}
if [[ "$RANK" == 0 && -f "$LEASE" ]]; then echo "the test-window lease is held ($LEASE): another window owns the Sparks"; exit 3; fi
MASTER=${MASTER:?set MASTER to rank 0 rendezvous address}
IMAGE=${IMAGE:?set IMAGE to the CUDA Python reference image}
CACHE_VOL=${CACHE_VOL:?set CACHE_VOL to the Python reference cache volume}
COMMIT=8474f3164ea60cb9c9f8e9bd5a4493fdf13eaa47
TF=${TF_TREE:?set TF_TREE to the Python reference checkout}; MODEL=${PACK:?set PACK to the model pack directory}
STEPS=${STEPS:-512}; SEEDS=${SEEDS:-4101}; PROMPT=${PROMPT:-1536}; BACKEND=${BACKEND:-roce}
# CED replay runs only in the server's shape: the last token a verify row
if [[ "${PREFILL_MODE:-full}" == replay ]]; then PROMPT_TAIL=verify; fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
log() { echo "[m2c2 r$RANK $(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
drop_caches() { sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || log "drop_caches refused"; }

# preflight: nothing else on the GPU, the kit intact, the tree and pack where prod keeps them
# PIECES_DRYRUN=1 (with PIECES=1 PIECES_ONLY=1, on any host): the stage's own code without containers: the run's assets
# dir, its env and pieces_check.py's preflight of every file and limit each Zig run boots with, on a fresh OUT too (the
# files the stage builds itself are "pend", the prompts' limits from PIECES_PROMPTS; an OUT a real run left is checked
# in full)
DRY=${PIECES_DRYRUN:-0}
if [[ "$DRY" != 1 ]]; then
if docker ps --format '{{.Names}}' | grep -q 'dsv41'; then log "a dsv41 container is running: stop prod first"; exit 2; fi
(cd "$KIT" && sha256sum -c --quiet SHA256SUMS) || { log "kit checksum mismatch"; exit 2; }
[[ -d "$TF/src" && -f "$MODEL/config.json" ]] || { log "missing $TF/src or $MODEL/config.json"; exit 2; }
fi
ulimit -l unlimited 2> /dev/null || log "warning: memlock $(ulimit -l) KiB"
{ hostname; nvidia-smi -L; free -g; } > "$OUT/host.txt" 2>&1
ARGS=(--rm --gpus all --ipc=host --network host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK
      -v "$TF:/dsv41-tf:ro" -v "$MODEL:/model:ro" -v "$CACHE_VOL:/cache" -v "$KIT:/kit:ro" -v "$OUT:/out"
      -e PYTHONPATH=/dsv41-tf/src:/cache/pylib -e PYTHONDONTWRITEBYTECODE=1 -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions
      -e TRITON_CACHE_DIR=/cache/triton -e CUDA_CACHE_PATH=/cache/nv/ComputeCache
      -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1} -e NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1}
      -e GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1})
KNOBS=(); while read -r kv; do [[ "$kv" =~ ^TF_ ]] && KNOBS+=(-e "$kv"); done < "$KIT/tools/zig/dsv41_m1/prod-knobs.env"
# R1=1 (default; prod-perf1 runs it): r1c-knobs.env on top, and ours follows the reference's env (block.zig Options.r1)
if [[ "${R1:-1}" == 1 ]]; then while read -r kv; do [[ "$kv" =~ ^TF_ ]] && KNOBS+=(-e "$kv"); done < "$KIT/tools/zig/dsv41_m1/r1c-knobs.env"; fi

port=29811
# PIECES_ONLY=1: only the PIECES=1 stage below (no seed's reference or decode gate)
[[ "${PIECES_ONLY:-0}" == 1 ]] && SEEDS=""
for seed in $SEEDS; do
    R=/out/seed$seed
    mkdir -p "$OUT/seed$seed"
    # 1. the reference: Python at TP=2 (this host's rank), state + tokens + digests + the AOT set + rope tables
    if [[ "${SKIP_REF:-0}" == 1 && -f "$OUT/seed$seed/ref/engram-host.bin" ]]; then
        log "seed $seed: reusing the reference in $OUT/seed$seed/ref (SKIP_REF=1)"
    else
    drop_caches
    log "seed $seed: Python reference (layers 0-39, prompt $PROMPT, $STEPS greedy steps, top-p 0.85)"
    t0=$(date +%s)
    docker run "${ARGS[@]}" "${KNOBS[@]}" -e TF_DSV41_EXPERT_TOPP=0.85 --name "dsv41-zig-ref-r$RANK" \
        --entrypoint python "$IMAGE" -u -B /kit/tools/zig/dsv41_m2b_ref.py --pack /model --out "$R/ref" --aot-prompts "${AOT_PROMPTS-1,2,9,16,17,33,45,64,65,96,173,300,1025,2049,2600}" \
        --layers 0-39 --prompt "$PROMPT" --steps "$STEPS" --seed "$seed" --rank "$RANK" --master "$MASTER" --port $port \
        ${PROF_ROWS:+--profile "$PROF_ROWS"} ${PROMPT_TAIL:+--prompt-tail "$PROMPT_TAIL"} ${PREFILL_MODE:+--prefill "$PREFILL_MODE"} \
        $( [[ "${PROF_PREFILL:-0}" == 1 ]] && echo --profile-prefill "$PROMPT" ) ${LIMIT:+--limit "$LIMIT"} > "$OUT/seed$seed/ref.log" 2>&1
    rc=$?; log "seed $seed: reference rc $rc in $(( $(date +%s) - t0 )) s ($(tail -1 "$OUT/seed$seed/ref.log"))"
    (( rc == 0 )) || exit 1
    docker run "${ARGS[@]}" --entrypoint python "$IMAGE" -B /kit/tools/zig/dsv41_engram_host.py --py-src /dsv41-tf/src \
        --config /model/config.json --tokenizer /model/tokenizer.json --out "$R/ref/engram-host.bin" > "$OUT/seed$seed/engram-host.log" 2>&1 \
        || { log "engram host tables failed"; exit 1; }
    fi
    # 2. ours: the same rank from the reference's state, over the hybrid collective
    if [[ "${DECODE:-1}" == 1 ]]; then
    drop_caches
    log "seed $seed: our forward (backend $BACKEND${RT:+, RT=$RT: $RT_KNOBS})"
    t0=$(date +%s)
    docker run "${ARGS[@]}" -e TF_TP_WORLD=2 -e TF_TP_RANK="$RANK" -e TF_TP_DEVICE=0 -e TF_TP_MASTER="$MASTER" \
        -e TF_TP_PORT=$((port + 10)) -e TF_DSV41_PLAN_TIMEOUT=1800 -e TF_COMM_BACKEND="$BACKEND" -e TF_TP_ROCE_FALLBACK=error \
        -e TF_TP_KERNEL_IMAGE=/kit/tp_mailbox.fatbin -e M2B_TOPP=0.85 -e TF_DSV41_FAILFAST=0 -e M2B_PROF_ROWS="${PROF_ROWS:-}" \
        $( [[ "${RT:-0}" == 1 && -n "$RT_KNOBS" ]] && printf -- '-e %s ' ${RT_KNOBS//,/ } ) \
        --name "dsv41-zig-m2c2-r$RANK" \
        --entrypoint bash "$IMAGE" -c 'export TF_NCCL_LIB=$(python -c "import nvidia.nccl, os; print(os.path.join(os.path.dirname(nvidia.nccl.__file__), \"lib\", \"libnccl.so.2\"))" 2>/dev/null || echo libnccl.so.2); exec /kit/tf-dsv41-m1 m2b /model '"$R/ref"' '"$R/zig" \
        > "$OUT/seed$seed/zig.log" 2>&1
    rc=$?; log "seed $seed: ours rc $rc in $(( $(date +%s) - t0 )) s: $(grep -h '^PASS\|^FAIL' "$OUT/seed$seed/zig.log" | tail -1)"
    fi
    # 2b. optional (PREFILL=1): Zig's own prefill from an empty slot (M2B_PREFILL=1): every state role == the
    #     reference's dump and its first token, then the same decode gate; about +10 min (one more load)
    if [[ "${PREFILL:-0}" == 1 ]]; then
        drop_caches
        log "seed $seed: Zig prefill gate"
        t0=$(date +%s)
        docker run "${ARGS[@]}" -e TF_TP_WORLD=2 -e TF_TP_RANK="$RANK" -e TF_TP_DEVICE=0 -e TF_TP_MASTER="$MASTER" \
            -e TF_TP_PORT=$((port + 13)) -e TF_DSV41_PLAN_TIMEOUT=1800 -e TF_COMM_BACKEND="$BACKEND" -e TF_TP_ROCE_FALLBACK=error \
            -e TF_TP_KERNEL_IMAGE=/kit/tp_mailbox.fatbin -e M2B_TOPP=0.85 -e M2B_PREFILL=1 -e TF_DSV41_FAILFAST=0 -e TF_DSV41_GM_V2="${GM_V2:-}" -e TF_DSV41_PREFILL="${PREFILL_MODE:-}" --name "dsv41-zig-pf-r$RANK" \
            $( [[ "${PROF_PREFILL:-0}" == 1 ]] && printf '%s %s' -e "TF_DSV41_PROFILE=$R/zig-prefill/prof-rank$RANK.json" ) \
            $( [[ -n "${PF_KNOBS:-}" ]] && printf -- '-e %s ' ${PF_KNOBS//,/ } ) \
            --entrypoint bash "$IMAGE" -c 'export TF_NCCL_LIB=$(python -c "import nvidia.nccl, os; print(os.path.join(os.path.dirname(nvidia.nccl.__file__), \"lib\", \"libnccl.so.2\"))" 2>/dev/null || echo libnccl.so.2); mkdir -p '"$R/zig-prefill"'; exec /kit/tf-dsv41-m1 m2b /model '"$R/ref"' '"$R/zig-prefill" \
            > "$OUT/seed$seed/prefill.log" 2>&1
        rc=$?; log "seed $seed: prefill gate rc $rc in $(( $(date +%s) - t0 )) s: $(grep -h '^PASS Zig\|^FAIL Zig' "$OUT/seed$seed/prefill.log" | tail -1) / $(grep -h '^PASS M2b\|^FAIL M2b' "$OUT/seed$seed/prefill.log" | tail -1)"
    fi
    # 3. optional (LANES=1): M3's first gate, the same decode through the lanes engine on rank 0 (drafts off), rank 1
    #    following over the plan link; tokens == the reference's
    if [[ "${LANES:-0}" == 1 ]]; then
        drop_caches
        log "seed $seed: lanes over the GPU target (M3)"
        t0=$(date +%s)
        docker run "${ARGS[@]}" -e TF_TP_WORLD=2 -e TF_TP_RANK="$RANK" -e TF_TP_DEVICE=0 -e TF_TP_MASTER="$MASTER" \
            -e TF_TP_PORT=$((port + 15)) -e TF_DSV41_PLAN_TIMEOUT=1800 -e TF_COMM_BACKEND="$BACKEND" -e TF_TP_ROCE_FALLBACK=error \
            -e TF_TP_KERNEL_IMAGE=/kit/tp_mailbox.fatbin -e M2B_TOPP=0.85 -e M2B_LANES=1 -e TF_DSV41_FAILFAST=0 --name "dsv41-zig-m3-r$RANK" \
            --entrypoint bash "$IMAGE" -c 'export TF_NCCL_LIB=$(python -c "import nvidia.nccl, os; print(os.path.join(os.path.dirname(nvidia.nccl.__file__), \"lib\", \"libnccl.so.2\"))" 2>/dev/null || echo libnccl.so.2); mkdir -p '"$R/zig-lanes"'; exec /kit/tf-dsv41-m1 m2b /model '"$R/ref"' '"$R/zig-lanes" \
            > "$OUT/seed$seed/lanes.log" 2>&1
        rc=$?; log "seed $seed: lanes rc $rc in $(( $(date +%s) - t0 )) s: $(grep -h '^PASS\|^FAIL' "$OUT/seed$seed/lanes.log" | tail -1)"
    fi
    port=$((port + 20))
done
# PIECES=1 (TF_DSV41_PIECE_RUNS / _REPLAY_RUNS, gate; the pod form is tools/zig/dsv41_pieces/job.sh): several
# prompts admitted in one round, at the served config (the kit's triton-fill/fill-prod.env: TP=2, 4 slots, 1M context,
# replay, R1's knobs) so the kit's AOT fill (built for that env, the runs' variants included) covers every launch.
#  1. Python at TP=2 with prod's knobs: the batched round (dsv41_m2b_ref.py --batched: one CED encoder run over every
#     prompt's piece, slots.prefill_runs) and the same prompts alone (Python's batched == alone, for the record);
#  2. ours serving the prompts at once through the served engine (rank 0 tf-dsv41-m1 sess4, rank 1 follow) with the
#     round planner, against the batched reference: runs 1 (TF_DSV41_PIECE_RUNS=1), 0 (a piece at a time), 2 (piece runs
#     and TF_DSV41_REPLAY_RUNS=1). Each must print "PASS sess4 turn 1 == Python"; runs 1 / 2 must log "piece runs: " /
#     "replay runs: ". The ASSETS' AOT set is the reference's plus the kit's fill.
# Every stage runs under a timeout (PIECES_REF_MIN 20, PIECES_RUN_MIN 12; its container removed on expiry) and the
# first failure stops the stage with exit 1 ("FAIL pieces: ..."); rank 0's last line is "PASS pieces" or "FAIL pieces".
# The two hosts share a status channel (tools/zig/dsv41_m2c2/peer.py, rank 0 serving on $MASTER:PIECES_PEER_PORT, 29890):
# each posts a heartbeat and, on its failure, "FAIL <reason>"; a watcher beside every container removes it as soon as
# the other rank fails or stops answering, so neither waits out a deadline on a dead peer. The AOT merges and the
# tokens' comparison run in the image as the references wrote their files (root), not as the host user.
# PIECES_PROMPTS (160 300 500 700: every piece past 16 rows, their rows within one 2,048-row run), PIECES_STEPS (24),
# PIECES_RUNS ("1 0 2", or "0,2": the runs to do). Each run starts when both ranks posted ready on the peer channel
# (PIECES_BARRIER_S, 900); rank 0 posts the run's pass once its own checks hold, and rank 1 takes it as the run's verdict
# (its follower's exit 70 when rank 0's process ends is then no failure; PIECES_PASS_S, 120). About 30 min.
if [[ "${PIECES:-0}" == 1 ]]; then
    P=/out/pieces; mkdir -p "$OUT/pieces"
    prompts=${PIECES_PROMPTS:-160 300 500 700}; psteps=${PIECES_STEPS:-24}
    fillenv=$KIT/triton-fill/fill-prod.env
    peerport=${PIECES_PEER_PORT:-29890}; peer=(python3 -B "$KIT/tools/zig/dsv41_m2c2/peer.py")
    srvpid=""; beatpid=""
    pstop() { [[ -n "$beatpid" ]] && kill "$beatpid" 2>/dev/null; [[ -n "$srvpid" ]] && kill "$srvpid" 2>/dev/null; beatpid=""; srvpid=""; }
    pfail() {
        log "FAIL pieces: $*"
        [[ -n "$beatpid" ]] && kill "$beatpid" 2>/dev/null; beatpid=""
        "${peer[@]}" set "$MASTER" "$peerport" "$RANK" "FAIL $*" > /dev/null 2>&1
        [[ "$RANK" == 0 ]] && sleep 15      # the server lives long enough for rank 1's watcher to read it
        pstop
        exit 1
    }
    trap pstop EXIT
    if [[ "$RANK" == 0 ]]; then "${peer[@]}" serve "$peerport" >> "$OUT/pieces/peer.log" 2>&1 & srvpid=$!; sleep 1; fi
    "${peer[@]}" beat "$MASTER" "$peerport" "$RANK" $$ >> "$OUT/pieces/peer.log" 2>&1 & beatpid=$!
    # a container under a deadline (removed on expiry: the client's exit does not stop it) and the peer's watch
    timed() {
        local min=$1 name=$2 wpid rc; shift 2
        rm -f "$OUT/pieces/peer-$name"
        "${peer[@]}" watch "$MASTER" "$peerport" "$RANK" "$name" > "$OUT/pieces/peer-$name" 2>&1 & wpid=$!
        timeout --kill-after=30 "$(( min * 60 ))" docker run --name "$name" "$@"
        rc=$?
        kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null
        docker rm -f "$name" > /dev/null 2>&1
        (( rc == 124 || rc == 137 )) && log "pieces: $name passed its $min min deadline"
        if [[ -s "$OUT/pieces/peer-$name" ]]; then pfail "$(cat "$OUT/pieces/peer-$name")"; fi
        return $rc
    }
    [[ -f "$fillenv" && -f "$KIT/triton-fill/aot/aot.json" ]] || pfail "the kit has no triton-fill (build-kit.sh TRITON_FILL=1 or TRITON_FILL_FROM)"
    grep -q '^TF_DSV41_PIECE_RUNS=1' "$fillenv" || pfail "the kit's fill was built without the runs' variants ($fillenv: TF_DSV41_PIECE_RUNS / _REPLAY_RUNS)"
    # the served assets' full rope table (prod's TF_DSV41_CONTEXT): the runs boot at the fill's 1M context
    served=${PIECES_SERVED_ASSETS:?set PIECES_SERVED_ASSETS to the served assets directory}
    [[ -f "$served/rope.bin" ]] || pfail "no served rope table at $served/rope.bin (PIECES_SERVED_ASSETS)"
    if [[ "$DRY" != 1 ]]; then
    refargs=(--pack /model --layers 0-39 --steps "$psteps" --prompts "$(echo $prompts | tr ' ' ',')" --prefill replay
             --prompt-tail verify --limit 4096 --rank "$RANK" --master "$MASTER")
    pport=$port
    for kind in batched alone; do
        pport=$(( pport + 2 ))
        drop_caches
        log "pieces: Python reference ($kind, prompts $prompts)"
        t0=$(date +%s)
        timed "${PIECES_REF_MIN:-20}" "dsv41-zig-pref-r$RANK" "${ARGS[@]}" "${KNOBS[@]}" -e TF_DSV41_EXPERT_TOPP=0.85 \
            --entrypoint python "$IMAGE" -u -B /kit/tools/zig/dsv41_m2b_ref.py "${refargs[@]}" \
            $( [[ $kind == batched ]] && echo --batched ) --out "$P/ref-$kind" --port "$pport" > "$OUT/pieces/ref-$kind.log" 2>&1
        rc=$?; log "pieces: $kind reference rc $rc in $(( $(date +%s) - t0 )) s ($(tail -1 "$OUT/pieces/ref-$kind.log"))"
        (( rc == 0 )) || pfail "the $kind reference failed (rc $rc, ref-$kind.log)"
    done
    timed 5 "dsv41-zig-peng-r$RANK" "${ARGS[@]}" --entrypoint python "$IMAGE" -B /kit/tools/zig/dsv41_engram_host.py \
        --py-src /dsv41-tf/src --config /model/config.json --tokenizer /model/tokenizer.json \
        --out "$P/ref-batched/engram-host.bin" > "$OUT/pieces/engram-host.log" 2>&1 || pfail "engram host tables"
    for src in "$P/ref-alone/aot" /kit/triton-fill/aot; do
        timed 3 "dsv41-zig-pmerge-r$RANK" "${ARGS[@]}" --entrypoint python "$IMAGE" -B /kit/tools/zig/dsv41_aot_merge.py \
            "$P/ref-batched/aot" "$src" >> "$OUT/pieces/aot-merge.log" 2>&1 \
            || pfail "merging $src into the reference's AOT set (aot-merge.log)"
    done
    timed 2 "dsv41-zig-pcmp-r$RANK" "${ARGS[@]}" --entrypoint python "$IMAGE" -B -c '
import json, sys
d, ps = sys.argv[1], sys.argv[2:]
same = sum(json.load(open(f"{d}/ref-batched/p{p}/ref.json"))["tokens"] == json.load(open(f"{d}/ref-alone/p{p}/ref.json"))["tokens"] for p in ps)
print(f"pieces: Python batched == alone, tokens: {same}/{len(ps)} (informational)")' "$P" $prompts > "$OUT/pieces/compare.log" 2>&1
    log "$(tail -1 "$OUT/pieces/compare.log")"
    fi
    # the runs' assets: the reference's AOT set (the kit's fill merged in) and Engram host tables, the served rope table
    # (mounted at /served-assets: the reference's covers its 4,096-token limit, the runs need TF_DSV41_CONTEXT + 2,048)
    mkdir -p "$OUT/pieces/assets"
    ln -sfn ../ref-batched/aot "$OUT/pieces/assets/aot"
    ln -sfn ../ref-batched/engram-host.bin "$OUT/pieces/assets/engram-host.bin"
    ln -sfn /served-assets/rope.bin "$OUT/pieces/assets/rope.bin"
    A=$P/assets
    refs=(); for p in $prompts; do refs+=("$P/ref-batched/p$p"); done
    runlist=${PIECES_RUNS:-1 0 2}; runlist=${runlist//,/ }
    other=$(( 1 - RANK ))
    # a mark on the peer channel (a run's ready / pass), retried while rank 0's server comes up
    ppost() {
        local i
        for i in $(seq 90); do "${peer[@]}" set "$MASTER" "$peerport" "$1" "$2" > /dev/null 2>&1 && return 0; sleep 2; done
        return 1
    }
    for runs in $runlist; do
        # both ranks ready before either starts the run (the other may still be in its previous run's ending)
        ppost "ready.$runs.$RANK" "ready" || pfail "run $runs: the peer channel at $MASTER:$peerport"
        msg=$("${peer[@]}" await "$MASTER" "$peerport" "ready.$runs.$other" "${PIECES_BARRIER_S:-900}" "$other") \
            || pfail "run $runs: rank $other not ready ($msg)"
        drop_caches
        pr=$(( runs > 0 )); rr=$(( runs == 2 ))
        log "pieces: ours, TF_DSV41_PIECE_RUNS=$pr TF_DSV41_REPLAY_RUNS=$rr"
        t0=$(date +%s)
        if [[ "$RANK" == 0 ]]; then cmd="sess4 /model $A $P/zig-runs$runs ${refs[*]}"; else cmd="follow /model $A"; fi
        # the run's env after the fill's (docker: the last value of a key wins); the preflight reads the same list
        zenv=(TF_TP_WORLD=2 TF_TP_RANK="$RANK" TF_TP_DEVICE=0 TF_TP_MASTER="$MASTER" TF_TP_PORT=$(( port + 5 + runs ))
              TF_DSV41_PLAN_TIMEOUT=600 TF_COMM_BACKEND="$BACKEND" TF_TP_ROCE_FALLBACK=error TF_TP_KERNEL_IMAGE=/kit/tp_mailbox.fatbin
              TF_DSV41_FAILFAST=1 TF_DSV41_ASSETS="$A" TF_DSV41_POOL_TOKENS=1400000 TF_DSV41_PROMPT_TAIL=verify
              TF_DSV41_OWN_PREFILL=1 TF_DSV41_PREFILL_PIECES=1 TF_DSV41_SESSIONS=0 TF_DSV41_GRAPHS=1
              TF_DSV41_PIECE_RUNS=$pr TF_DSV41_REPLAY_RUNS=$rr SESS4_TURNS=1 SESS4_STEPS="$psteps")
        mounts=(/out="$OUT" /kit="$KIT" /served-assets="$served" /model="$MODEL")
        names=(); [[ -f "$KIT/tools/zig/dsv41_m2c2/knob-names.txt" ]] && names=(--names "$KIT/tools/zig/dsv41_m2c2/knob-names.txt")
        python3 -B "$KIT/tools/zig/dsv41_m2c2/pieces_check.py" --env-file "$fillenv" $(printf -- '--set %s ' "${zenv[@]}") \
            $(printf -- '--map %s ' "${mounts[@]}") --assets "$A" --refs "${refs[@]}" --kit-aot "$KIT/triton-fill/aot" \
            --steps "$psteps" "${names[@]}" $( [[ "$DRY" == 1 ]] && echo --pending --prompt-lens "$(echo $prompts | tr ' ' ',')" ) \
            > "$OUT/pieces/check-runs$runs.log" 2>&1
        crc=$?; log "pieces: run $runs preflight: $(tail -1 "$OUT/pieces/check-runs$runs.log")"
        (( crc == 0 )) || pfail "run $runs preflight ($(grep -h '^pieces_check: FAIL [^0-9]' "$OUT/pieces/check-runs$runs.log" | head -3 | cut -c20- | tr '\n' ';'))"
        if [[ "$DRY" == 1 ]]; then
            echo "PASS sess4 turn 1 == Python: (dry run)" > "$OUT/pieces/zig-runs$runs.log"
            [[ "$RANK" == 0 ]] && { echo "piece runs: (dry)"; echo "replay runs: (dry)"; } >> "$OUT/pieces/zig-runs$runs.log"
            rc=${PIECES_DRY_RC:-0}     # the dry run's stand-in for the run's exit (70: a follower's fail-fast at rank 0's end)
        else
        timed "${PIECES_RUN_MIN:-12}" "dsv41-zig-pieces-r$RANK" "${ARGS[@]}" -v "$served:/served-assets:ro" --env-file "$fillenv" \
            $(printf -- '-e %s ' "${zenv[@]}") \
            --entrypoint bash "$IMAGE" -c 'export TF_NCCL_LIB=$(python -c "import nvidia.nccl, os; print(os.path.join(os.path.dirname(nvidia.nccl.__file__), \"lib\", \"libnccl.so.2\"))" 2>/dev/null || echo libnccl.so.2); mkdir -p '"$P/zig-runs$runs"'; exec /kit/tf-dsv41-m1 '"$cmd" \
            > "$OUT/pieces/zig-runs$runs.log" 2>&1
        rc=$?
        fi
        verdict=$(grep -h '^PASS sess4\|^FAIL sess4' "$OUT/pieces/zig-runs$runs.log" | tail -1)
        log "pieces: run $runs rc $rc in $(( $(date +%s) - t0 )) s: ${verdict:-no verdict}"
        if [[ "$RANK" == 0 ]]; then
            (( rc == 0 )) || pfail "run $runs (rc $rc, zig-runs$runs.log: $(grep -m1 -h 'error\|panic\|MissingTriton\|no captured' "$OUT/pieces/zig-runs$runs.log" | cut -c1-160))"
            [[ "$verdict" == PASS* ]] || pfail "run $runs: ${verdict:-no turn-1 verdict}"
            (( pr == 0 )) || grep -q "piece runs: " "$OUT/pieces/zig-runs$runs.log" || pfail "run $runs logged no multi-segment run (piece runs)"
            (( rr == 0 )) || grep -q "replay runs: " "$OUT/pieces/zig-runs$runs.log" || pfail "run $runs logged no batched decoder replay (replay runs)"
            # the run's verdict for rank 1: its follower ends when rank 0's does (fail-fast's exit 70 included)
            ppost "pass.$runs" "PASS run $runs" || pfail "run $runs: could not post its pass"
        else
            # the verdict is rank 0's: a follower's exit after rank 0 posted the pass (rc 70: fail-fast once rank 0's
            # process ended) is a pass; rank 0's FAIL, or no pass, is a failure
            msg=$("${peer[@]}" await "$MASTER" "$peerport" "pass.$runs" "${PIECES_PASS_S:-120}" 0) \
                || pfail "run $runs: rank 0's verdict ($msg; this rank's rc $rc, zig-runs$runs.log: $(grep -m1 -h 'error\|panic' "$OUT/pieces/zig-runs$runs.log" | cut -c1-120))"
            (( rc == 0 )) || log "pieces: run $runs: rc $rc after rank 0's pass (fail-fast at rank 0's exit): pass"
        fi
    done
    [[ "$DRY" == 1 ]] && { log "PASS pieces dry run: runs $runlist (preflight, barrier, verdict)"; [[ "$RANK" == 0 ]] && sleep 5; pstop; trap - EXIT; exit 0; }
    log "PASS pieces: runs $runlist == Python's batched reference"
    [[ "$RANK" == 0 ]] && sleep 10      # rank 1's last watcher reads the server once more before it goes
    pstop; trap - EXIT
    port=$((port + 20))
fi
drop_caches
log "done: $(grep -h '^PASS\|^FAIL' "$OUT"/seed*/zig.log | tr '\n' ' ')"
