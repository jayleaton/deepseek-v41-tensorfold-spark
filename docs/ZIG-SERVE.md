# CUDA Zig serving on two DGX Sparks

Recommended weights: **q28-v2 EXL3 2.8 bpw**,
[`jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw`](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw).
Publication is pending and that repository name is provisional. The port loads EXL3 directly, including q28-v2's
per-expert widths. There are two processes: `tensorfold-dsv41 serve` on rank 0 and `tf-dsv41-m1 follow` on rank 1.

## Availability and requirements

The measured engine is TensorFold 1.0.2 plus the CUDA DeepSeek port. **That source export and the generated AOT
asset set are not included in this public recipe yet.** Upstream TensorFold 1.0.2 alone does not contain this CUDA
port, and the existing Python patches here target 0.6.0. The instructions below require an exported port source
snapshot with `build.zig`, `zig/`, and its build dependencies. They are not a fresh-clone installation path until
that export is published. Do not substitute upstream 1.0.2 and expect these binaries or switches to exist.

Required on both nodes:

- Two DGX Sparks (GB10, 128 GB each), TP=2, cabled/addressed CX7 ports; Docker with NVIDIA Container Toolkit,
  RDMA devices and matching image contents. Set the interface/HCA values to your own link.
- Identical q28-v2 model files; rank-specific packed Engram shards on local NVMe. [Engram packing](INSTALL.md)
  writes `engram-l{1,14}-r{rank}of2.bin`. The weights download alone is not sufficient.
- Rank-specific `/assets/rope.bin`, `engram-host.bin`, and `aot/`: RoPE tables covering the configured context plus
  prefill rows, native host lookup data, and Triton programs for all emitted shapes. At four slots, include DSpark,
  row-mode up to 64 rows, CED replay, and the device chain variants. PF_4K requires the matching 4K fill.
- `/cache/dsv41-calib` holding Python's measured cost tables and `/cache/pfdense-table.json` (the published
  `config/pfdense-table.json`). `CALIB=real` reads a stored table in Zig; it does not measure one itself.
- Writable prepared and session directories on local NVMe. Plan roughly 100 GB weights, 95 GB Engram shards,
  another rank's weight-sized prepared cache and up to 128 GiB session files per node, plus image/build space.
  Keep other workloads off the GPUs and preserve the 5 GiB target / 4 GiB hard MemAvailable floor on both ranks.

The source kit's `tools/zig/dsv41_m2c2/build-kit.sh` builds the ARM64 binaries and checks the sm_121 fatbins.
It generates a `knob-names.txt` from the names actually read by Zig. Asset generation uses
`tools/zig/dsv41_rope_tables.py`, `dsv41_engram_host.py`, `triton_fill.py`, the engine's `aot-needs` command and
`dsv41_m4/fill-prod.env` (or the 4K fill). Python reference source, Torch, Triton 3.7.1 and ptxas 13.3.73 are
needed for that fill. Copy assets per rank and validate the fill manifest; a `MissingTritonVariant` is a missing
prerequisite, not a reason to keep running. These generator tools belong to the pending port export.

## Quick start

Run these steps only after the port source and generated assets are available. The build recipe and launcher
are provided here; hardware execution was not repeated for this documentation update.

1. Clone this public recipe and download the pack on both nodes:

   ```bash
   git clone --recurse-submodules https://github.com/jayleaton/deepseek-v41-tensorfold-spark.git
   cd deepseek-v41-tensorfold-spark
   hf download jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw --local-dir /srv/models/q28-v2
   cp config/prod-zig.env.example config/prod-zig.env
   $EDITOR config/prod-zig.env
   ```

   Set the image, rank 0's link address, NIC/HCA settings and each node's own absolute directories. The engine
   settings must match on both ranks. `config/prod-zig.env` is ignored by git. The template uses the conservative
   2K prefill profile; the pack URL remains provisional until publication completes.

2. Build on an ARM64 Spark from an **exported source snapshot**, not private git history:

   ```bash
   mkdir -p build/zig/src build/zig/fatbins
   # Copy the exported port tree into build/zig/src/.
   # Optionally copy its validated sm_121 *.fatbin files, including tp_mailbox.fatbin, into build/zig/fatbins/.
   touch build/zig/fatbins/.keep
   docker build -f docker/zig/Dockerfile --build-arg ZIG_COMMIT="$PORT_REVISION" \
     -t tensorfold-dsv41-zig:local build/zig
   docker save tensorfold-dsv41-zig:local -o build/zig-image.tar
   ```

   Set `PORT_REVISION` to the exported engine revision. Transfer the tar to rank 1 and run
   `docker load -i build/zig-image.tar` there. Compare the label and root filesystem layers on both nodes with
   `docker image inspect tensorfold-dsv41-zig:local`. Set `IMAGE=tensorfold-dsv41-zig:local` in both configs.

   The multi-stage image uses Zig 0.17.0, CUDA 13, ReleaseSafe ARM64, sm_121 kernels, NCCL and verbs providers.
   Its default nvcc image is CUDA 13.0.1; for prebuilt kernels use the exact CUDA devel/runtime versions validated
   with them (build args `CUDA_DEVEL` / `CUDA_RUNTIME`). Use `ZIG_URL` for the exact Zig compiler tarball if the
   default download URL is unavailable. The first start builds prepared files; do not reuse Python's cache.

3. Check the commands, then run rank 1 on its node followed promptly by rank 0 on its node:

   ```bash
   scripts/run-zig.sh 1 --dry-run    # on rank 1
   scripts/run-zig.sh 0 --dry-run    # on rank 0
   scripts/run-zig.sh 1             # rank 1: keep running in this terminal
   scripts/run-zig.sh 0             # rank 0: another terminal on its node
   ```

   The launcher passes the serving settings to both processes, sets TP world/rank/device explicitly, and uses
   Docker `--init` because the follower exits when orphaned. It validates local directories and basic assets;
   it does not replace the engine's memory checks, fill validation, or a production watchdog. Keep the same env
   across restarts. Stop both ranks together with `docker stop tensorfold-zig-r0` / `tensorfold-zig-r1` on their
   respective nodes. A failure on either rank should end both via fail-fast (exit 70).

4. Wait for the model to load and inspect the boot log: four slots, CED replay, row graphs, successful device-draft
   self-check and RoCE probe. First preparation can take several minutes. Then query rank 0:

   ```bash
   curl -fsS localhost:8000/v1/models
   curl -fsS localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"DeepSeek-V4.1-Flash-TF","messages":[{"role":"user","content":"What is 17 * 23?"}],"temperature":0,"reasoning_effort":"low"}'
   ```

   Check that the answer is 391; do not infer readiness from a live container alone.

## Serving settings

These are the exact environment names from the serving profile and the port's knob readers, with the measured
all-on additions. **Default means unset engine behavior**, not the template value. The template explicitly opts
into optimizations whose engine defaults remain off. Both ranks need the same settings. `1` means on and `0`
means off unless the row gives another meaning. The recommended column is what the exported template uses;
`verify (unset)` documents an intentional default. This table describes the inspected port snapshot, not generic
upstream TensorFold settings.

| Name | Unset engine default | Recommended profile | What it does |
| --- | --- | --- | --- |
| `TF_DSV41_R1` | 0 | `1` | Complete decode R1 bundle |
| `TF_DSV41_DRAFTS` | 1 | `1` | DSpark drafting |
| `TF_DSV41_OWN_PREFILL` | 1 | `1` | Native prompt prefill rather than decode windows |
| `TF_DSV41_GRAPHS` | 0 | `1` | CUDA graphs for decode windows |
| `TF_DSV41_PREFILL` | full | `replay` | CED encoder prefill and bounded decoder replay, matching Python |
| `TF_DSV41_PROMPT_TAIL` | verify | `verify (unset)` | Last prompt token enters the first decode window |
| `TF_DSV41_PREFILL_PIECES` | 0 | `1` | Interleave prompt pieces with other slots' decode |
| `TF_DSV41_SESSION_SLOTS` | 0 | `1` | Session store with several live slots |
| `TF_DSV41_TAIL_JOIN` | 0 | `1` | Join prompt tail to the first shared verify round; needs PREFILL_PIECES |
| `TF_DSV41_DRAFT_TIMED` | 0 | `1` | Price draft depth from measured round times |
| `TF_DSV41_SLOTS` | 1 | `4` | Concurrent live slots |
| `TF_DSV41_SLOT_DRAFTS` | 0 | `1` | Per-slot drafts with batched verification |
| `TF_DSV41_ROWS_CAP` | 16 | `64` | Maximum rows per decode forward; requires matching AOT |
| `TF_DSV41_POOL_TOKENS` | unset: no paged pool | `1400000` | Shared paged KV capacity, not capacity per slot |
| `TF_DSV41_SAMPLING` | 0 | `1` | Keyed sampling for nonzero temperature |
| `TF_DSV41_EXPERT_TOPP` | unset: no pruning | `0.85` | Decode routed-expert top-p pruning |
| `TF_DSV41_INDEX_BUDGET_MIB` | 256 | `64` | Index selection scratch budget |
| `TF_DSV41_PREFILL_CHUNK` | 2048 | `2048` | Segment row limit |
| `TF_DSV41_PREFILL_ROWS` | 2048 | `2048` | Second row limit; smaller of CHUNK and ROWS wins |
| `TF_DSV41_PREFETCH_AHEAD` | 0 | `1` | Read next segment's Engram rows ahead |
| `TF_DSV41_PF_DENSE` | off | `fused` | Fused dense EXL3 prefill GEMM |
| `TF_DSV41_PF_DENSE_TABLE` | unset: heuristic | `/cache/pfdense-table.json` | Shape-specific GEMM tuning table |
| `TF_DSV41_GM_V2` | 0 | `1` | x3gm v2 routed-expert prefill |
| `TF_DSV41_BRANCHES` | 0 | `1` | Indexer, compressor and applicable KV work on a side stream |
| `TF_DSV41_BRANCHES_PRIO` | main | `side` | Side stream priority (eager execution) |
| `TF_DSV41_MHC_DEFER` | 0 | `1` | R1 mHC coefficients on their own stream |
| `TF_DSV41_L2PF` | 0 | `1` | Decode L2 prefetch |
| `TF_DSV41_L2PF_MB` | 12 | `12` | Fixed prefetch budget; other values refused |
| `TF_DSV41_L2PF_PACE_GBPS` | 150 | `150` | Fixed prefetch pacing; other values refused |
| `TF_DSV41_ENGRAM_GATE` | 0 | `1` | GPU waits for decode Engram reads |
| `TF_DSV41_GREEDY_GPU` | 0 | `1` | Device greedy choice and device-started draft round |
| `TF_DSV41_SPEC_DRAFT` | 0 | `1` | Start next speculative DSpark pass after verify |
| `TF_DSV41_DRAFT_GRAPHS` | 0: host path | `1` | Device candidates and graphed drafts; boot self-check |
| `TF_DSV41_ROUTER_GROUP_ROT` | 0 | `1` | Fuse histogram grouping and input rotation |
| `TF_DSV41_MHC_PFDEC` | 0 | `17` | Use prefill mHC kernels for decode windows of at least 17 rows |
| `TF_DSV41_DRAFT_BATCH_PROJ` | 0 | `1` | Project and normalize all slots' main rows together |
| `TF_DSV41_DRAFT_CHAIN` | 0 | `1` | Device draft chain; needs DRAFT_GRAPHS and chain AOT variant |
| `TF_DSV41_DRAFT_OVERLAP` | 0 | `1` | Overlap draft ingests and window tail; needs GREEDY_GPU device start |
| `TF_DSV41_ROUND_GRAPH` | 0 | `1` | Split head/tail graphs and event-based Engram gate |
| `TF_DSV41_KV_NORM_STORE` | 0 | `1` | Fuse KV RMS, SWA RoPE and FP8 store |
| `TF_DSV41_LOOKUP_PLAN` | 0 | `1` | Cost-priced copy-draft lookup planning |
| `TF_DSV41_PF_4K` | 0 | `0 (optional 1)` | 4096-row prefill workspace; may fall back to 2K |
| `TF_DSV41_PF_4K_MEMORY_OK` | unset | `unset (optional 1)` | Required acknowledgment for 4K memory cost |
| `TF_DSV41_PF_WS_FLOOR_GIB` | FLOOR_HARD_GIB (4) | `unset (optional 3.5)` | Memory remaining after workspace allocation on every rank |
| `TF_DSV41_PIECE_RUNS` | 0 | `0` | Combine several slots' prompt pieces; gated, off |

### Data, serving and transport settings

| Name | Unset engine default | Recommended profile | What it does |
| --- | --- | --- | --- |
| `TF_DSV41_CONTEXT` | 4096 | `1048576` | Slot position limit, not a promise that every slot can fill it together |
| `TF_DSV41_ASSETS` | unset | `/assets` | Per-rank rope, host Engram lookup and AOT programs |
| `TF_DSV41_INDEX_KV` | FP8 fixed in this port | `fp8` | Compatibility setting; does not select a different Zig index type |
| `TF_DSV41_ENGRAM_DIR` | unset | `/engram` | Packed per-rank Engram shards |
| `TF_DSV41_CALIB` | cached | `real` | Read Python's stored costs; Zig does not perform Python's boot measurement |
| `TF_DSV41_CALIB_DIR` | user cache directory | `/cache/dsv41-calib` | Directory of matching Python cost tables; missing table uses built-in costs |
| `TF_DSV41_PREPARED` | unset | `/prepared` | Zig prepared layer images (different from Python prepared folders) |
| `TF_DSV41_PREPARED_WRITE` | 0 | `1` | Write missing prepared files |
| `TF_DSV41_PLAN_TIMEOUT` | 300 | `1800` | Seconds allowed for TP rendezvous connection |
| `TF_DSV41_PLAN_LINK` | blocking socket | `rdma` | Spin on the plan socket before blocking; 5000 μs when rdma |
| `TF_DSV41_PLAN_PIN` | off | `auto` | Pin plan threads to the best available CPU |
| `TF_DSV41_DEPTH_JOINT` | 1 | `0` | Disable joint draft depth policy, matching the Python reference |
| `TF_DSV41_SESSIONS` | 1 | `1` | Session reuse; SESSION_SLOTS additionally required at four slots |
| `TF_DSV41_SESSION_RAM_MIB` | 256 | `256` | Bounded session state RAM tier |
| `TF_DSV41_SESSION_DISK` | unset: RAM only | `/sessions` | NVMe session tier |
| `TF_DSV41_SESSION_DISK_GIB` | 64 | `128` | NVMe tier file budget |
| `TF_DSV41_SESSION_DISK_MIN` | 1024 | `1024` | Minimum token count for parking a session |
| `TF_DSV41_FLOOR_GIB` | max(5, hard floor) | `5` | Target MemAvailable in GiB per rank |
| `TF_DSV41_FLOOR_HARD_GIB` | 4 | `4` | Hard admission floor in GiB |
| `TF_DSV41_REQUEST_LOG` | unset: off | `/state/requests.jsonl` | Request metadata log |
| `TF_DSV41_DISCONNECT` | 1 | `1` | Cancel generation when client disconnects |
| `TF_DSV41_THINKING` | 1 | `1` | Default thinking mode; requests can override |
| `TF_DSV41_DEFAULT_EFFORT` | high | `high` | Default effort (high = 75) |
| `TF_DSV41_SCHEMA_PROMPT` | 1 | `1` | Include response-format schema in the prompt |
| `TF_DSV41_IMAGES` | placeholder | `placeholder` | Replace image parts with an omission notice; this profile does not serve native vision |
| `TF_DSV41_GRAMMAR` | 0 | `1` | Grammar-constrained structured output |
| `TF_DSV41_TOOL_GRAMMAR` | 0 | `0` | Automatic tool-schema constraint; remains off |

| Name | Unset engine default | Recommended profile | What it does |
| --- | --- | --- | --- |
| `GLM53_TF_ROCE_MAX_KB` | 256 | 1024 | RoCE message cap; keeps larger multi-slot windows on RoCE |
| `GLM53_TF_ROCE_FAST` | 0 | 1 | Fast RoCE gather path |
| `TF_TP_WORLD` | 1 | 2 (launcher) | Number of TP ranks |
| `TF_TP_RANK` | 0 | 0 or 1 (launcher) | This node's rank |
| `TF_TP_DEVICE` | 0 | 0 (launcher) | Local CUDA device |
| `TF_TP_MASTER` | loopback | your rank 0 link address | TP rendezvous address |
| `TF_TP_PORT` | 29500 | 29571 | Rendezvous port; plan port defaults to this plus 7 |
| `TF_COMM_BACKEND` | nccl | roce | Tensor exchange transport |
| `TF_TP_KERNEL_IMAGE` | unset | image's tp_mailbox.fatbin (launcher) | TP kernel image |
| `NCCL_SOCKET_IFNAME` | library auto-selection | your link interface | NCCL network interface |
| `NCCL_IB_HCA` | library auto-selection | your cabled RDMA devices | NCCL/RoCE HCA selection |

`IMAGE`, `MODEL_DIR`, `ENGRAM_DIR`, `ASSETS_DIR`, `PREPARED_DIR`, `STATE_DIR`, and `CACHE_DIR` are required
launcher inputs, not engine knobs. `HOST=localhost` and `PORT=8000` are launcher defaults for the HTTP listener.
The maximum reply is 32768 tokens and the served model ID is `DeepSeek-V4.1-Flash-TF` in this small launcher.

### Optional 4K prefill and gated piece runs

To try PF_4K, use a matching 4K AOT fill and set **all** of `TF_DSV41_PF_4K=1`,
`TF_DSV41_PF_4K_MEMORY_OK=1`, `TF_DSV41_PREFILL_CHUNK=4096`, and `TF_DSV41_PREFILL_ROWS=4096`.
It requires `GM_V2=1` and `PREFILL=replay`. The 4K workspace is transient and released after the CED finish.
Each prompt piece checks headroom across both ranks; if either cannot leave the workspace floor, both use 2K
segments with the same bits. The boot budget can also refuse a profile that cannot fit.

`TF_DSV41_PF_WS_FLOOR_GIB` defaults to the hard floor (4 GiB). The recorded 4K campaign used 3.5 GiB to avoid
persistent fallback on the more constrained rank; this reduces its workspace margin and is an optional tuning
choice, not the recommended all-on default. Do not enable two-batch prefill overlap (PF_TBO) alongside 4K without
a separate memory gate: the tested combination exhausted memory. The reported 4K numbers are from the c8 run.

`TF_DSV41_PIECE_RUNS=0` remains explicit: multi-slot prompt batching is still gated for this setup. It needs
`PREFILL_PIECES=1`, replay mode, matching multi-piece AOT shapes and a passing hardware gate before adoption.
Do not equate all-on with enabling every experimental knob.
