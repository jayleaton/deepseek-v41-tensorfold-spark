# Native CUDA serving on two DGX Sparks

This branch wires upstream TensorFold's `dsv41-serve` build target. Read [ENGINE](ENGINE.md) before use: the current
pin includes flash17 `dsq-1` admission but is not device/pack qualified. Historical rates and profiles describe
the previous deployed engine. Do not bypass checkpoint admission or treat this guide as a new serving receipt.

## Build

```bash
git clone https://github.com/jayleaton/deepseek-v41-tensorfold-spark.git
cd deepseek-v41-tensorfold-spark
TENSORFOLD_SOURCE=/absolute/TensorFold scripts/build-native.sh -t tensorfold-dsv41:local
scripts/check-zig-image.sh tensorfold-dsv41:local
docker save tensorfold-dsv41:local -o build-native-image.tar
```

Transfer the image to the worker and `docker load -i build-native-image.tar`. Both nodes must have the same image ID
and `org.tensorfold.revision` label matching `config/tensorfold.rev`. Build on ARM64 with Docker/NVIDIA build tools.
`--build-arg ZIG_URL=<compiler-url>` selects an exact compiler archive if the default URL is unavailable.
The runtime retains the library/NEEDED-chain and mlx5-provider check; that check must also pass on a loaded image.

## Configure

Copy `config/prod-zig-flash17.env.example` to ignored `config/prod-zig.env` on each node. Set IMAGE, model/Engram/assets,
prepared/state/cache paths, master address and NIC/HCA names. Keep pack, engine settings and image content identical.
Use each pack's own prepared/session/calibration directories. Copy `config/pfdense-table.json` into `CACHE_DIR`
when retaining the candidate profile's dense-table setting. The q28-v2 rollback profile stays separate.
The historic 1M/four-slot profile is a candidate for the new pin; preserve the memory floors and qualify it before use.

Profiles use the engine's own setting names (`config/tensorfold-settings.txt`); the launcher refuses renamed ones.
`TF_DSV41_THINKING=0` and `TF_DSV41_DEFAULT_EFFORT` become the head's `--no-thinking` and `--reasoning-effort` flags.
`TF_TP_ROCE_HCA` selects the rank-local RDMA port when more than one peer is cabled. It is forwarded unchanged,
including a value supplied from the caller environment. Credit: @rootsbymenda, #63. The launcher also keeps both-rank
`NCCL_CUMEM_ENABLE=0` / `NCCL_CUMEM_HOST_ENABLE=0` defaults from #41. Set topology only in ignored local config.

## Generate runtime assets

There are four independent inputs:

1. Pack config, tokenizer, index and all referenced EXL3 shards, byte-identical on both nodes.
2. Engram **weight** shards from that pack's original release, rank-specific on local NVMe; see [INSTALL](INSTALL.md).
3. `rope.bin`: `DSV41RP1`, two fp32 `[context + 2048, 64]` tables from the retained build-only reference arithmetic.
4. Native `engram-token-map.bin` (`DSENGR01`), `sigs.json`, exact-environment `needs.json` and a complete `aot/`.

The token map is produced by the pinned upstream `zig/dsv41_token_map.zig` tool. The pin includes shared
`dsq-1` metadata admission support. Generate and check assets against the actual selected pack; source admission
support alone does not establish complete asset or serving qualification.

RoPE and family Triton AOT still require Python/Torch/Triton in an isolated generator environment.
The existing generators under `engine/zig/tools/zig` and `engine/kernels` are retained unchanged;
their historical reference dependencies remain pending the upstream-equivalence audit. These produce files
consumed by native code; the serving image contains no Python inference runtime. Keep Triton 3.7.1 and ptxas 13.3.73
and source/toolchain hashes with the assets. A new kernel generator, including #71, requires separate arithmetic and
device qualification and regenerated AOT. Do not silently adopt it with a recipe update.

Enumerate with the actual native image and serving environment. `scripts/run-zig.sh N --check-aot` runs host-only
`tf-dsv41 aot-needs SIGS OUT` with no GPU exposed, then checks every specialization against the local AOT manifest
and required cubins. The native tool takes two file arguments; the old optional third-argument coverage behavior
is not assumed. Regenerate after changing context, slots, prefill, drafting, images or any engine control.

After generating the complete assets, on each node:

```bash
scripts/run-zig.sh 0 --check-aot       # use 1 on the worker
scripts/run-zig.sh 0 --identity        # compare with rank 1: must match
scripts/run-zig.sh 0 --dry-run
scripts/run-zig.sh 0 --effective-profile   # the container's engine settings; credential names only
```

The launcher checks the upstream image label and local directories, RoPE geometry/size, native token-map pack
provenance, rank-specific Engram headers/sizes and AOT cubin presence. The common identity hashes config, tokenizer,
index and common asset manifests plus the effective engine profile, image and transport settings. Verify model and
asset bytes against their release manifests before deploying; these metadata checks do not certify arithmetic or
hash every large weight/Engram file at each boot. Run --check-aot after every setting change.

`--identity` must print the same hash on both ranks; `serve.sh` preflight refuses the pair otherwise. Its inputs:

- the engine revision and image ID;
- the model config, tokenizer and weight index, and the common assets (`rope.bin`, `engram-token-map.bin`,
  `sigs.json`, `aot/aot.json`); each rank's own Engram shards are checked, but not hashed;
- `TF_COMM_BACKEND`, `TF_TP_MASTER` and `TF_TP_PORT`;
- every other forwarded engine setting (`config/tensorfold-settings.txt`), except `NCCL_*` and the rank-local
  settings below.

The engine's own transport digest (`Config.digest` in `fabric/tp/config.zig`: world size, backend,
`TF_TP_MAILBOX_MAX_KB`, `TF_TP_ROCE_CHUNKED_KB`, `TF_TP_FAILFAST`, `TF_TP_ROCE_WIRE`, `TF_TP_ROCE_FALLBACK`,
`TF_GRAPH_CACHE`) is only the minimum the ranks agree on after they connect. It is not the full list of what
must match to start. `Session.open` (`fabric/tp/tp.zig`) first runs `Bootstrap.open`: rank 0 listens on the
plan port and rank 1 connects to `TF_TP_MASTER` on it (`TF_TP_PLAN_PORT`, default `TF_TP_PORT` + 7). Mismatched
ports never reach the digest compare; rank 1 just waits out its plan-connect timeout. So the identity treats a
setting as common, including `TF_TP_PLAN_PORT`, `TF_TP_ROCE_HCAS`, `TF_TP_ROCE_TC` and the test switches
(`TF_TP_ALLOW_HOST_ONLY`, `TF_TP_LOCAL`, `TF_TP_STATS`, `TF_TP_TRACE`), unless the engine source shows it is
per-rank.

Rank-local settings are forwarded with their values but may differ per node. The engine source shows each is
per-rank:

- **placement on this host:** `TF_TP_ROCE_CPU` (the RoCE proxy thread), `TF_TP_PLAN_PIN`, `TF_TP_ROCE_HCA`,
  `TF_ROCE_HCA`, `TF_ROCE_GID_INDEX`, `TF_NCCL_LIB`;
- **the fast path's local tuning:** `TF_TP_ROCE_FAST`, `TF_TP_ROCE_POLLERS`, `TF_TP_ROCE_STAGGER_NS`,
  `TF_TP_ROCE_BLOCKS`, and the follower's plan polling `TF_TP_PLAN_LINK` and `TF_TP_PLAN_SPIN_US`;
- **this rank's own waits:** `TF_TP_PLAN_TIMEOUT` (the bootstrap deadline of each rank), `TF_TP_MAILBOX_TIMEOUT_S`,
  `TF_TP_FAILFAST_GRACE_S`, and `TF_TP_PEER_MEM_S` (rank 1's report period; rank 0 polls with no timeout).

The lists are `TRANSPORT_COMMON` and `RANK_LOCAL` in `scripts/engine_settings.py`. With `TENSORFOLD_SOURCE` set,
tests check them against the pinned engine: the digest fields, the `Config` fields `Bootstrap.open` reads, and
that only the bootstrap runs before the digest compare.

## Start and stop

Start `scripts/run-zig.sh 1` on the worker, then `scripts/run-zig.sh 0` on the head. These launch `tf-dsv41 follow`
and `tensorfold-dsv41 serve`; inspect the boot logs, `/health` and `/v1/models`, then check a real answer with
`scripts/canary.py`. A live container alone is insufficient. Stop rank 0 with enough time for the native drain:

```bash
docker stop -t 80 tensorfold-zig-r0
```

Observe rank 1 exiting on its node. A custom `CONTAINER_PREFIX` changes both names consistently. Native health,
wire semantics and fail-fast behavior belong to the selected engine; no retired Python stall watchdog is implied.
See [OPERATIONS](OPERATIONS.md) for supervisor and optional guard behavior.

## Optional preflight

`PREFLIGHT=1` runs `scripts/preflight.py` before starting a rank. Exit 1 warnings remain visible; exit 2 failures
stop startup. Native health shapes or graph-floor fields that are absent are explicitly unmeasured. This helper
does not certify the checkpoint or justify growing its pool.

For head-driven preflight/start/stop/status/watch commands, configure the worker
checkout and config paths and follow [native operations](OPERATIONS.md#native-pair-lifecycle-and-watchdog-58).
`serve.sh` uses the native launcher for both ranks; original source-checkpoint
transfer is unnecessary when [rank 1 is packed on the head](INSTALL.md#pack-rank-1-on-the-head-then-copy-the-output-69).
