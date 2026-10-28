# Native CUDA serving on two DGX Sparks

This branch wires upstream TensorFold's `dsv41-serve` build target. Read [ENGINE](ENGINE.md) before use: the initial
pin is not device/pack qualified and rejects the flash17 pack's `dsq-1` metadata. Historical rates and profiles describe
the previous deployed engine. Do not bypass checkpoint admission or treat this guide as a new serving receipt.

## Build

```bash
git clone https://github.com/jayleaton/deepseek-v41-tensorfold-spark.git
cd deepseek-v41-tensorfold-spark
scripts/build-native.sh -t tensorfold-dsv41:local
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

`TF_TP_ROCE_HCA` selects the rank-local RDMA port when more than one peer is cabled. It is forwarded unchanged,
including a value supplied from the caller environment. Credit: @rootsbymenda, #63. The launcher also keeps both-rank
`NCCL_CUMEM_ENABLE=0` / `NCCL_CUMEM_HOST_ENABLE=0` defaults from #41. Set topology only in ignored local config.

## Generate runtime assets

There are four independent inputs:

1. Pack config, tokenizer, index and all referenced EXL3 shards, byte-identical on both nodes.
2. Engram **weight** shards from that pack's original release, rank-specific on local NVMe; see [INSTALL](INSTALL.md).
3. `rope.bin`: `DSV41RP1`, two fp32 `[context + 2048, 64]` tables from the retained build-only reference arithmetic.
4. Native `engram-token-map.bin` (`DSENGR01`), `sigs.json`, exact-environment `needs.json` and a complete `aot/`.

The token map is produced by the pinned upstream `zig/dsv41_token_map.zig` tool. At the initial pin, its metadata
parser also refuses flash17's `dsq-1`; the separately reviewed engine update must resolve asset metadata admission
without weakening serving checks. Later pins may install it through `zig build dsv41-assets`.

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
```

The launcher checks the upstream image label and local directories, RoPE geometry/size, native token-map pack
provenance, rank-specific Engram headers/sizes and AOT cubin presence. The common identity hashes config, tokenizer,
index and common asset manifests plus the effective engine profile, image and transport settings. Verify model and
asset bytes against their release manifests before deploying; these metadata checks do not certify arithmetic or
hash every large weight/Engram file at each boot. Run --check-aot after every setting change.

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
