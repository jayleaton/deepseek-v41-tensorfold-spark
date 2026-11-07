# Install and run the native recipe

The recommended pack is flash17. Read [ENGINE](ENGINE.md) first: the pin includes
`dsq-1` admission support but is still local, and this migration has no new pack/device
qualification. Follow [ZIG-SERVE](ZIG-SERVE.md) for native rank assets and launch.
Use matching model and release-specific Engram tables on both nodes; q28-v2 keeps
its separate rollback profile. `serve.sh start` now invokes only the native pair
supervisor. It cannot accidentally enter the retired Python builder that asserted
uniform expert widths (#57). Native per-expert planning is source support, not a
receipt that this pin has loaded the recommended weights. Start with an
empty prepared directory for cold-start qualification; do not reuse prepared blobs
as a substitute for actual pack admission and width-class coverage on both ranks.

## Build-host prerequisites

Use an isolated ARM64 Linux build environment with Git, Bash, curl, xz, Docker/NVIDIA
build tools and writable model, asset and state directories. The native compiler is
Zig **0.17.0**, pinned by the engine's `.zig-version` and the native Dockerfile; the
Docker build installs its own compiler, not a host `zig` executable.
Install that exact ARM64 Zig compiler for the host token-map step as well. Set
`EDITOR` to your editor executable (for example `export EDITOR=vi`) before using
the README's copy-and-edit command.

For Engram packing and asset generation, retain the historical generator environment:
Python **3.12**, the NumPy and Torch supplied by `nvcr.io/nvidia/pytorch:26.07-py3`
(the base in `docker/Dockerfile` at recipe commit `391d677b`), Triton **3.7.1**, and
ptxas **13.3.73**. The retained generator documents the Python path, Triton and ptxas
in [`triton_fill.py`](../engine/zig/tools/zig/triton_fill.py); RoPE uses Torch's CPU
arithmetic. No independent exact NumPy or Torch wheel pin is recorded here: preserve
that base environment and record its actual package versions with generated assets,
rather than substituting an arbitrary current Torch wheel. This is a build-only
environment, separate from native serving and its sandbox interpreter.

Inside that generator environment, preserve its NumPy/Torch packages while creating
a separate environment for the CLI and pinned compiler dependencies:

```bash
python3.12 -m venv --system-site-packages /absolute/generator-venv
source /absolute/generator-venv/bin/activate
python -m pip install huggingface_hub 'triton==3.7.1' 'nvidia-cuda-nvcc==13.3.73'
hf --help
python -m pip freeze > /absolute/generator-requirements.txt
```

Set both `TRITON_PTXAS_PATH` and `TRITON_PTXAS_BLACKWELL_PATH` to the absolute
`ptxas` path from that compiler package (or the matching CUDA toolkit), and verify
`"$TRITON_PTXAS_PATH" --version` reports **13.3.73**. Follow
[asset generation](ZIG-SERVE.md#generate-runtime-assets) for the reference source,
native token-map tool and output directories. Download the selected model before
packing Engram weights; `$MODEL/config.json` below must already exist.

## Engram source setup (#68)

EXL3 packs do not carry the Engram tables (layers 1 and 14, about 101 GB each).
Use DeepSeek's original checkpoint from the **same model release**. Download its
index first, then discover and download exactly the repository-relative paths it
names. `--list` needs only Python's standard library; packing/checking also needs
NumPy. Install those tools in your isolated build environment, not in the runtime.

For the 1017 source inspected for this update, the index is pinned below. Replace
the example directories with absolute paths and run in a shell on the packing host:

```bash
SOURCE_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
SOURCE_REV=1bae0e1088993b9b82de08f59de5be5a34e42170
SRC=/absolute/source-1017
MODEL=/absolute/flash17-model
HEAD_ENGRAM=/absolute/engram-r0
RANK1_OUTPUT=/absolute/engram-r1-staging
mkdir -p "$SRC"
hf download "$SOURCE_REPO" model.safetensors.index.json --revision "$SOURCE_REV" --local-dir "$SRC"
printf '%s  %s\n' 74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8 "$SRC/model.safetensors.index.json" | sha256sum -c -
python3 scripts/pack_engram.py --src "$SRC" --list > "$SRC/engram-files.txt"
while IFS= read -r shard; do
    hf download "$SOURCE_REPO" "$shard" --revision "$SOURCE_REV" --local-dir "$SRC"
done < "$SRC/engram-files.txt"
python3 scripts/pack_engram.py --src "$SRC" --config "$MODEL/config.json" --out "$HEAD_ENGRAM" --rank 0
python3 scripts/pack_engram.py --src "$SRC" --config "$MODEL/config.json" --out "$RANK1_OUTPUT" --rank 1
python3 scripts/pack_engram.py --src "$SRC" --config "$MODEL/config.json" --check "$HEAD_ENGRAM" --rank 0
python3 scripts/pack_engram.py --src "$SRC" --config "$MODEL/config.json" --check "$RANK1_OUTPUT" --rank 1
```

The pinned index names `model-00047-of-00048.safetensors` and
`model-00048-of-00048.safetensors` at the repository root. Do not prepend `shards/`:
other release indexes may already include a directory, and discovery preserves it.
Missing indexes produce an actionable error before packing. The paths above were
verified against the pinned repository; no full source download or production pack
was performed for this migration. For another release, pin its own source/index
and inspect its output rather than reusing this index or its shard numbers.

Each rank keeps its half of the hash heads on local NVMe (about 47 GiB per layer,
94 GiB per node). The packer writes `engram-l{1,14}-r{rank}of2.bin` from raw source
bytes. `--check` verifies headers and sampled rows; it is not an exhaustive hash of
all source bytes. Synthetic tests cover both ranks and corruption detection. The
historical measured runs used the MiaAI-Lab kit's equivalent format; they do not
qualify a newly generated full checkpoint. Copy the rank-1 output to the worker's
`ENGRAM_DIR`, then perform native preflight and the actual model canary only when
the engine/pack migration is ready. This task performs no GPU or TP2 run.

## Pack rank 1 on the head, then copy the output (#69)

Both `--rank 0` and `--rank 1` commands above run on the **head/packing host**, against
its local source checkpoint and model config. Check rank 1 there before transfer:

```bash
python3 scripts/pack_engram.py --src "$SRC" --config "$MODEL/config.json" --check "$RANK1_OUTPUT" --rank 1
WORKER_SSH=user@worker.example
WORKER_ENGRAM=/absolute/worker-engram
ssh "$WORKER_SSH" "mkdir -p '$WORKER_ENGRAM'"
scp "$RANK1_OUTPUT"/engram-l*-r1of2.bin "$WORKER_SSH:$WORKER_ENGRAM/"
```

Set the worker's `ENGRAM_DIR` to that destination. Only the generated rank-1 Engram
files are transferred by this step. The worker needs neither the original DeepSeek
source checkpoint nor a recipe checkout **for packing**. Native serving still needs
its own matching model, image, native assets, writable state/prepared directories
and configured launcher checkout for head-driven lifecycle commands. Do not copy
rank-0 tables into the worker's Engram directory. An independent worker-side pack
is optional and requires its own source/index and packer setup. The copy procedure
is documented and covered by synthetic rank-format tests; a complete public install
and transfer with the full checkpoint remain unrun in this CPU-only task.
