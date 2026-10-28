# Install and run the native recipe

The recommended pack is flash17. Read [ENGINE](ENGINE.md) first: the initial upstream
pin rejects its `dsq-1` metadata, and this migration has no new pack/device
qualification. Follow [ZIG-SERVE](ZIG-SERVE.md) for native rank assets and launch.
Use matching model and release-specific Engram tables on both nodes; q28-v2 keeps
its separate rollback profile. `serve.sh start` now invokes only the native pair
supervisor. It cannot accidentally enter the retired Python builder that asserted
uniform expert widths (#57). Native per-expert planning is source support, not a
receipt that this initial pin has loaded the recommended weights. Start with an
empty prepared directory for cold-start qualification; do not reuse prepared blobs
as a substitute for actual pack admission and width-class coverage on both ranks.

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
