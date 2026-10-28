# Install and run the native recipe

The recommended pack is flash17. Follow [ZIG-SERVE](ZIG-SERVE.md) for the existing native build, rank assets,
image checks and launch. The retired Python build/cache/start workflow is no longer an installation option.
Use matching model and release-specific Engram tables on both nodes; q28-v2 keeps its separate rollback profile.

## Engram shards The EXL3 packs do not carry the Engram tables (layers 1 and 14, ~101 GB each). They come from
DeepSeek's original checkpoint; each rank keeps its half of the hash heads (~47 GiB a layer) on local NVMe:

```bash
mkdir -p <src> && hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model.safetensors.index.json --local-dir <src>
python3 scripts/pack_engram.py --src <src> --list                 # the shard files that hold the tables
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 <those files> --local-dir <src>
python3 scripts/pack_engram.py --src <src> --config <HEAD_MODEL>/config.json --out <HEAD_ENGRAM> --rank 0
python3 scripts/pack_engram.py --src <src> --config <HEAD_MODEL>/config.json --out <dir> --rank 1   # copy to <WORKER_ENGRAM> on the worker
python3 scripts/pack_engram.py --src <src> --config <HEAD_MODEL>/config.json --check <HEAD_ENGRAM> --rank 0
```

`pack_engram.py` writes the format the engine reads (`engram-l{1,14}-r{rank}of2.bin`) from the source tensors'
raw bytes; it is tested against the engine's reader on synthetic tables. The measured runs used shards packed by
the MiaAI-Lab kit (`./start.sh pack`), which writes the same format; `--check` compares either with the source.
