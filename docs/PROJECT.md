# Repository layout

| Path | Purpose |
| --- | --- |
| `engine/zig`, `docker/zig` | the existing native CUDA engine and image build |
| `scripts/run-zig.sh` | one local native rank; run separately on each node |
| `config/prod-zig*.env.example` | pack-specific native serving profiles |
| `scripts/pack_engram.py` | rank-local Engram weight packing |
| `engine/kernels`, `engine/reference`, `engine/zig/tools` | retained kernels, references and asset generators |
| `bench`, `tests` | clients and CPU/reference checks |
| `docs`, `results` | installation, measurements and historical receipts |

The old Python engine patches, serving Dockerfile and launcher are retired. Use [ZIG-SERVE](ZIG-SERVE.md).
