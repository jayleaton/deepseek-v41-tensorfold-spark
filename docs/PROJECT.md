# Repository layout

The serving implementation is pinned upstream TensorFold. See [ENGINE](ENGINE.md) for its build, qualification
limits and historical release provenance.

| Path | Purpose |
| --- | --- |
| `config/tensorfold.repo`, `config/tensorfold.rev` | the unmodified upstream TensorFold commit the engine starts from |
| `patches/`, `config/tensorfold.patches`, `config/tensorfold.tree` | the recipe's engine patch series, how much of it is applied, and the engine tree it must give |
| `config/tensorfold-settings.txt`, `scripts/engine_settings.py` | the pinned engine's setting names; launcher forwarding and legacy refusal |
| `docker/zig/Dockerfile`, `scripts/build-native.sh` | native CUDA pair and runtime library checks |
| `docker/compat/`, `scripts/build-compat-image.sh` | the CUDA 13.3 forward-compatibility layer over the runner image (pinned driver files, `tf-compat-entry`) |
| `scripts/run-zig.sh`, `scripts/serve.sh` | per-rank launch, identity, assets and operations |
| `config/prod-zig*.env.example` | pack-specific candidate profiles; local copies are ignored |
| `scripts/pack_engram.py` | raw-byte Engram weight-shard packing; NumPy, no inference engine |
| `scripts/check_native_assets.py` | native pack/image/profile identity and rank-asset checks |
| `scripts/check-public.sh`, `check_secrets.py`, `redact.py` | publication scanning and credential-safe output |
| `bench/`, `tests/` | HTTP clients and recipe-tool checks |
| `docs/`, `results/` | current instructions and explicitly historical measurements |
| `scripts/campaign/` | historical analysis tools; no serving dependency |

The old Python inference family and staged server are retired. The older native export and its build-time tools
remain in engine/ for the generator-equivalence audit.
Independent clients, security checks, results and required asset generators remain.
