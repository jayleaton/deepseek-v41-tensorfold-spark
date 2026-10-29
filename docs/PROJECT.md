# Repository layout

The serving implementation is pinned upstream TensorFold. See [ENGINE](ENGINE.md) for its build, qualification
limits and historical release provenance.

| Path | Purpose |
| --- | --- |
| `config/tensorfold.rev` | one reviewed upstream engine revision |
| `config/tensorfold-settings.txt`, `scripts/engine_settings.py` | the pinned engine's setting names; launcher forwarding and legacy refusal |
| `docker/zig/Dockerfile`, `scripts/build-native.sh` | native CUDA pair and runtime library checks |
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
