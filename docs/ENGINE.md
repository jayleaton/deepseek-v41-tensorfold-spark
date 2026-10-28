# Native TensorFold engine

The recipe builds upstream [TensorFold](https://github.com/ashhart/TensorFold) at the exact commit in
[`config/tensorfold.rev`](../config/tensorfold.rev). That file is the single engine pin. The Docker build fetches
and verifies it, builds the `dsv41-serve` target with both CUDA runtime and DeepSeek family images, and installs
`tensorfold-dsv41` and `tf-dsv41`. The serving build fetches upstream directly and applies no recipe patches. The older native export stays in
`engine/zig` solely to preserve the existing generator/reference dependencies pending their equivalence audit.

`tensorfold-dsv41 serve` owns rank 0; `tf-dsv41 follow` owns rank 1. `dsv41-serve` is the upstream build target,
not the executable name. The experimental family requires `TF_EXPERIMENTAL=deepseek`.

The initial pin is an upstream baseline, not a qualified replacement for the previous deployed export. In particular,
its checkpoint admission rejects the documented flash17 `dsq-1` format metadata. Do not edit a pack's config to bypass
that refusal. Move the single pin only after the engine changes and actual pack admission have been reviewed.
Native images, structured output, defaults, sessions and sampling also require their own migration qualification.

## Build and checks

[`scripts/build-native.sh`](../scripts/build-native.sh) supplies the pin to [`docker/zig/Dockerfile`](../docker/zig/Dockerfile).
The builder uses Zig 0.17.0, CUDA 13 and real sm_121 images; the runtime keeps NCCL, verbs/mlx5 and pinned cuBLAS
library verification. A host compile with empty or synthetic GPU images is insufficient for serving.

Follow the pinned upstream `AGENTS.md`, `CONTRIBUTING.md`, `.zig-version` and native contributor documentation for
engine changes. Upstream gates include fmt, lean with zero new finding identities, host/golden/security/fuzz/kernel
contracts and relevant device/model checks. This repository's retained tests exercise recipe tooling:

```bash
python3 -m pytest -q tests
bash scripts/check-public.sh --history
gitleaks git --redact
gitleaks dir --redact
```

Run compute in the authorized isolated environment. Those checks do not establish a checkpoint/device receipt.

## Assets and historical provenance

Native serving needs `rope.bin`, `engram-token-map.bin`, a complete `aot/` and local rank Engram weight shards.
The older export and all its kernel/reference generators remain under `engine/` pending a separate equivalence audit.
Python build-only asset generators remain necessary for RoPE and Triton kernels; they do not run inference or ship
in the serving image. Their retained dependencies and limitations are described in [ZIG-SERVE](ZIG-SERVE.md).

The earlier public recipe and measurements remain reproducible at
[60b6688](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/tree/60b66888a1a379fb38bf85c232b27ec60b007373).
No old result is relabelled as a measurement of this pin. TensorFold's source, model files and generated kernels keep
their copyright and license notices.

For the host-tooling regression gates, use Ruff **0.17.3** and set `RUFF` to its
executable if it is not on `PATH`. Set `BASH32` to a real GNU Bash 3.2 executable
for the publication portability tests; those tests verify the reported shell
version and execute clean, finding and empty-list cases. The all-ok preflight test
injects measured latencies; real disk latency remains an operator measurement,
with the production 300 µs warning threshold unchanged.

```bash
ruff check --isolated --select E4,E7,E9,F .
ruff format --isolated --check bench/speed.py scripts/preflight.py tests/bench/test_speed.py tests/scripts/test_preflight.py
```

Compare core-lint findings by identity against the base; a smaller total does not
compensate for newly introduced findings. Run the tooling and full retained suites
in the authorized compute environment. Keep the original 16 inherited compiler
errors explicit until their historical source/toolchain prerequisites are addressed.
