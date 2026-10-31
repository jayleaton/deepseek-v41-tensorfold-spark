# Native TensorFold engine

The recipe builds [TensorFold](https://github.com/ashhart/TensorFold) at the exact commit in
[`config/tensorfold.rev`](../config/tensorfold.rev), fetched from the repository in
[`config/tensorfold.repo`](../config/tensorfold.repo). The commit is the single engine pin; the repository only says
where it is published. The Docker build resolves
and verifies it, builds the `dsv41-serve` target with both CUDA runtime and DeepSeek family images, and installs
`tensorfold-dsv41` and `tf-dsv41`. The serving build uses upstream Git objects without recipe patches. The older native export stays in
`engine/zig` solely to preserve the existing generator/reference dependencies pending their equivalence audit.

[`config/tensorfold-settings.txt`](../config/tensorfold-settings.txt) lists every `TF_*` / `TENSORFOLD_*` setting
the pinned engine reads. The launcher forwards exactly the ones a profile sets, plus `NCCL_*`, never credential
names. It refuses a renamed or removed setting (for example `GLM53_TF_ROCE_FAST`, now `TF_TP_ROCE_FAST`) and warns
about any other engine-looking name the pin does not read, so an old profile cannot silently fall back to defaults.
Every pin bump regenerates the list from the new engine source, and a test fails when the two disagree:

```bash
python3 scripts/engine_settings.py generate /path/to/tensorfold-checkout   # after editing config/tensorfold.rev
```

`tensorfold-dsv41 serve` owns rank 0; `tf-dsv41 follow` owns rank 1. `dsv41-serve` is the upstream build target,
not the executable name. The experimental family requires `TF_EXPERIMENTAL=deepseek`.

The current pin is release candidate G, `58bece1d3` on `dsv41/candidate-g`. G is candidate F (`56200688d`, the
fallback pin) plus the compute lane (`TF_DSV41_COMPUTE_LANE`, off by default; [OPERATIONS](OPERATIONS.md)) and its
executor. F carries the opt-in settings the flash17 profile turns on (the KV mirror, both rotation folds, the indexer
row split, Engram shard validation and five others; [FLASH-1017](FLASH-1017.md)). Both descend from stable head H
(`20ad77878`, the previous pin), which has flash17 `dsq-1` checkpoint admission from `98f2a3896` and boots the real
flash17 pack on TP2. H added:
- DeepSeek-V4.x chat rendering, sampling defaults and tool-schema holding;
- the Spark stats keys in replies (on by default for DeepSeek; `TENSORFOLD_WIRE=native` turns them off);
- since `314c7a911`: the explicit non-strict tool fallback, pool-growth agreement only on page growth, DSpark statics
  without a context drain, KV transactions that back up only the ratio-2 carries, and grammar masks staged and sent
  only for constrained rows. `TF_DSV41_GRAMMAR` is honoured again: unset, empty or `1` (the default) serves
  structured output, `0` refuses it, and any other value stops the boot.

G is not on upstream main yet, so `config/tensorfold.repo` points at JJ's public fork
(`jayleaton/TensorFold`, branch `dsv41/candidate-g`), where the commit is published. To fall back to F, put
`56200688d52af54062fadbab12de36e95165ad72` (an ancestor of G on the same branch) in `config/tensorfold.rev`, unset
`TF_DSV41_COMPUTE_LANE` and regenerate the settings list. Once the upstream PRs merge
and the pin is on `ashhart/TensorFold`, the file points back there. A local checkout or a bundle remains the
fallback, as described below. This source support is not a new
pack/device qualification. Do not edit a pack config to bypass admission checks. Native images, structured output,
defaults, sessions and sampling still require migration qualification; historical production results do not
qualify this pin.

## Results at this pin (2026-10-29/30)

These are measured on one pair of Sparks against reference A, the engine export of public recipe `60b6688`. They are
paired spot checks, not a full qualification.

- **A/C correctness** (phase 2, C = engine `314c7a911`, two quiet ABBA pairs):
  - 97/97 gates in every selected job.
  - The cross-arm token differences were only in forced `tool_choice` cells. For named, C renders only the named
    tool. For required, the prompt is the same; the token path differs and the calls are the same.
- **H on TP2** (the benchmark harness's real-pack boot):
  - ready, smoke "391", canary pass, strict tools pass;
  - 2048-token greedy decode in 52.5 s;
  - no missing Triton variant on either rank.
- **H spot ABBA vs A** (two pairs, all four jobs CPU-quiet): 18/18 gates in every job, and 77 token comparisons per
  pair, 0 different.
- **This recipe's launcher at H on TP2** (`scripts/serve.sh`, the flash17 profile, image built from a bundle with
  key-file authentication):
  - both effective profiles carry `TF_TP_ROCE_FAST=1`, `TF_TP_MAILBOX_MAX_KB=1024`, `TF_TP_PLAN_LINK=rdma` and
    `TF_DSV41_GRAMMAR=1`, and the running engines' environments have the same values;
  - `--check-aot` passes on both ranks, and preflight passes;
  - each rank logs `roce fast on`, a 1024 KiB mailbox region (`len 4202496`) and its plan link pinned to a CPU;
  - ready in 335 s (319 s on a second boot);
  - 401 without the key, smoke "391", and one strict required tool call with schema-valid arguments;
  - a clean `serve.sh stop`, and the key appears in none of the receipts.

| arm/A (higher is better) | C at `314c7a911` | H, pair 1 | H, pair 2 | H, median |
| --- | ---: | ---: | ---: | ---: |
| prefill 8K tok/s | 0.864 | 1.057 | 1.069 | **1.063** |
| native decode, code T0, 1 stream | 0.975 | 0.992 | 1.017 | **1.004** |
| native decode, code T0, 2 streams | 0.959 | 0.991 | 0.990 | **0.991** |
| native decode, code T0, 4 streams | 0.958 | 0.968 | 0.977 | **0.973** |
| tools auto, 1 stream, per stream | 0.881 | 1.058 | 0.987 | **1.022** |

Still open at H: 4-stream decode is 2.7% behind A. Auto-tool calls still pay for a synchronous grammar fill on rank 0
and whole-mask messages to the peer, which needs a lane-core redesign.

## Build and checks

[`scripts/build-native.sh`](../scripts/build-native.sh) supplies the pin to [`docker/zig/Dockerfile`](../docker/zig/Dockerfile).
The builder uses Zig 0.17.0, CUDA 13 and real sm_121 images; the runtime keeps NCCL, verbs/mlx5 and pinned cuBLAS
library verification. A host compile with empty or synthetic GPU images is insufficient for serving.
The image also builds the three libraries the engine loads from `lib/` beside `bin/` (upstream
`packaging/RUNTIME.md`). Each is from a pinned source, and each is checked for the engine's entry points and its version:
- `libtfgrammar.so` (`zig build grammar` over xgrammar v0.2.8 at its tag commit) for structured output and required
  or named tool calls; without it those requests get HTTP 400;
- libjpeg-turbo 3.1.4.1 and libwebp 1.6.0 decoders (by tarball sha256) for JPEG and WebP image input.

The image's library proof refuses any other library name the binaries can open, except HIP RCCL and FFmpeg video,
which this image does not request.

`scripts/build-native.sh -t tensorfold-dsv41:local` fetches the pin from that repository. To build without the
network, or from your own copy, supply a checkout containing the pin:

```bash
TENSORFOLD_SOURCE=/absolute/TensorFold scripts/build-native.sh -t tensorfold-dsv41:local
```

The script reads the exact revision from the pin file, fetches only that revision's
history into temporary scratch, and creates a self-contained bundle. It does not
modify the supplied checkout, use its working-tree edits, or require its HEAD to
match the pin. For transfer to another builder, such as a Spark that has no engine checkout, create a bundle
containing the upgrade branch and supply the bundle file instead:

```bash
git -C /absolute/TensorFold bundle create /absolute/tensorfold.bundle dsv41/upgrade-latest
git -C /absolute/TensorFold bundle verify /absolute/tensorfold.bundle   # must list the pin's history
TENSORFOLD_SOURCE=/absolute/tensorfold.bundle scripts/build-native.sh -t tensorfold-dsv41:local
```

The bundle must contain the pinned commit and all of its history (`git bundle create ... <branch>`, not a range).

Bundle prerequisites and exact commit resolution are verified locally and again in
the Docker builder. A supplied invalid/incomplete bundle fails without falling back
to the network. Only the Dockerfile, source resolver, shared credential boundary and selected bundle enter the
temporary build context; it is removed when the command exits. The bundle never
enters the runtime image. Without `TENSORFOLD_SOURCE`, the builder fetches the same
pin, by commit, from `config/tensorfold.repo`; it does not follow a moving branch.

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

The subprocess architecture regression gate parses launcher Bash syntax with
`tree-sitter==0.25.2` and `tree-sitter-bash==0.25.1`. Install these alongside pytest
in the isolated check environment. It rejects direct external commands and exec
in the host watchdog/launchers, including nested substitutions; `safe_shell.sh`
is the single adapter to the checked `safe_exec.py` supervisor. The build stage
includes Python only for that boundary; the serving runtime remains native.

The syntax/AST gate guards against accidental bypasses in the repository's own
code. It is not a sandbox against deliberately obfuscated code or trusted operator
configuration. Executable payloads must be statically auditable; unresolvable
dispatch is rejected outside the explicitly reviewed adapter.

All host tooling shell quoting goes through `scripts/safe_quote.py`. It checks raw
values with the shared credential rule before encoding shell syntax, including
nested SSH commands and the worker supervisor path. The checked spawn boundary
remains in place. The source gate rejects direct quoting outside this adapter.

Diagnostic subprocesses use `safe_exec.child_environment`: credential names and
credential-bearing aliases are absent, and tainted operational controls are
refused rather than dropped into defaults. Shell configuration keeps known values
unexported; validators receive them through private descriptors confined to exact reviewed
supervisor entries. These descriptors are consumed and closed before diagnostic
children start. Typed process ownership and deadline metadata remain supervisor
controls, independent of credential-bearing settings. Credentials are never carried in a registry
environment variable or executable argv. Serving authentication uses the native
read-only key-file transport described in OPERATIONS.

Literal/known-variant masking remains an output backstop, never the boundary.
Output is strictly line-based and stateless: each complete physical line is masked
in place and released immediately. Nothing is decoded, buffered, reconstructed or
withheld; credential isolation, rather than output framing, protects environment
dumps of every encoding. Ordinary settings, metrics and following errors remain
readable. Complete JSON documents are masked field by field. See SECURITY.md.
Docker state evidence uses a typed State allowlist and never captures Config.Env.
The reference SSH loader also uses the checked argv/quoting supervisor; binary
model bytes remain a data transport, and failed/unparseable responses are not
saved as metadata or diagnostics. The copied historical TP transport harness is
an unavailable stub; use the pinned upstream checkout for qualified transport work.
The quoting gate discovers executable code across the entire recipe, including
reference CLIs. Synthetic test quoting has exact, reviewed call-site exceptions.

Checked children also drop shell startup hooks (`BASH_ENV`, `ENV`,
`PROMPT_COMMAND`, exported Bash functions, `SHELLOPTS`, `BASHOPTS` and `PS4`).
Controlled Bash children disable profile and rc loading. Explicit operator CONFIG
loading remains trusted, quiet and separate from implicit shell startup. Remote
diagnostics run through the worker's checked supervisor under `env -i` with a
fixed system PATH; SSH configured RemoteCommand is disabled for these commands.
The worker checkout must contain the same shared boundary scripts. Serving keys
remain in the separately mounted key file, never in remote session
environment or diagnostic arguments.
