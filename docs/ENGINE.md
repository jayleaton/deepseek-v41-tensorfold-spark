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

The current pin is `5d25d4ff4` (`5d25d4ff4e9fe7c5c7430afc63194f60c508da04`), the G port. It is the 40 commits of
release candidate G (`58bece1d3`, below) ported onto `5074ee87d`, the line that carries the in-model runner. That line
adds the `TF_DSV41_BIAS_VL` and `TF_DSV41_NO_FORCED_DRAFTS` settings fixes, the forced-rows hold (a held reply's
result rows land whole), the graph bank's teardown frees, the runner proof fixes and the runner itself, and it holds
upstream #2182 (`f1402df75`, `6cb6ca1c9`; [Session restore](#session-restore-86)). Three commits follow the port:
`0e547b124` reads the ported image, Engram and memo settings through the shared settings table (a lean fixup),
`5114cb28d` closes a test before the ported health wire test, and `5d25d4ff4` makes the vision engine wrapper forward
`resume_held` to its inner engine. Without that last one, the compute lane refused at boot whenever image input was on.
Six commits of the port name bertholomus as co-author ([CREDITS](CREDITS.md)). Production serving on two Sparks runs
this pin since 2026-11-11, on the CUDA forward-compatibility layer
([OPERATIONS](OPERATIONS.md#cuda-forward-compatibility-tf_cuda_compat)); the measurements are under [Results at this
pin (2026-11-11)](#results-at-this-pin-2026-11-11).

The lane and runner keep their DeepSeek-V4.1 names (`TF_DSV41_COMPUTE_LANE*`, `TF_DSV41_RUNNER*`). At this pin they are
a family binding's aliases of the generic `TF_COMPUTE_LANE*` and `TF_RUNNER*` (`zig/src/core/compute_binding.zig`,
`runner_binding.zig`); a boot sets one family of names or the other, never both. The settings list holds both.

The previous pin is `7b79ef92d` on `dsv41/compute-lane-release-s86`: `07106059c`, the exact engine source of the compute-lane
studies ([compute lane](compute-lane.md)), plus seven commits. `07106059c`'s tree is byte-identical to the measured
source `3a429d6f1` (tree `c3636e6da`), with the same history re-recorded under one author. It is G plus `deadline_s`,
the engine-held compute lane with its sandbox, and the experimental in-model runner, which stays off unless its settings
are set. The first added commit (`e5172c4a2`) is `TF_DSV41_COMPUTE_LANE_DEFAULT` (unset: no change), which gives every
thinking chat the runner's guide and, when its reply has room for a call, the built-in tool, and reports
`compute_executed`; the flash17 profile turns it on with the runner
([compute lane](compute-lane.md#compute-in-every-chat)). The second (`75d774470`) changes only what that setting
does: a client's own `python` function keeps the name for its request (the built-in tool steps aside), and
`compute_executed.calls` counts only calls that ran. The third and fourth (`012767b62`, `8a6ce56a3`) offer the built-in
tool only to a thinking chat whose reply has room for a call (`max_tokens` at least 1024, no stop strings, logprobs or
structured output; the guide stays on every thinking chat), and never return a built-in call the engine could not run
([compute lane](compute-lane.md#when-the-built-in-tool-is-offered)). The fifth (`01a62132c`) makes the runner's
inline cells abstain after a TeX or worded operator (`\(4 \times 49 - 11 =` had got the value of `49 - 11`). The sixth
and seventh (`5f3e98939`, `7b79ef92d`) are upstream #2182 by Sergey Tretyak (goshi), cherry-picked unchanged from
`f1402df75` and `6cb6ca1c9`: `Disk.validate` now agrees with the stash contract and the replay drafter, so a session
parked on the NVMe tier under `TF_DSV41_PREFILL=replay` is restored instead of failing with `BadBlob` and prefilling
again ([#86](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/issues/86), reported by crystal8194; the
measurement is under [Session restore](#session-restore-86)). [compute
lane](compute-lane.md) lists the measured G commits'
counterparts on that branch.

Before it the recipe pinned `01a62132c` (that branch without #2182), and earlier release candidate G, `58bece1d3` on `dsv41/candidate-g` (its counterpart on that branch is `3adb042bb`,
the same tree). G is candidate F (`56200688d`, the
fallback pin) plus the compute lane (`TF_DSV41_COMPUTE_LANE`, off in the engine, on in the flash17 profile; [OPERATIONS](OPERATIONS.md)) and its
executor. F carries the opt-in settings the flash17 profile turns on (the KV mirror, both rotation folds, the indexer
row split, Engram shard validation and five others; [FLASH-1017](FLASH-1017.md)). Both descend from stable head H
(`20ad77878`, an earlier pin), which has flash17 `dsq-1` checkpoint admission from `98f2a3896` and boots the real
flash17 pack on TP2. H added:
- DeepSeek-V4.x chat rendering, sampling defaults and tool-schema holding;
- the Spark stats keys in replies (on by default for DeepSeek; `TENSORFOLD_WIRE=native` turns them off);
- since `314c7a911`: the explicit non-strict tool fallback, pool-growth agreement only on page growth, DSpark statics
  without a context drain, KV transactions that back up only the ratio-2 carries, and grammar masks staged and sent
  only for constrained rows. `TF_DSV41_GRAMMAR` is honoured again: unset, empty or `1` (the default) serves
  structured output, `0` refuses it, and any other value stops the boot.

The pin is not on upstream main, and it is not yet published on JJ's public fork (`jayleaton/TensorFold`, the
repository in `config/tensorfold.repo`) either. It must be published there before this branch can merge. Until then
the default build cannot fetch it: set `TENSORFOLD_SOURCE` to a local checkout or bundle that contains it, as described
under [Build and checks](#build-and-checks). To fall back to the previous pin, put
`7b79ef92d846138d51638b677c473392a56e240c` (branch `dsv41/compute-lane-release-s86`, published) in
`config/tensorfold.rev` and regenerate the settings list; it reads the profile's settings under the same names. To fall
back to G, put `58bece1d3184269059fd3f7f755c52f4f295113d` (branch `dsv41/candidate-g`) in `config/tensorfold.rev` and
regenerate the settings list; G does not read the profile's runner lines, so the launcher warns and does not forward
them. To fall back to F, put `56200688d52af54062fadbab12de36e95165ad72` (an ancestor of G on that branch) in
`config/tensorfold.rev`, also unset `TF_DSV41_COMPUTE_LANE` and regenerate the settings list. Once the upstream PRs
merge and the pin is on `ashhart/TensorFold`, the file points back there. A local checkout or a bundle remains the
fallback, as described below. This source support is not a new
pack/device qualification. Do not edit a pack config to bypass admission checks. Native images, structured output,
defaults, sessions and sampling still require migration qualification; historical production results do not
qualify this pin.

## Session restore (#86)

[#86](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/issues/86): with the replay prefill and no
`TF_DSV41_SESSION_DELTA`, every session restore off the NVMe tier failed `Disk.validate` with `BadBlob`, so a
returning long conversation paid its whole prefill again. Upstream #2182 (goshi) relaxes the two checks that no writer
honours; crystal8194 measured it on G (`58bece1d3` plus #2182) at 16 `resumed ... from NVMe`, 0 `BadBlob` and a warm
4x60k TTFT of 0.4 s instead of 30.6 s. The measurements below are of `7b79ef92d`, which first pinned the fix; the
current pin carries the same two upstream commits, and they were not repeated on it.

On `7b79ef92d`, one TP2 boot of the flash17 default profile (`r-7b79ef92d846-runner`, 2026-11-08), `vllm bench serve`
prefix_repetition, 4 prefixes x 60,000 tokens + 512 suffix, 16 prompts, `--max-concurrency 1`, `--seed 0`, 256 output
tokens, `--ignore-eos`, cold then the same 16 prompts warm (TTFT from the engine's `done` lines, all 16 requests):

| | issue, G (`58bece1d3`) | `7b79ef92d`, cold | `7b79ef92d`, warm |
| --- | --- | --- | --- |
| TTFT median | 30.6 s cold, 30.6 s warm | 25.24 s | **0.300 s** (0.289-0.314) |
| `BadBlob`, both ranks | 16 | 0 | **0** |
| `resumed ... from NVMe` / `from RAM` (rank 0) | 0 / 0 | 0 / 0 | 0 / **16** (60,496 of 60,512 tokens each) |

The issue's G figures come from crystal8194's profile, not this one. The warm cell resumed from the RAM tier. vLLM's client marked 2 of the 16 cold requests failed ("never received a
valid chunk") although the engine logged all 16 complete (256 tokens, HTTP 200); its own cold median over the other 14
is 28.73 s.

To force the NVMe tier, the same profile with only `TF_DSV41_SESSION_RAM_MIB=16` added (no 60k entry fits in RAM;
goshi's recipe on #86) was booted on `01a62132c` and on `7b79ef92d`, the same day: 6 distinct prompts of 60,000 +
512 tokens cold, then the same 6 warm, otherwise as above (engine TTFT, all 6 requests of each cell):

| `SESSION_RAM_MIB=16` | `01a62132c` (without the fix) | `7b79ef92d` (with it) |
| --- | --- | --- |
| cold TTFT median | 24.84 s | 25.61 s |
| warm TTFT median | 25.23 s (prefilled from 0) | **0.353 s** (0.300-0.371) |
| `BadBlob`, rank 0 / rank 1 | 6 / 6 | **0 / 0** |
| warm `resumed ... from NVMe` / `from RAM` (rank 0) | 0 / 0 | **5** / 1 |
| entries that "did not come back" (rank 0) | 6 | 0 |

`01a62132c` reproduces #86 on this pair; `7b79ef92d` restores five of the six from NVMe (60,496 of 60,512 tokens
each) and the sixth from RAM. Rank 1 logs `BadBlob` but not resumes, so resumes are counted on rank 0. The upstream
host tests add 13 `session_codec` cases (`zig build test-dsv41-device-host` 240/240 here against 227/227 before).
vLLM's client marked 2, 4 and 1 requests failed in `01a62132c`'s cold and warm cells and `7b79ef92d`'s cold cell; the
engine logged every request complete.

## Results at this pin (2026-11-11)

Measured on 2026-11-11, TP2 on two DGX Sparks, the flash17 K6 MTP pack, the flash17 profile with the compute lane on.
The tables are in [BENCHMARKS](BENCHMARKS.md#the-g-port-5d25d4ff4).

- **Speed against production G** (`58bece1d3`): the switch boot's decode geometric mean over four cells was **1.024**
  of the median of 26 G restores that day; a second boot, the first restore cycle, measured 1.021.
- **Tokens**: replies token-identical to G's on 12 of 12 in both boots. Greedy 2K and exact 8K prefill token IDs equal
  the H reference. The thinking-off set scored 19 of 20, the same rows as G, with all 20 replies identical to G's; a
  159-item task subset scored 151 of 159, the same items correct as G.
- **Serving checks**: a native image request and the compute-lane smoke pass. Before shutdown, both ranks logged 0
  allocator reports and 0 error lines. After SIGTERM each rank logs 1629 allocator reports: the known teardown pattern
  of the runner's graph bank, not a serving leak.
- **CUDA forward compatibility** (an earlier window that day, on this pin's image with the compat layer, one
  boot on the image's driver and one with `TF_CUDA_COMPAT=0`): decode **+3.5 %** faster with the compat driver, paired
  on identical tokens (faster on 14 of 15 requests, one-sided sign test p = 0.0005), and the thinking-off set **9.5 %**
  faster by wall time.

## Results at H (2026-10-29/30)

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

`scripts/build-runner-image.sh` layers the in-model runner over that image, and `scripts/build-compat-image.sh` the CUDA
forward-compatibility driver over the runner image ([OPERATIONS](OPERATIONS.md#cuda-forward-compatibility-tf_cuda_compat));
neither adds an engine binary.

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

The core lint command reads `.ruff.toml`, which excludes only the frozen,
hash-pinned study records in `bench/compute-lane/`. Do not add `--isolated` to that
command: it would ignore the archive policy. The archive's import, link, manifest
and oracle checks remain separate gates; the exclusion does not qualify them.

```bash
ruff check --select E4,E7,E9,F .
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
