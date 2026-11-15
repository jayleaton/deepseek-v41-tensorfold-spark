# Native TensorFold engine

The recipe builds unmodified upstream [TensorFold](https://github.com/ashhart/TensorFold) at the exact commit in
[`config/tensorfold.rev`](../config/tensorfold.rev) (`f212eeabd`, on upstream `main`), fetched from
[`config/tensorfold.repo`](../config/tensorfold.repo), and applies this recipe's patch series over it:
[`patches/`](../patches/) `0001` to the number in [`config/tensorfold.patches`](../config/tensorfold.patches), in
order, with `git am`. The result must be the engine tree in [`config/tensorfold.tree`](../config/tensorfold.tree), or
the build stops. The Docker build resolves and verifies the commit, applies the series, builds the `dsv41-serve` target
with both CUDA runtime and DeepSeek family images, and installs `tensorfold-dsv41` and `tf-dsv41`. The image carries
the upstream commit as `org.tensorfold.revision` and the tree as `org.tensorfold.tree`, and the launcher refuses an
image whose labels differ from the two files. The older native export stays in `engine/zig` solely to preserve the
existing generator/reference dependencies pending their equivalence audit.

## The patch series

The 72 patches are the DeepSeek-V4.1 engine the pair serves (G, `6c4eeccda`) that is not on upstream `main` yet: G's
first-parent history over its upstream base `f212eeabd`, one patch per commit, each with its author. Two of those
commits were merges of a side branch holding one commit each, whose diff is exactly that commit's; each is one patch
carrying that commit's message, author and date, with a note that it came from a merge (`0013`, Flash-Next's compute
binding in its family; `0010`, engine-owned reply bytes that survive every strip). As this work reaches upstream
`main`, the pin moves forward and the series gets shorter. Applied to `f212eeabd`, `0001`-`0072` give tree
`8ea3edd87`, which is the content of `6c4eeccda` except for 24 comment and notes lines in 6 files that named internal
machines, tools and people. The checkout's last commit is `986206947` with the fixed committer `fetch-source.sh` uses.

To get the patched engine source, for example to regenerate the settings list or build by hand:

```bash
bash docker/zig/fetch-source.sh "$(cat config/tensorfold.rev)" /absolute/new-dir /nonexistent \
  "$(cat config/tensorfold.repo)" "$PWD/patches" "$(cat config/tensorfold.patches)" "$(cat config/tensorfold.tree)"
```

The previous recipe engine (`12257eb51`: upstream `0d51120a8` plus `0001`-`0207` of the earlier series, tree
`1d4482025`) and the engines before it (G `58bece1d3`, F `56200688d`, H `20ad77878`) are prefixes of that earlier
series, not of this one. To roll back, check out the recipe's previous commit: its `config/` and `patches/` pin it,
and its ENGINE has the rollback table for the older engines.

### This engine against the previous recipe engine

`6c4eeccda` sits on upstream `f212eeabd` (2026-11-09), 2,517 commits newer than the previous pin `0d51120a8`.
Among the upstream changes: tool-call ids keyed over the request (a replay names its calls again;
`TENSORFOLD_TOOL_ID_KEY_FILE`), opt-in chunked prefill admission for long prompts (`TENSORFOLD_PREFILL_POLICY`,
`_ORDER`, `_TOKENS`), `TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS`, and Engram shards whose direct reads refuse a sector read
buffered instead of failing. Flash17's `dsq-1` checkpoint admission is on upstream itself at this base (`b284ce1e2`,
the upstream form of the previous series' `98f2a3896`). The G line adds:

- the compute lane and the in-thinking runner in TensorFold's shared core, each family binding its own names: the
  generic `TF_COMPUTE_LANE*` / `TF_RUNNER*`, or DeepSeek-V4.1's `TF_DSV41_COMPUTE_LANE*` / `TF_DSV41_RUNNER*` as
  before (a boot sets one or the other; both is a startup error). The flash17 profile keeps the `TF_DSV41_*` names;
- engine-owned reply bytes (the runner's written values) that survive every prose strip, cut at each format's own call
  openers;
- `TF_DSV41_ENGRAM_READERS` and `TF_DSV41_PROMPT_PROF`;
- the image's CUDA 13.3 forward-compatibility layer (below).

The image does not ship PDFium, so PDF inputs (`--vision-pdf`, upstream) stay unavailable, as on G's own image; the
image's library check lists `libpdfium.so` with the HIP runtimes as not requested.

Both engines have IDX_SPLIT, IDX_IOTA, ROT_FOLD, KV_MIRROR, DEPTH_OVERHEAD_MS, `TF_DSV41_SESSION_LCP_RESUME` (#80),
the NVMe session-restore fix (#86), `deadline_s`, `compute_executed`, `TF_DSV41_COMPUTE_LANE_DEFAULT`, the runner's
COMMIT / ABSTAINS / INLINE_CHAIN / INLINE_MARK, and undeclared tool calls returned whole. Only the previous engine has
`TF_DSV41_RUNNER_SPEC_HOLD` (speculation that survives a runner fire, off in the profile); the launcher refuses
nothing for it but warns that this engine does not read it.

### CUDA forward compatibility

G serves on NVIDIA's CUDA 13.3 user-mode driver (libcuda 610.43.02) over the Sparks' older kernel module.
[`scripts/build-compat-image.sh`](../scripts/build-compat-image.sh) layers it over the runner image: it copies the
driver files out of NVIDIA's NGC PyTorch 26.07 container (by digest, the base of Mia / MiaAI-Lab's dual-Spark image;
`COMPAT_IMAGE` / `COMPAT_PATH` take another image holding the same files), requires exactly the files and links in
[`docker/compat/compat.sha256`](../docker/compat/compat.sha256) and
[`compat.links`](../docker/compat/compat.links), and sets the entrypoint
[`tf-compat-entry`](../docker/compat/tf-compat-entry). The entry puts the driver first on the loader path only while
the host kernel module is older than 610 and `cuInit` succeeds through it, and logs one line saying which driver runs;
`TF_CUDA_COMPAT=0` in the profile forces the host driver (`run-zig.sh` accepts 0 or 1 and passes it to the
entrypoint; the engine does not read it). The engine files and labels are the runner image's.

[`config/tensorfold-settings.txt`](../config/tensorfold-settings.txt) lists every `TF_*` / `TENSORFOLD_*` setting
the pinned engine reads. The launcher forwards exactly the ones a profile sets, plus `NCCL_*`, never credential
names. It refuses a renamed or removed setting (for example `GLM53_TF_ROCE_FAST`, now `TF_TP_ROCE_FAST`) and warns
about any other engine-looking name the pin does not read, so an old profile cannot silently fall back to defaults.
Every pin bump regenerates the list from the new engine source, and a test fails when the two disagree:

```bash
python3 scripts/engine_settings.py generate /absolute/new-dir   # the patched checkout above, after any pin or series change
```

`tensorfold-dsv41 serve` owns rank 0; `tf-dsv41 follow` owns rank 1. `dsv41-serve` is the upstream build target,
not the executable name. The experimental family requires `TF_EXPERIMENTAL=deepseek`.

The previous recipe engine was patches `0001`-`0207` of the earlier series (`12257eb51`): `1ea71240a` plus two
server commits (see
[Undeclared tool calls](#undeclared-tool-calls-at-this-pin)). `1ea71240a` (`0205`) is `7b79ef92d`
plus two commits for [#80](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/issues/80) (see
[Rewritten histories](#rewritten-histories-at-this-pin)).
`7b79ef92d` (`0203`) is `07106059c`, the exact engine source of the compute-lane
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
measurement is under [Session restore at this pin](#session-restore-at-this-pin)). [compute
lane](compute-lane.md) lists the measured G commits'
counterparts in this history.

The recipe pinned `01a62132c` (`0201`, without #2182) before; earlier it pinned release candidate G, `58bece1d3` (patch
`0050` gives its tree, apart from the comment lines above). G is candidate F (`56200688d`, the
fallback pin) plus the compute lane (`TF_DSV41_COMPUTE_LANE`, off in the engine, on in the flash17 profile; [OPERATIONS](OPERATIONS.md)) and its
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

A local checkout or a bundle remains the
fallback, as described below. This source support is not a new
pack/device qualification. Do not edit a pack config to bypass admission checks. Native images, structured output,
defaults, sessions and sampling still require migration qualification; historical production results do not
qualify this pin.

> The sections from here on measured the previous recipe engines: "this pin" in them means the engine each one
> names (`7b79ef92d`, `1ea71240a`, `12257eb51`, H), not `6c4eeccda`.

## Session restore at this pin

[#86](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/issues/86): with the replay prefill and no
`TF_DSV41_SESSION_DELTA`, every session restore off the NVMe tier failed `Disk.validate` with `BadBlob`, so a
returning long conversation paid its whole prefill again. Upstream #2182 (goshi) relaxes the two checks that no writer
honours; crystal8194 measured it on G (`58bece1d3` plus #2182) at 16 `resumed ... from NVMe`, 0 `BadBlob` and a warm
4x60k TTFT of 0.4 s instead of 30.6 s.

On this pin, one TP2 boot of the flash17 default profile (`r-7b79ef92d846-runner`, 2026-11-08), `vllm bench serve`
prefix_repetition, 4 prefixes x 60,000 tokens + 512 suffix, 16 prompts, `--max-concurrency 1`, `--seed 0`, 256 output
tokens, `--ignore-eos`, cold then the same 16 prompts warm (TTFT from the engine's `done` lines, all 16 requests):

| | issue, G (`58bece1d3`) | this pin, cold | this pin, warm |
| --- | --- | --- | --- |
| TTFT median | 30.6 s cold, 30.6 s warm | 25.24 s | **0.300 s** (0.289-0.314) |
| `BadBlob`, both ranks | 16 | 0 | **0** |
| `resumed ... from NVMe` / `from RAM` (rank 0) | 0 / 0 | 0 / 0 | 0 / **16** (60,496 of 60,512 tokens each) |

The issue's G figures come from crystal8194's profile, not this one. The warm cell resumed from the RAM tier. vLLM's client marked 2 of the 16 cold requests failed ("never received a
valid chunk") although the engine logged all 16 complete (256 tokens, HTTP 200); its own cold median over the other 14
is 28.73 s.

To force the NVMe tier, the same profile with only `TF_DSV41_SESSION_RAM_MIB=16` added (no 60k entry fits in RAM;
goshi's recipe on #86) was booted on the previous pin and on this one, the same day: 6 distinct prompts of 60,000 +
512 tokens cold, then the same 6 warm, otherwise as above (engine TTFT, all 6 requests of each cell):

| `SESSION_RAM_MIB=16` | `01a62132c` (previous pin) | `7b79ef92d` (this pin) |
| --- | --- | --- |
| cold TTFT median | 24.84 s | 25.61 s |
| warm TTFT median | 25.23 s (prefilled from 0) | **0.353 s** (0.300-0.371) |
| `BadBlob`, rank 0 / rank 1 | 6 / 6 | **0 / 0** |
| warm `resumed ... from NVMe` / `from RAM` (rank 0) | 0 / 0 | **5** / 1 |
| entries that "did not come back" (rank 0) | 6 | 0 |

The previous pin reproduces #86 on this pair; this pin restores five of the six from NVMe (60,496 of 60,512 tokens
each) and the sixth from RAM. Rank 1 logs `BadBlob` but not resumes, so resumes are counted on rank 0. The upstream
host tests add 13 `session_codec` cases (`zig build test-dsv41-device-host` 240/240 here against 227/227 before).
vLLM's client marked 2, 4 and 1 requests failed in the old pin's cold and warm cells and the new pin's cold cell; the
engine logged every request complete.

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

`scripts/build-native.sh -t tensorfold-dsv41:local` fetches the upstream pin from that repository and applies the
series. To build without the network, or from your own copy, supply an upstream checkout containing the pin (the
series is applied the same way):

```bash
TENSORFOLD_SOURCE=/absolute/TensorFold scripts/build-native.sh -t tensorfold-dsv41:local
```

The script reads the exact revision from the pin file, fetches only that revision's
history into temporary scratch, and creates a self-contained bundle. It does not
modify the supplied checkout, use its working-tree edits, or require its HEAD to
match the pin. For transfer to another builder, such as a Spark that has no engine checkout, create a bundle
containing upstream `main` and supply the bundle file instead:

```bash
git -C /absolute/TensorFold bundle create /absolute/tensorfold.bundle main
git -C /absolute/TensorFold bundle verify /absolute/tensorfold.bundle   # must list the pin's history
TENSORFOLD_SOURCE=/absolute/tensorfold.bundle scripts/build-native.sh -t tensorfold-dsv41:local
```

The bundle must contain the pinned commit and all of its history (`git bundle create ... <branch>`, not a range).

Bundle prerequisites and exact commit resolution are verified locally and again in
the Docker builder. A supplied invalid/incomplete bundle fails without falling back
to the network. Only the Dockerfile, source resolver, shared credential boundary and selected bundle enter the
temporary build context; it is removed when the command exits. The bundle never
enters the runtime image. Without `TENSORFOLD_SOURCE`, the builder fetches the same
pin, by commit, from `config/tensorfold.repo`; it does not follow a moving branch. Only the patches the series
names enter the build context, and a missing or duplicated number stops the build.

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

## Rewritten histories at this pin

A session resumes only from a stored entry that is an exact prefix of the new prompt, so a client that rewrites the
middle of a history (drops an old image, edits a message, compacts) prefills from the last entry before the change:
on a long agent session often the whole context ([#80](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/issues/80)).
A longer entry cannot be cut at the change: every layer's sliding-window ring, the ratio-2 carries and the replay stash
at that token depend on the history, and only a snapshot taken there holds them.

`a3cf60d67` and `1ea71240a` add, all off unless `TF_DSV41_SESSION_LCP_RESUME=1`:

- **Checkpoints.** The prefill planner also snapshots a prompt at every multiple of `TF_DSV41_SESSION_LCP_EVERY`
  tokens (default 32768, a multiple of 16), before its replay point; the first is the first multiple at or past the
  64-token snapshot minimum (`1ea71240a`: a spacing of 16, 32 or 48 made none before). A rewritten history resumes
  from the last checkpoint before the change. Each one is an ordinary RAM-tier entry (a few MB of bounded state, no
  extra pool pages).
- **Prerequisite.** Only the round planner (`TF_DSV41_PREFILL_PIECES=1`, which needs `TF_DSV41_OWN_PREFILL=1` and
  `TF_DSV41_PROMPT_TAIL=verify`, both defaults) saves checkpoints. Whole-prompt mode (`TF_DSV41_PREFILL_PIECES` off)
  snapshots each prompt once, so there the setting makes no checkpoints. The shipped profiles use the planner.
- **Retention.** Checkpoints take their conversation's recency on every resume, so the RAM budget evicts a chat's older
  turns before them; an evicted checkpoint is dropped, never written to the NVMe tier.
- **Images.** A resume never starts inside an image's positions; it moves back to an entry at or before the image.

With the setting unset the planner's pieces, the snapshots and every resumed row are those of `7b79ef92d` (a host test
compares a trace of both). Whatever the setting, an admission that loses at least 1,024 tokens of reuse logs
`sessions: slot S: a N-token prompt diverged at token D (stored M); it prefills from P`.
[IMAGES](IMAGES.md#long-histories-with-many-images) explains why a history with many images is rewritten.

Measured on two Sparks with the same commit ported to the serving engine `5d25d4ff4` (2026-11-11, G's recipe, knob on;
nine turns, a 67.5K-token log then eight screenshots; the ninth turn drops the oldest screenshot): the rewritten
68,632-token turn resumed 65,536 tokens from RAM and reached its first token in 1.97 s, against 43.99 s for a cold
prefill of the same prompt with the knob off, and its greedy reply equals the cold one token for token.

## Undeclared tool calls at this pin

A model sometimes calls a tool by a name the request did not declare: an agent client such as OpenCode 2 declares
its shell tool as `shell`, and DeepSeek-V4.1 can write a call to `bash` after a history of such calls. Up to
`1ea71240a` the server kept that well-formed call as the reply's text, so the reply ended with `finish_reason: "stop"`
and no `tool_calls`, and the client took the DSML block as its final answer and ended the turn. An outside OpenCode
2.2.7 benchmark of this recipe at `176024b` lost 4 of its 30 trials this way, by the reporter's reading of the
transcripts; [the report, its 30 rows and what each source shows](../results/undeclared-tool-20261112/README.md). By
that report, vLLM returns the call, the client answers that the tool is unknown, and the model retries; we have not
measured that recovery.

`3ca06dd4b` returns it the same way, and `12257eb51` requires the call to be whole: a whole call block naming an undeclared tool is a `tool_calls` entry with
`finish_reason: "tool_calls"`, streamed or not, when the name is 1 to 64 letters, digits, `_` or `-` and the arguments
are a JSON object. Everything else stays as before: a declared tool keeps its declared spelling; a name with spaces,
dots or other characters, malformed or unclosed blocks (an undeclared XML or GLM call with stray text in its markup
too, in either call mode), and bare JSON that names no declared tool (a structured answer) stay text; a request without tools never gets a tool call; a call inside the reasoning is never made; and a
call cut by the end token or `max_tokens` is completed only for a declared tool. The engine's upstream line kept
undeclared calls as text to match its frozen Python answers (`7be90a36b`), although its API docs describe this
behaviour, so the change is a recipe patch (`0206`, `0207`) with its own host tests: they fail on `1ea71240a`, and one replays
DeepSeek-V4.1's own reply captured on the Sparks (a request declaring only `shell`, after two `bash` calls in the
history, answered with a DSML `bash` call and `finish_reason: "stop"`).
