# Native TensorFold engine

The recipe builds upstream [TensorFold](https://github.com/ashhart/TensorFold) at the exact commit in
[`config/tensorfold.rev`](../config/tensorfold.rev). That file is the single engine pin. The Docker build resolves
and verifies it, builds the `dsv41-serve` target with both CUDA runtime and DeepSeek family images, and installs
`tensorfold-dsv41` and `tf-dsv41`. The serving build uses upstream Git objects without recipe patches. The older native export stays in
`engine/zig` solely to preserve the existing generator/reference dependencies pending their equivalence audit.

`tensorfold-dsv41 serve` owns rank 0; `tf-dsv41 follow` owns rank 1. `dsv41-serve` is the upstream build target,
not the executable name. The experimental family requires `TF_EXPERIMENTAL=deepseek`.

The current pin is the local `dsv41/upgrade-latest` commit `5368c7a35`, including
flash17 `dsq-1` checkpoint admission from `98f2a3896`. It is not on upstream main yet.
Use a local checkout or self-contained bundle as described below; upstream fetching
will work once that exact commit is published. This source support is not a new
pack/device qualification. Do not edit a pack config to bypass admission checks.
Native images, structured output, defaults, sessions and sampling still require
migration qualification; historical production results do not qualify this pin.

## Build and checks

[`scripts/build-native.sh`](../scripts/build-native.sh) supplies the pin to [`docker/zig/Dockerfile`](../docker/zig/Dockerfile).
The builder uses Zig 0.17.0, CUDA 13 and real sm_121 images; the runtime keeps NCCL, verbs/mlx5 and pinned cuBLAS
library verification. A host compile with empty or synthetic GPU images is insufficient for serving.

Until the commit is available upstream, supply a checkout containing the pin:

```bash
TENSORFOLD_SOURCE=/absolute/TensorFold scripts/build-native.sh -t tensorfold-dsv41:local
```

The script reads the exact revision from the pin file, fetches only that revision's
history into temporary scratch, and creates a self-contained bundle. It does not
modify the supplied checkout, use its working-tree edits, or require its HEAD to
match the pin. For transfer to another builder, create a bundle containing the
upgrade branch and supply the bundle file instead:

```bash
git -C /absolute/TensorFold bundle create /absolute/tensorfold.bundle dsv41/upgrade-latest
TENSORFOLD_SOURCE=/absolute/tensorfold.bundle scripts/build-native.sh -t tensorfold-dsv41:local
```

Bundle prerequisites and exact commit resolution are verified locally and again in
the Docker builder. A supplied invalid/incomplete bundle fails without falling back
to the network. Only the Dockerfile, source resolver, shared credential boundary and selected bundle enter the
temporary build context; it is removed when the command exits. The bundle never
enters the runtime image. Without `TENSORFOLD_SOURCE`, the builder fetches the same
pin from upstream; it does not follow a moving branch. No GitHub push is required.

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
remain in the separately mounted operator-owned file, never in remote session
environment or diagnostic arguments.
