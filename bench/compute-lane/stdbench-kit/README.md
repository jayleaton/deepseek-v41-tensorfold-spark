# Serial CPU/research benchmark kit

Python 3.10+ and standard library only; Linux is required for coding resource limits.
Implementation was written and statically inspected locally. **No Python, selftest,
dataset fetch, live request, or coding completion has been executed by the author.**
The parent must run the verification below on its selected guarded node (<rank0-host> or
<rank1-host>), inspect the artifacts, and deploy. No server boot, queue, SSH, or endpoint
is implicit in any command. There is no default endpoint.

Standards inspection: Projects and the target ancestry contained no AGENTS.md,
LOCAL-AGENT.md, CONTRIBUTING, or lint/build configuration. This directory is not a git
repository. No project-mandated check commands were found. The following checks
are the handoff verification commands, not claims of successful execution.

## Verification on the parent's node only

Run commands from the copied `kit/` directory. Choose fresh output paths; runs,
normalized tasks, and graded JSONL deliberately refuse overwrite. Use one process
at a time; there are no worker pools or retries. Keep artifacts for review and
remove only the exact scratch output directories the parent created afterward.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 -m py_compile bench.py grader_worker.py
python3 bench.py selftest --out selftest-artifacts --bootstrap 20000
```

Inspect `selftest-artifacts/selftest.json`, `report.json`, both `*-graded.jsonl`,
and both `*-run/responses.jsonl`. Selftest uses the **same serialized request,
response parser, grading functions, and report implementation** as real/replayed
runs. It never constructs a network transport. Only its hardcoded trusted
HumanEval-style add/subtract fixtures execute in subprocesses; there is no CLI
option to trust arbitrary completions. Fixtures include correct, wrong,
generation timeout, HTTP error, truncated, missing usage, and manual-review pairs
at seeds 101/102. Each arm's resolved accuracy must be 4/14, not all passing;
primary aggregate accuracy must be null because one suite needs manual review.
`guard-fixtures.json` additionally proves a trusted `os._exit(0)` completion is
rejected despite exit code zero, a stop/text response containing `tool_calls` is
rejected, and unsupported symbolic MATH remains pending.
The usage fixture is 10 prompt + 20 completion = 30 total, with 12 reasoning
already included in completion. Selftest asserts that accounting, balanced boxed
fraction extraction, rational grading, and hard failure on duplicate/missing
pairs. Real subprocess success/failure is required by its expected accuracies.
A report is generated only if all checks pass; these are expected results until
the parent actually runs it.

Replay through the CLI without any network (including no localhost requests):

```sh
python3 bench.py run --tasks selftest-artifacts/tasks.jsonl --arm plain --max-tokens 128 --replay selftest-artifacts/plain-replay.jsonl --out replay-plain
python3 bench.py run --tasks selftest-artifacts/tasks.jsonl --arm lane --max-tokens 128 --replay selftest-artifacts/lane-replay.jsonl --out replay-lane
python3 bench.py grade --run replay-plain --out replay-plain-graded.jsonl
python3 bench.py grade --run replay-lane --out replay-lane-graded.jsonl
python3 bench.py report --plain replay-plain-graded.jsonl --lane replay-lane-graded.jsonl --plain-run replay-plain --lane-run replay-lane --out replay-report.json
```

CLI replay does not bypass the coding isolation gate: without a proper container,
its otherwise successful coding responses become manual review. Consequently its
resolved denominator differs from the trusted built-in selftest. Recorded timing
is used for reproducible reports, while actual monotonic replay overhead is also
recorded separately. Replay can read a prior `responses.jsonl` directly; every
requested key must occur exactly once and extra replay keys fail.

## Locked sources and normalization

The parent supplies the three original files under `kit/sources/`:
`gsm8k.jsonl`, `math500.jsonl`, and `humaneval.jsonl.gz`. Their immutable upstream
revisions, URLs, formats, and compressed/original-byte SHA256 values are in
`datasets.lock.json`. Normalization is offline and verifies every selected source
hash before reading it. No dataset is bundled or downloaded here.

```sh
python3 bench.py normalize --suites gsm8k --limit 48 --out gsm8k-n48.jsonl
python3 bench.py normalize --suites math500 --limit 20 --out math500-n20.jsonl
python3 bench.py normalize --suites humaneval --limit 24 --out humaneval-n24.jsonl
```

Subset rank is exactly ascending SHA256 of UTF-8 `stdbench-v1:` followed by the
original stable id; take the first n within each suite, with id as tie-breaker.
GSM8K uses its original zero-based JSONL row index when upstream has no id;
HumanEval uses `task_id`; MATH-500 uses `unique_id` (then id, then row index).
No random shuffle selects the subset. Run order separately shuffles the selected
task list with Python `random.Random(20261104)`, preserving seed order 101,102
inside each task; both arms therefore receive the same order. Retain normalized
files and manifests to preserve exact ordering across Python versions.

GSM8K maps question/answer, extracts reference text after the final `####`, and
adds only `Return final answer after ####.` to the question. HumanEval maps
prompt/test/entry_point/task_id from gz JSONL and adds only a complete-function
single-Python-fence instruction; base tests are retained separately and **never
included in the model request**. Original prompt is retained as `code_prefix` for
native body completions. MATH-500 maps problem/answer/unique_id; it is not a full
symbolic-equivalence evaluation. HumanEval means original **base** HumanEval,
not HumanEval+. No AIME suite is selected.

Optional parent-authorized fetching is separate and always requires the explicit
network flag, even to verify cached files:

```sh
python3 bench.py fetch --fetch-network
```

Dataset fetch permits HTTPS-only redirects (needed by HF resolve/cache), sends no
API authorization header, validates the final bytes against SHA256, and never
replaces an existing file. Live inference refuses all redirects. Lock entries
must contain a full 40-hex immutable revision in the URL path and a 64-hex SHA256;
source paths cannot escape the lock directory. A new source needs its own lock
entry, not an unpinned branch/tag URL.

Generic JSONL is supported with lock format `normalized`: fields `id`, `suite`,
`prompt`, and `kind`, plus `answer` for `integer`, `numeric`, or `math_manual`, or
`test` and `entry_point` for `humaneval`/`mbpp`. `math_manual` means conservative
MATH grading, not that every item is manual. MBPP source format also accepts
`task_id`/`id`, `text`/`prompt`, `test_list`/`test`, and an explicit `entry_point`.
Supply any required base-test setup/imports in `test`; no plus/hidden tests or
solutions belong in prompts. No MBPP data or AIME data is selected in this lock.

## Inference protocol and explicit live opt-in

Both arms use model alias **DeepSeek-V4.1-Flash-TF**, intended underlying build
**1017**. The parent must independently verify the server's actual build before
live work: an alias in a request is not build provenance. Exact 1017 published
scores for these suites have not been established here. Any historical/base
model reference scores belong to their own model/version/protocol and must not
be presented as published 1017 numbers or as equivalent to this small subset.

Defaults: seeds 101,102; temperature 1; top_p .95; top_k 20;
`reasoning_effort: "high"`; `chat_template_kwargs: {enable_thinking: true}`;
max_tokens 8192; HTTP socket and whole-request timeout 300s; server `deadline_s: 270`.
The larger token cap and 300s/270s deadlines are explicit stdbench protocol
choices/deviations from the earlier AB3 protocol, not assumed identical settings.
The server must honor the thinking, effort, top_k, and deadline extensions;
errors are recorded with no retry and unsupported fields are never silently
removed. A Linux SIGALRM interval timer in the main serial thread bounds the
entire HTTP request including slowly arriving response/error-body bytes, in
addition to the socket timeout. The timer and prior handler are cleared/restored
after every request; an existing alarm timer is rejected. Server deadline
enforcement must also be checked by the parent. Monotonic elapsed time covers request serialization already prepared,
HTTP transport, response body read, and connection cleanup (not later grading).

Plain has no `tools` key. Lane adds exactly:

```json
{"tools": [{"type": "code_interpreter"}]}
```

There is no system guide, external runner, tool-response loop, or client execution
of tool calls. Lane relies on the server's sole native tool declaration. The
client does not expose or manage any server-side native tool execution. If the
server returns any message containing `tool_calls` (even with stop and correct
text), or a tool-call finish instead of a final text completion, the record
is retained and graded incorrect as unsupported finish/content. All failures,
timeouts, truncations, and missing usage remain records. A partial interrupted
run fails complete-manifest grading rather than silently dropping absent tasks.

There is no live inference unless **all three** of `--live`, an explicit full
chat-completions endpoint, and `--go-file` are supplied. The file must contain
exactly `GO` after whitespace stripping and is checked before every request.
The kit never creates this file. The parent must supply its approved endpoint
and GO path; this example intentionally contains no actual endpoint:

```sh
python3 bench.py run --tasks gsm8k-n48.jsonl --arm plain --live --endpoint "$APPROVED_CHAT_COMPLETIONS_URL" --go-file "$APPROVED_GO_FILE" --out gsm8k-plain
python3 bench.py run --tasks gsm8k-n48.jsonl --arm lane --live --endpoint "$APPROVED_CHAT_COMPLETIONS_URL" --go-file "$APPROVED_GO_FILE" --out gsm8k-lane
```

Repeat serially for math500-n20 and humaneval-n24 with separate output directories;
never run both arms concurrently. `OPENAI_API_KEY` is optional and read from the
environment, never written to artifacts. `--max-tokens`, `--request-timeout`,
`--deadline-s`, `--seeds`, and `--order-seed` are explicit overrides. Preserve the
raw JSON text response and manifest, not just extracted answers. Missing usage
stays null; totals are prompt + completion only when both exist, never plus
reasoning. Raw server-reported totals and all response fields remain available
in the raw response. HTTP bodies and transport errors are retained.

## Grading and reports

```sh
python3 bench.py grade --run gsm8k-plain --out gsm8k-plain-graded.jsonl
python3 bench.py grade --run gsm8k-lane --out gsm8k-lane-graded.jsonl
python3 bench.py report --plain gsm8k-plain-graded.jsonl --lane gsm8k-lane-graded.jsonl --plain-run gsm8k-plain --lane-run gsm8k-lane --out gsm8k-report.json
```

Use the corresponding math500 run paths similarly. Numeric grading uses final
balanced `\\boxed{...}`, `####`, an explicit final-answer label, or an answer-only
response, with exact Decimal comparison and conservative thousands separators.
It never selects an arbitrary number from reasoning, evaluates expressions, or
uses rounding/tolerance. Integer kind supports AIME-like integer checks if a
future parent-selected lock supplies them. MATH accepts canonical string equality
and exact numeric/rational equality (decimal, integer, simple a/b or LaTeX frac).
Different parseable numeric values are incorrect; unsupported symbolic
comparisons need manual review. General MATH equivalence is unsupported.

**Primary accuracy and accuracy delta are withheld (null) whenever any paired
manual review remains.** `resolved_subset` is explicitly separate. Errors,
timeouts, and length truncations are incorrect. Manual decisions must be resolved
in a separately audited workflow before claiming full MATH accuracy; there is no
automatic pass on an unparsed symbolic answer.

Real HumanEval/MBPP completions execute only after `--isolated-container` plus a
container marker. Markers are merely a gate, **not a sandbox implementation**.
The parent must provide a dedicated unprivileged, network-disabled, read-only
container with no secrets/host sockets, only necessary read-only input and
separate writable output/tmp, dropped capabilities and no-new-privileges, memory
and CPU limits, and a PID budget permitting only benchmark parent + its one
worker (no candidate-created descendants). Do not execute real completions on the
host. Container provisioning is outside this kit; the option does not create one.
No real grading command with a pretend isolation flag is provided here.

Within that container run `grade` serially with `--isolated-container`, default
3s wall timeout and 256MiB worker address-space cap (configurable). Each task
gets a separate subprocess/temp directory; Linux rlimits bound CPU, address
space, output file size, open descriptors, process count, and core dumps. Output
is discarded and temp directories removed. Timeout kills only the exact worker
PID via `proc.kill()`, never a process group. Test success requires subprocess
exit 0 **and** a pipe receipt emitted by the worker only after test execution
returns. This rejects accidental early `os._exit(0)`/`sys.exit(0)` false passes.
Untrusted Python can introspect the worker and forge the receipt or alter tests;
these are base functional tests, not adversarially tamper-proof tests. Isolation
and the prohibition on extra processes must be enforced by the parent container.
Without the gate, code is retained for manual review and never executed.

Reports include per-suite and overall paired task+seed outcomes, accuracy deltas,
common-pair token/time sums, means, lane/plain ratios and deltas, per-arm mean and
median wall time, generation errors/timeouts, grading timeouts, truncations, and
missing usage. Token comparisons use only pairs with both values present;
missing-pair counts show the reduced denominator. No missing usage becomes zero.
Primary exact McNemar/binomial uses **seed 101 only**, one observation per task,
and is withheld if those tasks need manual review. All-seed task+seed McNemar is
marked descriptive because seeds repeat tasks.

Bootstrap defaults to **20,000** replicates, RNG **20261104**, 95% percentile CIs
for accuracy delta and token/time deltas and ratios. It resamples tasks within
suite strata, retaining all seeds for every selected task. This preserves
within-task dependence rather than treating seeds as independent tasks.
Null/undefined replicates remain excluded with valid-replicate counts shown.
Small pilot subsets and selected-suite mixtures are not full-dataset scores.
Both run manifests are required: duplicate, missing, unexpected, changed-task,
unequal-settings, and incomplete-seed pairs hard fail, including pairs missing
from both result files. There are no silent drops or automatic retries.
