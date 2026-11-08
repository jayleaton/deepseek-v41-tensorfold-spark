# Phase-2 analysis and blinded MATH review API

These offline, stdlib-only tools implement T5 spec SHA256
`9c7327007cfd15825a608f2f6b06f2b39e33b88548c5bb7deb1b209b7395d240`.
Analysis/review/activation execute no candidate code; the full grader uses Jail
for HumanEval. No module sends inference requests. Do not use the old kit
`make_report`, automatic MATH proposals, or its blanket length-finish grading
in place of this protocol. Freeze all four module hashes/dependencies before collection.
This document is the README for these modules; the shared phase2 README
belongs to the parent implementation and was left untouched.

## Analysis input and CLI

```sh
python3 analysis.py report --manifest frozen/analysis-plain.json \
  --graded gsm-graded.jsonl humaneval-graded.jsonl math-reviewed.jsonl \
  --out new-analysis.json
```

`--graded` files together must contain exactly one canonical row for every
scheduled `(suite,id,seed,arm)`, 368 rows overall. Files can combine arms or suites;
replace provisional MATH rows with merge output, rather than adding duplicates.
Canonical rows require:

```json
{"suite":"gsm8k","id":"stable-original-id","seed":101,"arm":"plain",
 "grade":"correct","status":"ok","finish_reason":"stop",
 "elapsed_seconds":1.25,"tokens":{"completion_tokens":17},
 "native_accounting":{"complete":true,"returned_tokens":23}}
```

Grades are `correct`, `incorrect`, `manual_review`; arms are `plain`, `lane`.
Every non-`ok` status is scored incorrect, regardless of grade. Missing/error
responses still require rows with actual elapsed wall time. A `length` finish
is censored and can be correct if the delivered final answer was graded correct.
The tool consumes grades; it never grades or repairs final answers itself.
`manual_review` is conservative incorrect in every scheduled denominator.

Manifest formats: the kit's flat `{"tasks":[...],"seeds":[101,102]}` (optional
`tasks_sha256` is checked), or freeze.py's `analysis-{arm}.json`, with
`seeds` and `suites: {name: {tasks:[...],tasks_sha256:"..."}}`. Each nested hash
is checked. Task suite/ID pairs must be unique. Production requires exactly
three suites with 48/20/24 tasks. The externally pinned freeze establishes the
specific selected IDs, not merely these sizes. All missing/duplicate/unexpected
rows hard-fail. An incomplete run is retained externally and analyzed after
completing the fixed cohort; this tool refuses to silently reduce the sample.
Output paths must be new; existing files are never overwritten.

`native_accounting` comes from the parent's independently qualified native-ID
reconciliation. `complete:true` attests chronological all-round native IDs,
including tool/protocol/forced returned output exactly once, prompt excluded;
`returned_tokens` must then be a nonnegative integer. Incomplete accounting may
supply a nonnegative integer observed lower bound, or null. Missing accounting
is allowed as missing evidence, never zero. Completion usage under `tokens`
is preserved in source hashes but never substituted for native returned IDs.
The analyzer cannot authenticate that attestation itself. Preserve ID/hash,
inserted-result reconciliation, reasoning/content/prompt counts and server/tool
timing in the source rows; unknown extra fields are retained in source hashes.

Reports are **per suite**, without pooled scores. Primary differences average
both seeds per task. Exact sign-flip DP uses integer weights `abs(d_i)` where
`d_i` sums the two seed correctness differences. Zero weights cancel; ties in
absolute tails are included and the all-zero p-value is one. Integer numerator
and denominator are exported. Holm adjusts the three primary suite p-values at
familywise .05. Bootstrap uses 20,000 draws of n task indices with replacement,
sorted task IDs, fresh `random.Random(20261104)` per suite and `randrange(n)`
for each sampled task. Both seeds and arms remain unchanged. The same draws
feed accuracy, token and time statistics. Sorted percentile endpoints are
indices 500 and 19499. An undefined denominator in **any** replicate withholds
the interval; undefined draws are never silently discarded.

Token/time metrics report paired-complete means, medians, sums ratio and mean
difference, plus ratio/difference CIs. Time includes failed/deadline elapsed
costs. Native-token paired completeness is explicitly counted; per-arm native
coverage and observed lower-bound totals are separate. Any incomplete native
pair withholds the full-cohort token claim. Missing-token sensitivities retain
the original n task clusters in resampling and select observed pairs within
each sampled cluster. They are not asserted representative of missing rows.
A separate sensitivity requires both generations `ok`, uncensored, and observed.

For each metric savings means plain minus lane. The net numerator sums signed
savings from either-arm censored pairs; its denominator is total net savings.
The positive numerator sums positive pair savings from censored pairs; its
denominator is all positive pair savings. Fractions are null if their denominator
is nonpositive. Both contributions and a strict `>25%` warning are exported.
A point-ratio reversal in the uncensored successful sensitivity is flagged.
No favorable uncensored speedup claim is made when censor contributions exceed
25%. Individual paired solve losses are always listed with original ID/seed.

`local_efficiency_numerical_conditions_met` checks the numerical T5 conditions
including adverse unresolved assignments. `primary_statistical_benefit_supported`
checks positive adverse difference with adverse Holm p<=.05; original conservative
primary p-values remain unchanged and separately reported. Neither flag qualifies
efficacy: `qualified_efficacy_claim` is always false until the parent audits native
execution safety, immutable manifests, accounting and completed blinded MATH
reviews. No nonsignificance/non-inferiority inference or historical-baseline
retention comparison is emitted.

Python API: `analyze(rows, manifest) -> report`; `validate_cohort(...) -> index`;
`exact_sign_flip(list[int]) -> {p_two_sided,tail_assignments,assignments,...}`;
`holm({suite:p}) -> {suite:adjusted_p}`. `unresolved_bounds(clusters)` accepts
lists of per-task `(plain_row,lane_row)` seed pairs and is shared with review.

## Seal all MATH answers before review

Before collection, create the private mapping from tasks alone:

```sh
python3 math_review.py preseal --manifest frozen/analysis-plain.json \
  --private-key precollection-mapping.json
```

The JSON manifest can contain the full frozen task set or only the selected20
normalized MATH tasks, with seeds101/102. No responses enter `preseal`. Pin the
printed canonical `preseal_sha256` externally before collection. The mode0600
mapping fixes candidateID->suite/ID/seed/arm and packet order with RNG20261105.
Production sealing now requires that exact file/hash; it joins replies without
rekeying. Modified mapping/order/task bytes fail hash validation.

```sh
python3 math_review.py seal --tasks normalized-tasks.jsonl \
  --responses plain-responses.jsonl lane-responses.jsonl \
  --preseal precollection-mapping.json --preseal-sha256 PRECOLLECTION_CANONICAL_HASH \
  --packet public-packet.json --private-key private-key.json
```

Tasks use kit normalized `kind:math_manual`, `prompt`, `answer`, `suite`, `id`.
Response files use bench's `content` for the actual final delivered message,
plus `suite,id,seed,arm,status`; retain full canonical graded-row fields in those
rows if merge output will go directly to analysis. Reasoning/tool messages do
not enter the packet. Non-MATH suite rows may coexist in inputs and are ignored.
Exactly 20 unique MATH tasks, both seeds and both arms (80 replies), are mandatory,
including failures and capped replies. No outcome filtering is possible through
the public CLI. Failed replies with null content appear as empty final text.

The public JSON array contains **only** `candidate_id`, `problem`,
`official_answer`, `candidate_final_content`. Opaque candidate IDs and packet
order use the presealed RNG20261105, independent of outcomes. Arm, seed, original
IDs, grades, ownership, timing, status and paired outcomes stay in the private
key. The postcollection key is created exclusively with mode0600; it contains the original rows,
per-row hashes, task hash, response-cohort hash, packet hash and mappings. Seal
prints canonical JSON SHA256 values for packet/key, not aggregates. Canonical
hashing is sorted-key compact UTF-8 JSON with unescaped Unicode, no newline,
and no NaN. These hashes differ from hashes of pretty printed on-disk bytes.
Record the printed postcollection key hash externally before review; restrict key access.
This second key embeds/authenticates the already pinned precollection mapping.
Distribute only the public packet to the two independent reviewers. Independence
and withholding aggregate scores are operator obligations; reviewer IDs cannot
prove human independence. Preserve sealed original labels before any merge.

Each reviewer supplies CSV or JSON array with fields:

```csv
candidate_id,reviewer_id,label,reason
C-opaque-id,reviewer-A,correct,Exact equivalent final answer
```

Labels are `correct`, `incorrect`, `unresolved`; every label requires a nonempty
reason. Both initial reviewers must label **every** candidate exactly once and
use distinct IDs. Prefer the last balanced boxed final answer; absent a box,
require an unambiguous explicit final statement. Exact equivalence is permitted;
approximation only when the problem/reference requests it. Reviewers see original
final content without rewriting or model queries. No aggregate scores may be
shared until reviews and any third-review labels are sealed.

```sh
python3 math_review.py merge --packet public-packet.json \
  --private-key private-key.json --key-sha256 PRESEALED_CANONICAL_KEY_HASH \
  --first reviewer-a.csv --first-id reviewer-A \
  --second reviewer-b.json --second-id reviewer-B \
  --third reviewer-c.csv --third-id reviewer-C \
  --out math-reviewed.jsonl --audit private-review-audit.json
```

Omit both third arguments when unavailable. Third labels may cover any subset
of initial disagreements and must not label agreed candidates. Their ID must
be distinct. Route disagreements to the third reviewer using only the relevant
public packet items; the audit contains private mappings and aggregates and
must never serve as a reviewer packet. Two matching correct/incorrect labels
resolve; otherwise retain unresolved. Initial agreement on unresolved stays
unresolved. Missing third votes leave disagreements unresolved. Generation errors
remain incorrect even if reviewers mislabel their content. Output rows preserve
original fields, replace `grade`, and attach `math_review` sealing metadata.
Merged JSONL and audit are created mode0600, exclusively. Audit retains every
original reviewer label/reason, reviewer hashes, disputed IDs and decisions.
Key/packet hashes, per-row hashes and full private cohort coverage are rechecked.

Unresolved primary scoring is incorrect. Bounds report per-arm verified-correct
and all-unresolved-correct rates. Difference adverse assigns lane unresolved
wrong/plain unresolved correct; favorable reverses assignments. Both arms
unresolved correct is **never** used as a difference bound. While unresolved
remain, the label is **verified-correct rate**. Full analysis repeats these
bounds and computes adverse bootstrap and Holm sensitivity; a benefit statement
cannot rely merely on the conservative point estimate.

Python API: `preseal(tasks,key_path) -> hashes`; `build_preseal(tasks) -> mapping`;
`seal(tasks,rows,packet_path,key_path,preseal_path=...,preseal_sha256=...) -> hashes`;
`merge(packet,private,first,second,first_id,second_id,key_sha256=...,third=None,
third_id=None) -> (graded_rows,audit)`. JSONL output writes occur only in CLI.
Keep the phase2 modules together for imports.

## Standards, verification and work ledger

2026-11-05: read exact frozen T5 spec and kit/bench.py. Explicit ancestry checks
from `/` through `stdbench/phase2` found no AGENTS.md, LOCAL-AGENT.md, CONTRIBUTING,
pyproject.toml, setup.cfg, tox.ini, .flake8 or Makefile. No git checkout was found
in this artifact ancestry. Initial work wrote `analysis.py`, `math_review.py`
and this `analysis.md`. Follow-up added `grade_all.py`, `activation_report.py`
and the precollection MATH mapping. No subhelpers, Python execution, tests,
SSH, endpoint/GPU/queue operations or repository edits were performed. Source was
read back for static inspection under the assigned prove-it-works skill; runtime
verification remains pending and is explicitly delegated to the parent.

Parent-only verification on the already authorized remote CPU host, with its
2GiB MemoryMax/no-swap, nice19, ionice class3, timeout7200 and single-worker
constraints (do **not** run these on the dev machine):

```sh
python3 -m py_compile analysis.py math_review.py grade_all.py activation_report.py
python3 analysis.py selftest
python3 math_review.py selftest
python3 grade_all.py --selftest
python3 activation_report.py --selftest
```

Selftests execute only trusted synthetic data and never candidate code. They
cover exact DP versus exhaustive enumeration of small weights, all ties, known
Holm values, deterministic cluster sampling, percentile indices, capped correct
scoring, censor contributions, incomplete native evidence, missing/duplicate
identities, deterministic MATH blinding, independent merge, third adjudication,
unresolved arm-difference bounds, private-file mode, preseal hash mutation,
GSM negative/currency/fraction/ambiguity and capped-correct cases, and declared
but unused activation. Synthetic review fixtures
use two tasks/eight answers only through internal `_trusted_selftest=True`; no
production seal/merge/report CLI permits shrinking the registered cohort.
Synthetic temporary private keys are cleaned automatically. Parent should retain
remote receipts and remove generated bytecode after recording its check result.

## Full-cohort grading API

```sh
python3 grade_all.py --image-id sha256:PINNED_64_HEX_IMAGE_ID \
  --inputlist pilot-plain/responses.jsonl pilot-lane/responses.jsonl OTHER_BOOT_RESPONSES \
  --manifest frozen/analysis-plain.json --out all-graded.jsonl
```

`--inputlist` accepts one or more JSONL paths covering all turns and both arms.
Exactly368 scheduled replies are required, with no diagnostics, missing,
duplicate or unexpected identities. Per-row frozen task hashes are checked when
present. Validation finishes before Jail initialization. Output is exclusive
mode0600 canonical JSONL, preserving source costs/accounting and adding grades,
`grading` evidence, source/manifest hashes and `censored`. CLI requires a full
immutable image ID. Freeze this source, kit bench.py, Jail/worker/extraction
hashes before collection. No candidate code runs on the host.

GSM uses final delivered content only: discard a closed think prefix and reject
an unclosed one. Prefer the last balanced boxed answer, then the suffix after
last `####`, then a `Final answer:` line, else require the entire answer to be
numeric. Strict grammar accepts signed integers, decimals, valid thousands
grouping, signed integer fractions, and integer LaTeX frac/dfrac forms. A leading
currency dollar sign, before or after sign, and one terminal sentence period are
allowed. Units, malformed grouping, arbitrary prose numbers, alternative answers,
trailing guesses and rounding are rejected. Exact bench.rational Fraction
comparison publishes extracted/reference pairs per row. Length/deadline finishes
retain censoring and can pass if an actual delivered final is correct; non-ok
generation statuses remain incorrect. HumanEval requires Jail correct grade,
ok status and a completion receipt, and retains the full Jail evidence. Every
MATH row is manual_review regardless of auto proposals/status, so all80 enter
blind review; merge/analysis still force generation errors incorrect.

Replace the provisional80 MATH rows with reviewed merge output before analysis;
do not add the reviewed rows alongside provisional duplicates.

API: `validate_responses(rows,manifest) -> (task_map,index)`;
`grade_all(rows,manifest,image_id) -> canonical_rows`;
`gsm_grade(task,content) -> evidence`; `gsm_final(content) -> extracted_string`.
Trusted GSM selftest fixtures never initialize Jail.

## Dataset activation API

```sh
python3 activation_report.py --manifest frozen/analysis-plain.json \
  --inputlist BOOT_RESPONSE_PATHS --safety BOOT_SAFETY_PATHS --out activation.json
```

Requires the same full response cohort, retaining failed/censored outcomes.
Per suite/seed/arm it counts native calls, authenticated delivered results,
successful native-bound results, independent reference-verified successes,
replies using tools/fraction, executor identities, generation status counts and
missing/failed safety evidence. Duplicate/unexpected safety rows and
foreign/duplicate call/reference IDs fail. Missing safety remains missing
rather than a manufactured success. Controller safety rows omit arm, so CLI
joins only via a shared boot directory with a one-arm response file. API callers
must attach the correct arm themselves. Diagnostic safety rows are counted
separately and excluded; diagnostic response rows are refused.

Zero lane dataset calls produce **DECLARED BUT UNUSED**. Other activation is
labelled **NATIVE CALLS OBSERVED**, with no efficacy/benefit claim. Plain calls
are separately flagged. Nothing is filtered based on grades or outcomes.
API: `report(rows,safety_rows,manifest) -> report`.
