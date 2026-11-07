# Source-only handoff — NEEDS-T5

No Python, tests, analysis, SSH, <other-host>, GPU, queue, endpoint, posting, engine or git
operations were run for this implementation. No child helpers or reviewer were
used. The parent owns execution on Sparks <rank0-host>/<rank1-host> under its existing
2 GiB/no-swap, nice19, ionice-c3, single-worker, QUIET/admission/no-boot guard.
This handoff neither implements nor changes that guard or the parent runtime/prep.

## Exact API

Python 3 standard library only. Import `reporter.py` by path or place this directory
on the import path. No work occurs on import.

- `load_grader(path) -> module`: checks the frozen SHA256, loads the unchanged
  legacy grader with a unique module name, without inserting into `sys.modules`.
- **Report function: `analyze(rows, tasks, answers, audit, grader) -> dict`.** Raw observation list,
  60 public task dicts, private answer dict, separate independent parent audit,
  and module returned by `load_grader`. Executes one shared 20,000-draw bootstrap.
  It does not perform file-hash binding; an embedding caller must bind the inputs
  independently before calling. The CLI enforces binding itself.
- `write_report(report, output_directory)`: writes `RESULTS.md`, `report.json`,
  `per-observation.csv`. JSON contains summaries, pair IDs, missing keys, costs,
  audit, bootstrap method/intervals and the full 360-key ledger. CSV has one row
  per scheduled key, including unknown costs as empty fields. References and
  selected answer values are not emitted by the reporter. Parent audit must not
  contain private answers. JSON is the authoritative nested call/cost coverage.
- `classify(row, task, answer, grader)`: unchanged strict grading plus independent
  selected-span classification, invoking the **same** `grader.extract` and
  `grader.valid_value`. No answer-driven fallback, repair or regrading sensitivity.
- `expected_grid(tasks)`, `normalize(rows, tasks, answers, grader)`,
  `observation_ledger(tasks, indexed)` and `acceptance(...)` are exposed for
  handwritten controls. No separate model execution is performed.

## Files supplied by parent

`tasks.public.jsonl`: exactly 60 tasks, one JSON object per line, suite
`compute_dl`, family in integer/modular/division/dp/graph/aggregation, IDs exactly
`compute-format-v1-{family}-{001..010}`. Existing input/prompt fields pass through;
there must be no answer/reference fields. It is the same instance for both wordings.

`answers.json`: private JSON object, keys exactly the 60 task IDs, each value the
legacy grader's typed integer, division pair, or A–F object. One answer per task,
shared by seeds and variants; no generation/recomputation here.

`rows.jsonl`: **raw**, not pregraded, normalized retained observations. Required:

- `taskid` or `id` (if both, they must agree), `family`, integer `seed` 101/102,
  `study_arm` exactly `L-old`, `L-new`, `P-old`, `P-new`, explicit `status`.
- Physical `arm`, if supplied, is `lane`/`L` or `plain`/`P` and must match the
  study arm's leading letter. It never substitutes for `study_arm`.
- Legacy grade inputs: `content`, `answer_eligible`, `external_tool_calls`,
  `finish_reason` retain their meanings and bytes. An empty/error/capped reply is
  still a supplied observation. Do not invent a row for an unattempted key.
  Diagnostics, exclusions, duplicates, unexpected seeds/IDs/arms and P arithmetic
  are rejected. The strict `grade` and `grader` fields of a raw row are ignored.
- `native_accounting: {complete: true, returned_tokens: nonnegative_integer}`.
- `initial_prompt_reconciliation`: `qualified: true` (or
  `qualified_under_proposed_count_contract: true`) and `initial_prompt_tokens`
  (or `server_initial_prompt_tokens`). This plus returned is explicitly SUBTOTAL.
- `elapsed_seconds`: measured full client wall, finite/nonnegative.
- Optional full multi-round `input_accounting: {qualified: true, complete: true,
  provenance: <nonempty evidence locator>, processed_input_tokens: <integer>}`.
  Missing/unqualified accounting remains unknown and withholds full input ratios.
- `call_accounting`: `complete: true` or `coverage[field]: true` plus integer
  values for `attempted`, `blocked_dispatches`, `successful_executions`,
  `failed_executions`, `delivered_authenticated_results`, `reply_used_result`.
  Unknown call counts remain unknown; zero is valid only when explicitly covered.
- Delivered L results: `tensorfold.compute.calls` with `inserted: true`, string
  `id`, exact `code`, explicit native `status` and exact delivered `output` text. The delivered call census
  must agree with qualified `delivered_authenticated_results`, including zero.

## Independent `audit.json` contract

This is supplied by the parent **after independent real-collection qualification**,
never copied from a row's `safety` boolean or inferred from the model's answer.
The reporter validates this receipt contract, not the external truth of an audit.
Do not label synthetic fixtures `qualified-real-collection` for a report run.

Top level:

- `kind: "qualified-real-collection"`.
- `spec_sha256`: `b47ede697eed3a38d244ad5ff7447690732bd12945ff4592554d45bc2d8527ec`.
- `grader_sha256`: `1a2249a1b5e0b06f03a4300995d0c320e5a21630d261b3791501f7c83740fb9f`.
- `input_sha256`: SHA256 of the exact files, keyed `rows`, `tasks`, `answers`.
  The CLI rejects mismatches before writing any report.
- `scheduled_keys`: the exact sorted `expected_grid(tasks)` rendered as objects
  `{id, seed, study_arm}`, all 360, even if collection is incomplete.
- `integrity`: `qualified: true`, nonempty `provenance`, and each of
  `sealed_inputs`, `disjointness`, `prompt_neutrality`, `configuration`,
  `dispatch_boundary`, `source_image_rank_environment`, `post_run_bindings`
  a dict `{qualified: true, provenance: <nonempty evidence locator>}`. These
  attest the corresponding frozen spec gates, including sealed source/task/
  reference/wire hashes, disjointness inventory, exact prompts/boot arrays,
  source/image/rank qualifications and post-run native/tool-source bindings.
  Additional cache/boot/held-time/restoration disclosures are retained in JSON.
- `r7`: `qualified: true`, nonempty honest `provenance`,
  `scope: "owned_runner_cells"`, `all_optional_runners_off: true`,
  `oracle_pinned: true`, `oracle_sha256` identifying the parent's actual pinned
  oracle, and explicit integer zeros for `owned_cell_count`,
  `checked_owned_cell_count`, `wrong_values`, `wrong_bytes`, `not_in_text`,
  `cuts`, `unknowns`. All optional runners OFF means zero owned runner cells.
  Parent runs the pinned oracle and records this zero-owned coverage honestly.
  **r7 cannot see or validate native compute code.** Its zero counters are not
  evidence of native code safety; that verification is entirely separate below.
- `native_replay`: `qualified: true`, nonempty `provenance`,
  `independently_isolated_cpython: true`, `presentation_token_limit: 512`,
  `clipping_policy_qualified: true`, `policy_source_sha256`, `tokenizer_sha256`
  identifying the qualified pinned production policy and tokenizer, explicit
  integer zeros for `wrong_values`, `wrong_bytes`, `not_in_text`,
  `unresolved_cuts`, `unknowns`, integer `policy_clipped_results` equal to the
  verified clipped receipt count, and `receipts` list. Do not translate all known
  policy clips into unresolved cuts. Legacy/raw `cuts` records may be retained;
  their total is not required to be zero. Preserve wrong/cut/unknown evidence.
- Exactly one replay receipt per delivered L result, no extras/duplicates:
  `id`, `seed`, `study_arm`, `call_id`, `qualified: true`, nonempty `provenance`,
  `presentation_verified: true`, `full_reference_verified: true`,
  `policy_transform_verified: true`, `frame_binding_verified: true`,
  `source_binding_verified: true`, `native_prefix_exact: true`,
  `matches_bytes: true`, `in_text: true`, `native_frame_sha256`, `native_status`
  and `reference_status` both matching the call's native `status`.
  Hashes/lengths: `source_sha256` of UTF-8 code, `landed_output_sha256` of UTF-8
  landed output, `expected_presentation_sha256` equal to the landed hash,
  `full_reference_sha256`, `full_reference_bytes`, `landed_output_bytes`,
  `full_reference_body_tokens`, `retained_prefix_bytes`, `retained_prefix_tokens`,
  and `marker_bytes` (a JSON string preserving the exact UTF-8 marker bytes).
  Every SHA256 field is 64 lowercase hex characters. Full reference hashes and
  lengths refer to the complete reference body after pinned native status/body/
  normalization and preceding byte/time limits, before token presentation.
- **Known policy clip:** `clipped: true`, `complete_value: false`, reference body
  tokens >512, retained tokens ≤512, exact marker
  `\n[output truncated at 512 tokens]` after a nonempty retained prefix, or
  `[output truncated at 512 tokens]` with no prefix. Landed length must equal
  prefix bytes + marker bytes, with that exact marker suffix. Parent must have
  independently reproduced the pinned policy, including UTF-8 seam backoff,
  all preceding native limits and exact source/status/frame/native-ID binding.
  A marker or generic matching prefix alone is insufficient evidence.
- **Unclipped:** `clipped: false`, `complete_value: true`, reference body tokens
  ≤512, retained tokens equal reference tokens, retained/full-reference/landed
  byte lengths equal, full-reference hash equal landed hash, and `marker_bytes`
  empty (no policy-added marker). A naturally printed marker below the cap is
  ordinary complete output, not evidence of clipping.

The parent must independently run isolated Jail/CPython with the exact source/
state, stdout/stderr/exit/status, native frames and qualified native codec/policy
before emitting these receipts. The reporter validates receipt consistency and
hash joins; it neither executes native code nor establishes the parent's external
audit truth. The real A1 511/512/513-token, UTF-8 seam, status, frame and landed-byte
mutation controls remain parent-owned; handwritten metadata controls do not
replace those real policy controls.

A proven policy clip is **presentation-verified clipped**, never a verified
complete value. `full_reference_verified: true` concerns the independent full
reference, not invisible native tail bytes. A reference change entirely beyond
the retained prefix can be observationally indistinguishable; do not claim it
was detected. Unresolved cuts, unsupported policy branches, unknowns, wrong landed
bytes or missing source/status/frame attribution block qualification. Native
verification never rescues the separately graded final answer or format gate.
JSON/RESULTS expose verified policy clip counts, complete value counts, unresolved
counts, r7 zero-owned scope and each receipt's full-reference/landed hashes,
lengths, prefix and marker. The report API signature above is unchanged.

Safety/provenance failure gives `status: BLOCKED` even if collection is incomplete;
`collection_status: INCOMPLETE` and all missing keys remain separately visible.
Qualified but missing observations give `status: INCOMPLETE`. With all 360 and
qualified evidence, apply the frozen acceptance rule; no aggregate P substitute.

## Parent commands, not executed here

Run only inside the parent's already-admitted guarded node session, with G
restored for offline CPU. Use one worker; retain check stdout separately from
production results. Set `A` to this transferred analysis directory and `ROOT` to
the transferred `stdbench` directory. Run inherited positive/negative grader
fixtures as required by the parent's preparation seal as well; this slice does
not replace them.

AST source check (no bytecode files):

```sh
python3 -c 'import ast,pathlib,sys; [ast.parse(p.read_text(), filename=str(p)) for p in pathlib.Path(sys.argv[1]).glob("*.py")]' "$A"
```

Handwritten controls (one 20k stream, no result files; bytecode disabled):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 "$A/handwritten_checks.py" --grader "$ROOT/compute-dl/grade.py"
```

Actual real-input report, after parent CPU checks and real qualification:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 "$A/reporter.py" \
  --rows /parent/supplied/rows.jsonl \
  --tasks /parent/supplied/tasks.public.jsonl \
  --answers /parent/private/answers.json \
  --audit /parent/supplied/audit.json \
  --grader "$ROOT/compute-dl/grade.py" \
  --spec /parent/supplied/T5-FORMAT-RETEST-SPEC.md \
  --out /parent/designated/real-report
```

Replace the explicit parent paths with actual supplied files. Keep private answer
files private. No synthetic production report is needed to check this source.
After execution inspect the real readable report, JSON and CSV; check all 360
ledger keys and audit qualifiers before claiming results. No correctness claim
is made here without that execution.

## Statistical contract

One `Random(20261107)` stream, alphabetically sorted families and task IDs, 10
`randrange(10)` replacement task draws per family per replicate, 20,000 replicates.
Both seeds and wordings remain together. The same drawn task lists drive primary
L-new minus L-old format rates, all six primary family strata, secondary P wording
strict-rate difference, secondary L-new minus P-new strict-rate difference on the
three P families, and paired cost ratios of sums. CI95 uses sorted indices 500
and 19499. Any unknown cost in a contrast withholds its interval and full total;
undefined zero-denominator bootstrap ratios also withhold their interval. Cost
known subtotals retain explicit coverage. Missing paired observations withhold
that contrast's point estimate/CI; missing rows still count against the format
gate and remain in all arm/family denominators.

Handwritten fixtures cover correct/shape/ineligible/typed-wrong/invalid/duplicate
JSON, selected-span behavior, caps retained, physical arm validation, duplicate
keys, P paired loss despite unchanged aggregate, L strict decline, threshold
boundaries, missing ledger/costs, self-report safety rejection, module isolation,
known policy clip qualification without a complete-value claim, unresolved cut/
unknown rejection, source/status/frame/prefix/marker/hash mutation rejection,
natural markers below cap, honest undetectable tail differences, r7 zero-owned
scope, and one complete synthetic bootstrap. A synthetic report is always BLOCKED,
prints NEEDS-T5 and writes no production artifacts.
