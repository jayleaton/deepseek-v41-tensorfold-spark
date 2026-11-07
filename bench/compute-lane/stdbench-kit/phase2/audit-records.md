# Offline retained-record audit

`audit_records.py` reads only the explicit freeze, tokenizer and the fourteen
boot directories under `--data-root`. It writes one **new JSON file** outside
the data and freeze roots. It sends no requests, imports no controller or jail,
executes no candidate code and never reruns references. It does not read MATH
review labels, sealing keys, grades, outcomes or scores. No graders are changed.

The parent must run all verification and the actual audit on Spark, in the
authorized 2 GiB/no-swap scope with nice19, ionice class3, timeout7200 and one
worker (`-j1` for tools that accept it). No dev execution is authorized.
The standalone ancestry standards search found no AGENTS/assistant/CONTRIBUTING
or lint/build configuration; `../LEDGER.md` supplies the checks: py_compile,
offline source review and synthetic selftest. This author performed source
work only; neither compilation, tests nor actual data inspection was run.

Within that parent-managed scope, the CLI shape is:

```text
python3 -B /path/to/audit_records.py \
  --data-root <rank1-work>/stdbench/phase2/post-data-v1 \
  --freeze <rank1-work>/stdbench/phase2/frozen-v3.14 \
  --tokenizer /path/to/pinned/tokenizer.json \
  --tokenizer-sha <externally-pinned-sha256> \
  --out /path/outside-input-roots/new-audit.json

python3 -B /path/to/audit_records.py --selftest --out /path/new-synthetic-receipt.json
```

`--out` is a filename whose parent already exists; it is opened exclusively.
Exit 0 means qualified audit, or passed synthetic helper tests in selftest mode;
selftest always reports `qualified: false`. Exit 1 means unqualified or fatal
input/schema failure. Stdout contains only tiny verdict/count fields. JSON
contains identifiers, hashes, coverage, costs and gap descriptions; it never
exports response text, source, outputs, native token IDs or keys. Fatal errors
withhold exception payloads because those could include model content.

The fixed layout is:

| Boot | Relative data directory |
| --- | --- |
| pilot-plain | stdbench-20261105-v1-pilot/pilot-plain |
| pilot-lane | stdbench-20261105-v2-pilot/pilot-lane |
| t01 through t06, plain/lane | stdbench-20261105-v2-tNN/tNN-plain or tNN-lane |

Each needs `responses.jsonl`, `diagnostics.jsonl`, `safety.jsonl`, `rank0.log`.
The freeze's manifest must bind spec SHA
`9c7327007cfd15825a608f2f6b06f2b39e33b88548c5bb7deb1b209b7395d240`.
Its internal inventory, consumed frozen-file hashes, boot hashes and frozen
body/task hashes are checked. The parent must compare the exported freeze
manifest SHA against its independently retained receipt: a self-consistent
inventory alone cannot establish external authenticity. All consumed input
hashes are exported and rechecked for changes during the audit. Environment
proof belongs to the parent and requires no extra files here.

The auditor requires the full 368 unique dataset identities (48 GSM8K, 20
MATH-500, 24 HumanEval tasks, both seeds and arms), 28 diagnostic identities,
and exactly matching safety identities. It checks each row's boot, arm, seed,
task SHA and hash of the actual request against the frozen boot template.
Dataset and diagnostics stay separate. Missing files, duplicates, errors,
incomplete native evidence and safety failures block overall qualification.

API assumptions are the existing frozen-v3.14 source contracts:

- `manifest.json` has `files_sha256`, `boots` and `code_sources`; boot files have
  `boot`, `arm`, `requests`, `diagnostics`, each template carrying `task`,
  `task_sha256`, `body_template`, `body_sha256` and its row identity.
- Frozen `client.py` supplies unchanged `canonical`, `sha_ids`, `accounting`
  and `NativeDecoder`; the tokenizer is SHA-checked and loaded once. The
  auditor loads those exact definitions without importing transport code.
- Frozen `native_frames.py` supplies exact `authenticate` and its constants
  and helpers. The auditor replaces self-reported accounting with recomputed
  `client.accounting` before calling it. Native source/output frame binding,
  optional-feature rejection, inserted/prefix reconciliation and executor
  checks are therefore the frozen implementation, not a substring surrogate.
- Frozen `controller.reference_matches` is loaded as its exact AST definition
  with only its `base64` dependency, without importing controller side effects.
  Each delivered call needs exactly one retained `safety.references` result
  with matching `call_id`; duplicate call/reference IDs are rejected. Matching
  is recomputed, and its Jail `payload_sha256` must bind exactly
  `{"mode":"reference","code":<exact call source>}`. Worker SHA must match
  frozen `jail_worker.py`, image ID must be one of the two pinned rank images,
  and revision must match the frozen engine. Retained binding must equal the
  newly authenticated binding. No Jail object is created.

Both diagnostic responses must accept `enable_thinking=true` and
`reasoning_effort=high`. Plain permits no calls in any row. Each lane boot
needs at least one successful, delivered, native-bound, independently matched
diagnostic result equal to its frozen 323 or 55 reference. Execution errors
are retained gaps; they cannot satisfy activation. Per-suite/seed dataset
activation reports calls, successful deliveries, executor and usage coverage;
complete zero-call lane evidence is labelled `DECLARED BUT UNUSED`.

HTTP parsing deliberately accepts only anchored lines of the form:

```text
[tensorfold] 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" 200 -
```

The IP may be any literal loopback IPv4/IPv6 address. All recognized access
lines are checked for foreign/nonloopback IPs. Every boot needs exactly its
frozen dataset size plus two successful completion POSTs, with no other POST
paths/statuses or unrecognized POST lines. A missing/different access format
is unqualified with a parent-inspection reason; counts are never inferred
from response rows. Parent must inspect the actual format if this gate fails.

Dataset and diagnostic costs separately expose native-ID, completion and
inserted totals as **observed sums with row coverage**. Missing evidence
stays null; a zero inserted count is accepted only through the frozen engine's
zero-call omission contract and complete authentication. Known partial counts
from failed records are lower bounds, not successful/full-cohort evidence.
Optional descriptive fields use only actual numeric metadata: client
`elapsed_seconds`, usage `prompt_tokens`, completion details
`reasoning_tokens`/`content_tokens`, and tensorfold
`server_seconds`/`tool_seconds`. Other spellings are not guessed; absent
fields remain null. No reasoning/content split is derived by subtraction.

Synthetic selftest covers valid/missing/foreign/unexpected/duplicate HTTP,
body drift, missing retained references and missing-cost null handling. It
is helper coverage, not an end-to-end campaign or real-tokenizer receipt.
The parent must compile, run these fixtures, review the actual exported
verdicts/gaps and inspect real log/metadata formats before claiming verified
operation. Overall qualification requires every boot and row gate to pass.
