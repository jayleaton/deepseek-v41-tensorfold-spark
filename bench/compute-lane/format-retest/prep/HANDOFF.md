# Source-only FORMAT preparation handoff

No Python, tests, AST parsing, generation, inference, SSH, queue operations, builds,
<other-host>, engines or GPU operations were executed by this author. All files here
are new source. No repository or historical artifact was edited. Ancestry
AGENTS.md/LOCAL-AGENT.md/CONTRIBUTING and project check configurations were searched;
none apply to this nonGit artifact tree. No helpers were launched, as requested.

Read inputs: Implementer B's complete DIAGNOSIS.md, the complete frozen FORMAT spec,
compute-dl generate_tasks.py, independent reference_a.py/reference_b.py,
disjointness.py, exclusion_sources.py, original A2 request_adapter.py and its
client3/reconciliation/phase2 native-evidence dependencies. The inherited
exclusion_sources loader alone does NOT cover the new specification's complete
historical inventory, so generation consumes the parent's expanded pinned
inventory through the original canonical CollisionIndex instead.

Frozen spec SHA256:
`b47ede697eed3a38d244ad5ff7447690732bd12945ff4592554d45bc2d8527ec`.
Spec bytes are checked against the external supplied SHA AND this frozen SHA.
No analysis endpoint or analysis implementation is included.

## Parent-run checks

Only on <rank0-host>/<rank1-host> within the existing 2G/no-swap, nice19, ionice-c3,
single-worker CPU enclosure; G restored, QUIET/admission/no-boot rules apply.
Parent runs its queue turn-preflight at every new shell before the following.
These commands are instructions, NOT recorded executions or PASS claims:

```sh
python3 -B prep/checks.py
python3 -B prep/prep.py checks
python3 -B -c 'import ast,pathlib; [ast.parse(p.read_bytes(),filename=str(p)) for p in pathlib.Path("prep").glob("*.py")]'
```

The first two are equivalent entrypoints; run one. checks.py uses a handwritten
one-cell DP control and fake native decoder/response, never the factory. It
checks all four logical identities, same seeds/settings and L/P tools-only
differences, wrong/missing native hashes/IDs, wrong/missing prompt counts, native
generation errors, missing DONE, exact-wire drift, retained effective-OFF
errors, unknown input costs staying null, duplicate/missing/malformed schedule
keys and duplicate JSON keys. It also checks imported legacy names and sys.path
are untouched. Parent must inspect the actual generated distribution and wires,
run inherited real positive/negative grader fixtures and fake native response
controls, render every CPU prompt, qualify count evidence and seal runtime.

## File/API map

- common.py: strict JSON (duplicate keys/nonfinite rejected), pinned byte reads,
  exact suffixes and read_tasks(bytes). Requires precisely 60 records and all
  six families/slots0..9. Public task fields are the original eight fields;
  prompt remains OLD. The NEW prompt exists only in request templates.
- generation.py: verify_pins(...), generate(config, retain_collision) returns
  `(tasks, reference_a_by_id, reference_b_by_id, audit)`. `_candidate` is private;
  never call it directly. Draw sequence/distributions and mathematical prompt
  bodies are copied from compute-dl's factory, with only the specified label,
  date/master, slots and IDs changed. No random salt/date choice exists.
- legacy.py: Legacy().load(name) loads the UNMODIFIED original files at their
  original absolute paths. Each Legacy instance owns a private import graph,
  custom builtins import and private sys.path copy. No short legacy names are
  published to sys.modules and no existing module is replaced. Unknown ordinary
  imports retain standard Python behavior. Do not copy historical modules into
  prep or flatten their original paths when staging on nodes.
- request_adapter.py: Adapter(settings,tools,plain_families=...),
  build_request(task,seed,logical_arm), validate_template(task,template),
  validate_neutrality(adapter,task,seed), layout(tasks,adapter), validate_keys(...).
  Production stage fixes P policy to dp/graph/aggregation; generic adapter policy
  remains configurable. Logical arm is always L-old/L-new/P-old/P-new in
  `logical_arm` and `request_identity=[suite,id,seed,logical_arm]`. Original A2
  `arm` stays lane/plain and its legacy `key=[suite,id,seed]` is preserved.
- prep.py: one CLI with checks, generate, stage. It never calls a transport or
  collector. All output paths must be new; files are exclusively created.

## Before ANY generation

Parent must record external Python version and source hashes BEFORE invoking
generate. There is no pin-creation command that silently trusts current source.
`source_pins` must contain EVERY prep/*.py basename and SHA256. It therefore
covers factory, manifest validators, checks, loader, adapter and CLI together.
Also pin original reference_a, reference_b and disjointness sources. Record the
inventory hash and config hash externally before use. No config, inventory,
tasks or pin receipt was created here.

Generation config (JSON object, real absolute staged paths and trusted hashes):

```json
{
  "spec_path": "/absolute/T5-FORMAT-RETEST-SPEC.md",
  "spec_sha256": "b47ede697eed3a38d244ad5ff7447690732bd12945ff4592554d45bc2d8527ec",
  "python_version": "PARENT_PREPINNED_EXACT_VERSION",
  "source_pins": {"EVERY_PREP_PY_BASENAME": "EXTERNAL_SHA256"},
  "legacy_source_pins": {
    "reference_a": "EXTERNAL_SHA256",
    "reference_b": "EXTERNAL_SHA256",
    "disjointness": "EXTERNAL_SHA256"
  },
  "inventory_path": "/absolute/inventory.json",
  "inventory_sha256": "EXTERNAL_SHA256"
}
```

Inventory schema (adapter format, not a claim that T5 prescribes this JSON):

```json
{
  "complete": true,
  "required_source_ids": ["compute-dl-v1", "PARENT_ENUMERATED_OTHER_AUTHORITATIVE_SOURCES"],
  "sources": [
    {"id":"compute-dl-v1","path":"/absolute/original-v1.jsonl","sha256":"EXTERNAL_SHA256","count":120,"coverage":["compute-dl-v1"]}
  ],
  "resolutions": {}
}
```

The abbreviated example is deliberately incomplete and WILL FAIL. Enumerate all
sources available at freeze, with one unique source entry for every required ID.
The source entries must collectively cover these tags: W5,W6,W7,W8,W9,census,
guide,trace,pilot-eleven,AB3,guide-S3,GSM8K,MATH,HumanEval,stdbench,
declaration-control,compute-dl-v1. A source may cover several tags only with
actual authoritative evidence. Tags are coverage assertions, not discovery or
proof by themselves: parent owns the completeness attestation and trusted hash.
Every file is a nonempty original public JSON array or JSONL manifest, pinned by
SHA and exact count. Missing/unreadable files, duplicate JSON keys or source IDs,
empty manifests, source count mismatch or any missing coverage tag STOP.
The compute-dl-v1 source ID must contain exactly all120 expected IDs plus typed
inputs. Retain all original evidence; reused descendant IDs are deduplicated only
if prompt/family/inputs/spec evidence agrees exactly. Differing evidence blocks
for parent resolution rather than silently overwriting. No historical answers
or test code is executed. Paths are explicit; this code discovers no exclusions.

Unstructured historical records pass through the ORIGINAL CollisionIndex and
require its original prompt-bound resolutions (prompt_sha256, original_evidence
path/hash/location, reason, signature or explicit nonoverlap decision).
Unresolved overlaps stop before generation; no heuristic blanket nonoverlap.
Collision checks cover ID, typed inputs, normalized full prompt, and mathematical
operation/operands independent of wording. New accepted tasks enter the index.
Only a retained documented collision advances attempt; reference exceptions or
disagreement abort immediately. Exactly 1000 attempts are permitted per slot.

```sh
python3 -B prep/prep.py generate --config /absolute/generation-config.json --config-sha256 EXTERNAL_SHA256 --output /absolute/NEW-generation-directory
```

Output: tasks.jsonl, private reference-a.json, private reference-b.json,
generation-audit.json and collisions.jsonl. References use original algorithms
on identical public inputs. Collision log flush+fsync occurs before next attempt;
it remains if any later failure occurs. Existing output directories are rejected.
Preparation exceptions also write preparation-failure.json and are re-raised;
that receipt authorizes no regeneration.
Failure is never a replacement authorization. Audit records source/Python/spec
and inventory pins and task/reference file hashes. This is NOT the runtime seal.

## Stage EXISTING tasks, separately

Stage config fields: spec_path/spec_sha256; tasks_path/tasks_sha256;
generation_audit_path/generation_audit_sha256; references object mapping
`reference-a.json` and `reference-b.json` to absolute paths; settings_path/
settings_sha256; tools_path/tools_sha256; client_wait_s >=330;
legacy_source_pins for a2/client3/client/reconciliation/native_frames.
Optional plain_families must equal `["dp","graph","aggregation"]` for this spec.
Parent supplies the ORIGINAL pinned settings object and canonical tools array,
with insertion order preserved. SETTINGS in prep.py matches the original A2
runtime_stage.py settings exactly; no extra request settings are allowed. The
canonical declaration's authoritative SHA must come from the parent baseline,
not a newly invented declaration. The inherited strict grader is checked against
T5's exact SHA before stage returns. Independent reference files must have the
exact task ID set, agree and match the pinned generation audit.

```sh
python3 -B prep/prep.py stage --config /absolute/stage-config.json --config-sha256 EXTERNAL_SHA256 --output /absolute/NEW-stage.json
```

Output boots.L has240 requests, boots.P has120. Each boot independently starts
from sorted(taskID,seed) pairs and fresh Random(20261107). A balanced list begins
with half old→new (False), half new→old (True), then fresh Random(20261108)
shuffles it. Each pair stays consecutive. Stage retains explicit full templates,
wires, SHA256, identity, original adapter arm/key and execution policy per entry.
The stage serializer intentionally preserves object insertion order; NEVER
reserialize stage data with sort_keys=True, as A2 exact wire checks will reject
the reordered bodies/schemas. No CPU rendering is claimed: stage explicitly says
PENDING_PARENT_CPU_RENDER_AND_COUNT_QUALIFICATION and NOT_SEALED.

## Observation boundary for the parent collector

For each scheduled request, CPU sidecar must contain original A2 `key`, the
four-part `request_identity`, `wire_sha256`, and the corresponding original
`plain_ids` or `declared_lane_ids` prompt array. Old/new cannot share a sidecar
without an exact identity and wire match. Parent renders these arrays and seals
them with request arrays before admitting any collection.

```python
retained = adapter.observe(raw_observation, old_public_task, template, cpu_sidecar, decoder)
persist_and_fsync(retained)        # parent implementation; REQUIRED first
require_accepted(retained)         # reject/stop only after raw+diagnostics survive
```

observe calls original A2 classify_observation with its original lane/plain arm.
It retains a deep copy of raw observations and every available native diagnostic.
On A2 exceptions it recovers diagnostic stages using the original helpers, with
each error retained. Wrong/missing native evidence, prompt count/wire mismatch,
generation/framing errors and effective-OFF violations reject; costs are never
manufactured. Multi-round full input costs remain original A2 unknown/null unless
separately measured; do not make unknown input-accounting a zero-cost observation.
`accepted` is an evidence-eligibility result, NOT a format/strict correctness
score or completed-cohort PASS. Retain every scheduled missing key and attempted
reply; parent owns strict grading, safety, runtime/environment qualification,
budget/admission, boots/restoration, final seal and all analysis/pass decisions.
