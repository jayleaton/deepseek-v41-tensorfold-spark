# Phase-2 freeze helper

`freeze.py` is an offline stdlib-only artifact writer. It does not read responses,
interpret T5 acceptance criteria, adjust sizes, execute a driver/grader, fetch
datasets, contact endpoints, use SSH, or manipulate queues/locks/services.
No Python execution or tests were performed on the dev machine. No repository
was edited. Ancestry standards search found no AGENTS.md, LOCAL-AGENT.md,
CONTRIBUTING, or Python build/lint configuration; see `../LEDGER.md`.
Parent verification is scoped <rank1-host> `py_compile` and the offline dry run below.

The operator must read the exact T5 specification and supply its externally
obtained SHA256, three explicit counts, and locked normalized input hashes.
Counts are fixed at **48/20/24**, each with both seeds in both arms (368
dataset replies). Other counts fail; pilot results cannot shrink the cohort.
Counts must still be supplied explicitly. Inputs are uncompressed normalized UTF-8
JSONL, one suite per file, with kit task fields and source_sha256/source_revision
matching the supplied dataset lock. The helper checks provenance metadata;
the upstream kit normalization must have verified original dataset bytes.
It cannot establish an ID's original dataset position from normalized data alone.

Use a hash-pinned configuration to avoid a long command line. Paths in config
are interpreted relative to the invocation directory; use absolute paths for
reproducibility. Example structure (replace all placeholders after reading T5):

```json
{
  "spec": "/absolute/T5-spec.md",
  "spec_sha": "EXTERNALLY_SUPPLIED_64_HEX_SHA256",
  "kit": "/absolute/stdbench/kit",
  "lock": "/absolute/stdbench/kit/datasets.lock.json",
  "lock_sha": "LOCK_64_HEX_SHA256",
  "inputs": {
    "gsm8k": {"path": "/absolute/gsm8k.jsonl", "sha256": "INPUT_SHA256"},
    "math500": {"path": "/absolute/math500.jsonl", "sha256": "INPUT_SHA256"},
    "humaneval": {"path": "/absolute/humaneval.jsonl", "sha256": "INPUT_SHA256"}
  },
  "counts": {"gsm8k": 48, "math500": 20, "humaneval": 24},
  "settings": {
    "model": "DeepSeek-V4.1-Flash-TF",
    "reasoning_effort": "high", "temperature": 1.0,
    "top_p": 0.95, "top_k": 20, "max_tokens": 8192,
    "deadline_s": 270, "request_timeout": 360,
    "return_token_ids": true, "stream": true,
    "stream_options": {"include_usage": true}
  },
  "code_sha_file": {"path": "/absolute/code-hashes.json", "sha256": "MAPPING_SHA256"}
}
```

```sh
python3 freeze.py --config /absolute/operator.json --config-sha CONFIG_SHA256 --out /absolute/new-freeze
```

Config and CLI settings cannot be mixed. Equivalent direct flags are `--spec`,
`--spec-sha`, `--kit`, `--lock`, `--lock-sha`, and repeated `--input suite=path`,
`--input-sha suite=sha256`, `--count suite=n`; `--settings` accepts a JSON object
with the documented settings keys. The frozen T5 protocol fixes all settings
above except client request_timeout, which must be an integer >=330 seconds
(default360). Conflicting settings fail. Thinking stays enabled; plain has no tools, lane has only
`[{"type":"code_interpreter"}]`. Bodies come from kit `request_payload`, then
the explicit T5 settings are applied. Every dataset and diagnostic body has
return_token_ids=true, stream=true and stream_options.include_usage=true.
The parent must use its stream/native-ID-aware phase2 client; the old kit
transport/parser alone does not establish native-ID/all-round accounting.
No guide, external tool loop, or response-driven adjustment exists.

Before final freeze, include phase2 client, grader, analysis, and any reference
or test artifacts in the code hash mapping. Optional direct flags are
`--code-sha-file /absolute/code-hashes.json --code-sha-file-sha MAPPING_SHA256`.
Mapping JSON is `{"/absolute/client.py":"64_HEX_SHA256", ...}`. Both the
mapping file and every listed code file must match externally supplied hashes.
Config may instead contain `"code_sha_files": {"/absolute/client.py":"SHA256", ...}`;
that direct mapping is pinned by the config SHA. Do not use both forms.
Verified bytes are copied to sources/code and bound in manifest.code_sources
and files_sha256. Missing/incorrect hashes fail before creating output.
The mapping is optional for preparation; parent must ensure final collection
includes every required client/grader/analysis/reference source before GO.
This helper neither grants GO nor treats the local spec hash as external approval.

Output must not exist, including an empty directory or dangling symlink. Input
hash/schema errors create no output. A later filesystem failure can leave a
partial directory; treat a missing/invalid inventory as incomplete, preserve
evidence, and choose a new output directory. The helper never resumes or deletes.

Selection is the kit's SHA256(`stdbench-v1:` + original stable ID), then ID,
ascending; frozen full task objects retain IDs unchanged. Blocks are contiguous
halves of that order, first block receiving the extra task for odd counts.
Both seeds 101 and 102 belong to every selected task in the final denominator.
The main six-turn order is fixed:

1. GSM block 1: plain, lane
2. MATH block 1: lane, plain
3. HumanEval block 1: plain, lane
4. GSM block 2: lane, plain
5. MATH block 2: plain, lane
6. HumanEval block 2: lane, plain

Pilot uses the first four selected tasks per suite, seed 101, in two aggregate
boots: pilot-plain then pilot-lane, each containing all three suites. Those pairs count in
final results and are excluded from the main schedule in **both** arms. This
does not change counts or caps. All seed-102 pairs remain scheduled. Actual main launch selection must follow the read T5 spec
and frozen manifest; this helper makes no GO decision or scheduling mutation.

The parent controller API is `manifest.json` with `boots` entries containing
`{boot, arm, file, sha256}`. Its `turns` entries are `{name, boots}` for `pilot`,
then `t01` through `t06`, with boot names listed in execution order.
`boots/<boot>.json` is `{boot, arm, diagnostics, requests}`; each dataset request has
`{suite, id, seed, task, task_sha256, body_template, body_sha256}`. Task is the
full normalized object, body_template is the actual OpenAI request body, and
both object hashes use kit canonical serialization. Parent may send precisely
that canonical body through its qualified stream/native-ID-aware client.
No request ordering or body mutation by this helper depends on responses.
Before **any dataset request in every newly configured boot**, run the two
diagnostics in list order, seed101, using the exact prompts:

1. `Use the Python tool to print 17*19, then report the result.`
2. `Use the Python tool to print sum(range(11)), then report the result.`

Diagnostic rows have the same task/body/hash fields, reference323 or55, and
included_in_cohort=false. They use identical full settings and user messages
across arms; only lane declares tools. No extra instruction is appended.
There are 28 diagnostic requests across 14 boots, separate from all dataset
grids/denominators. The parent must enforce diagnostic activation/safety gates
before dataset requests, preserve failures, and report diagnostic costs separately.

Boot files are authoritative launch inputs. For each full block, sort
`(taskID, seed)` tuples, then apply a fresh Python Random(20261104).shuffle.
Paired boots use the same order. Frozen `block*-ordered-pairs.jsonl` files
retain the full shuffled block grids with hashes in manifest.block_orders.
Main boot requests filter pilot pairs from that full order without reshuffling.
Pilot boot requests use the same order filtered to the first four selected
tasks/seed101 per suite; the three suites retain GSM/MATH/HumanEval order.
Each boot's raw-file SHA256 is in manifest.

`manifest.json` also has explicit per-run task/seed grids, canonical pair hashes,
arm order, and `main_run_groups[].arm_runs`. These per-seed files are
compatibility grids for deduplication/inspection, not the registered mixed-seed
execution order: do not execute the flat runs with separate seed shuffles.
Use boot.requests for exact collection order. Selected/block task JSONL
preserves hash-selection order, distinct from shuffled pair-order JSONL.
Do not feed a whole block with both
seeds after pilot; that would duplicate pilot pairs. Do not replay pilot when
resuming main. No automatic completed-response discovery occurs.

`expected-pairs.jsonl` is the complete pilot-plus-main arm grid, checked for
exact coverage and duplicates. `requests.jsonl` has one full body per arm pair;
`body_sha256` is SHA256 of kit canonical UTF-8 JSON (sorted keys, compact
separators, Unicode unescaped, no trailing newline), exactly the kit wire
serialization. `analysis-plain.json` and `analysis-lane.json` carry matching full
task/seed manifests for final aggregation, **not launch instructions** and not
drop-in kit run manifests. Merge pilot and main records by suite/ID/seed/arm,
reject duplicates/missing pairs, then feed matching complete task sets to
analysis. `COUNTS.md` states all final/pilot/main denominators.

The freeze copies the exact spec, config when used, normalized inputs, kit
bench source, dataset lock, helper source, README, plan and supplied code files. It records locked
original dataset revisions/hashes without pretending to re-fetch them.
Manifest `files_sha256` pins all content files except manifest itself and the
two inventory files, listed explicitly in `files_sha256_excludes` (self hashes
cannot be embedded). `file-hashes.json` pins every other output file including
the manifest; `file-hashes.sha256` pins that
inventory. Verify both and keep the inventory root hash externally. This is
content-addressed evidence, not filesystem write protection. The inventory
and its sidecar intentionally cannot include their own hashes. No timestamps
or output paths enter generated content.

Authorized parent preflight only, on scoped <rank1-host> (not <dev-host>):

```sh
python3 -m py_compile freeze.py
python3 freeze.py --dry-run --kit /absolute/stdbench/kit --lock-sha LOCK_SHA256 --out /absolute/new-fake-freeze
```

Dry run fabricates count+2 tasks per suite in a temporary directory, selects
48/20/24, and invokes the normal freeze twice. It asserts identical inventories,
exact pilot/main coverage, 96/40/48 final pairs per suite/arm (four pilot pairs
included in each), 368 body hashes, diagnostics/stream/native-ID fields,
registered block shuffle, bad spec hash rejection, existing output rejection,
and file inventory integrity. It also exercises copied code hashing with a
fake artifact. It writes the fake freeze and
an adjacent `.dry-run-receipt.json`; temporary fixtures and repeat output are
removed automatically. Receipt is deliberately outside the freeze inventory.
Keep or remove the fake output/receipt under the parent's scratch policy.
No endpoint, driver, grader, replay, or subprocess is executed by dry run.

## Completed collection

The frozen spec is SHA256 `9c7327007cfd15825a608f2f6b06f2b39e33b88548c5bb7deb1b209b7395d240`; completed campaign is `campaign-v2.json`, source/request freeze `frozen-v3/`. The retained initial plain pilot comes from campaign-v1/frozen-v2. All 368 scheduled replies and 28 diagnostics are complete; all six main turn verdicts qualified and G was restored. Existing boot/output paths deliberately reject repeats.

Current results and evidence are in `post-v1/`, with canonical Markdown rendering in `report-v2/`. `analysis.py report --manifest frozen-v3/analysis-plain.json --graded post-v1/nonmath-graded.jsonl post-v1/math-reviewed.jsonl --out NEW.json` reproduces statistics. Run this only on a Spark through the existing CPU guard, using a new output path. See `analysis.md` for the frozen grader, blinded review and activation APIs. `audit_records.py` and `audit_envs.py` audit retained evidence without inference. `export_check.py` checks kit syntax and report integrity; its recorded invocation has finished. `final_verify.py` records direct G process liveness and cleanup state.

All three dataset lanes were **DECLARED BUT UNUSED**. Read the top-level `RESULTS.md` and `NEEDS-T5.md` for the interpretation, strict GSM extraction limitation, HumanEval censoring sensitivity, MATH agent-review method, published-reference caveats and recovered pilot cleanup incident.
