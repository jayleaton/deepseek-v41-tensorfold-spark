# Compute-lane benchmark evidence

This directory holds the kits, frozen specifications, analysis code, results, per-problem tables and independent
adjudications behind the compute-lane claims of this recipe: the lane with the native Python compute tool enabled
(`L` / `lane`), compared with the same model with no tool (`P` / `plain`) and with the tool declared but execution
disabled (`D` / `declared_off`). Each study was registered in a frozen spec before collection, run on the two-Spark
TP2 serving setup, and then audited by an independent adjudicator (`T5`) that did not run the collection.

**Paths, host names and login names are placeholders.** Scripts were copied with only these values replaced (see
[REDACTIONS.md](REDACTIONS.md)). Commands below are the commands as they were run, with paths turned into
placeholders: `<rank0-work>` / `<rank1-work>` are the work directories on the rank-0 / rank-1 Spark (the studies lived
in `<rank0-work>/stdbench/`), `<rank0-host>` / `<rank1-host>` the two hosts, `<devwork>` the workspace where the
adjudicator worked. A tool that built a per-rank path from the rank number (`/home/<user>0{rank + 1}/work/...`) now
builds `<rank{rank}-work>/...`; point these at your own directories before rerunning anything.

## Index files

| file | what it is |
|---|---|
| `MANIFEST.sha256` | sha256 of every file here (`sha256sum -c MANIFEST.sha256` from this directory) |
| `REDACTIONS.md` | every file whose bytes differ from the original: original sha256, staged sha256, kinds of values replaced; the frozen-spec hash table; whitelisted grep false positives |
| `RAW-RECEIPTS.sha256` | sha256, size and original path of everything behind these studies that is not copied here: raw request/response logs, per-boot snapshots, run state, step and boot logs, telemetry, caches, reports over 1 MB, the engine-source snapshot used by the dispatch proof, superseded or per-turn kit copies, and byte-identical duplicates of included files |

Files not listed in `REDACTIONS.md` are byte-identical to the originals, so their hashes match the frozen manifests,
seals and adjudications. A redacted file keeps its original hash in `REDACTIONS.md`. Byte-identical copies that
appeared at several source paths are included once; the other paths are listed in `RAW-RECEIPTS.sha256` as
`duplicate-of:<path here>`.

## Layout

| directory | contents |
|---|---|
| `stdbench-kit/` | the standard-benchmark kit (`bench.py`, `grader_worker.py`, `datasets.lock.json`, pinned `sources/` task files), `phase2/`: the shared phase-2 modules the compute-lane studies import (client, grader `grade_all.py`, jail worker, rank operations, audit records, native-frame checks, CPU guard, analysis) and the phase-2 spec; `project/`: the project README, PLAN and LEDGER (standards, operating constraints, per-turn receipts) |
| `decl-control/` | declaration-control study: generators, controller/client/launcher (`*3.py`, `launch3.sh`, `turn3.sh`), grader, analysis, native stubs, frozen manifests, CPU gate verification, `RESULTS.md`, `finalized-a1-v1/per-request.csv`, `canonical-python-tools.json`, `spec/`, T5 adjudication and audit |
| `compute-dl/` | computation-demanding D/L/P study: task generators and references, graders, analysis, controller and request adapters, A1 and A2 continuation kits (`a1-*`, `a2-*`), frozen specs, results at each stage, `per-problem.csv`, T5 adjudication and audit |
| `format-retest/` | final-answer format re-test: dataset generation, prep, runtime kit, `settings-v1.json`, audit and r7 post-checks, `results-v1/` (`RESULTS.md`, `per-observation.csv`), final evidence, `spec/`, T5 adjudication and audit |
| `ab3/` | AB3 token/solve study: `AB3-FREEZE.json`, the frozen kit `kit2/` (scripts, boot configs, units, request templates, `guides.json`, per-task `census/`), per-turn generated drivers `turns/t0N/imr-measure.py`, `results/` (`AB3-RESULTS.md`, `ab3-final.json`, window reports, tool re-check), `safety/`, `rendered-tool-schema.json`, `ab3-unit-ids.txt`, spec and T5 adjudication |
| `owned-oracle-r7/` | the r7 owned-value oracle and its checkers (`oracle.py`, `recheck.py`, `gate.sh`, `selftest.sh`, window and W9 reporters, `detector.json`) with its synthetic self-test fixtures and the 2026-11-02 validation summaries |
| `configs/` | the measured configurations: arm L's effective container environment, the lane studies' request body, AB3 tool-full's 18 engine lines and request body ([below](#configs)) |

Within each study directory the source layout is kept (subdirectory names as on the host, with agent-name prefixes
removed). `summary/` holds the project-level result summaries for that study as they stood at each stage;
`t5-workers/` holds the adjudicator's CPU worker scripts that were kept beside the study rather than in its audit
folder.

## Studies

### Declaration control (`decl-control/`)

Whether declaring the Python tool, as opposed to executing it, explains the lane's token differences on standard suites (GSM8K,
MATH-500, HumanEval). Three arms per turn (`plain`, `declared_off`, `lane`), 552 fresh replies over three turns.

- Specs: parent `decl-control/spec/T5-DECL-CONTROL-SPEC.md`, original sha256
  `adb05167fff21469ee7251555cb243f1f109c4d8e3de237fdcaa59ada8eb55e6`; amendment A1
  `decl-control/spec/T5-DECL-CONTROL-SPEC-A1.md`, original sha256
  `e7308621c645aef4402924a1b47c1a52e05baabe00239f915011f8431bbd0e89`.
- Results: `decl-control/RESULTS.md`; the project-level summary `RESULTS-DECL-CONTROL-A1.md` is byte-identical to
  `compute-dl/PREVIOUS-DECL-CONTROL-SUMMARY.md` (kept there); per-request table
  `decl-control/finalized-a1-v1/per-request.csv`; analysis `finalized-a1-v1/analysis.json`; finalize receipt and
  math-review audit beside it.
- Adjudication: `decl-control/T5-DECL-CONTROL-ADJUDICATION.md` (2026-11-05, **AGREE under A1**: all three suites
  declared but unused; no detected incremental execution-enabled effect on this cohort). Bound to parent `adb05167…`
  plus A1 `e7308621…`, freeze manifest `785a8f5b9625427d50fc1e18b26bbdd40198ddd55ca18d3a635741edaab94d21`. Audit:
  `decl-control/t5-decl-control-adjudication/` (`audit/summary.json`, `audit/input-hashes.json`, support audit), worker
  scripts in `decl-control/t5-workers/`.
- Rerun (as run): the campaign is `campaign-a1-v1.json` (bound to CPU gate `cpu-final-a1-v2/verification.json`), one
  launch per turn from `<rank0-work>/stdbench/decl-control/`: `bash launch3.sh campaign-a1-v1.json t01` (then `t02`,
  `t03`). Postcollection preparation, grading and analysis run in the guarded CPU scope; the analysis CLI is
  `python3 analysis3.py --manifest <frozen-a1-v1/analysis-manifest.json> --inputlist <graded inputs...> --out <new path>`.
  See `RUN.md` and `FEASIBILITY.md`.

### Computation-demanding D/L/P (`compute-dl/`)

A generated 120-task, six-family cohort (integer, modular, division, DP, graph, aggregation) that needs exact
computation; arms D (declared, execution off), L (lane) and P (plain), two seeds. The original registration stopped
at 342/480 replies; amendment A1 continued D/L, amendment A2 added P. Final: 720/720 replies.

- Specs: `compute-dl/spec/T5-COMPUTE-DL-SPEC.md` original sha256
  `c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141`; `T5-COMPUTE-DL-SPEC-A1.md`
  `0dca1fa52838388ea1e7afbc3a4c5e9747775f64e871640df7a66496369e5453`; `T5-COMPUTE-DL-SPEC-A2.md`
  `8eaa9575592f519b504d08022d400602e4b552874c8424b02c59c34f4af174e2`.
- Results: final `compute-dl/a2-results-v1/RESULTS.md` and `a2-results-v1/per-problem.csv`; summary
  `compute-dl/summary/RESULTS.before-format-20261107.md` (L 187/240 vs P 120/240, +27.92 points, 95% CI [22.92, 32.92]);
  earlier stages `compute-dl/RESULTS.md` + `compute-dl/per-problem.csv` (original 342/480), `a1-results/RESULTS.md`,
  `report-v1/RESULTS.md`, `publication-v1/RESULTS.md`, `summary/RESULTS.before-A2-20261106.md`. The machine-readable
  `a2-results-v1/report.json` (7.5 MB), `analysis.json` (2.3 MB) and the graded JSONL are hash-listed in `RAW-RECEIPTS.sha256`.
- Adjudication: `compute-dl/T5-COMPUTE-DL-ADJUDICATION.md` (2026-11-06, **CONFIRM** the completed amended study and its
  local accuracy benefit; the efficiency condition does not pass; outcome-informed amended study, not the original
  confirmatory comparison). Bound to parent `c3e6da46…`, A1 `0dca1fa5…`, A2 `8eaa9575…`. Audit:
  `compute-dl/t5-compute-final-adjudication/` (`summary.json`, `residuals.json`, `support.json` and the worker scripts).
- Rerun (as run), from `<rank0-work>/stdbench/compute-dl/`: original turns `bash launch.sh <campaign> <turn>`;
  A1 `bash a1-continuation/launch.sh FINAL_CAMPAIGN A1-resume`, then `... A1-plain`; A2
  `bash a2-continuation/launch.sh FINAL_CAMPAIGN A2-plain`, each after the staged steps in the matching `HANDOFF.md`
  (`runtime_stage.py stage|materialize`, `render_prompts.py`, `cpu_check.py`, `qualify_rank.py`, `lifecycle-launch.sh
  success|failure`, `finalize_cpu.py`). Tasks: `generate_tasks.py`, references `reference_a.py` / `reference_b.py`,
  public task file `dataset-v1/tasks-public.jsonl`; see `TASKSET.md`.

### Format re-test (`format-retest/`)

Whether new final-answer wording removes the lane's format failures without losing plain solves: L-old/L-new and
P-old/P-new on a 60-task, six-template development cohort, 360 fresh replies.

- Spec: `format-retest/spec/T5-FORMAT-RETEST-SPEC.md`, original sha256
  `b47ede697eed3a38d244ad5ff7447690732bd12945ff4592554d45bc2d8527ec`. Settings: `format-retest/settings-v1.json`.
- Results: `format-retest/RESULTS.md` (L-new 120/120 with 0 format failures vs L-old 86/120 with 34; P-new 60/60 vs
  P-old 59/60, 0 paired P losses), `format-retest/results-v1/RESULTS.md`, `results-v1/per-observation.csv` (360
  rows), `analysis/RESULTS.md`, `summary/RESULTS.md`; independent raw audit `audit-v2/raw-audit.json`; final artifact
  inspection, step log, budget and G-restoration verification under `final-evidence/`; pinned r7 coverage
  `final-evidence/r7/`. `results-v1/report.json` (1.5 MB) is hash-listed.
- Adjudication: `format-retest/T5-FORMAT-RETEST-ADJUDICATION.md` (2026-11-07, **CONFIRM / COMPLETE / PASS**), bound to
  `b47ede69…`. Audit: `format-retest/t5-format-retest-adjudication/audit.json` and its worker script.
- Rerun (as run), from `<rank0-work>/stdbench/format-retest/`: `bash runtime/launch.sh FINAL_CAMPAIGN format` (one turn,
  cold L then cold P), then `audit_collection.py` / `audit_collection_post_v2.py`, `r7_audit.py` with the r7 oracle,
  `analysis/reporter.py`, `publish.py --campaign --audit --r7 --out`. See `HANDOFF-review.md` and `runtime/HANDOFF.md`.

### AB3 (`ab3/`)

Token-cost study on 60 units across four boot configurations (plain, lane, full, only; seeds 101–103; six turns, 720
receipts), registered before the first request.

- Spec: `ab3/T5-AB3-SPEC.md`, original sha256 `81b34c1a79a754efca01ebee85f47c74490a029f23e4e0533799205b60dd524e`.
  Freeze: `ab3/AB3-FREEZE.json`, original sha256 `93417d27c61d8c52069ec1b33cb83d79f26e6cd3fdc0b8d33127e784b6315e7a`; sealed
  analysis `ab3/kit2/ab3_analysis.py`, sha256 `26d7bec86ca8f860a26109bc66f50821a7fdbacf682c375572e6e2c03cf6acca`.
- Results: `ab3/results/AB3-RESULTS.md` (**DOES NOT MEET**), `ab3/results/ab3-final.json`, per-turn window reports,
  `toolcheck-verify.json/.md`, `ab3/safety/`, tool re-run outputs. No per-problem CSV exists for AB3; the per-pair
  table is `ab3/t5-ab3-adjudication/ab3-adjudicate-read4/audit/pairs.json`.
- Adjudication: `ab3/T5-AB3-ADJUDICATION.md` (2026-11-04, **DOES NOT MEET**: the primary token CI fails and a
  recorded-but-undelivered trace newline fails safety/provenance; solves and censoring pass), bound to `81b34c1a…`.
  Audit: `ab3/t5-ab3-adjudication/` (summary, pairs, tool bindings, raw-hash list, metadata, script).
- Rerun (as run): the six turns ran the generated `turns/t0N/imr-measure.py` drivers built by
  `kit2/make_window_driver.py` and wrapped by `kit2/window.sh`; the analysis ran once after the last turn:
  `python3 -I ab3_analysis.py LAYOUT UNITS TOOLCHECK_JSON LEDGER_JSON OUT_PREFIX RUN_DIR...` (with `ab3-layout.json`,
  `ab3-units.jsonl`, the tool re-check JSON and the ledger). The six per-turn kit directories on the host are
  byte-identical copies of `kit2/` plus the receipts and logs listed in `RAW-RECEIPTS.sha256`; the earlier `kit/` is
  superseded and hash-listed.

### Owned-value oracle r7 (`owned-oracle-r7/`)

The checker that verifies every engine-owned value in a reply against an independent recomputation (wrong values,
wrong bytes, values not in the text). `oracle.py` sha256 `35f2bdf0243df7917b2b61098a1ff4a91bf1044854634912ab4800495db853fc`
is byte-identical to the copy pinned by the format re-test (`stdbench/format-retest/r7/oracle.py`, listed as a duplicate)
and to the one used under `<rank0-work>/owned-oracle/tool/`.
Rerun (as run): `bash selftest.sh` (must print `selftest PASS`), `bash run.sh <out dir> <receipt dir>...`,
`gate.sh OUT ROOT...` (exit 0 pass, 1 wrong engine write, 2 oracle did not run); see its `README.md`. In the format
re-test, r7 covered 0 owned runner cells because the optional runners were off; it gives no native-compute coverage
there.

## What these results do and do not show

Each RESULTS file and adjudication states its own limits. In short: the compute-DL accuracy benefit is an
outcome-informed amended study on a generated mixture, not a public-benchmark or serving-default result. The format
re-test supports clearer final formatting on a 60-task development cohort. Declaration control found no detected
incremental execution effect on standard suites. AB3 did not meet its registered criteria. No result here is a
matched comparison with DeepSeek's published scores.

## Rerunning

The kits expect two Spark hosts serving the recipe in TP2, a GPU lock per host, a CPU guard (2 GiB scopes) for every
CPU step, and the frozen inputs listed in each study's manifests. Raw receipts are not here: to re-audit from raw
data, obtain the files by the hashes in `RAW-RECEIPTS.sha256`. To re-derive the statistics, use the included
per-problem tables and analysis scripts. Writing a fresh output path is required throughout; existing collection
paths reject repeats.

## Configs

[`configs/README.md`](configs/README.md) lists the files. `configs/arm-L-effective.env` was read from the arm-L boot's
container snapshot (format-retest, rank 0). The recipe defaults to tool-full plus forced tool declaration;
arm L is the commented opt-in. Their shared base and compute-setting differences are described in
[docs/compute-lane.md](../../docs/compute-lane.md#the-flash17-profile-against-the-measured-arms).

## Published-layout adaptations

The archive preserves frozen study code and historical hashes; it is excluded only from the core
Ruff style gate in the recipe's `.ruff.toml`. It still needs its import/help, link, manifest and
oracle checks. Historical source hashes in the specs, adjudications and REDACTIONS table remain
historical; they are not silently replaced by claims about this publication repair.
`MANIFEST.sha256` pins the **current published bytes**, including the adaptations below. Its rows
are regular files; Git additionally pins the relative directory aliases `kit -> stdbench-kit` and
`phase2 -> stdbench-kit/phase2`. These aliases restore the historical import layout without copying
or reformatting the analysis/grader implementations. Minimal adapters supply omitted shared modules.

Safe `--help` exits prevent archival host scripts from launching collection, reading private state
or writing outputs merely to inspect usage. Modules with a normal main guard are imported instead.
This proves source/import availability, not that historical campaigns can run without their frozen
inputs, authorized hosts, queue controllers and omitted raw receipts. It grants no collection approval.
The statistical methods, seeds, datasets, strict grading and `owned-oracle-r7/oracle.py` are unchanged.
The portable oracle wrappers are publication adaptations; the evaluation algorithm remains the frozen one.
Its restored synthetic cases and EXPECTED files are independently listed in
[`owned-oracle-r7/selftest/fixtures.sha256`](owned-oracle-r7/selftest/fixtures.sha256). The initial
RAW-RECEIPTS exclusion inventory is retained for provenance even for those now-restored test inputs.

Exact files adapted or restored in this repair (paths relative to this directory):

| File | Why |
| --- | --- |
| `MANIFEST.sha256` | Refresh current published-byte hashes and include restored inputs/adapters; omit the manifest itself and directory aliases. |
| `RAW-RECEIPTS.sha256` | Mark initial export hashes/exclusions as historical; retain original entries unchanged. |
| `README.md` | Correct exported navigation/configuration wording and record this exact adaptation inventory. |
| `REDACTIONS.md` | Mark initial export hashes/exclusions as historical; retain original entries unchanged. |
| `ab3/T5-AB3-ADJUDICATION.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `ab3/ab3_freeze.py` | safe --help exit before historical top-level work |
| `ab3/kit2/ab3_check2.py` | safe --help exit before historical top-level work |
| `ab3/kit2/make_window_driver.py` | safe --help exit before historical top-level work |
| `ab3/kit2/run_cells.py` | safe --help exit before historical top-level work |
| `ab3/kit2/stale_live.py` | safe --help exit before historical top-level work |
| `ab3/kit2/test_ab3_analysis.py` | safe --help exit before historical top-level work |
| `ab3/kit2/tokalign.py` | safe --help exit before historical top-level work |
| `ab3/kit2/window_boot.py` | safe --help exit before historical top-level work |
| `ab3/results/ab3_turnsafety.py` | safe --help exit before historical top-level work |
| `ab3/t5-ab3-adjudication/ab3-adjudicate-read4/audit/script.py` | safe --help exit before historical top-level work |
| `ab3/t5-ab3-adjudication/ab3-adjudicate-stats1/rank1-host/audit/script.py` | safe --help exit before historical top-level work |
| `ab3/turns/t01/imr-measure.py` | safe --help exit before historical top-level work |
| `ab3/turns/t02/imr-measure.py` | safe --help exit before historical top-level work |
| `ab3/turns/t03/imr-measure.py` | safe --help exit before historical top-level work |
| `ab3/turns/t04/imr-measure.py` | safe --help exit before historical top-level work |
| `ab3/turns/t05/imr-measure.py` | safe --help exit before historical top-level work |
| `ab3/turns/t06/imr-measure.py` | safe --help exit before historical top-level work |
| `compute-dl/PREVIOUS-DECL-CONTROL-SUMMARY.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `compute-dl/RESULTS.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `compute-dl/ROOT-SUMMARY.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `compute-dl/a1-continuation/lifecycle_probe.py` | safe --help exit before historical top-level work |
| `compute-dl/a1-continuation/qualify_rank.py` | safe --help exit before historical top-level work |
| `compute-dl/a1-prep/env_check.py` | safe --help exit before historical top-level work |
| `compute-dl/a1-prep/p_verify_stage.py` | safe --help exit before historical top-level work |
| `compute-dl/a1-prep/preflight_check.py` | safe --help exit before historical top-level work |
| `compute-dl/a1-results/RESULTS.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `compute-dl/a1-runtime/qualify_rank.py` | safe --help exit before historical top-level work |
| `compute-dl/a1-runtime/rank_ops.py` | Repair the published import path; preserve experiment computation. |
| `compute-dl/a2-continuation/a1_budget.py` | resolve retained dependencies at the published layout without changing study algorithms |
| `compute-dl/a2-continuation/coordination.py` | resolve retained dependencies at the published layout without changing study algorithms |
| `compute-dl/a2-continuation/pre-mirror-source-v1/adoption.py` | resolve shared parent modules while preserving snapshot-local precedence |
| `compute-dl/a2-continuation/pre-mirror-source-v1/cpu_check.py` | resolve shared parent modules while preserving snapshot-local precedence |
| `compute-dl/a2-continuation/pre-mirror-source-v1/finalize_cpu.py` | resolve shared parent modules while preserving snapshot-local precedence |
| `compute-dl/a2-continuation/pre-opaque-mirror-source-v2/cpu_check.py` | resolve shared parent modules while preserving snapshot-local precedence |
| `compute-dl/a2-continuation/qualify_rank.py` | safe --help exit before historical top-level work |
| `compute-dl/a2-continuation/request_adapter.py` | resolve retained dependencies at the published layout without changing study algorithms |
| `compute-dl/a2-results-v1/RESULTS.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `compute-dl/baseline_check.py` | safe --help exit before historical top-level work |
| `compute-dl/pilot_pin.py` | safe --help exit before historical top-level work |
| `compute-dl/preflight_cpu.py` | safe --help exit before historical top-level work |
| `compute-dl/prepare_adjudication.py` | safe --help exit before historical top-level work |
| `compute-dl/publication-v1/RESULTS.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `compute-dl/source_inventory.py` | safe --help exit before historical top-level work |
| `compute-dl/summary/RESULTS.before-A2-20261106.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `compute-dl/summary/RESULTS.before-format-20261107.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `configs/README.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `configs/tool-full-ab3.env` | Replace the rank-0 host's name in a comment with its role; the engine lines are unchanged. |
| `decl-control/RESULTS.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `decl-control/T5-DECL-CONTROL-ADJUDICATION.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `decl-control/clashes3.py` | safe --help exit before historical top-level work |
| `decl-control/cleanup_compiler_caches3.py` | safe --help exit before historical top-level work |
| `decl-control/cpu_probe.py` | safe --help exit before historical top-level work |
| `decl-control/cpu_seal3.py` | safe --help exit before historical top-level work |
| `decl-control/cpu_stage3.py` | safe --help exit before historical top-level work |
| `decl-control/deploy_verify3.py` | safe --help exit before historical top-level work |
| `decl-control/dispatch-high/run_dispatch.py` | safe --help exit before historical top-level work |
| `decl-control/environment_audit3.py` | safe --help exit before historical top-level work |
| `decl-control/export_csv3.py` | safe --help exit before historical top-level work |
| `decl-control/identity_check3.py` | safe --help exit before historical top-level work |
| `decl-control/post-comparison-check3.py` | safe --help exit before historical top-level work |
| `decl-control/prepare_campaign3.py` | safe --help exit before historical top-level work |
| `decl-control/prepare_dispatch.py` | safe --help exit before historical top-level work |
| `decl-control/prepare_dispatch_high.py` | safe --help exit before historical top-level work |
| `decl-control/rank_preflight3.py` | safe --help exit before historical top-level work |
| `decl-control/restored_native3.py` | safe --help exit before historical top-level work |
| `decl-control/summary/NEEDS-T5-DECL-CONTROL-A1.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `decl-control/t5-workers/t5-decl-adjudicate.py` | safe --help exit before historical top-level work |
| `decl-control/t5-workers/t5-decl-support.py` | safe --help exit before historical top-level work |
| `decl-control/verify_exports3.py` | safe --help exit before historical top-level work |
| `decl-control/verify_feasibility.py` | safe --help exit before historical top-level work |
| `format-retest/NEEDS-T5.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `format-retest/RESULTS.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `format-retest/results-v1/NEEDS-T5.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `format-retest/runtime/coordination.py` | resolve retained dependencies at the published layout without changing study algorithms |
| `format-retest/runtime/lifecycle_probe.py` | safe --help exit before historical top-level work |
| `format-retest/spec/T5-FORMAT-RETEST-SPEC.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `format-retest/summary/RESULTS.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `kit` | Restore published relative layout alias → `stdbench-kit`; Git pins the link target. |
| `owned-oracle-r7/README.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `owned-oracle-r7/cpu-scope.sh` | Portable bounded CPU entry point, explicit image/QUIET handling, owned cleanup or fail-closed fixture preflight. |
| `owned-oracle-r7/gate.sh` | Portable bounded CPU entry point, explicit image/QUIET handling, owned cleanup or fail-closed fixture preflight. |
| `owned-oracle-r7/run.sh` | Portable bounded CPU entry point, explicit image/QUIET handling, owned cleanup or fail-closed fixture preflight. |
| `owned-oracle-r7/selftest.sh` | Portable bounded CPU entry point, explicit image/QUIET handling, owned cleanup or fail-closed fixture preflight. |
| `owned-oracle-r7/selftest/fixtures.sha256` | Pin the complete synthetic input set; self-test fails closed before evaluation. |
| `owned-oracle-r7/selftest/make_check_receipts.py` | safe --help exit before historical top-level work |
| `owned-oracle-r7/selftest/make_cut_receipts.py` | safe --help exit before historical top-level work |
| `owned-oracle-r7/selftest/receipts/bad-inline.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-list-fence.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-list-repl.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-measure-inline.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-capcut-deadline.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-capcut-divmod-check1000.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-capcut-divmod-check9.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-capcut-divmod-mod1000.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-capcut-divmod-mod9.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-capcut-divmod-quotient.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-capcut-divmod-remainder-word.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-capcut-divmod-remainder.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-capcut-fence.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-capcut-nocap.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-capcut-notlast.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-capcut-sign.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-capcut-span.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-capcut-timeout.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-capcut-trace.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-capcut-wrong.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-cut.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-divmod.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-fence.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-mark-shifted.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-mark.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-missing.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-span-divmod-display-cut.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-span-divmod-display-whole.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-span-divmod-echo-whole.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-span-plain-display-whole.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-span-plain-echo-whole.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-span.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-trace-new.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-unparsed.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-record-value.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-trace-engine.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/bad-trace.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/cut-record-deadline.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/cut-record-digits.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/cut-record-divmod-check1000.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/cut-record-divmod-check9.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/cut-record-divmod-mod1000.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/cut-record-divmod-mod9.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/cut-record-divmod-quotient.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/cut-record-divmod-remainder-word.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/cut-record-divmod-remainder.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/cut-record-fence.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/cut-record-mark.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/cut-record-s1.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/cut-record-trace-head.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/cut-record-trace.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/good-inline.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/good-list-fence.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/good-list-repl.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/good-mark-imitation-norecords.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/good-mark-imitation-wrong.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/good-mark-imitation.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/good-record-divmod.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/good-record-fence.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/good-record-inline.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/good-record-mark.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/good-record-model.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/good-record-span-divmod-whole.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/good-record-trace-new.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/good-record-trace-wording3.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/good-spaced-word.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/good-trace-engine.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/good-trace.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/live-record-divmod.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/receipts/live-record-timeout.json` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/t5-r5/EXPECTED` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `owned-oracle-r7/selftest/t5-r6/EXPECTED` | Restore original synthetic oracle fixture/expectations, not a model-run transcript. |
| `phase2` | Restore published relative layout alias → `stdbench-kit/phase2`; Git pins the link target. |
| `stdbench-kit/phase2/T5-STDBENCH-SPEC.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `stdbench-kit/phase2/audit_envs.py` | safe --help exit before historical top-level work |
| `stdbench-kit/phase2/bench.py` | resolve retained dependencies at the published layout without changing study algorithms |
| `stdbench-kit/phase2/cpu_final.py` | safe --help exit before historical top-level work |
| `stdbench-kit/phase2/cpu_postreview.py` | safe --help exit before historical top-level work |
| `stdbench-kit/phase2/cpu_stage1.py` | safe --help exit before historical top-level work |
| `stdbench-kit/phase2/cpu_stage2.py` | safe --help exit before historical top-level work |
| `stdbench-kit/phase2/cpu_stage3.py` | safe --help exit before historical top-level work |
| `stdbench-kit/phase2/export_check.py` | safe --help exit before historical top-level work |
| `stdbench-kit/phase2/finalize.py` | safe --help exit before historical top-level work |
| `stdbench-kit/phase2/prepare_campaign.py` | safe --help exit before historical top-level work |
| `stdbench-kit/phase2/rank_preflight.py` | safe --help exit before historical top-level work |
| `stdbench-kit/phase2/recovery_cpu.py` | safe --help exit before historical top-level work |
| `stdbench-kit/phase2/render_results.py` | safe --help exit before historical top-level work |
| `stdbench-kit/phase2/report_details.py` | safe --help exit before historical top-level work |
| `stdbench-kit/phase2/review_disputes.py` | safe --help exit before historical top-level work |
| `stdbench-kit/phase2/seal_math_early.py` | safe --help exit before historical top-level work |
| `stdbench-kit/project/PLAN.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
| `stdbench-kit/project/README.md` | Correct published links/omission notices or configuration labels; preserve numerical findings and historical spec hashes. |
