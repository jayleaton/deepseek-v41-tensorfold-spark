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
container snapshot (format-retest, rank 0); the recipe profile differs from it only as
[docs/compute-lane.md](../../docs/compute-lane.md#the-flash17-profile-against-arm-l) lists.
