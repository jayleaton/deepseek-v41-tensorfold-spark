# A1 source-only analysis handoff

This directory is new source only. The original strict grader,342 outcomes, failed historical audit/stop and D/L report are immutable. No functional/AST Python checks were run on the development machine. No GPU, SSH, queue, <other-host>, Lead, push or publishing operations were performed. No files in a1-runtime/a1-prep were edited.

Read frozen amendment0dca1fa52838388ea1e7afbc3a4c5e9747775f64e871640df7a66496369e5453 and original c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141. The pinned scalar selector is extracted as the exact `gsm_final` AST from phase2/grade_all.py dcc8ef766247a1b02c993c9a50186b70b68c4fb5b6493e1bb42b3f7727186d36. The selector's unrelated imports/Jail never execute. The original compute-dl/grade.py1a2249a1b5e0b06f03a4300995d0c320e5a21630d261b3791501f7c83740fb9f is imported read-only from its absolute resolved filepath, verified before execution. Its grade() API supports each arm without modifying the old grade_all() restriction.

## Parent API

Prepend this directory to sys.path in the isolated parent process (before importing any other `analysis`, `grade`, `pins` or `collect_report` module). Do not import this new collect_report by the old module's name in an already populated interpreter. `pins.load_local(name)` explicitly binds sibling source paths, avoiding accidental reuse of an old analysis/grade module.

```python
import collect_report
report = collect_report.collect(rows, tasks, answers, metadata={
    'integrity': {
        'qualified': True,
        'spec_sha256': '0dca1fa52838388ea1e7afbc3a4c5e9747775f64e871640df7a66496369e5453',
        'provenance': 'parent audit evidence path/hash',
    },
    'budget': parent_budget_evidence,
    'audit': parent_source_and_mask_evidence,
})
markdown = collect_report.render_results(report)
```

No writes occur; the parent writes report JSON/Markdown/row JSONL at its owned destination. No CLI collection entrypoint or inference capability exists here.

`analysis.analyze(rows, tasks, answers=None, integrity=None)` accepts raw rows plus private typed answers, or strict graded rows when answers=None. `grade.grade_all` validates the observed3-arm grid, calls the original grade(row, task, answer) on every row and preserves raw aliases. `grade.sensitivity_all` recomputes strict grades and adds secondary grades; it never changes strict grader metadata. `collect` returns `strict_rows`, `secondary_rows`, `analysis`, `secondary_analysis`, `secondary` (coverage, all changed rows, all residual incorrect rows), presentation receipts, metadata and current source hashes.

Rows accept declared_off/D, lane/L, plain/P. Tasks and answer-map shapes are unchanged.120 tasks,20 per original family, seeds101/102 give720 scheduled keys; no selection uses an outcome. Diagnostics, duplicate alias keys, invalid seeds and excluded rows are rejected. Unobserved rows stay missing, costs unknown, solve bounds worst/best. Any missing scheduled key disables completed-study strict efficacy inference even if L/P alone is complete. Scalar sensitivity always has160 scheduled rows per arm and combined sensitivity240, with non-scalar strict grades preserved. Changed/residual lists contain observed rows; missing rows appear separately in analysis.

Contrast keys are L_minus_P (sole strict amended primary), L_minus_D (mechanism), D_minus_P (descriptive). Each exposes overall, scalar_only and six families. Dynamic point/cost field names reflect the actual comparator, e.g. solve_difference_L_minus_P, ratio_L_over_P, P_sum/L_sum. Arrays/maps never receive a numeral rescue. Sensitivity tests are all labelled nonprimary and cannot qualify efficacy.

## Trust boundary with parent audit

Parent owns rigorous source/native-frame/worker/image/wire/prompt binding, actual clipping implementation, isolated replay, uniform requalification of all342 original and future calls, exact original mask/no retry, runtime qualifications, cumulative budget and source seals. `integrity.qualified` must be true only after those gates pass. Merely copying the old historical integrity verdict is invalid.

Parent-audited row schema is unchanged: native_binding.qualified and calls with id/native_bound/delivered/status/source_sha256/output_sha256; safety.qualified and references with call_id/matches/result; result.status must equal call.status. Analysis re-joins call IDs and exact source/output hashes, checks status, and consumes the NEW uniformly revalidated `matches` verdict. It does not rerun the native codec, independent replay or old output comparator. This delegation permits clipped presentations while retaining the original full-reference bytes. Parent must never pass the failed historical matches verdict as newly qualified.

Attach the a1-prep/reference_policy.presentation() result as each reference's `presentation` to publish clipped/unclipped counts, full-reference/landed lengths and hashes, prefix length and marker bytes. When present, its qualified verdict, landed SHA and landed byte count are checked again. Do not replace normalized_body_base64 with a landed prefix: full reference evidence stays intact. Report `presentation_summary.retained_receipts` coverage: no presentation entry means unavailable presentation telemetry, never proof of an unclipped result. A clipped tool fragment is never counted as a complete final answer; final content alone controls both graders.

Call accounting, input accounting and initial prompt reconciliation retain original qualified/complete/coverage semantics. Actual returned totals and full client elapsed wall are used. Unknown full multi-input totals suppress full cost ratio/CI; initial+returned remains a separately named SUBTOTAL. D attempted/blocked/native execution totals come from receipt fields; policy terminations, final-answer replies and strict solves are separate nonexclusive classifications by family/seed. Native D/P calls/frames or execution counters block safety. Activation requires30 distinct successful delivered authenticated verified L tasks; multiple calls/seeds count once.

## Statistics and claim guards

pair_analysis.py is a source copy of the original analysis71101f372473fc5be4a7f9896fb5b3e8ef375620c1dca87f48bb8559ddb292ae, generalized with a scalar report group and new parent-audited presentation receipt consumption. Its old two-arm qualification/activation report and claim rules were removed. Internal D/L are positional calculation labels only; analysis.py relabels all exposed fields and identity lists and applies the amended three-arm qualification once.

Each contrast restarts the same fresh Random(20261105)20k-draw generator with the same fixed sorted120 IDs. Thus each draw selects exactly the same20 replacement task IDs per family for ALL contrasts, seeds, arms, scalar/family/overall estimates, endpoints and both grading views. This is deterministic shared sampling, without holding a huge3-arm draw tensor. Percentiles500/19499 and exact two-sided task-level sign-flip DP are unchanged. Family/cost/subset intervals and all sensitivity tests are descriptive/marginal. The120 task clusters belong to six fixed templates, not120 independent task families.

Local benefit thresholds apply only to strict L/P, with complete720, parent integrity, safe zero-execution D/P, adequate L activation, original returned/wall censor guards and original numerical criteria. Unknown full input costs never permit total-token-saving claims. >25% net/positive censor contribution or complete-both reversal blocks uncensored efficiency. Missing outcomes do not become observed incorrect points or zero costs. Outcome-informed amendment,222 earlier L/18 resumed L, pause, extra boots/cache resets and mostly L-before-P order remain disclosed; retention_batch is descriptive only.

## Standards and verification

Before editing, checked ancestry AGENTS.md/LOCAL-AGENT.md/CONTRIBUTING and lint/format/build configuration at /, /home, <dev-home>, Documents, Projects, <agent-dir>, lead, dsv41-update, stdbench and compute-dl; absent, matching original LEDGER.md. Artifact tree is not Git; no branch, commit, kit, export or PR is created. There is no applicable repository check configuration. The bounded source-only implementation is kept in one owner directory; no helper was launched because this assigned slice shares the same math/API contract and the parent owns independent verification.

Parent must seal ALL seven files here (six Python sources plus this README) using SOURCE-SHA256.txt before any fresh collection. Also bind immutable original grader, analysis, both specs and phase2 selector; source_hashes() is current-state telemetry and does not replace the precollection seal.

Parent runs AST parsing and `python -B /path/to/a1-analysis/fakecheck.py` only inside the already authorized Sparks CPU scope:2GiB/noSwap/nice19/ionice-c3/timeout7200/single worker, with parent admission rules. No dev Python runs. fakecheck uses synthetic observations, NEVER executes fixture code, and writes no outputs. It checks selector/grammar positives/negatives, strict equality, status/eligibility/tool-call controls, non-scalar preservation, complete/missing masks, duplicate aliases and seeds, exact DP, real20k bootstrap, scalar/combined denominators, presentation mutations, activation, unknown costs, censor dominance/reversal, receipt-derived D classification and rendering/strict JSON serialization. Parent additionally tests real retained342 compatibility and source bindings before accepting this handoff. Any finding requires fixing/resealing before collection; no PASS claim is made here.
