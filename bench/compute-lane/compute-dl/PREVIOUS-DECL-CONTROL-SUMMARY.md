# Declaration-control A1 results — NEEDS-T5

> Export navigation: paths marked omitted retain their historical names. Consult the
> [raw-receipt hash inventory](../RAW-RECEIPTS.sha256) for excluded receipts and duplicate destinations;
> historical documents without an inventory entry are not bundled or hash-attested here.

**No detected incremental execution-enabled effect on this cohort.** All three suites are **DECLARED BUT UNUSED**: zero dataset calls, despite six successful, authenticated and independently checked lane diagnostic calls. Declared-off (D) and declared+lane (L) have the same strict accuracy and returned-token totals. Their wall-time ratio intervals include 1. This does not establish equivalence or an execution benefit.

All **552 fresh scheduled replies** are retained: 184 per arm, with seeds 101/102 over 48 GSM8K, 20 MATH-500 and 24 original HumanEval tasks. These tasks/seeds were previously observed; this is a fixed-cohort configuration study. Eighteen excluded diagnostics and all three scheduled GPU turns completed. No reserve turn was used. Each turn restored G; final direct checks passed on both ranks, with all 18 experiment container names absent and no experiment job pending.

Base spec SHA256: `adb05167fff21469ee7251555cb243f1f109c4d8e3de237fdcaa59ada8eb55e6`.

Amendment A1 SHA256: `e7308621c645aef4402924a1b47c1a52e05baabe00239f915011f8431bbd0e89`.

Served checkpoint: **DeepSeek-V4.1-Flash-1017**, upstream revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, EXL3 2.80 bpw, TP2. Both ranks used per-SHA image `tensorfold-dsv41-imr:r-3a429d6f122d`, engine revision `3a429d6f122d9eb2900c43e622933d311e4be44f`. P has no declaration/execution OFF; D has the native explicit Python declaration/execution OFF; L has the identical declaration/body/execution ON. Runner, guide, S3, trace and other optional features were OFF, verified in all 18 actual native-process snapshots.

Thinking/high, temperature 1, top_p .95, top_k 20; concurrency 1; aggregate returned native-ID cap 8192; server deadline 270 s; client wait 360 s. Orders were P→D→L, D→L→P, L→P→D. Tokens/problem and wall seconds/problem below are means over every scheduled task/seed reply, including the capped reply. Wall is client end-to-end time; boot, restore and offline grading are separate.

| Suite | Arm | Strict solved / replies | Accuracy | Returned tokens/problem | Wall s/problem | Timeouts | Caps | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GSM8K | P | 54/96 | 56.25% | 280.7188 | 4.6790 | 0 | 0 | 0 |
| GSM8K | D | 52/96 | 54.17% | 214.1979 | 3.5259 | 0 | 0 | 0 |
| GSM8K | L | 52/96 | 54.17% | 214.1979 | 3.5213 | 0 | 0 | 0 |
| MATH-500 | P | 40/40 | 100.00% | 754.4000 | 11.0493 | 0 | 0 | 0 |
| MATH-500 | D | 40/40 | 100.00% | 589.0500 | 8.4520 | 0 | 0 | 0 |
| MATH-500 | L | 40/40 | 100.00% | 589.0500 | 8.4890 | 0 | 0 | 0 |
| HumanEval | P | 47/48 | 97.92% | 628.4792 | 9.9536 | 0 | 1 | 0 |
| HumanEval | D | 48/48 | 100.00% | 456.5833 | 6.8882 | 0 | 0 | 0 |
| HumanEval | L | 48/48 | 100.00% | 456.5833 | 6.9172 | 0 | 0 | 0 |

All 552 returned-ID counts/hashes and initial CPU/server prompt counts reconcile. All 190 D/L request pairs (184 dataset plus six repeated diagnostic pairs) match exact HTTP body bytes and served initial counts. Together with CPU reconstruction, this supports **CPU-matched declaration prompts, identical request bodies and matching served prompt counts**. Same-length server token substitutions or an unmodelled rendering branch remain possible. Direct served-ID equality and fully isolated execution-only causality are not established.

Before collection, the pinned-source CPU routing harness passed whole/SSE forced-call controls: D produced the client-visible call with zero dispatch/insertion/resume; L dispatched once and followed real insertion/resumption. A compiled absent-pool-guard mutant was caught behaviorally. This is source-path proof, not instrumentation of the deployed binary. Actual D environments and retained native evidence independently showed zero executions. No deployed engine source or image was changed.

Strict accuracy remains primary. Exact two-sided task-cluster sign-flip tests retain both seeds; Holm correction covers the six D−P/L−D primary tests. Shared 20,000 task bootstrap draws use seed 20261104, sorted task IDs and percentile indices 500/19499. There is no pooled suite score.

| Suite | Primary contrast | Accuracy difference, pp | Bootstrap 95% CI, pp | Exact p | Holm p |
| --- | --- | --- | --- | --- | --- |
| GSM8K | D−P | −2.0833 | [−8.3333, 4.1667] | 0.7539 | 1.0000 |
| GSM8K | L−D | 0.0000 | [0.0000, 0.0000] | 1.0000 | 1.0000 |
| MATH-500 | D−P | 0.0000 | [0.0000, 0.0000] | 1.0000 | 1.0000 |
| MATH-500 | L−D | 0.0000 | [0.0000, 0.0000] | 1.0000 | 1.0000 |
| HumanEval | D−P | 2.0833 | [0.0000, 6.2500] | 1.0000 | 1.0000 |
| HumanEval | L−D | 0.0000 | [0.0000, 0.0000] | 1.0000 | 1.0000 |

| Suite | L/D returned-token ratio, 95% CI | L/D wall-time ratio, 95% CI | Dataset activation |
| --- | --- | --- | --- |
| GSM8K | 1.0000 [1.0000, 1.0000] | 0.9987 [0.9941, 1.0021] | DECLARED BUT UNUSED |
| MATH-500 | 1.0000 [1.0000, 1.0000] | 1.0044 [0.9993, 1.0081] | DECLARED BUT UNUSED |
| HumanEval | 1.0000 [1.0000, 1.0000] | 1.0042 [0.9993, 1.0078] | DECLARED BUT UNUSED |

Reduced returned tokens versus P already occur with execution OFF. The declaration adds 306 initial prompt tokens to each reply: D/P full prompt+returned ratios are 1.6343 (GSM8K), 1.1655 (MATH-500), and 1.1692 (HumanEval). No total-token-saving claim qualifies. All dataset replies have complete, source-qualified single-round input accounting. Full multi-round input totals for activation diagnostics are withheld under A1; missing input is never imputed as zero.

GSM strict accuracy is sensitive to answer formatting. The preregistered one-numeral final-span secondary changes D−P to +2.0833 pp (marginal exact p=.5); it does not replace the primary score. All changed/residual extractions are retained. The single capped plain HumanEval reply contributes 89.5% of the net returned-token savings and more than 82% of the apparent wall savings versus P; the registered censor guard blocks an uncensored efficiency claim. All 120 MATH finals received two independent blinded reviewer **agent** reviews, with no disagreements or unresolved cases; no human review is claimed.

Published figures below are **REFERENCE ONLY**, retrieved 2026-11-04. They are historical BASE protocols, not matched instruct/quantized baselines; MATH is not MATH-500. Unpublished sampling/caps are unknown. Full source dates/settings and the exact current Flash/full-model distinction are in `PUBLISHED.md` (omitted from this export).

| Published checkpoint / protocol | GSM8K | MATH | Original HumanEval |
| --- | --- | --- | --- |
| Exact Flash-1017 instruct, these local suites | Not published | MATH-500 not published | Not published |
| [September V4.1-Flash-BASE](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md), historical reference | 93.0, 8-shot exact match | 61.1, 4-shot exact match | 79.4, 0-shot pass@1 |
| [V4-Pro-BASE preview card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/README.md), historical reference | 92.6, 8-shot exact match | 64.5, 4-shot exact match | 76.8, 0-shot pass@1 |

The [full generated report](../decl-control/RESULTS.md) includes L−P secondary comparisons, all metric intervals, per-seed outcomes, activation/unknown coverage, censor sensitivities, boot/restore timing and grader/source hashes. [Per-request CSV](../decl-control/finalized-a1-v1/per-request.csv), [analysis](../decl-control/finalized-a1-v1/analysis.json), retained-evidence audit (`decl-control/finalized-a1-v1/audit.json`, omitted from this export), [finalization receipt](../decl-control/finalized-a1-v1/finalize-receipt.json), and [run ledger](LEDGER.md) are local artifacts. The earlier two-arm study (`RESULTS-PHASE2-TWO-ARM.md`, omitted from this export) is preserved separately and is not pooled with these results.

Postcollection reporting adapters are outside the 87-file precollection seal. A reporting-only tuple/list persistence comparison was corrected after preserving its failed receipt; all 570 corrected attestation checks and changed-count/order negative controls passed. Repeated frozen grading produced identical grades; HumanEval grading metadata changed on rerun. Frozen collection, grading, extraction, statistical and MATH-seal mechanisms remained unchanged. All checks ran in guarded 2 GiB/no-swap Spark scopes. Status: **NEEDS-T5**, local handoff only.
