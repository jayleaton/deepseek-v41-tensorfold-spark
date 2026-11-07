# Computation-demanding D/L study — INCOMPLETE

> Export navigation: paths marked omitted retain their historical names. Consult the
> [raw-receipt hash inventory](../../RAW-RECEIPTS.sha256) for excluded receipts and duplicate destinations;
> historical documents without an inventory entry are not bundled or hash-attested here.

**NEEDS-T5. Collection stopped at 342/480 replies; 138 remain uncollected. G is restored.**
Spec SHA256: `c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141`.
D = declared tool, executor OFF. L = identical declaration/request, lane ON. No plain arm.
120 deterministic instances, six fixed templates, two seeds; fixed-mixture screen, not a public benchmark or 120 independent task families.

## Stop and diagnosis

`compute-dl-v1-aggregation-015`, seed101, stopped turn2 at the frozen tool-result/reference gate. The native result was clipped at512 tokens; the comparator used full554-token stdout. Three isolated CPython replays reproduced the full stdout. A pinned native-tokenizer CPU proof exactly reconciles the delivered bytes with the production512-token/UTF-8 clipping policy. The complete first output line matches the independent task reference.
The failing receipt and frozen audit remain unchanged. No observed reply was replaced or regraded, and collection GO is revoked on both ranks. A T5 amendment is needed for a source-aware verifier and a prospective layout for the remaining18 L +120 D replies.
[Truncation proof](../diagnosis-v1/truncation-proof.json) · [diagnostic replays](../diagnosis-v1/diagnosis.json) · [frozen audit](../report-v1/audit.json).

## Observed results and missingness

Every family schedules40 replies per arm. Solved/observed and cost means below describe retained observations only; different coverage prevents a full-cohort point comparison. Missing costs are unknown, never zero.

| Family | D solved/observed | L solved/observed | D returned IDs/problem | L returned IDs/problem | D wall s/problem | L wall s/problem | observed timeouts D/L |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 1/120 | 169/222 | 269.650 (n=120) | 357.279 (n=222) | 3.147 (n=120) | 4.427 (n=222) | 0/0 |
| integer | 0/20 | 28/40 | 92.000 (n=20) | 179.675 (n=40) | 1.382 (n=20) | 2.579 (n=40) | 0/0 |
| modular | 0/20 | 21/40 | 106.650 (n=20) | 148.700 (n=40) | 1.661 (n=20) | 2.329 (n=40) | 0/0 |
| division | 0/20 | 40/40 | 119.700 (n=20) | 216.875 (n=40) | 1.763 (n=20) | 3.004 (n=40) | 0/0 |
| dp | 0/20 | 31/40 | 436.700 (n=20) | 568.775 (n=40) | 4.771 (n=20) | 6.637 (n=40) | 0/0 |
| graph | 1/20 | 28/40 | 393.450 (n=20) | 487.550 (n=40) | 4.279 (n=20) | 5.658 (n=40) | 0/0 |
| aggregation | 0/20 | 21/22 | 469.400 (n=20) | 693.318 (n=22) | 5.027 (n=20) | 7.936 (n=22) | 0/0 |

All240 scheduled replies per arm: D solve bounds [1, 121]/240; L solve bounds [169, 187]/240. L−D logical bounds: 20.0 to 77.5 percentage points. These are missing-outcome bounds, not a confidence interval.
The full-cohort primary difference,95%CI and exact sign-flip p-value are withheld. No completed-study efficacy claim is made. No serving-default promotion or total-token-saving claim is made.

## Paired statistics and accounting

The frozen analysis uses20,000 shared task-cluster bootstrap draws, stratified by family, Random(20261105), percentile indices500/19499; both seeds/arms stay together. The exact task sign-flip is the sole overall primary test. Family/cost/sensitivity intervals are marginal and descriptive.

- returned_ids: full-cohort ratio/CI withheld. Selected complete-pair sensitivity n=120: L/D=1.447;95%CI=[1.3760257579211306, 1.533785221435302]. This selected sample is not representative of missing cases.
- wall_seconds: full-cohort ratio/CI withheld. Selected complete-pair sensitivity n=120: L/D=1.520;95%CI=[1.449793843273254, 1.6043014529368174]. This selected sample is not representative of missing cases.

Only1 paired observation is complete-both and uncensored. Censoring blocks an uncensored efficiency claim. Cap/deadline/policy rates and net/positive censor contributions are retained in the full analysis.
Multi-round processed-input totals remain withheld. Initial-prompt-plus-returned is a separately labelled subtotal, not full all-round token cost. Full-cohort processed-input-plus-returned ratios/CIs are withheld.
[All sums, means, medians, coverage, censor contributions and paired solve gains/losses](../report-v1/analysis.json) · [480-row per-problem ledger](../per-problem.csv).

## Activation and diagnostics

Activation: 111/120 distinct L task IDs (ADEQUATE; threshold30). Each credited task has an original successful, delivered, authenticated, replay-matched native execution. D made zero native executions. Six of eight planned diagnostics were collected; all three completed boots passed their diagnostics.
| Family | verified L task IDs | seed101 | seed102 | D/L format rejections |
|---|---:|---:|---:|---:|
| integer | 20 | 20 | 20 | 0/12 |
| modular | 20 | 20 | 20 | 0/19 |
| division | 20 | 20 | 20 | 0/0 |
| dp | 20 | 20 | 20 | 0/9 |
| graph | 20 | 18 | 20 | 0/12 |
| aggregation | 11 | 10 | 11 | 0/1 |

Per-family/per-seed attempted, blocked, successful/failed execution and result-use counts are in [the frozen analysis](../report-v1/analysis.json). Successful activation does not override incomplete coverage or the failed frozen gate.

## Supplementary native tool timing

The native report exposes call_ms (translate/queue/run) and exec_ms (executor run). These descriptive means use original frozen-qualified rows only; they do not replace the frozen analysis. Complete server end-to-end time is unavailable.
| Family | D executor s/problem | L executor s/problem | D tool-call s/problem | L tool-call s/problem |
|---|---:|---:|---:|---:|
| overall | 0.000 (n=120) | 0.015 (n=221) | 0.000 (n=120) | 0.000 (n=2) |
| integer | 0.000 (n=20) | 0.016 (n=40) | 0.000 (n=20) | unknown (n=0) |
| modular | 0.000 (n=20) | 0.014 (n=40) | 0.000 (n=20) | unknown (n=0) |
| division | 0.000 (n=20) | 0.015 (n=40) | 0.000 (n=20) | unknown (n=0) |
| dp | 0.000 (n=20) | 0.015 (n=40) | 0.000 (n=20) | unknown (n=0) |
| graph | 0.000 (n=20) | 0.015 (n=40) | 0.000 (n=20) | 0.000 (n=2) |
| aggregation | 0.000 (n=20) | 0.016 (n=21) | 0.000 (n=20) | unknown (n=0) |

Timing sums, means, medians and coverage (`supplementary-telemetry.json`, omitted from this export).

## Operational evidence and limits

Cumulative operator-held GPU time: 3413.519s (56.89min), including boots, diagnostics, independent replay and both G restorations; overrun0s. Turn1 QUALIFIED; turn2 STOPPED. Budget was not exhausted. No GPU or CPU job remains pending.
CPU D/L prompt arrays matched for242 keys; source/image/environment and sent-wire/initial-count checks are retained. Served initial arrays were not directly observed; same-length server rendering differences remain unobserved. Full planned244 wire/count comparisons cannot qualify with138 replies missing.
Inherited declaration-control base SHA `adb05167fff21469ee7251555cb243f1f109c4d8e3de237fdcaa59ada8eb55e6`; A1 SHA `e7308621c645aef4402924a1b47c1a52e05baabe00239f915011f8431bbd0e89`.
Public task manifest SHA `3e1e07d7a573eb194c0568afac2ef2112011b2b2cada9012fbbce4e6f47f7548`; reference A/B agree120/120, zero disjointness collisions. Frozen grader/statistics and all original receipts are preserved.
DeepSeek published scores in PUBLISHED.md (`../PUBLISHED.md`, omitted from this export) are labelled historical reference only. This generated task set has no matched DeepSeek published score; no vendor number is treated as a study arm.

**NEEDS-T5** — adjudicate the clipping-aware verifier and completion layout before any further collection.
