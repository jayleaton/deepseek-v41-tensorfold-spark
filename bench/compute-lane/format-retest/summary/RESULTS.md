# Format re-test — PASS / COMPLETE / NEEDS-T5

> Export navigation: paths marked omitted retain their historical names. Consult the
> [raw-receipt hash inventory](../../RAW-RECEIPTS.sha256) for excluded receipts and duplicate destinations;
> historical documents without an inventory entry are not bundled or hash-attested here.

All **360/360** fresh replies are collected and independently audited under T5's frozen rule. **L-new solved120/120 with0 format failures**, versus L-old86/120 with34 format failures. L gained34 strict solves with0 paired losses. **P-new solved60/60**, versus P-old59/60, with1 gain and **0 paired P losses**. Safety/provenance qualification passed; the strict grader is unchanged.

Spec SHA256: `b47ede697eed3a38d244ad5ff7447690732bd12945ff4592554d45bc2d8527ec`.

| Arm | Strict solves | Format failures | Returned native IDs/reply | Full wall seconds/reply | Initial+returned subtotal/reply | Delivered native results |
|---|---:|---:|---:|---:|---:|---:|
| L-old |86/120 (71.67%)|34|380.483|4.822|889.983|123|
| L-new |120/120 (100%)|0|351.642|4.469|880.642|120|
| P-old |59/60 (98.33%)|1|2338.567|25.617|2656.100|0|
| P-new |60/60 (100%)|0|2466.117|26.936|2802.983|0|

L covers six families; P covers only DP/graph/aggregation. All arms had0 caps,0 deadlines and0 complete typed-but-wrong final values. No missing replies, exclusions, outcome retries, answer repair or task replacements were introduced. L-old had1 execution-error result, independently matched to its isolated reference; it is retained. All120 L-new executions succeeded.

| Family (20 replies per arm) | L-old strict / format failures | L-new strict / format failures | P-old strict / format failures | P-new strict / format failures |
|---|---:|---:|---:|---:|
| Integer |10 /10|20 /0|—|—|
| Modular |13 /7|20 /0|—|—|
| Division |20 /0|20 /0|—|—|
| DP |15 /5|20 /0|20 /0|20 /0|
| Graph |9 /11|20 /0|19 /1|20 /0|
| Aggregation |19 /1|20 /0|20 /0|20 /0|

The primary L-new minus L-old format-failure difference is **−28.33 percentage points**,95% family-stratified task-cluster bootstrap interval **[−36.67,−20.00] points** (20,000 frozen draws). L-new/L-old returned-ID ratio is0.92420 (CI[0.87400,0.97528]); wall ratio0.92681 (CI[0.87900,0.97663]). New wording increases initial prompt length; prompt-length change is part of this intervention.

On the three-family matched secondary comparison, L-new and P-new both solve60/60. L-new/P-new returned-ID ratio is0.21668 (CI[0.19519,0.23991]); wall ratio0.23465 (CI[0.21078,0.25978]). These are observed costs on this generated cohort. L multi-round processed-input totals remain unknown; initial+returned quantities are explicitly subtotals. No full-token/FLOP-saving claim follows.

**G is restored.** Actual step logs record COMPLETE collection, both study boots drained, G SERVING_VERIFIED and turn QUALIFIED. Independent verification confirms `/v1/models`200 and both expected running G images, source `58bece1d3184269059fd3f7f755c52f4f295113d`, run `efast2-20261106T203019Z`. Total charged held time **5590.183 seconds (93m10s)** includes boots, diagnostics, drain and restoration;0 overrun against6300s. One ORDER turn used two cold boots, with old/new interleaved within each. GO is revoked on both nodes.

All245 delivered native presentations (243 dataset results plus2 diagnostics) passed source/frame/status and independently isolated CPython checks; none were clipped. P diagnostics made0 native calls. Pinned r7 reported0 wrong values/bytes/not-in-text over **0 owned runner cells** because optional runners wereOFF; r7 provides no native-compute coverage. Native-compute verification is the separate audit.

A postcollection audit schema bug was corrected in a new versioned postprocessor: family lookup now uses the hash-pinned public manifest instead of an absent nested template field. The original sealed collection code, strict grader and receipts remain unchanged. The original failure log, focused360-template/failure-retention controls and corrected audit are preserved.

## Review evidence

- [Frozen analysis, per-family counts, paired identities and cost intervals](../results-v1/RESULTS.md)
- Full machine-readable report, per-family costs and safety provenance (`format-retest/results-v1/report.json`, omitted from this export)
- [360-row observation ledger](../results-v1/per-observation.csv)
- [Independent raw audit](../audit-v2/raw-audit.json)
- [Final artifact inspection](../final-inspection.json)
- [Actual step log](../final-evidence/steps.jsonl),[budget](../stdbench-format-20261107-v1/budget.json),G restoration (`format-retest/final-evidence/G-restore.log`, omitted from this export) and [independent G verification](../final-G.json)
- [Pinned r7 coverage](../r7-final-v1/r7.json)
- [Postcollection correction controls](../audit-post-check-v2.json) and original failure (`format-retest/final-evidence/audit-v1.log`, omitted from this export)
- [Task ledger](../LEDGER.md) and [NEEDS-T5 handoff](../NEEDS-T5.md)
- [Previous compute-DL A2 summary, preserved](../../compute-dl/summary/RESULTS.before-format-20261107.md)
- DeepSeek published benchmarks, historical reference only (`PUBLISHED.md`, omitted from this export)

This PASS supports clearer final formatting on a60-task,six-template development cohort; it does not establish a turn-boundary mechanism, public-benchmark superiority or a serving-default change. The observed0/120 L-new failures does not prove a population≤1.67% failure rate, and0/60 paired P losses does not prove zero population risk. Actual served prompt arrays were not captured; CPU arrays, exact wires and server initial-count reconciliation leave same-length rendering differences possible. Old study outcomes are not pooled. No matched DeepSeek published score exists for this generated cohort.
