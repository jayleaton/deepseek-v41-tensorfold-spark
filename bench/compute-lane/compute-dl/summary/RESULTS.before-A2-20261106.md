# Compute study A1 — BLOCKED / INCOMPLETE (2026-11-06)

> Export navigation: paths marked omitted retain their historical names. Consult the
> [raw-receipt hash inventory](../../RAW-RECEIPTS.sha256) for excluded receipts and duplicate destinations;
> historical documents without an inventory entry are not bundled or hash-attested here.

**NEEDS-T5. G is not currently restored.** The A1 continuation stopped before any
new inference because <rank1-host> `/tmp/<rank1-host>-gpu.lock` is absent. The queued G
restoration failed on the same required read-only open. Both new GO markers are
revoked; no owned job is running. The failed restoration entry remains at queue
head, and its held-time interval is unresolved.

All342 retained replies passed uniform clipping-aware revalidation;112 distinct
L tasks have verified activation. All378 new replies remain uncollected
(18 L,120 D,240 P), so the amended L/P comparison is withheld.

Parent SHA: `c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141`.
A1 SHA: `0dca1fa52838388ea1e7afbc3a4c5e9747775f64e871640df7a66496369e5453`.

- [A1 results, CPU qualification and failed restoration evidence](../a1-results/RESULTS.md)
- [NEEDS-T5 and required infrastructure recovery](../a1-results/NEEDS-T5.md)

The following original summary is preserved as a **historical snapshot before
A1**. Its serving-state statement does not describe the current state above.

# Computation-demanding D/L study — INCOMPLETE

**NEEDS-T5.** Collection stopped at 342/480 replies; 138 remain uncollected.
The verifier omitted the native 512-token tool-result clipping policy and
compared a clipped result with full stdout. The pinned native-tokenizer CPU
proof exactly reconciles those bytes. The frozen failing receipt is preserved.

G is restored on both ranks. Cumulative held time was 3413.519 seconds
(56.89 minutes), including both restorations, with no budget overrun.
Collection GO is revoked on both ranks; no benchmark/check job is pending.

Observed strict solves: D 1/120, L 169/222. All 53 observed L non-solves are
format rejections under the frozen extraction rule; no rescue grading was
applied. Verified L execution exposure covers 111 distinct tasks. These partial
observations do not support a completed-study efficacy claim; the primary
point estimate, confidence interval and test are withheld.

Spec SHA256: `c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141`.

- [Per-family results, tokens, wall time, missingness and paired sensitivities](../RESULTS.md)
- [480-row per-problem ledger](../per-problem.csv)
- [Exact native clipping proof](../diagnosis-v1/truncation-proof.json)
- [T5 adjudication request and completion constraints](../NEEDS-T5.md)
- [138 remaining original requests](../remaining-collection.json)
- [Prior declaration-control results](../../decl-control/RESULTS.md)
- DeepSeek published scores — historical reference only (`PUBLISHED.md`, omitted from this export)

**Default: keep collection stopped until JJ steers a frozen T5 amendment SHA.**
