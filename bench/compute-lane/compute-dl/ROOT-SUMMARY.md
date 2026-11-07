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

- [Per-family results, tokens, wall time, missingness and paired sensitivities](compute-dl/RESULTS.md)
- [480-row per-problem ledger](compute-dl/per-problem.csv)
- [Exact native clipping proof](compute-dl/diagnosis-v1/truncation-proof.json)
- [T5 adjudication request and completion constraints](compute-dl/NEEDS-T5.md)
- [138 remaining original requests](compute-dl/remaining-collection.json)
- [Prior declaration-control results](decl-control/RESULTS.md)
- [DeepSeek published scores — historical reference only](PUBLISHED.md)

**Default: keep collection stopped until JJ steers a frozen T5 amendment SHA.**
