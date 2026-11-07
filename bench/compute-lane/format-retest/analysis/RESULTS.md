# Format retest — NEEDS-T5

Implementation handoff only. No analysis or handwritten controls have been run.
Real collection, private references, independent audit, and parent-run CPU checks
are still required. No production observations or efficacy/safety result is asserted.

Frozen spec SHA256: `b47ede697eed3a38d244ad5ff7447690732bd12945ff4592554d45bc2d8527ec`.
Unchanged strict grader SHA256: `1a2249a1b5e0b06f03a4300995d0c320e5a21630d261b3791501f7c83740fb9f`.

PASS only if all are true: collection/provenance complete; L-new has≤2/120 format failures; L-new strict solves≥L-old strict solves; zero P paired losses (P-old correct/P-new incorrect) across its60 pairs; safety passes.

Missing observations mean INCOMPLETE. Safety/provenance failures BLOCK regardless
of format outcomes. L-new 3–6 format failures are INCONCLUSIVE; ≥7 fails, as do any
P paired loss or net L strict decline. The reporter retains all 360 scheduled keys,
including missing/error/capped/deadline observations. Unknown cost is never zero.

Native verification follows inherited A1: independently proven 512-token policy
clips may qualify as `presentation_verified=true`, `complete_value=false`,
`clipped=true`, `full_reference_verified=true`. Reports distinguish policy clips
from unresolved cuts, and preserve full-reference and landed hashes/lengths.
Clipped fragments never count as complete values or rescue final-answer grading.
Invisible native tail bytes are not authenticated by a matching presentation.
r7 covers only owned runner cells (zero with all optional runners OFF); native
compute requires separate independent Jail/CPython, frame and policy proof.

The parent invocation in HANDOFF.md produces readable RESULTS.md, report.json and
per-observation.csv from real supplied inputs. No synthetic production JSON/CSV is
provided at handoff; handwritten controls never write result artifacts.

This is a limited development cohort of six generated templates and two seeds.
Prompt length changes with wording; cold boots and prompt-cache state confound
KV-state comparisons. P arithmetic is omitted. The observed ≤2 acceptance target
is not population proof, and zero P losses does not prove zero population risk.
A PASS cannot establish the proposed turn-boundary mechanism, general benchmark
superiority, or a serving-default change. No old compute-DL outcomes are pooled.
