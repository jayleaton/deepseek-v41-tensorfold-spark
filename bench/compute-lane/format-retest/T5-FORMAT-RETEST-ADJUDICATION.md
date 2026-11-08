# Format retest adjudication — PASS

2026-11-07. **CONFIRM / COMPLETE / PASS** against frozen spec `b47ede697eed3a38d244ad5ff7447690732bd12945ff4592554d45bc2d8527ec`. [Independent raw audit](t5-format-retest-adjudication/audit.json), [reproducible worker](t5-format-retest-adjudication/t5-format-audit-worker.sh); collection evidence in `../format-retest/`.

## Frozen rule

Independent node CPU audit checks481 pinned input hashes,360 unique raw replies and their audited copies, exact native returned-ID counts, unchanged strict grader SHA `1a2249a1…`, private A/B reference agreement, exact old/new prompt substitutions, matched P/L bodies apart from declaration, and the frozen interleaved random order. No answer repair or cohort pooling. Preparation/source seals include the fresh-task/disjointness inventory; this review does not claim a new independent regeneration of every earlier exclusion set.

| Gate | Observed | Ruling |
|---|---|---|
| Complete/provenance |360/360, no missing; source seals and raw identities reconcile|PASS|
| L-new format failures≤2/120 |0/120; L-old34/120|PASS|
| L-new strict solves≥L-old |120/120 versus86/120;34 paired gains,0 losses|PASS|
| Zero paired P losses |P-new60/60 versusP-old59/60;1 gain,0 losses|PASS|
| Safety |245 exact, unclipped native result presentations, including2 diagnostics; no unresolved mismatch|PASS|

The primary format-rate difference is−28.33pp, task-cluster/family-stratified CI95[−36.67,−20.00], independently reproduced with the fixed20,000 draws. All34 old-L strict failures are format failures; there are no complete typed-but-wrong final values in any arm. No caps or deadlines occurred. All observations remain, including L-old's independently matched execution-error result.

Safety has two different coverage sources: pinned r7 reports zero wrong/not-in-text over **zero runner-owned cells**, since optional runners were off; it is not the native-tool verifier. The separate source/frame/status-bound isolated CPython audit covers243 dataset tool results plus2 diagnostics. Independent review confirms all245 full-reference/landed-body hashes match and none clipped; no new jail executions were performed for this adjudication.

The versioned postcollection schema fix changes family lookup to the hash-pinned public manifest, not grading or collection. The original failure and correction controls remain. Independent raw regrading here bypasses that lookup and confirms the results. The charged5590.183s, including boots/restoration, is below6300s. CPU-matched/count-checked prompts retain the disclosed absence of actual served-array capture.

## New-wording L versus plain: use the matched subset

The three shared families are DP, graph and aggregation:30 fresh tasks×two seeds,60 replies per arm. **Both solve60/60.** All P and L replies finish uncensored:zero caps,zero deadlines,zero missing. The comparison is an uncensored secondary result on this cohort.

| Matched new-wording arm | Returned IDs/reply | Full client wall s/reply |
|---|---:|---:|
| L-new |534.37|6.320|
| P-new |2466.12|26.936|

L/P returned-ID ratio **0.21668**, CI95[0.19519,0.23991]:78.33% fewer returned IDs. Wall ratio **0.23465**, CI95[0.21078,0.25978]:76.54% less measured response time. Independent raw sums and fixed bootstrap reproduce these exactly. The352 IDs/4.5s headline is L-new across **all six** families; comparing it with three-family P would be unmatched.

These data support lower returned-ID and wall cost with no observed strict-solve loss on this fresh generated subset. They do not establish population accuracy equivalence, full processed-token/FLOP savings or uncensored P performance on arithmetic, which was not retested. L's multi-round processed inputs remain unmeasured. L and P use separate cold boots; cache/temporal effects remain possible. Neutral new wording is identical in both arms and changes only the requested final representation; its extra prompt length is part of the intervention.

## Public wording and publication readiness

Suggested wording:

> On120 generated computation-heavy tasks with two seeds, our Python execution lane with its tool declaration produced187/240 strictly formatted correct answers versus120/240 for plain under an8192-ID cap. Half the plain replies capped, accounting for most of the roughly93% observed returned-ID and time savings. We then prospectively tested clearer final-answer wording, applied identically to every arm, on60 fresh tasks. The lane solved120/120 versus86/120 with the old wording, eliminating the observed format failures. On the30 fresh DP/graph/aggregation tasks also run plain, both new-wording arms solved60/60, with no caps; the lane returned78% fewer IDs and took77% less measured response time.

Publish these as **two separate studies**, name the qualified quantized Flash-1017 EXL3 2.80bpw configuration and Python tool declaration, and disclose compute-DL's outcome-informed amendments. Do not rewrite its187/240 result, pool it with the retest, call this an engine formatting fix, or claim the turn-boundary explanation is causally proven. These are generated six-template development cohorts, not standard benchmark scores. Zero observed failures/losses does not prove zero future risk; the earlier no-call benchmark result remains unchanged.

**No additional experiment is needed before JJ publishes those qualified descriptive claims.** Keep the frozen specs, new wording, task/reference hashes, caps and matched denominators with the evidence. Broader benchmark, population reliability or serving-default claims would require separate evidence.
