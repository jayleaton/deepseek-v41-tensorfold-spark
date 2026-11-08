# AB3 frozen replication rule

2026-11-04. Prospective rule, fixed before T5 inspects any AB3 outcome. The SHA256/time entry in STATUS-T5 is the registration. lane must freeze the manifest, rendered requests, references, order and immutable image hashes before the first AB3 request; record its timestamp against this registration. If collection already started, stop calling this pre-first-request registration: disclose the chronology and require a fresh prospective run for confirmation. Do not retrofit these rules to already observed AB3 outcomes.

## Comparison and units

Four arms, one immutable qualified image/bank/tokenizer, same user prompts and references:

| Arm | Enabled behavior |
|---|---|
| plain | No tool declaration, in-model runner off, S3 off, no runner guide. |
| lane | Built-in `code_interpreter`/`python` declaration; existing out-of-thinking compute lane and sandbox; in-model runner off, S3 off, no runner guide. |
| tool-full | Same tool declaration as lane; in-model runner and trace on, S3 runner-first execution with sandbox fallback on, exact pilot R+N trust-guide bytes. |
| tool-only | Same tool declaration and in-model runner/trace settings as tool-full, **no runner guide and S3 off**; calls use the sandbox. |

“No guide” means no request or environment fallback guide. Pin the actual rendered tool schema/description; tool-only must not silently remove the namespace. COMMIT=0 in all arms; other experimental knobs, including STATED, CHAIN, INLINE_MARK and SYNC_GRAPH, stay off. Compile budget 10,522. Tool-full versus tool-only varies the **joint guide+S3 addition**, not either component separately. Lane versus tool-only varies the in-model runner configuration. Only tool-full versus lane is confirmatory.

Use **60 distinct W9 task specifications**, balanced approximately to W9's family weights: add30 11, algo 5, bignum 3, div_big 8, mbpp 20, mul30 3, mul_big 10. Exclude all 11 pilot task IDs, in every seed. Within each family, sort remaining frozen W9 IDs by SHA256 of UTF-8 `AB3-v1:` followed by the ID, break ties by ID, and take the first required count. No admission, solve, length or favorable-outcome filtering. This deterministic reuse is a W9-subset replication, not a new unseen-task benchmark.

Run seeds **101,102,103**, temperature 1, top_p .95, top_k 20, concurrency 4, thinking enabled: 180 task/seed units per arm, **720 logical replies** total. All four arms use identical seed/task pairing. Split into three fixed family-balanced 20-task blocks. Arm boot orders are plain/lane/full/only, lane/full/only/plain, full/only/plain/lane across blocks; freeze within-block task/seed order using Python `random.Random(20261104).shuffle` on sorted task/seed tuples, shared across arms. Pin cache policy and use equally cold arm boots; no always-full-last policy. Publish all request/config/order hashes before execution. No outcome-selected replacement unit or favorable seed subset.

## Primary token endpoint

For tool-full F and lane L, let T be the count of the complete chronological returned native token-ID stream, **including forced/inserted engine and tool-result tokens exactly once**, excluding user/system/tool-description prompt tokens. Count all tool rounds, reasoning, code, final content and protocol tokens that appear in that returned stream. Do not concatenate only reasoning and final content: tool output may be absent from both. Keep separate model-sampled, forced/result, reasoning/content and prompt totals where authenticated; do not silently call them interchangeable. This matches the pilot's all-returned-ID endpoint, not its smaller `usage.completion_tokens` total.

Primary change is `100*(sum(T_F)/sum(T_L)-1)`, all 180 paired units retained. **Token gate passes only if point change ≤−10% and the upper endpoint of the two-sided 95% CI ≤−5%.**

CI: 20,000 paired hierarchical draws, Python RNG seed 20261104; iterate families in lexicographic order, resample that family's task IDs with replacement at its frozen task count, then draw three seeds with replacement within each selected task. Share each draw's task/seed indices across arms. Pool token sums with the original stratum counts, not equal family weights. Sort draws; endpoints are indices 500 and 19,499 (zero based). Freeze the implementation before outcomes. Secondary plain and tool-only contrasts use the same draws and are explicitly exploratory; no alternate contrast may rescue a failed primary.

Report per-family/per-seed totals, median task paired change and leave-one-task-out results. Each seed's primary point change must be ≤0. No single family or alternate estimator replaces the pooled gate. This is a precision gate, not a claimed power guarantee from 11 pilot units.

## Solves: an operational non-inferiority gate

Use the frozen W9 reference tests/sign-aware final-answer grader, blind to arm, and independently verify references before collection. A complete correct final answer must actually land; a correct intermediate, cut, tool value or hypothetical continuation is insufficient. Grade all 180 units/arm, including empty, cap and deadline outcomes.

For the primary pair, define **L** as lane-correct/full-incorrect and **G** as full-correct/lane-incorrect. Require all of:

- **L ≤3/180 (1.67%)**; gains never erase this separate loss cap.
- Full's solved count is at least lane's (`G≥L`).
- No task has a lane-only loss in two or three seeds; repeated task-specific regressions block even if aggregate counts pass.

Three isolated losses is a predeclared small finite-cohort tolerance for stochastic replies at T=1, instead of W9's zero-loss condition. The no-net-decrease and repeat-regression guard prevent a large systematic deficit from hiding behind gains. This is an **operational fixed-cohort non-inferiority criterion**, not a powered population non-inferiority test at a 1.67% margin. Report L/G, affected tasks and paired differences with task-cluster uncertainty; do not sell the three correlated seeds as 180 independent tasks. An all-tied bootstrap's zero-width solve interval is not evidence of zero population risk. In particular, even zero affected tasks among 60 only yields a conservative approximately 4.87% one-sided binomial upper bound under an independent exchangeable-task model. A public statistical population non-inferiority claim requires a separately designed larger study.

Audit all four arms' losses, but do not substitute an exploratory arm for tool-full after seeing scores.

## Safety, deadlines and censoring

Pin **r7 SHA256 `35f2bdf0243df7917b2b61098a1ff4a91bf1044854634912ab4800495db853fc`** for native cell adjudication: **zero wrong landed bytes**, including partial output; zero wrong intended spans/stale claims. Unknown, unparsed, unplaced or missing provenance is not a pass. Record-authenticated partial numeric prefixes are cut, never complete verified values. Cuts must be ≤1% of raw owned occurrences per arm, every landed byte authenticated and correct; whitespace-only output verifies nothing. Model-written engine labels are imitations, not owned writes.

**r7 does not validate S3 per-call reports.** Independently execute/reference-check every delivered successful tool result, S3 and sandbox/fallback alike, using exact call source, inputs, state where relevant, actual delivered bytes and the pinned transport normalization. Authenticate those bytes against the native returned stream. Errors must be correctly attributed failures, never asserted numeric results. No unverified tool result may enter a safety PASS. A coverage gap is INCONCLUSIVE, not “0 wrong.” Record all fallbacks and executor identities. No new checker driven by AB3 outcomes; any repair must use independent fixtures and a new disclosed protocol decision.

One **300-second end-to-end server budget per logical task** and 32,768 aggregate returned-token cap, identical across arms, including all tool rounds/execution. Do not reset 300 seconds for each tool call. Native receipts must preserve partial streams, IDs/counts, deadlines, cells and per-call records live. Client timeout at least 360 seconds; early client close, missing receipt or telemetry is an integrity failure, not an exclusion. Never complete a timed-out response by replay. A complete answer already landed before a cap can still solve, but remains censored.

Censored means any deadline, token cap, client truncation or incomplete logical response. For F/L require:

- Each arm has ≤18/180 censored units (10%); full has at most **three more** censored units than lane (≤1.67pp increase), and no seed has more than one extra full censor.
- On complete-in-both pairs, token reduction is ≥5%; report its selected-subset CI without replacing the primary cohort. After removing pairs with a cut in either arm, the token point change remains ≤0.
- Let `s_i=T_L−T_F`. Require **both** `sum_censored(s_i)/sum_all(s_i) ≤25%` and `sum_censored(max(s_i,0))/sum_all(max(s_i,0)) ≤25%`. Denominators must be positive. The second rule prevents negative contributions elsewhere from concealing concentration. Retain all original censored observations and report the complete-both denominator.

These guards directly address W9's 55.9% censor contribution. Failure cannot be waived using a favorable complete-only result. An expanded deadline/cap run would be a newly registered study, not a repair. Foreign traffic, image/config drift or unbalanced cache/concurrency invalidates confirmatory qualification of the affected run; retain originals, do not delete only affected outcomes. No causal latency/compute claim from token counts. Report end-to-end wall and tool execution separately.

## Decision, registration and public claim

**MEETS AB3** only if primary token, operational solve, safety, censoring, complete provenance and execution-integrity gates all pass. Failed numerical/safety gate: DOES NOT MEET. Missing required evidence: INCONCLUSIVE, never pass. No automatic serving-default change follows; JJ decides. Do not peek for an attractive interval or extend N after outcomes. All methods, scripts, guide/schema bytes and qualified-image hashes must be sealed before collection; later deviations remain visible and cannot become the confirmatory analysis.

A passing claim may describe **the built configuration combining the in-model runner, declared Python tool, S3 runner-first execution with sandbox fallback, and the trust guide**, versus the existing sandbox lane on this sampled W9 mix at T=1. It cannot isolate the namespace, guide, S3 or trace as the cause; cannot claim general Python-on-circuit support, elimination of the sandbox, training, or efficacy of answer-commit (off here; zero pilot firings). State N/tasks/seeds, all-returned-token accounting, CI, losses and censoring. Tool-only supplies a limited ablation of guide+S3 jointly. Report all four arms regardless of outcome. No instruction to shorten reasoning or answers is part of this treatment.

CPU preparation/checks only on <rank0-host>/<rank1-host> under <4 scopes, ≥6 GiB available, no launch during boot, MemoryMax=2G/MemorySwapMax=0, nice19, ionice-c3, one CPU/-j1, session QUIET pause. This spec requests no source edits, new image or unapproved GPU turn.
