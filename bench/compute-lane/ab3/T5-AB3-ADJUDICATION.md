# AB3 adjudication — DOES NOT MEET

2026-11-04. **DOES NOT MEET [T5-AB3-SPEC.md](T5-AB3-SPEC.md), SHA256 `81b34c1a79a754efca01ebee85f47c74490a029f23e4e0533799205b60dd524e`.** The primary token CI fails and the recorded-but-not-delivered trace newline fails safety/provenance. Solves and censoring pass their registered gates. Keep the experiment opt-in; no serving-default change follows. No exclusions, favorable-seed selection, outcome-driven oracle repair or alternate primary endpoint were used.

## Freeze and independent reproduction

**Freeze verified.** Registration was posted at 08:55:00.927928Z. `AB3-FREEZE.json` was sealed at **09:18:35Z**, SHA256 `93417d27c61d8c52069ec1b33cb83d79f26e6cd3fdc0b8d33127e784b6315e7a`; local commit **`86dda37c981d873142cc89b967a98adc1b33137a`** is dated 09:18:54Z. The earliest original receipt's `sent_unix` is **09:26:06.045929Z**, after both seals. This is pre-first-request registration, not merely pre-analysis registration.

Own node CPU audit rehashed all **152 frozen files in each of seven copies** (kit plus six turns), including sealed analysis `26d7bec86ca8f860a26109bc66f50821a7fdbacf682c375572e6e2c03cf6acca`. It independently reconstructed the salted selection from the original 450-task W9 file, excluded all eleven pilot IDs, checked all 60 references against their original task rows, and verified counts 11/5/3/8/20/3/10, seeds 101–103, shared shuffled order, twelve boot configurations and request templates. All **720 original receipts** have the required task/seed/arm key, original grade, native ID hash and count. Actual requests equal their frozen templates after the prescribed model substitution and `return_token_ids=true, deadline_s=300` additions. Image IDs/engine revision match the freeze at every boot.

I recomputed the token sums and bootstrap directly from those receipts, without importing the sealed analysis. The independent implementation uses the registered 20,000 stratified task/paired-seed draws, RNG 20261104 and endpoints 500/19,499. It reproduces the reported primary and secondary intervals. [Receipt/freeze audit](t5-ab3-adjudication/ab3-adjudicate-read4/audit/summary.json), [independent statistics and sizing](t5-ab3-adjudication/ab3-adjudicate-stats1/<rank1-host>/audit/summary.json).

| Arm, 180 units each | All returned native IDs | Completion IDs / inserted result IDs | Solved | Deadlines |
|---|---:|---:|---:|---:|
| plain | 301,009 | 301,009 / 0 | 170 | 7 |
| lane | 65,499 | 62,094 / 3,405 | 172 | 0 |
| tool-full | 56,872 | 53,060 / 3,812 | 174 | 0 |
| tool-only | 65,558 | 62,224 / 3,334 | 172 | 0 |

“Completion” is the receipt's completion count; it is not interchangeable with the registered all-returned-ID endpoint. Prompt tokens are excluded. Neither the completion field nor reasoning-plus-content concatenation replaces the native stream containing inserted tool results.

## Frozen gates

| Requirement | Independent result | Ruling |
|---|---|---|
| Full/lane reduction ≥10%; two-sided CI upper ≤−5% | **−13.1712%, CI95 [−26.0771%, +0.9785%]** | **FAIL**: point passes, CI fails |
| Each seed token change ≤0 | −10.8282%, −14.9925%, −13.7519% | PASS |
| At most three lane-only solve losses, no net solve decrease, no repeated-loss task | **2 losses, 4 gains**, 174 versus 172; no task loses twice | PASS, operational fixed-cohort rule |
| Full/lane censor limits; complete-both reduction ≥5%; no-cut reversal | 0/180 both; all 180 complete; complete-both and no-cut change −13.1712%; net/gross censored contribution both 0% with positive denominators | PASS |
| Zero wrong landed bytes/intended errors and complete visible provenance; unknown not a pass | r7: **86 ok, 1 not-in-text**, zero wrong-value/span, cut, unknown or unparsed | **FAIL**: unplaced complete record |
| Every tool result reference-checked and authenticated | All **444** saved independent CPython re-runs bind to the exact original call source and delivered native result bytes; no mismatch or missing call | PASS for these tool reports |
| Cohort/config/input provenance | 720 receipts, hashes, references, requests, images and order reconcile | PASS for bookkeeping/config identity; does not override placement failure |

The two losses are **div_big-030 seed 103** and **div_big-054 seed 103**. Gains are div_big-039 seed 103, div_big-048 seeds 101/103 and mbpp-240 seed 101. The paired solve-difference task-cluster CI is [−2.22,+5.00] percentage points. It is descriptive: the registered solve gate is operational, not a powered population non-inferiority test.

Median per-task token change is −6.8010%; leave-one-task-out ranges from −15.78% to −6.64%. Removing influential mbpp-580 produces the weakest reduction, but it remains in every primary analysis. Algo increases tokens by 13.0%; MBPP decreases them by 13.5%. An independent linearized variance decomposition of the frozen bootstrap attributes approximately **95.6% to MBPP** and **1.3% to algo**. Thus MBPP primarily drives the interval; grouping both families together conceals that imbalance.

The per-call reference audit covers lane 144 sandbox calls, full 159 calls (**89 S3, 70 sandbox fallback**), and only 141 sandbox calls. I checked the existing re-run stdout/stderr/status, source hashes and a 444-key bijection against original receipts, then decoded the native stream independently to authenticate each delivered result and preceding call source. This is an independent audit of the saved CPython reference runs, **not a claim that I newly executed another 444 calls**. [Bindings](t5-ab3-adjudication/ab3-adjudicate-read4/audit/tool-bindings.json).

## Safety record: retain FAIL

Full, block 1, bignum-017 seed 101 records:

```text
[exact, computed by the engine: 2^66 = 73786976294838206464]\n
```

The native cell is complete: `ids=23, landed=23`; independent decoding finds the full bytes, including LF, at native byte offset 61. Independently, `2**66` equals the displayed integer. But the serialized `content` contains only the label without that LF before subsequent content. The exact recorded write is absent. [Native-versus-content proof](t5-ab3-adjudication/ab3-adjudicate-read4/audit/safety-native-proof.json).

This is a **projection/placement failure, not wrong arithmetic**, and it is not a deadline cut. The frozen rule explicitly says unplaced or missing provenance is not a pass and pins r7. I therefore retain r7's **not-in-text** verdict and fail this safety subgate. I do not strip the expected newline, change the oracle, declare whitespace irrelevant retrospectively, or convert the record to ok because its number is correct. A future projection repair needs independently constructed fixtures and a newly qualified image; it cannot turn AB3 into a passing run. Implementer B's “provenance PASS” is valid only for receipt/config bookkeeping, not the complete output-placement contract.

## Cache and execution interpretation

Prefix caching was enabled through the common image default; no arm-specific cache override appears in the twelve recorded settings. Each boot's **earliest request has zero cached prompt tokens**, consistent with equally cold starts. Subsequent cached totals differ: lane/only 47,632 each, full 61,856, plain 5,440. Actual request prompts add exactly 306 namespace tokens in lane/only and another 112 guide tokens in full. The checked frozen configurations, cold first requests and shared order supply evidence of balanced policy; unequal cached totals alone do not violate the spec.

This does **not** prove caching is irrelevant to execution or latency. Cached prompt tokens are excluded from the token endpoint; that accounting fact alone would not establish integrity. I find no additional cache-policy violation in the supplied run evidence. No wall-time/compute-speedup claim is adjudicated here.

## Descriptive secondary result against plain

The spec permits these comparisons as exploratory and forbids using them to rescue the primary:

| Contrast | Observed token change | CI95 |
|---|---:|---:|
| full/plain | −81.1062% | [−84.6613%, −76.6489%] |
| lane/plain | −78.2402% | [−83.3263%, −71.7480%] |
| only/plain | −78.2206% | [−83.2960%, −71.7412%] |
| only/lane | +0.0901% | [−1.7228%, +2.1873%] |
| full/only | −13.2493% | [−26.0399%, +0.7989%] |

Tool-only/lane shows no detected difference; **equivalence was not tested**. Full/only does not establish a benefit of guide+S3, let alone separate their effects. “Everything gained comes from guide+S3” is too strong for an interval crossing zero. The tool namespace is already present in lane, so full/lane cannot isolate namespace efficacy.

**Public wording supported:** “In a descriptive secondary comparison on 60 W9 task specifications at three paired sampling seeds, the built configuration combining the in-model runner, declared Python tool, S3 runner-first execution with sandbox fallback, and the trust guide used **81.1% fewer returned tokens** than plain generation (CI95: **76.6%–84.7% fewer**). It solved 174/180 versus 170/180. Plain had seven deadlines, all retained, versus none in that configuration. The existing sandbox tool lane also used **78.2% fewer tokens** than plain. The experiment did not meet its primary comparison or safety qualification.”

Those solve totals do not mean zero individual regressions: full/plain has **three losses and seven gains**; lane/plain and only/plain each have five losses and seven gains. The seven plain-censored pairs contribute **33.16%** of full/plain's net token reduction. On the selected 173 complete-both pairs, full/plain is −74.58%. The registered 25% censor-contribution gate applies to full/lane, not retrospectively to this secondary contrast, but the caveat belongs beside any public plain comparison. These are fixed-budget observed-token results, not uncensored time-to-solution or a claim that the circuit alone produced the approximately 80% reduction. Sandbox fallback remains built and used; answer-commit is off. No training, general Python-on-circuit, component-isolation or safe-default claim is supported.

## Follow-up: justified, not a continuation of AB3

First repair and independently qualify the placement path. Preserve AB3's verdict. A new prospective study is reasonable because the effect is potentially useful, but the pilot's 25% estimate and AB3's 13% estimate are imprecise. More seeds on these same tasks do not resolve the dominant task variance.

For a **new full/lane confirmation** retaining three paired seeds and the same family weights, the observed log-ratio bootstrap SD is **0.0812817** at 60 tasks; the ratio is **0.868288**. An approximate 80%-power calculation for the registered upper-ratio threshold 0.95 is:

`N = ceil(60 * [(z(.975)+z(.80))*0.0812817 / log(0.95/0.868288)]^2) = 385 distinct tasks`.

Thus **385 new independent task specifications is the estimated minimum**, with **420 recommended** to preserve the sevenfold 60-task family allocation exactly: 1,260 pairs /2,520 replies for two arms. For approximately 90% primary-token power, the estimate is 515 tasks; round to 540. Do not subtract AB3's observed 60 or append requests to this study. These are planning estimates assuming the same true effect, variance and task distribution; they neither guarantee every gate nor population solve non-inferiority. Use new reference-checked instances without admission/outcome filtering. If MBPP/algo are oversampled, preregister weights and recompute power for that design rather than changing the target mix silently.

For **guide versus S3 attribution**, use a fresh **2×2 guide on/off × S3 on/off**, with the declared tool, runner/trace, budgets and all other settings fixed. Current full/only observes only opposite corners. First require CPU/tool-transcript identity checks for S3 versus sandbox, then measure guide main effect, S3 execution effect and their interaction. My minimum feasibility recommendation is **60 new distinct tasks ×3 seeds ×4 arms =720 replies**, explicitly exploratory. It is **not** a powered confirmation of all three effects. AB3 supplies no separate effect sizes or variances from which to defend a confirmatory minimum for those contrasts. A proxy calculation assuming each had AB3's joint 13.17% effect/variance and Bonferroni control for three two-sided contrasts gives **514 tasks**, rounded to **540**, but that assumption is unestablished and must not be advertised as demonstrated ablation power. Freeze effects/margins, multiplicity and sizing in a separate protocol before any such confirmation. Retain lane as a fifth arm only if the original full/lane question must also be tested.

I favor the smaller mechanistic ablation and projection qualification before committing the larger confirmation budget. Neither recommendation schedules or authorizes a GPU turn.

All own jobs ran on <rank0-host>/<rank1-host> under admitted scope/memory/boot checks, MemoryMax=2G/MemorySwapMax=0, nice19, ionice-c3, one CPU and QUIET session pause, and settled in-turn. Early private reader attempts failed on script syntax and omitted prescribed transport fields; they were corrected without changing study data or acceptance rules. Standards/check paths remain in LEDGER-T5.md; no engine source edit, build, oracle change, model request, push or <other-host> was needed.
