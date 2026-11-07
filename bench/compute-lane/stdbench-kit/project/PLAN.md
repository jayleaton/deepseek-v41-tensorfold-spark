# Phase 2: normal public benchmarks, CPU-prepared only

Status: **NO GPU GO**. Nothing is queued. Start only after JJ explicitly accepts the current guide×S3 ablation as promising and authorizes phase 2. This prep does not adjudicate that ablation. Read [PUBLISHED](PUBLISHED.md) before interpreting historical numbers: no exact-1017 scores were found for these short suites. This is a useful paired local experiment, not a reproduction of DeepSeek's scorecard.

## Suites and budget

Use unchanged public test problems, zero-shot prompts, one independent answer per task and seed. No guide, calculator instruction, custom arithmetic families, prompt selection based on outcomes, repair or execution feedback. HumanEval asks for a complete Python function in a fence, includes the original signature/docstring but no hidden tests. Keep all tasks, seeds, prompts and body hashes frozen before GO.

| Suite | Tasks × seeds per arm | Mean output planning allowance | At 30–49 returned tokens/s, plus 5s/problem | Grading |
| --- | ---: | ---: | ---: | --- |
| GSM8K test | 48 × 2 = 96 | 1,500 | 0.95–1.47h | Exact rational/integer final answer after #### or boxed extraction; compare official #### gold |
| MATH-500 test | 20 × 2 = 40 | 4,000 | 0.96–1.54h | Conservative boxed exact equivalence; unresolved symbolic answers require blinded manual adjudication |
| HumanEval original | 24 × 2 = 48 | 3,000 | 0.88–1.40h | Original `test` + `check(entry_point)`; pass@1, no plus label |

Throughput evidence is the **actual AB3 plain rank log**, not the older ~100tok/s marketing/short-code measurements: <rank0-host> `~/work/dsv41-bench/out/imr-measure-20261104T091946Z/ab3-b1-1-plain/rank0.log` includes 29.8, 30.7, 31.5 and 48.6tok/s. This is workload-dependent, not a verified throughput for these datasets. Estimate formula: `n*seeds*(expected_output/tps + TTFT) / 3600`, add 10–15min for each boot and recovery. Total ~3–5h per arm across three suites, not 1–2h for the entire battery.

These are expected, not worst-case durations. Common max_tokens=8192, per-request timeout=300s, server deadline=270s, concurrency=1. Capped output can affect accuracy; report it. Before the full battery after GO, use the first four already-selected tasks per suite and seed 101 (count them in the final results), with balanced arm order, to validate mean length/TTFT and lane activation. If projection exceeds two hours per suite/arm, reduce only by a predeclared hash-order prefix, equally in both arms, before further responses are inspected. Record the retained IDs and why. Never silently stop one arm early and analyze survivors. A suite wall budget hit marks the run incomplete; restore service and schedule the missing matching block only after a further authorized GPU turn. A 300s timeout per sample gives worst cases of 8h/3.3h/4h respectively; do not promise a hard two-hour bound without the pilot.

Select task IDs by SHA256(`stdbench-v1:` + original stable ID), take first n. No answer-dependent or W9/AB3 overlap exclusion. Freeze subset hashes. Seeds **101,102**; subset/order seed **20261104**. Same task order and seed schedule in paired boots. Duplicate/missing pairs fail the paired report. GSM8K IDs are original test row indices, MATH its unique_id, HumanEval task_id. All source revisions and SHA256s are in the kit lock. Public fetch is HTTPS only and hashes fail closed.

## Arms and request contract

Read-only AB3 source: `~/work/ab3-20261104/kit2/ab3_kit2.py`, `~/work/dsv41-bench/arms-ab3-b1-*-*.json`, rank config files. AB3 plain/lane both had `TF_DSV41_COMPUTE_LANE=1` and `TF_DSV41_RUNNER_COMMIT=0`; request-level code_interpreter distinguished them. For JJ's explicit OFF/ON comparison, phase 2 uses:

| Setting | plain | lane-only |
| --- | --- | --- |
| TF_DSV41_COMPUTE_LANE | 0 | 1 |
| TF_DSV41_RUNNER | 0 | 0 |
| TF_DSV41_COMPUTE_LANE_RUNNER (S3) | 0 | 0 |
| TF_DSV41_RUNNER_COMMIT | 0 | 0 |
| TF_DSV41_RUNNER_STATED / INLINE_CHAIN / INLINE_MARK / SYNC_GRAPH | 0 | 0 |
| TF_DSV41_RUNNER_FORMS | fence,repl,inline | fence,repl,inline |
| runner_guide in HTTP body | absent | absent |
| tools in HTTP body | absent | [{"type":"code_interpreter"}] |
| trace form | absent | absent |

Remove inherited runner circuit/bank/backend paths and all trace/guide overrides when deriving fresh rank envs; explicitly record final effective env on both ranks. Do not edit existing configs. All ordinary G engine optimizations, model pack, expert top-p .85, context, draft settings, clocks and calibration policy are held identical; “other features OFF” means the experimental compute/runner/guide/S3/trace stack, not every serving optimization. Both arms use the same qualified image, chosen after the ablation; AB3 historical engine SHA `3bf584af1c5407006f154e0c70d05c6f495d0487` is a reference, not an instruction to roll back. Freeze selected image IDs and revision before GO run.

Model `DeepSeek-V4.1-Flash-TF`; chat-template thinking enabled; explicit reasoning_effort=high (stable local comparison, not published max). T=1, top_p=.95, top_k=20 match AB3 sampling; no relay-added thinking budget. Both arms retain identical cap/deadlines. No client-driven tool loop or runner; the declaration enables the native lane. Reject unexpected external tool_calls as protocol failures. The tool declaration is an unavoidable prompt/protocol difference from AB3 and must be disclosed; do not claim a pure identical-prompt kernel comparison. Optional later negative-control: lane disabled plus identical tool declaration, separately authorized and budgeted.

Activation evidence after GO: native lane call/owned/abstain counters or request logs for lane, zero native compute calls for plain, zero runner/S3/trace activity in both. An all-abstain or never-triggered lane run is reported as such, never called an effective lane test. Do not enable trace merely to count activation. The current client saves returned content/reasoning separately and response usage; internal hidden computation not exposed by API is not invented as returned tokens.

## Metrics and paired analysis

Primary: within-suite accuracy difference (lane−plain) over all scheduled task×seed responses, timeout/protocol errors scored incorrect; manual-review fraction separately visible. Do not finalize MATH accuracy while unresolved reviews remain. Secondary: mean API completion tokens/problem including reasoning once, final/reasoning breakdown where provided, mean and median monotonic wall seconds/problem (includes HTTP/TTFT/lane work), timeout and finish=length rates, lane calls where available. Missing usage stays null; report token coverage and paired-complete token results separately. Never use characters/4. Include failed requests' elapsed time, label token totals as a lower bound when usage is missing. No rewards for timeouts returning fewer tokens.

Pair by suite + task ID + generation seed. Exact McNemar on discordant correctness, with seeds reduced to a predeclared task-level outcome for a valid independent-task exact test; per-sample version is descriptive because seeds cluster. Primary uncertainty is task-cluster bootstrap (20,000 draws, RNG 20261104), retaining both seeds in each sampled task; 95% percentile CI for accuracy delta, ratio of paired sums and paired mean differences for tokens/time. Bootstrap each suite independently; no averaged score pretending the subsets equal full benchmarks. Also show gains/losses, raw denominators and Wilson arm intervals. Apply Holm adjustment across the three primary suite accuracy tests if making significance claims. Small subsets have low power; unchanged accuracy is not noninferiority. Predeclare a practical success gate: no net accuracy decrease and lower tokens/time with CIs, accompanied by all isolated losses; treat this as exploratory, not a powered acceptance test.

DeepSeek figures appear as separate historical reference rows with model/checkpoint, suite and shot differences. Exact 1017 field is “not published”; never compute a retention ratio to a base or differently dated subset. No claims about published tokens/time (not supplied for these exact comparisons).

## ORDER layout (proposal only; never appended by this prep)

Partition each suite's hash-selected tasks into two balanced blocks, both seeds in each. Proposed six GPU turns, each at most ~1–2h **total** on expectation: GSM block1 plain→lane; MATH block1 lane→plain; HumanEval block1 plain→lane; GSM block2 lane→plain; MATH block2 plain→lane; HumanEval block2 lane→plain. Each block/arm is a separately booted env, equal calibration and clocks, randomized request order shared with its partner. If a turn exceeds the allowed window, split at a predeclared task boundary; preserve paired task groups. Across two blocks this gives ABBA/BAAB order per suite. No cross-arm prefix cache sharing; record boot duration separately from per-problem wall time.

At actual GO, operator runs queue turn-preflight on newly written launch scripts and helpers, appends only that turn at the tail, acquires the existing GPU protocol only at head, and restores G after every turn before releasing. Do not reserve six future turns blindly. This prep supplies no script that mutates ORDER/locks, boots ranks, changes service or automatically promotes itself on a GO file.

## CPU verification / delivery

Kit lives at <rank0-host> `~/work/stdbench/`; source mirror is local `stdbench/kit/`. Dry run uses only fabricated API responses, including wrong, truncated, timeout and missing-usage cases. Dataset fetch/normalization does not make inference requests. See kit README for exact commands and proof receipt. CPU verification occurs on <rank1-host> via <rank0-host> because <rank0-host> already has three user scopes counting init and Chrome. Every job stays in one MemoryMax=2G/MemorySwapMax=0 user scope, nice19/ionice3, outer timeout7200, internal timeout1200, single worker. Check <rank0-host> QUIET/ORDER plus both memory floors at admission and during work. No <dev-host> tests, installs or heavy compute. No engines or repository builds.
