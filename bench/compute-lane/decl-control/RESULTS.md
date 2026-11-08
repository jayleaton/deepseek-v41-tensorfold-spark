# Declaration-control A1 results — NEEDS-T5

> Export navigation: paths marked omitted retain their historical names. Consult the
> [raw-receipt hash inventory](../RAW-RECEIPTS.sha256) for excluded receipts and duplicate destinations;
> historical documents without an inventory entry are not bundled or hash-attested here.

Base protocol SHA256: `adb05167fff21469ee7251555cb243f1f109c4d8e3de237fdcaa59ada8eb55e6`.

Amendment A1 SHA256: `e7308621c645aef4402924a1b47c1a52e05baabe00239f915011f8431bbd0e89`.

All552 scheduled replies are retained (184 per arm): GSM8K48, MATH-50020 and original HumanEval24, seeds101/102. These public tasks and seeds were already observed. This fixed-cohort configuration study has no pooled suite score or population non-inferiority/equivalence verdict.

Served model: DeepSeek-V4.1-Flash-1017, revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, EXL3 2.80bpw quantization. Engine/image revision `3a429d6f122d9eb2900c43e622933d311e4be44f` (`r-3a429d6f122d`). P has no declaration and execution OFF; D has the explicit Python declaration and execution OFF; L has the identical declaration/body and execution ON.

Thinking/high, temperature1, top_p.95, top_k20, concurrency1; aggregate returned native-ID cap8192, server deadline270s and client wait360s. Three turns use P→D→L, D→L→P, L→P→D. Eighteen boot diagnostics are excluded from endpoints. Errors, timeouts, policy terminations and capped answers remain scheduled observations.

Independent retained-evidence audit qualified: **yes**. Environment audit qualified: **yes**. Any evidence failure withholds qualified mechanism/efficacy attribution without deleting results.

A1 supports only “CPU-matched declaration prompts, identical request bodies and matching served prompt counts” when the corresponding audit passes. Same-length server substitutions or an unmodelled rendering branch remain possible. Directly verified served-ID equality and fully isolated execution-only causality are not established.

Strict accuracy is primary. Returned native IDs and client end-to-end wall are descriptive secondary endpoints; wall includes generation and tool rounds. Diagnostics, boot/restoration and offline references are separate. Input+returned accounting is neither billing nor GPU-work equivalence.

| Suite | Arm | Strict solved/scheduled | Accuracy % | Returned IDs mean | ID coverage | Wall seconds mean | Timeouts | Caps | Errors | Policy terminations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | P | 54/96 | 56.2500 | 280.7188 | 96/96 | 4.6790 | 0 | 0 | 0 | 0 |
| gsm8k | D | 52/96 | 54.1667 | 214.1979 | 96/96 | 3.5259 | 0 | 0 | 0 | 0 |
| gsm8k | L | 52/96 | 54.1667 | 214.1979 | 96/96 | 3.5213 | 0 | 0 | 0 | 0 |
| humaneval | P | 47/48 | 97.9167 | 628.4792 | 48/48 | 9.9536 | 0 | 1 | 0 | 0 |
| humaneval | D | 48/48 | 100.0000 | 456.5833 | 48/48 | 6.8882 | 0 | 0 | 0 | 0 |
| humaneval | L | 48/48 | 100.0000 | 456.5833 | 48/48 | 6.9172 | 0 | 0 | 0 | 0 |
| math500 | P | 40/40 | 100.0000 | 754.4000 | 40/40 | 11.0493 | 0 | 0 | 0 | 0 |
| math500 | D | 40/40 | 100.0000 | 589.0500 | 40/40 | 8.4520 | 0 | 0 | 0 | 0 |
| math500 | L | 40/40 | 100.0000 | 589.0500 | 40/40 | 8.4890 | 0 | 0 | 0 | 0 |

GSM strict extraction and original HumanEval official tests plus check(entry_point) use sealed grade3. HumanEval candidates run only inside the actual pinned <rank1-host> Jail. MATH labels come from all120 blinded answers in namespace decl-control-v1: two independent reviews, third for disagreements; unresolved answers retain adverse/favorable bounds. Reviewer types below explicitly distinguish agents from humans.

| MATH reviewer | Reviewer type |
| --- | --- |
| a1-math-independent-a | agent |
| a1-math-independent-b | agent |

MATH disputed candidates: 0; third-reviewed: 0. A reviewer labelled agent is an agent review, with no human-review claim.
Paired task clusters retain both seeds. Exact two-sided task sign-flip tests use Holm across the six D−P/L−D primary tests. All contrasts/metrics share20000 task bootstrap draws, RNG20261104, sorted IDs, percentile indices500/19499. L−P and GSM sensitivity are secondary/marginal. Nonsignificance is not equivalence.

| Suite | Contrast | Role | Accuracy Δ pp | CI95 pp | Gains/losses | Exact p | Holm family6 p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | D-P | planned primary | -2.0833 | [-8.3333, 4.1667] | 6/8 | 0.7539 | 1.0000 |
| gsm8k | L-D | planned primary | 0.0000 | [0.0000, 0.0000] | 0/0 | 1.0000 | 1.0000 |
| gsm8k | L-P | secondary/marginal | -2.0833 | [-8.3333, 4.1667] | 6/8 | 0.7539 | unavailable |
| humaneval | D-P | planned primary | 2.0833 | [0.0000, 6.2500] | 1/0 | 1.0000 | 1.0000 |
| humaneval | L-D | planned primary | 0.0000 | [0.0000, 0.0000] | 0/0 | 1.0000 | 1.0000 |
| humaneval | L-P | secondary/marginal | 2.0833 | [0.0000, 6.2500] | 1/0 | 1.0000 | unavailable |
| math500 | D-P | planned primary | 0.0000 | [0.0000, 0.0000] | 0/0 | 1.0000 | 1.0000 |
| math500 | L-D | planned primary | 0.0000 | [0.0000, 0.0000] | 0/0 | 1.0000 | 1.0000 |
| math500 | L-P | secondary/marginal | 0.0000 | [0.0000, 0.0000] | 0/0 | 1.0000 | unavailable |


| Suite | Contrast | Unresolved by arm | Per-arm rate bounds | Adverse Δ pp | Favorable Δ pp | Adverse Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| gsm8k | D-P | {"base":0,"comparison":0} | {"base":[0.5625,0.5625],"comparison":[0.5416666666666666,0.5416666666666666]} | -2.0833 | -2.0833 | 1.0000 |
| gsm8k | L-D | {"base":0,"comparison":0} | {"base":[0.5416666666666666,0.5416666666666666],"comparison":[0.5416666666666666,0.5416666666666666]} | 0.0000 | 0.0000 | 1.0000 |
| gsm8k | L-P | {"base":0,"comparison":0} | {"base":[0.5625,0.5625],"comparison":[0.5416666666666666,0.5416666666666666]} | -2.0833 | -2.0833 | unavailable |
| humaneval | D-P | {"base":0,"comparison":0} | {"base":[0.9791666666666666,0.9791666666666666],"comparison":[1.0,1.0]} | 2.0833 | 2.0833 | 1.0000 |
| humaneval | L-D | {"base":0,"comparison":0} | {"base":[1.0,1.0],"comparison":[1.0,1.0]} | 0.0000 | 0.0000 | 1.0000 |
| humaneval | L-P | {"base":0,"comparison":0} | {"base":[0.9791666666666666,0.9791666666666666],"comparison":[1.0,1.0]} | 2.0833 | 2.0833 | unavailable |
| math500 | D-P | {"base":0,"comparison":0} | {"base":[1.0,1.0],"comparison":[1.0,1.0]} | 0.0000 | 0.0000 | 1.0000 |
| math500 | L-D | {"base":0,"comparison":0} | {"base":[1.0,1.0],"comparison":[1.0,1.0]} | 0.0000 | 0.0000 | 1.0000 |
| math500 | L-P | {"base":0,"comparison":0} | {"base":[1.0,1.0],"comparison":[1.0,1.0]} | 0.0000 | 0.0000 | unavailable |


| Suite | Contrast | Metric | Complete pairs | Full-cohort ratio | Ratio CI95 | Mean Δ | Δ CI95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | D-P | returned_ids | 96/96 | 0.7630 | [0.6094, 0.9778] | -66.5208 | [-145.0521, -4.1146] |
| gsm8k | D-P | initial_prompt_ids | 96/96 | 4.1607 | [3.9733, 4.3580] | 306.0000 | [306.0000, 306.0000] |
| gsm8k | D-P | processed_input_ids | 96/96 | 4.1607 | [3.9733, 4.3580] | 306.0000 | [306.0000, 306.0000] |
| gsm8k | D-P | prompt_plus_returned | 96/96 | 1.6343 | [1.3151, 2.1328] | 239.4792 | [160.9479, 301.8854] |
| gsm8k | D-P | wall_seconds | 96/96 | 0.7536 | [0.6009, 0.9699] | -1.1531 | [-2.5000, -0.0915] |
| gsm8k | L-D | returned_ids | 96/96 | 1.0000 | [1.0000, 1.0000] | 0.0000 | [0.0000, 0.0000] |
| gsm8k | L-D | initial_prompt_ids | 96/96 | 1.0000 | [1.0000, 1.0000] | 0.0000 | [0.0000, 0.0000] |
| gsm8k | L-D | processed_input_ids | 96/96 | 1.0000 | [1.0000, 1.0000] | 0.0000 | [0.0000, 0.0000] |
| gsm8k | L-D | prompt_plus_returned | 96/96 | 1.0000 | [1.0000, 1.0000] | 0.0000 | [0.0000, 0.0000] |
| gsm8k | L-D | wall_seconds | 96/96 | 0.9987 | [0.9941, 1.0021] | -0.0046 | [-0.0172, 0.0080] |
| gsm8k | L-P | returned_ids | 96/96 | 0.7630 | [0.6094, 0.9778] | -66.5208 | [-145.0521, -4.1146] |
| gsm8k | L-P | initial_prompt_ids | 96/96 | 4.1607 | [3.9733, 4.3580] | 306.0000 | [306.0000, 306.0000] |
| gsm8k | L-P | processed_input_ids | 96/96 | 4.1607 | [3.9733, 4.3580] | 306.0000 | [306.0000, 306.0000] |
| gsm8k | L-P | prompt_plus_returned | 96/96 | 1.6343 | [1.3151, 2.1328] | 239.4792 | [160.9479, 301.8854] |
| gsm8k | L-P | wall_seconds | 96/96 | 0.7526 | [0.6000, 0.9681] | -1.1577 | [-2.5031, -0.0999] |
| humaneval | D-P | returned_ids | 48/48 | 0.7265 | [0.6059, 1.0035] | -171.8958 | [-390.2708, 1.5625] |
| humaneval | D-P | initial_prompt_ids | 48/48 | 2.8635 | [2.5938, 3.2028] | 306.0000 | [306.0000, 306.0000] |
| humaneval | D-P | processed_input_ids | 48/48 | 2.8635 | [2.5938, 3.2028] | 306.0000 | [306.0000, 306.0000] |
| humaneval | D-P | prompt_plus_returned | 48/48 | 1.1692 | [0.9299, 1.5907] | 134.1042 | [-84.2708, 307.5625] |
| humaneval | D-P | wall_seconds | 48/48 | 0.6920 | [0.5640, 0.9889] | -3.0654 | [-7.0011, -0.0699] |
| humaneval | L-D | returned_ids | 48/48 | 1.0000 | [1.0000, 1.0000] | 0.0000 | [0.0000, 0.0000] |
| humaneval | L-D | initial_prompt_ids | 48/48 | 1.0000 | [1.0000, 1.0000] | 0.0000 | [0.0000, 0.0000] |
| humaneval | L-D | processed_input_ids | 48/48 | 1.0000 | [1.0000, 1.0000] | 0.0000 | [0.0000, 0.0000] |
| humaneval | L-D | prompt_plus_returned | 48/48 | 1.0000 | [1.0000, 1.0000] | 0.0000 | [0.0000, 0.0000] |
| humaneval | L-D | wall_seconds | 48/48 | 1.0042 | [0.9993, 1.0078] | 0.0290 | [-0.0042, 0.0683] |
| humaneval | L-P | returned_ids | 48/48 | 0.7265 | [0.6059, 1.0035] | -171.8958 | [-390.2708, 1.5625] |
| humaneval | L-P | initial_prompt_ids | 48/48 | 2.8635 | [2.5938, 3.2028] | 306.0000 | [306.0000, 306.0000] |
| humaneval | L-P | processed_input_ids | 48/48 | 2.8635 | [2.5938, 3.2028] | 306.0000 | [306.0000, 306.0000] |
| humaneval | L-P | prompt_plus_returned | 48/48 | 1.1692 | [0.9299, 1.5907] | 134.1042 | [-84.2708, 307.5625] |
| humaneval | L-P | wall_seconds | 48/48 | 0.6949 | [0.5680, 0.9881] | -3.0364 | [-6.9430, -0.0713] |
| math500 | D-P | returned_ids | 40/40 | 0.7808 | [0.6959, 0.8745] | -165.3500 | [-312.1750, -55.0000] |
| math500 | D-P | initial_prompt_ids | 40/40 | 4.1975 | [3.4305, 5.3068] | 306.0000 | [306.0000, 306.0000] |
| math500 | D-P | processed_input_ids | 40/40 | 4.1975 | [3.4305, 5.3068] | 306.0000 | [306.0000, 306.0000] |
| math500 | D-P | prompt_plus_returned | 40/40 | 1.1655 | [0.9955, 1.5705] | 140.6500 | [-6.1750, 251.0000] |
| math500 | D-P | wall_seconds | 40/40 | 0.7649 | [0.6666, 0.8749] | -2.5973 | [-5.0490, -0.7356] |
| math500 | L-D | returned_ids | 40/40 | 1.0000 | [1.0000, 1.0000] | 0.0000 | [0.0000, 0.0000] |
| math500 | L-D | initial_prompt_ids | 40/40 | 1.0000 | [1.0000, 1.0000] | 0.0000 | [0.0000, 0.0000] |
| math500 | L-D | processed_input_ids | 40/40 | 1.0000 | [1.0000, 1.0000] | 0.0000 | [0.0000, 0.0000] |
| math500 | L-D | prompt_plus_returned | 40/40 | 1.0000 | [1.0000, 1.0000] | 0.0000 | [0.0000, 0.0000] |
| math500 | L-D | wall_seconds | 40/40 | 1.0044 | [0.9993, 1.0081] | 0.0370 | [-0.0038, 0.0851] |
| math500 | L-P | returned_ids | 40/40 | 0.7808 | [0.6959, 0.8745] | -165.3500 | [-312.1750, -55.0000] |
| math500 | L-P | initial_prompt_ids | 40/40 | 4.1975 | [3.4305, 5.3068] | 306.0000 | [306.0000, 306.0000] |
| math500 | L-P | processed_input_ids | 40/40 | 4.1975 | [3.4305, 5.3068] | 306.0000 | [306.0000, 306.0000] |
| math500 | L-P | prompt_plus_returned | 40/40 | 1.1655 | [0.9955, 1.5705] | 140.6500 | [-6.1750, 251.0000] |
| math500 | L-P | wall_seconds | 40/40 | 0.7683 | [0.6682, 0.8797] | -2.5603 | [-4.9944, -0.7082] |

Missing all-round input accounting is never zero. Any missing reply withholds that comparison’s full-cohort prompt+returned ratio/CI and total-token-saving claim. Known totals and initial counts remain separate. Selected accounting-complete sensitivities retain original task clusters and are not representative of missing cases.

| Suite | Contrast | Selected metric | Selected ratio | Selected CI95 | Selected mean Δ | Selected Δ CI95 |
| --- | --- | --- | --- | --- | --- | --- |


| Suite | Contrast | Metric | Uncensored complete pairs | Uncensored ratio | Net savings | Positive savings | Censored/net % | Censored/positive % | Censor reversal | >25% warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | D-P | returned_ids | 96 | 0.7630 | 6386.0000 | 8976.0000 | 0.0000 | 0.0000 | no | no |
| gsm8k | D-P | initial_prompt_ids | 96 | 4.1607 | -29376.0000 | 0.0000 | unavailable | unavailable | no | no |
| gsm8k | D-P | processed_input_ids | 96 | 4.1607 | -29376.0000 | 0.0000 | unavailable | unavailable | no | no |
| gsm8k | D-P | prompt_plus_returned | 96 | 1.6343 | -22990.0000 | 4784.0000 | unavailable | 0.0000 | no | no |
| gsm8k | D-P | wall_seconds | 96 | 0.7536 | 110.7014 | 146.8463 | 0.0000 | 0.0000 | no | no |
| gsm8k | L-D | returned_ids | 96 | 1.0000 | 0.0000 | 0.0000 | unavailable | unavailable | no | no |
| gsm8k | L-D | initial_prompt_ids | 96 | 1.0000 | 0.0000 | 0.0000 | unavailable | unavailable | no | no |
| gsm8k | L-D | processed_input_ids | 96 | 1.0000 | 0.0000 | 0.0000 | unavailable | unavailable | no | no |
| gsm8k | L-D | prompt_plus_returned | 96 | 1.0000 | 0.0000 | 0.0000 | unavailable | unavailable | no | no |
| gsm8k | L-D | wall_seconds | 96 | 0.9987 | 0.4381 | 2.2589 | 0.0000 | 0.0000 | no | no |
| gsm8k | L-P | returned_ids | 96 | 0.7630 | 6386.0000 | 8976.0000 | 0.0000 | 0.0000 | no | no |
| gsm8k | L-P | initial_prompt_ids | 96 | 4.1607 | -29376.0000 | 0.0000 | unavailable | unavailable | no | no |
| gsm8k | L-P | processed_input_ids | 96 | 4.1607 | -29376.0000 | 0.0000 | unavailable | unavailable | no | no |
| gsm8k | L-P | prompt_plus_returned | 96 | 1.6343 | -22990.0000 | 4784.0000 | unavailable | 0.0000 | no | no |
| gsm8k | L-P | wall_seconds | 96 | 0.7526 | 111.1395 | 147.1721 | 0.0000 | 0.0000 | no | no |
| humaneval | D-P | returned_ids | 47 | 0.9606 | 8251.0000 | 14546.0000 | 89.5043 | 50.7700 | no | yes |
| humaneval | D-P | initial_prompt_ids | 47 | 2.8589 | -14688.0000 | 0.0000 | unavailable | unavailable | no | no |
| humaneval | D-P | processed_input_ids | 47 | 2.8589 | -14688.0000 | 0.0000 | unavailable | unavailable | no | no |
| humaneval | D-P | prompt_plus_returned | 47 | 1.4549 | -6437.0000 | 10070.0000 | unavailable | 70.2979 | no | yes |
| humaneval | D-P | wall_seconds | 47 | 0.9263 | 147.1413 | 239.9460 | 82.7225 | 50.7276 | no | yes |
| humaneval | L-D | returned_ids | 48 | 1.0000 | 0.0000 | 0.0000 | unavailable | unavailable | no | no |
| humaneval | L-D | initial_prompt_ids | 48 | 1.0000 | 0.0000 | 0.0000 | unavailable | unavailable | no | no |
| humaneval | L-D | processed_input_ids | 48 | 1.0000 | 0.0000 | 0.0000 | unavailable | unavailable | no | no |
| humaneval | L-D | prompt_plus_returned | 48 | 1.0000 | 0.0000 | 0.0000 | unavailable | unavailable | no | no |
| humaneval | L-D | wall_seconds | 48 | 1.0042 | -1.3934 | 1.0378 | unavailable | 0.0000 | no | no |
| humaneval | L-P | returned_ids | 47 | 0.9606 | 8251.0000 | 14546.0000 | 89.5043 | 50.7700 | no | yes |
| humaneval | L-P | initial_prompt_ids | 47 | 2.8589 | -14688.0000 | 0.0000 | unavailable | unavailable | no | no |
| humaneval | L-P | processed_input_ids | 47 | 2.8589 | -14688.0000 | 0.0000 | unavailable | unavailable | no | no |
| humaneval | L-P | prompt_plus_returned | 47 | 1.4549 | -6437.0000 | 10070.0000 | unavailable | 70.2979 | no | yes |
| humaneval | L-P | wall_seconds | 47 | 0.9298 | 145.7479 | 238.8841 | 83.3950 | 50.8809 | no | yes |
| math500 | D-P | returned_ids | 40 | 0.7808 | 6614.0000 | 8059.0000 | 0.0000 | 0.0000 | no | no |
| math500 | D-P | initial_prompt_ids | 40 | 4.1975 | -12240.0000 | 0.0000 | unavailable | unavailable | no | no |
| math500 | D-P | processed_input_ids | 40 | 4.1975 | -12240.0000 | 0.0000 | unavailable | unavailable | no | no |
| math500 | D-P | prompt_plus_returned | 40 | 1.1655 | -5626.0000 | 4796.0000 | unavailable | 0.0000 | no | no |
| math500 | D-P | wall_seconds | 40 | 0.7649 | 103.8928 | 127.0362 | 0.0000 | 0.0000 | no | no |
| math500 | L-D | returned_ids | 40 | 1.0000 | 0.0000 | 0.0000 | unavailable | unavailable | no | no |
| math500 | L-D | initial_prompt_ids | 40 | 1.0000 | 0.0000 | 0.0000 | unavailable | unavailable | no | no |
| math500 | L-D | processed_input_ids | 40 | 1.0000 | 0.0000 | 0.0000 | unavailable | unavailable | no | no |
| math500 | L-D | prompt_plus_returned | 40 | 1.0000 | 0.0000 | 0.0000 | unavailable | unavailable | no | no |
| math500 | L-D | wall_seconds | 40 | 1.0044 | -1.4795 | 1.0165 | unavailable | 0.0000 | no | no |
| math500 | L-P | returned_ids | 40 | 0.7808 | 6614.0000 | 8059.0000 | 0.0000 | 0.0000 | no | no |
| math500 | L-P | initial_prompt_ids | 40 | 4.1975 | -12240.0000 | 0.0000 | unavailable | unavailable | no | no |
| math500 | L-P | processed_input_ids | 40 | 4.1975 | -12240.0000 | 0.0000 | unavailable | unavailable | no | no |
| math500 | L-P | prompt_plus_returned | 40 | 1.1655 | -5626.0000 | 4796.0000 | unavailable | 0.0000 | no | no |
| math500 | L-P | wall_seconds | 40 | 0.7683 | 102.4134 | 125.9473 | 0.0000 | 0.0000 | no | no |


| Suite | Arm | Metric | Coverage | Known sum | Known mean | Full total |
| --- | --- | --- | --- | --- | --- | --- |
| gsm8k | P | initial_prompt_ids | 96/96 | 9294.0000 | 96.8125 | complete |
| gsm8k | P | processed_input_ids | 96/96 | 9294.0000 | 96.8125 | complete |
| gsm8k | P | prompt_plus_returned | 96/96 | 36243.0000 | 377.5312 | complete |
| gsm8k | P | returned_ids | 96/96 | 26949.0000 | 280.7188 | complete |
| gsm8k | D | initial_prompt_ids | 96/96 | 38670.0000 | 402.8125 | complete |
| gsm8k | D | processed_input_ids | 96/96 | 38670.0000 | 402.8125 | complete |
| gsm8k | D | prompt_plus_returned | 96/96 | 59233.0000 | 617.0104 | complete |
| gsm8k | D | returned_ids | 96/96 | 20563.0000 | 214.1979 | complete |
| gsm8k | L | initial_prompt_ids | 96/96 | 38670.0000 | 402.8125 | complete |
| gsm8k | L | processed_input_ids | 96/96 | 38670.0000 | 402.8125 | complete |
| gsm8k | L | prompt_plus_returned | 96/96 | 59233.0000 | 617.0104 | complete |
| gsm8k | L | returned_ids | 96/96 | 20563.0000 | 214.1979 | complete |
| humaneval | P | initial_prompt_ids | 48/48 | 7882.0000 | 164.2083 | complete |
| humaneval | P | processed_input_ids | 48/48 | 7882.0000 | 164.2083 | complete |
| humaneval | P | prompt_plus_returned | 48/48 | 38049.0000 | 792.6875 | complete |
| humaneval | P | returned_ids | 48/48 | 30167.0000 | 628.4792 | complete |
| humaneval | D | initial_prompt_ids | 48/48 | 22570.0000 | 470.2083 | complete |
| humaneval | D | processed_input_ids | 48/48 | 22570.0000 | 470.2083 | complete |
| humaneval | D | prompt_plus_returned | 48/48 | 44486.0000 | 926.7917 | complete |
| humaneval | D | returned_ids | 48/48 | 21916.0000 | 456.5833 | complete |
| humaneval | L | initial_prompt_ids | 48/48 | 22570.0000 | 470.2083 | complete |
| humaneval | L | processed_input_ids | 48/48 | 22570.0000 | 470.2083 | complete |
| humaneval | L | prompt_plus_returned | 48/48 | 44486.0000 | 926.7917 | complete |
| humaneval | L | returned_ids | 48/48 | 21916.0000 | 456.5833 | complete |
| math500 | P | initial_prompt_ids | 40/40 | 3828.0000 | 95.7000 | complete |
| math500 | P | processed_input_ids | 40/40 | 3828.0000 | 95.7000 | complete |
| math500 | P | prompt_plus_returned | 40/40 | 34004.0000 | 850.1000 | complete |
| math500 | P | returned_ids | 40/40 | 30176.0000 | 754.4000 | complete |
| math500 | D | initial_prompt_ids | 40/40 | 16068.0000 | 401.7000 | complete |
| math500 | D | processed_input_ids | 40/40 | 16068.0000 | 401.7000 | complete |
| math500 | D | prompt_plus_returned | 40/40 | 39630.0000 | 990.7500 | complete |
| math500 | D | returned_ids | 40/40 | 23562.0000 | 589.0500 | complete |
| math500 | L | initial_prompt_ids | 40/40 | 16068.0000 | 401.7000 | complete |
| math500 | L | processed_input_ids | 40/40 | 16068.0000 | 401.7000 | complete |
| math500 | L | prompt_plus_returned | 40/40 | 39630.0000 | 990.7500 | complete |
| math500 | L | returned_ids | 40/40 | 23562.0000 | 589.0500 | complete |

Missing identities and input-accounting classifications are retained in analysis.json and audit.json. Full totals are unavailable wherever coverage is incomplete; no original prompt count is multiplied by decode batches or imputed as resumed input.

| Suite | Arm | Seed | Solved/scheduled | Errors | Timeouts | Caps |
| --- | --- | --- | --- | --- | --- | --- |
| gsm8k | P | 101 | 26/48 | 0 | 0 | 0 |
| gsm8k | P | 102 | 28/48 | 0 | 0 | 0 |
| gsm8k | D | 101 | 24/48 | 0 | 0 | 0 |
| gsm8k | D | 102 | 28/48 | 0 | 0 | 0 |
| gsm8k | L | 101 | 24/48 | 0 | 0 | 0 |
| gsm8k | L | 102 | 28/48 | 0 | 0 | 0 |
| humaneval | P | 101 | 23/24 | 0 | 0 | 1 |
| humaneval | P | 102 | 24/24 | 0 | 0 | 0 |
| humaneval | D | 101 | 24/24 | 0 | 0 | 0 |
| humaneval | D | 102 | 24/24 | 0 | 0 | 0 |
| humaneval | L | 101 | 24/24 | 0 | 0 | 0 |
| humaneval | L | 102 | 24/24 | 0 | 0 | 0 |
| math500 | P | 101 | 20/20 | 0 | 0 | 0 |
| math500 | P | 102 | 20/20 | 0 | 0 | 0 |
| math500 | D | 101 | 20/20 | 0 | 0 | 0 |
| math500 | D | 102 | 20/20 | 0 | 0 | 0 |
| math500 | L | 101 | 20/20 | 0 | 0 | 0 |
| math500 | L | 102 | 20/20 | 0 | 0 | 0 |


| Scope | Suite | Arm | Seed | Replies | Attempts | Prevented D attempts | Successful executions | Failed executions | Authenticated results | Replies using result | Activation label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset | gsm8k | P | 101 | 48 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | OBSERVED |
| Dataset | gsm8k | P | 102 | 48 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | OBSERVED |
| Dataset | gsm8k | D | 101 | 48 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | OBSERVED |
| Dataset | gsm8k | D | 102 | 48 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | OBSERVED |
| Dataset | gsm8k | L | 101 | 48 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | DECLARED BUT UNUSED |
| Dataset | gsm8k | L | 102 | 48 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | DECLARED BUT UNUSED |
| Dataset | humaneval | P | 101 | 24 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | OBSERVED |
| Dataset | humaneval | P | 102 | 24 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | OBSERVED |
| Dataset | humaneval | D | 101 | 24 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | OBSERVED |
| Dataset | humaneval | D | 102 | 24 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | OBSERVED |
| Dataset | humaneval | L | 101 | 24 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | DECLARED BUT UNUSED |
| Dataset | humaneval | L | 102 | 24 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | DECLARED BUT UNUSED |
| Dataset | math500 | P | 101 | 20 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | OBSERVED |
| Dataset | math500 | P | 102 | 20 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | OBSERVED |
| Dataset | math500 | D | 101 | 20 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | OBSERVED |
| Dataset | math500 | D | 102 | 20 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | OBSERVED |
| Dataset | math500 | L | 101 | 20 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | DECLARED BUT UNUSED |
| Dataset | math500 | L | 102 | 20 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | DECLARED BUT UNUSED |


| Scope | Suite | Arm | Seed | Replies | Attempts | Prevented D attempts | Successful executions | Failed executions | Authenticated results | Replies using result | Activation label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Excluded diagnostics | diagnostic | P | 101 | 6 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | OBSERVED |
| Excluded diagnostics | diagnostic | P | 102 | 0 | unknown; unknown rows=0 | unknown; unknown rows=0 | unknown; unknown rows=0 | unknown; unknown rows=0 | unknown; unknown rows=0 | unknown; unknown rows=0 | OBSERVED |
| Excluded diagnostics | diagnostic | D | 101 | 6 | 6; unknown rows=0 | 6; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | 0; unknown rows=0 | OBSERVED |
| Excluded diagnostics | diagnostic | D | 102 | 0 | unknown; unknown rows=0 | unknown; unknown rows=0 | unknown; unknown rows=0 | unknown; unknown rows=0 | unknown; unknown rows=0 | unknown; unknown rows=0 | OBSERVED |
| Excluded diagnostics | diagnostic | L | 101 | 6 | 6; unknown rows=0 | 0; unknown rows=0 | 6; unknown rows=0 | 0; unknown rows=0 | 6; unknown rows=0 | 6; unknown rows=0 | OBSERVED |
| Excluded diagnostics | diagnostic | L | 102 | 0 | unknown; unknown rows=0 | unknown; unknown rows=0 | unknown; unknown rows=0 | unknown; unknown rows=0 | unknown; unknown rows=0 | unknown; unknown rows=0 | OBSERVED |

D prevented-call counts are source-inferred attempts under absent-pool configuration, not deployed dispatch-spy counters. Unknown coverage is explicit. DECLARED BUT UNUSED applies only where all L execution counts are known and zero; no offloading benefit is demonstrated there. Calls using subgroups are exploratory and never replace the scheduled cohort.

| Boot | Arm | Verified native/reference diagnostic calls | HTTP audit | Boot evidence qualified |
| --- | --- | --- | --- | --- |
| t01-plain | P | 0 | yes | yes |
| t01-declared_off | D | 0 | yes | yes |
| t01-lane | L | 2 | yes | yes |
| t02-declared_off | D | 0 | yes | yes |
| t02-lane | L | 2 | yes | yes |
| t02-plain | P | 0 | yes | yes |
| t03-lane | L | 2 | yes | yes |
| t03-plain | P | 0 | yes | yes |
| t03-declared_off | D | 0 | yes | yes |


| Turn | Boot | Admission/load to ready seconds | Separate G restoration seconds | G restored run |
| --- | --- | --- | --- | --- |
| t01 | t01-plain | 319.5825 | 451.5073 | efast2-20261105T013127Z |
| t01 | t01-declared_off | 321.3404 | 451.5073 | efast2-20261105T013127Z |
| t01 | t01-lane | 319.3487 | 451.5073 | efast2-20261105T013127Z |
| t02 | t02-declared_off | 321.5298 | 436.4882 | efast2-20261105T021649Z |
| t02 | t02-lane | 319.2915 | 436.4882 | efast2-20261105T021649Z |
| t02 | t02-plain | 319.3139 | 436.4882 | efast2-20261105T021649Z |
| t03 | t03-lane | 323.5434 | 436.5130 | efast2-20261105T030303Z |
| t03 | t03-plain | 317.3304 | 436.5130 | efast2-20261105T030303Z |
| t03 | t03-declared_off | 319.3216 | 436.5130 | efast2-20261105T030303Z |

Restoration time is one interval per turn, repeated for its three boot rows for context; do not sum the repeated column. Times derive from retained steps, with G restoration requiring SERVING_VERIFIED after release and POPPED in the restore log, never outer return code alone.
GSM ONE-NUMERAL FINAL-SPAN is the preregistered secondary extraction sensitivity; strict accuracy remains primary. Every changed and residual grade appears below; the exact extracted spans remain in analysis.json.

| Suite | Contrast | Secondary accuracy Δ pp | CI95 pp | Marginal exact p | Gains/losses |
| --- | --- | --- | --- | --- | --- |
| gsm8k | D-P | 2.0833 | [0.0000, 5.2083] | 0.5000 | 2/0 |
| gsm8k | L-D | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0/0 |
| gsm8k | L-P | 2.0833 | [0.0000, 5.2083] | 0.5000 | 2/0 |


| GSM identity | Strict | Secondary | Disposition |
| --- | --- | --- | --- |
| ["gsm8k","105",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","105",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","105",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","105",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","105",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","105",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","1051",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","1051",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","1051",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","1051",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","1051",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","1051",102,"plain"] | incorrect | correct | changed |
| ["gsm8k","1083",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","1083",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","1083",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","1083",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","1083",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","1083",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","1084",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","1084",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","1084",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","1084",102,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","1084",102,"lane"] | incorrect | correct | changed |
| ["gsm8k","1084",102,"plain"] | incorrect | correct | changed |
| ["gsm8k","1133",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","1133",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","1133",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","1133",102,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","1133",102,"lane"] | incorrect | correct | changed |
| ["gsm8k","1133",102,"plain"] | incorrect | correct | changed |
| ["gsm8k","1148",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","1148",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","1148",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","1148",102,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","1148",102,"lane"] | incorrect | correct | changed |
| ["gsm8k","1148",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","1151",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","1151",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","1151",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","1151",102,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","1151",102,"lane"] | incorrect | correct | changed |
| ["gsm8k","1151",102,"plain"] | incorrect | correct | changed |
| ["gsm8k","1187",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","1187",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","1187",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","1187",102,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","1187",102,"lane"] | incorrect | correct | changed |
| ["gsm8k","1187",102,"plain"] | incorrect | correct | changed |
| ["gsm8k","12",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","12",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","12",101,"plain"] | incorrect | incorrect | residual incorrect |
| ["gsm8k","12",102,"declared_off"] | incorrect | incorrect | residual incorrect |
| ["gsm8k","12",102,"lane"] | incorrect | incorrect | residual incorrect |
| ["gsm8k","12",102,"plain"] | incorrect | incorrect | residual incorrect |
| ["gsm8k","1232",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","1232",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","1232",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","1232",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","1232",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","1232",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","1259",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","1259",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","1259",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","1259",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","1259",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","1259",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","1260",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","1260",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","1260",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","1260",102,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","1260",102,"lane"] | incorrect | correct | changed |
| ["gsm8k","1260",102,"plain"] | incorrect | correct | changed |
| ["gsm8k","1283",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","1283",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","1283",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","1283",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","1283",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","1283",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","147",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","147",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","147",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","147",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","147",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","147",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","188",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","188",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","188",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","188",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","188",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","188",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","203",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","203",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","203",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","203",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","203",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","203",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","207",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","207",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","207",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","207",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","207",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","207",102,"plain"] | incorrect | correct | changed |
| ["gsm8k","244",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","244",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","244",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","244",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","244",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","244",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","274",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","274",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","274",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","274",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","274",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","274",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","320",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","320",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","320",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","320",102,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","320",102,"lane"] | incorrect | correct | changed |
| ["gsm8k","320",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","325",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","325",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","325",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","325",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","325",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","325",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","355",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","355",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","355",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","355",102,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","355",102,"lane"] | incorrect | correct | changed |
| ["gsm8k","355",102,"plain"] | incorrect | correct | changed |
| ["gsm8k","416",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","416",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","416",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","416",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","416",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","416",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","422",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","422",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","422",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","422",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","422",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","422",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","443",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","443",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","443",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","443",102,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","443",102,"lane"] | incorrect | correct | changed |
| ["gsm8k","443",102,"plain"] | incorrect | correct | changed |
| ["gsm8k","502",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","502",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","502",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","502",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","502",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","502",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","551",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","551",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","551",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","551",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","551",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","551",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","559",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","559",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","559",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","559",102,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","559",102,"lane"] | incorrect | correct | changed |
| ["gsm8k","559",102,"plain"] | incorrect | correct | changed |
| ["gsm8k","577",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","577",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","577",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","577",102,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","577",102,"lane"] | incorrect | correct | changed |
| ["gsm8k","577",102,"plain"] | incorrect | correct | changed |
| ["gsm8k","586",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","586",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","586",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","586",102,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","586",102,"lane"] | incorrect | correct | changed |
| ["gsm8k","586",102,"plain"] | incorrect | correct | changed |
| ["gsm8k","599",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","599",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","599",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","599",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","599",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","599",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","604",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","604",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","604",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","604",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","604",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","604",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","641",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","641",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","641",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","641",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","641",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","641",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","654",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","654",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","654",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","654",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","654",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","654",102,"plain"] | incorrect | correct | changed |
| ["gsm8k","667",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","667",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","667",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","667",102,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","667",102,"lane"] | incorrect | correct | changed |
| ["gsm8k","667",102,"plain"] | incorrect | correct | changed |
| ["gsm8k","684",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","684",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","684",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","684",102,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","684",102,"lane"] | incorrect | correct | changed |
| ["gsm8k","684",102,"plain"] | incorrect | correct | changed |
| ["gsm8k","688",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","688",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","688",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","688",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","688",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","688",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","697",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","697",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","697",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","697",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","697",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","697",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","7",101,"declared_off"] | incorrect | incorrect | residual incorrect |
| ["gsm8k","7",101,"lane"] | incorrect | incorrect | residual incorrect |
| ["gsm8k","7",101,"plain"] | incorrect | incorrect | residual incorrect |
| ["gsm8k","7",102,"declared_off"] | incorrect | incorrect | residual incorrect |
| ["gsm8k","7",102,"lane"] | incorrect | incorrect | residual incorrect |
| ["gsm8k","7",102,"plain"] | incorrect | incorrect | residual incorrect |
| ["gsm8k","711",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","711",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","711",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","711",102,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","711",102,"lane"] | incorrect | correct | changed |
| ["gsm8k","711",102,"plain"] | incorrect | incorrect | residual incorrect |
| ["gsm8k","785",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","785",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","785",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","785",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","785",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","785",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","79",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","79",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","79",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","79",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","79",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","79",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","819",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","819",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","819",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","819",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","819",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","819",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","830",101,"declared_off"] | incorrect | incorrect | residual incorrect |
| ["gsm8k","830",101,"lane"] | incorrect | incorrect | residual incorrect |
| ["gsm8k","830",101,"plain"] | incorrect | incorrect | residual incorrect |
| ["gsm8k","830",102,"declared_off"] | incorrect | incorrect | residual incorrect |
| ["gsm8k","830",102,"lane"] | incorrect | incorrect | residual incorrect |
| ["gsm8k","830",102,"plain"] | incorrect | incorrect | residual incorrect |
| ["gsm8k","856",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","856",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","856",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","856",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","856",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","856",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","869",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","869",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","869",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","869",102,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","869",102,"lane"] | correct | correct | strict preserved |
| ["gsm8k","869",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","953",101,"declared_off"] | correct | correct | strict preserved |
| ["gsm8k","953",101,"lane"] | correct | correct | strict preserved |
| ["gsm8k","953",101,"plain"] | correct | correct | strict preserved |
| ["gsm8k","953",102,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","953",102,"lane"] | incorrect | correct | changed |
| ["gsm8k","953",102,"plain"] | correct | correct | strict preserved |
| ["gsm8k","994",101,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","994",101,"lane"] | incorrect | correct | changed |
| ["gsm8k","994",101,"plain"] | incorrect | correct | changed |
| ["gsm8k","994",102,"declared_off"] | incorrect | correct | changed |
| ["gsm8k","994",102,"lane"] | incorrect | correct | changed |
| ["gsm8k","994",102,"plain"] | incorrect | correct | changed |


| Suite | Contrast | Numerical local-efficiency conditions | Numerical total-token conditions | Qualified claim |
| --- | --- | --- | --- | --- |
| gsm8k | D-P | no | no | withheld / NEEDS-T5 |
| gsm8k | L-D | no | no | withheld / NEEDS-T5 |
| gsm8k | L-P | no | no | withheld / NEEDS-T5 |
| humaneval | D-P | no | no | withheld / NEEDS-T5 |
| humaneval | L-D | no | no | withheld / NEEDS-T5 |
| humaneval | L-P | no | no | withheld / NEEDS-T5 |
| math500 | D-P | yes | no | withheld / NEEDS-T5 |
| math500 | L-D | no | no | withheld / NEEDS-T5 |
| math500 | L-P | yes | no | withheld / NEEDS-T5 |

Numerical flags are the frozen analysis outputs, not a safety or serving-default verdict. Qualified local efficiency also requires full safety/accounting, no net primary solve decrease, returned-ID/wall CI uppers<1, no censor reversal and ≤25% net/positive censor contribution. Total-token savings additionally require a complete prompt+returned ratio CI upper<1. Where L/D is flat, use “no detected incremental execution-enabled effect on this cohort”; neither a flat result nor a nonsignificant test establishes equivalence.

Published reference comparison — REFERENCE ONLY. Retrieved2026-11-04; see PUBLISHED.md (`../PUBLISHED.md`, omitted from this export), [September Flash card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md), [Pro card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/README.md), and [exact1017 revision card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/README.md).

| Checkpoint / protocol | GSM8K | MATH / MATH-500 | Original HumanEval |
| --- | --- | --- | --- |
| Exact Flash-1017 instruct (our3 suites) | N/A — not published | MATH-500: N/A — not published | N/A — not published |
| September V4.1-Flash-BASE — historical REFERENCE ONLY | 93.0 exact match,8-shot | 61.1 exact match,4-shot; MATH, NOT MATH-500 | 79.4 pass@1,0-shot |
| V4-Pro-BASE preview-card — historical REFERENCE ONLY | 92.6 exact match,8-shot | 64.5 exact match,4-shot; MATH, NOT MATH-500 | 76.8 pass@1,0-shot |

The historical references use BASE checkpoints/internal evaluation, not Pro GA instruct, not our quantization and not matched local baselines. Sampling, generation counts and output caps are not published there. No retention target, accuracy-loss inference or token/time baseline is derived from them.

Freeze manifest SHA256: `785a8f5b9625427d50fc1e18b26bbdd40198ddd55ca18d3a635741edaab94d21`. CPU receipt SHA256: `1702014a7ec18314ad0a435beaeb70e5e9f62a0d46c1d5eb44eea69dc116855f`. Graded rows canonical SHA256: `8a14a275ce2b87b7b924453335957915d8ace9fedcc3b5f827d2fd9e2d222ee5`.

The post3/render_results3 wrappers are postcollection I/O/reporting additions, outside the87-file precollection seal. Frozen grade3/analysis3/math3 and the outcome mechanism remain unchanged. This report is local only; status **NEEDS-T5**.
