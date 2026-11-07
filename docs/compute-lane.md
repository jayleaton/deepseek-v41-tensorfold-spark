# Compute lane

The compute lane is DeepSeek-V4.1's built-in `python` tool, run by the engine itself. When the model calls it, the
reply does not end at the call: the engine holds the stream at its EOS (the slot and its KV stay; other streams keep
decoding), runs the code in a sandbox, renders the chat template's own tool-result turn and resumes the stream with
those tokens. The model continues from the same ids a client round trip would send, without a re-prefill and without
the client in the loop. The client gets one reply: the answer, plus the calls in a non-standard `compute` field.

[OPERATIONS](OPERATIONS.md#compute-lane-tf_dsv41_compute_lane-on-in-the-flash17-profile) covers the sandbox, its
limits, the boot self-test and how to turn the lane off. This page covers the measured configurations, what they
showed and how to rerun them. The kits, specs, checkers and results are in
[`bench/compute-lane/`](../bench/compute-lane/README.md).

## Engine

| | |
| --- | --- |
| Pin | `07106059c0d985dc352e126fe960941fbadc0166`, branch `dsv41/compute-lane-release` of `jayleaton/TensorFold` |
| Tree | `c3636e6da0d871470490830bb7860bb1a379c577`, identical to the measured source `3a429d6f1` |
| Measured image | `tensorfold-dsv41-imr:r-3a429d6f122d` (G's runtime image plus this source's two binaries and the runner's frontend layer, circuit and graph banks) |

The release branch holds the same 303 commits over upstream `0d51120a8` as the measured branch, with the same trees,
messages and dates, re-recorded under one author. Counterparts of commits named in this recipe's docs:

| measured | release branch | what |
| --- | --- | --- |
| `3a429d6f1` | `07106059c` | the studies' source (lane-only studies) |
| `3bf584af1` | `00b66d9f6` | AB3's source (tool-full) |
| `58bece1d3` | `3adb042bb` | release candidate G (previous pin) |
| `56200688d` | `3c4f2dba9` | candidate F |
| `20ad77878` | `b073317ab` | stable head H |

## Measured configurations

Every study ran on two GB10 Sparks (TP2) with the flash17 K6 MTP pack (DeepSeek-V4.1-Flash-1017, EXL3 2.80 bpw),
four slots, 1M context and the release candidate G settings of the flash17 profile.

**Requests** (all lane-only studies): thinking on, `reasoning_effort` high, temperature 1.0, top-p 0.95, top-k 20,
`max_tokens` 8192, `"deadline_s": 270`, seeds 101 and 102, one request at a time, streamed with
`return_token_ids`. The tool declaration is a client function named `python`, byte-equal to the engine's built-in
definition ([`decl-control/canonical-python-tools.json`](../bench/compute-lane/decl-control/canonical-python-tools.json)),
which arms the lane when it is on.

| arm | engine | request | what it measures |
| --- | --- | --- | --- |
| **P** plain | `TF_DSV41_COMPUTE_LANE=0` | no tools | the model alone |
| **D** declared, off | `TF_DSV41_COMPUTE_LANE=0` | the `python` declaration | the declaration without execution (the model's calls end the reply) |
| **L** lane | `TF_DSV41_COMPUTE_LANE=1` | the `python` declaration | the lane |

All three arms set the runner and its options off: `TF_DSV41_COMPUTE_LANE_RUNNER=0`, `TF_DSV41_RUNNER=0`,
`TF_DSV41_RUNNER_{COMMIT,STATED,INLINE_CHAIN,INLINE_MARK,SYNC_GRAPH}=0` and `TF_DSV41_RUNNER_FORMS=fence,repl,inline`
(read only by the runner). Arms differ only in the lines above; P and L differ in exactly one engine line. The
effective container environment of the L boot is in
[`bench/compute-lane/configs/`](../bench/compute-lane/README.md#configs), with paths replaced.

**Tool-full** (AB3 only, engine `3bf584af1`, image `tensorfold-dsv41-imr:r-3bf584af1c54`): L plus the in-model runner,
its calculator circuit and graph bank, S3 (calls go to the runner first, with the sandbox as fallback) and the trust
guide sent as the request's `runner_guide` (sha256 `53ed328f…`). The 18 engine lines are the commented
"tool-full" block of [`config/prod-zig-flash17.env.example`](../config/prod-zig-flash17.env.example); the requests
used `tools: [{"type": "code_interpreter"}]` (tool definition sha256 `eec0577c…`, about 306 prompt tokens),
`max_tokens` 32768, `deadline_s` 300, three seeds and four streams at a time. Its arms were plain, lane (L's engine
lines with this request shape), full and runner-only.

### The flash17 profile against arm L

The profile's default is arm L. Its engine settings equal L's effective environment except:

| setting | L | profile | why |
| --- | --- | --- | --- |
| `TF_DSV41_PF_WS_FLOOR_GIB` | 3.5 | 4.75 | the guard refuses a workspace floor under the 4 GiB hard floor ([FLASH-1017](FLASH-1017.md)); it only picks 4K or 2K prefill segments, same bits |
| `TF_DSV41_PREPARED_WRITE` | 0 | 1 | the studies mounted an already written prepared pack read-only; a first boot has to write it |
| `TF_DSV41_ACCEPT_LOG` | `/state/acclog.jsonl` | unset | draft-acceptance telemetry for the harness |

`TF_CUDA_SCHED` is unset in the profile because the studies ran with it unset. Names the bench launcher passed that
this engine no longer reads (`GLM53_TF_ROCE_*`, `TF_DSV41_PLAN_*`, `TF_DSV41_DRAIN_S` and the like) have their
current names in the profile with the same values, or are fixed engine behaviour; `TF_TP_ROCE_POLLERS=1` and
`TF_TP_ROCE_STAGGER_NS=150` are the engine defaults.

## Results

Each result below is the independent adjudicator's allowed public wording, verbatim apart from restored spacing,
followed by the configuration that produced it. Every adjudication names its frozen spec by sha256; both are in
[`bench/compute-lane/`](../bench/compute-lane/README.md).

### Computation-heavy tasks (compute-DL): lane only, runner off

Configuration: arms P, D and L above, image `r-3a429d6f122d`. 720 replies (240 per arm).

> On 120 newly generated computation-heavy tasks with two seeds, our Python execution lane with its tool declaration
> delivered 187/240 strictly formatted correct answers versus 120/240 for the plain model under an 8192-ID cap.
> Observed returned IDs and response time were about 93% lower. Half the plain replies hit the cap and accounted for
> most savings; the lane also lost 22 strictly graded answers through formatting on tasks plain solved. This amended,
> fixed-mixture study does not establish general benchmark or uncensored speed gains.

This is an outcome-informed amended study (amendments A1 and A2) with boot-order limitations; the amended accuracy
condition passed and the uncensored efficiency condition did not.

### Final-answer wording retest (format-retest): lane only, runner off

Configuration: arms P and L above, image `r-3a429d6f122d`, with clearer final-answer wording applied identically to
every arm. 360 replies. A separate prospective study, not pooled with compute-DL.

> We then prospectively tested clearer final-answer wording, applied identically to every arm, on 60 fresh tasks. The
> lane solved 120/120 versus 86/120 with the old wording, eliminating the observed format failures. On the 30 fresh
> DP/graph/aggregation tasks also run plain, both new-wording arms solved 60/60, with no caps; the lane returned 78%
> fewer IDs and took 77% less measured response time.

Safety had two sources: the pinned r7 owned oracle reported zero wrong or not-in-text over zero runner-owned cells
(the runner was off), and a separate isolated CPython audit matched all 245 tool results (243 dataset results and 2
diagnostics) to their full references, none clipped.

### Declaration control, standard suites (decl-control): lane only, runner off

Configuration: arms P, D and L above, image `r-3a429d6f122d`. 552 replies.

> On fixed subsets of 48 GSM8K, 20 MATH-500 and 24 original HumanEval tasks with two seeds, the served 2.80bpw
> DeepSeek-V4.1-Flash-1017 model produced exactly the same native outputs with the Python declaration whether
> execution was enabled or disabled. No dataset reply called Python. Adding the declaration with execution disabled
> reproduced the previously observed 23.7%, 21.9% and 27.4% reductions in returned tokens versus no declaration.
> Enabling execution added no detected benefit on this cohort. Prompt-plus-output totals increased; GSM strict-format
> accuracy fell slightly and HumanEval cost savings were dominated by one capped response.

### AB3: tool-full, runner on

Configuration: the tool-full arm above, engine `3bf584af1`. **AB3 did not meet its frozen spec**: its primary token
comparison (full against lane) failed its confidence interval, and a recorded-but-not-delivered trace newline failed
its safety/provenance gate. The figure below is a descriptive secondary comparison:

> In a descriptive secondary comparison on 60 W9 task specifications at three paired sampling seeds, the built
> configuration combining the in-model runner, declared Python tool, S3 runner-first execution with sandbox fallback,
> and the trust guide used 81.1% fewer returned tokens than plain generation (CI95: 76.6%–84.7% fewer). It solved
> 174/180 versus 170/180. Plain had seven deadlines, all retained, versus none in that configuration. The existing
> sandbox tool lane also used 78.2% fewer tokens than plain. The experiment did not meet its primary comparison or
> safety qualification.

Full against plain had three losses and seven gains; the seven plain-censored pairs contribute 33.16% of the net
token reduction. The pin adds two commits after AB3's engine (`730aa177d` and `3a429d6f1`, release branch `267add1d2`
and `07106059c`), which keep runner-owned bytes through the strips beside tool calls and keep trim limits off code
point boundaries. Tool-full has not been measured again at the pin.

### Which configuration produced which number

| result | configuration | engine |
| --- | --- | --- |
| 187/240 vs 120/240, ~93% fewer IDs and less time | lane only (L vs P), runner off | `3a429d6f1` |
| 120/120 vs 86/120; 78% fewer IDs, 77% less time | lane only (L vs P), runner off | `3a429d6f1` |
| identical outputs with execution on or off; 23.7/21.9/27.4% from the declaration | D vs L vs P, runner off | `3a429d6f1` |
| 81.1% fewer returned tokens than plain; 174/180 vs 170/180 | tool-full, runner on | `3bf584af1` |
| lane 78.2% fewer tokens than plain (AB3) | AB3's lane arm, runner off | `3bf584af1` |

## Limits

- These are generated six-template development cohorts and short fixed subsets, not standard benchmark scores.
- Returned IDs are not full processed-token or compute savings: the lane's multi-round inputs were not measured.
- No serving-default claim follows from any of these studies; the profile turns the lane on as an operator choice
  made before them ([OPERATIONS](OPERATIONS.md#compute-lane-tf_dsv41_compute_lane-on-in-the-flash17-profile)).
- One pair of Sparks, one pack.

## Experimental settings

Off in the flash17 profile; measured only where stated:

| setting | status |
| --- | --- |
| `TF_DSV41_RUNNER` and `TF_DSV41_RUNNER_*` | the in-model runner; on only in AB3's tool-full arm |
| `TF_DSV41_COMPUTE_LANE_RUNNER` and `_*` | S3, runner-first lane calls; on only in tool-full |
| request `runner_guide` | the trust guide; tool-full only, read only with the runner on |
| `TF_DSV41_RUNNER_TRACE*`, `_SPEC_HOLD`, `_INLINE_ARITH`, `_INLINE_MOD`, `_SETTLE_US`, `_COMMIT` (on) | never on in an adjudicated arm |

The lane itself (`TF_DSV41_COMPUTE_LANE` with its default limits) and `deadline_s` are the measured path.

## Reproducing

[`bench/compute-lane/README.md`](../bench/compute-lane/README.md) lists every kit with the commands as they were run;
host paths are placeholders. In short:

1. Build the image at the pin (`scripts/build-native.sh`, [ENGINE](ENGINE.md)) and boot the pair with the flash17
   profile, or with the measured environment in `bench/compute-lane/configs/` for a strict replay. Tool-full also
   needs the runner layer, circuit and graph bank (`packaging/runner/` in the engine; the shas are in
   `bench/compute-lane/ab3/AB3-FREEZE.json`).
2. Check the task sets, references and specs against `bench/compute-lane/MANIFEST.sha256`.
3. Run each study's controller for its arms and seeds, then its grader and analysis. Raw receipts are listed by sha256
   in `bench/compute-lane/RAW-RECEIPTS.sha256`.
4. The format-retest and AB3 checks use the r7 owned oracle in `bench/compute-lane/owned-oracle-r7/`.
