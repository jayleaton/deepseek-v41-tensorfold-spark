# Compute lane

The compute lane is DeepSeek-V4.1's built-in `python` tool, run by the engine itself. When the model calls it, the
reply does not end at the call: the engine holds the stream at its EOS (the slot and its KV stay; other streams keep
decoding), runs the code in a sandbox, renders the chat template's own tool-result turn and resumes the stream with
those tokens. The model continues from the same ids a client round trip would send, without a re-prefill and without
the client in the loop. The client gets one reply: the answer, plus the calls in a non-standard `compute` field.

**The flash17 profile's default is the in-thinking "tool-full" profile** with `TF_DSV41_COMPUTE_LANE_DEFAULT=1`: for
every thinking chat whatever the client sends, the model computes inside its thinking (the in-model runner writes the
values of the cells it can run) and, when its reply has room for a call, through the built-in tool ([Compute in every
chat](#compute-in-every-chat)).
Arm L, the lane alone with the runner off, is an opt-in block in the same profile.

[OPERATIONS](OPERATIONS.md#compute-lane-tf_dsv41_compute_lane-on-in-the-flash17-profile) covers the sandbox, its
limits, the boot self-test and how to turn the lane off. This page covers the default, the measured configurations,
what they showed and how to rerun them. The kits, specs, checkers and results are in
[`bench/compute-lane/`](../bench/compute-lane/README.md).

## Engine

| | |
| --- | --- |
| Pin | `3ca06dd4bf2e0243cf6e7cb6338a2501f78d59fe`, branch `dsv41/undeclared-tool-passthrough` of `jayleaton/TensorFold` (`1ea71240a` plus the undeclared-tool fix; `1ea71240a` is `7b79ef92d` plus the opt-in checkpoints of [#80](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/issues/80)); before them `7b79ef92d846138d51638b677c473392a56e240c`, branch `dsv41/compute-lane-release-s86` (`01a62132c` plus the session-restore fix of upstream #2182, [#86](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/issues/86)) |
| Measured source | `07106059c` on that branch: tree `c3636e6da0d871470490830bb7860bb1a379c577`, identical to `3a429d6f1` |
| Measured image | `tensorfold-dsv41-imr:r-3a429d6f122d` (G's runtime image plus that source's two binaries and the runner's frontend layer, circuit and graph banks) |

The pin is `07106059c` plus seven commits: `e5172c4a2` adds `TF_DSV41_COMPUTE_LANE_DEFAULT` ([Compute in every
chat](#compute-in-every-chat)), `75d774470` refines it (a client's own `python`, the `compute_executed` count),
`012767b62` and `8a6ce56a3` keep it out of replies that cannot run a call and out of thinking-off chats ([When the
built-in tool is offered](#when-the-built-in-tool-is-offered)), and `01a62132c` stops the runner's inline cells from
taking the right operand of an operator they do not read (`\times`, `\bmod`) as the whole expression; `5f3e98939` and
`7b79ef92d` (upstream #2182) let NVMe session restores through `Disk.validate`. With that setting
unset it renders and serves requests exactly as `07106059c` does, apart from the session restores; the inline fix applies whenever the runner is on.
Below it, the release branch holds the same 303 commits over upstream `0d51120a8` as the measured branch, with the
same trees, messages and dates, re-recorded under one author. Counterparts of commits named in this recipe's docs:

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
`max_tokens` 8192, `"deadline_s": 270`, seeds 101 and 102, one request at a time, streamed with `return_token_ids`.
The tool declaration is a client function named `python`, byte-equal to the engine's built-in definition
([`decl-control/canonical-python-tools.json`](../bench/compute-lane/decl-control/canonical-python-tools.json)), which
arms the lane when it is on.

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
guide sent as the request's `runner_guide` (sha256 `53ed328f…`). The 18 engine lines are the active default
"tool-full" block of [`config/prod-zig-flash17.env.example`](../config/prod-zig-flash17.env.example), before
`TF_DSV41_COMPUTE_LANE_DEFAULT=1`; the requests
used `tools: [{"type": "code_interpreter"}]` (tool definition sha256 `eec0577c…`, about 306 prompt tokens),
`max_tokens` 32768, `deadline_s` 300, three seeds and four streams at a time. Its arms were plain, lane (L's engine
lines with this request shape), full and runner-only.

### The flash17 profile against the measured arms

The profile's default is the tool-full arm's 18 engine lines plus `TF_DSV41_COMPUTE_LANE_DEFAULT=1`. AB3's requests sent
`tools: [{"type": "code_interpreter"}]` and the trust guide as `runner_guide`; with the default setting the engine
supplies the guide to every thinking chat and the tool to a thinking chat with room for a call, so a client need not
send either. `max_tokens` and `deadline_s` stay the client's (AB3
sent 32768 and 300). The engine is AB3's source plus the two owned-bytes commits and the default commit, so the default
has not itself been measured at the pin. Arm L is the commented opt-in block below it (the default's nineteen names with
the runner off, so no default value survives from an exported shell) and runs on the plain image.

Both blocks share the rest of the profile, which equals L's effective environment except:

| setting | L | profile | why |
| --- | --- | --- | --- |
| `TF_DSV41_PF_WS_FLOOR_GIB` | 3.5 | 4.75 | the guard refuses a workspace floor under the 4 GiB hard floor ([FLASH-1017](FLASH-1017.md)); it only picks 4K or 2K prefill segments, same bits |
| `TF_DSV41_PREPARED_WRITE` | 0 | 1 | the studies mounted an already written prepared pack read-only; a first boot has to write it |
| `TF_DSV41_ACCEPT_LOG` | `/state/acclog.jsonl` | unset | draft-acceptance telemetry for the harness |

`TF_CUDA_SCHED` is unset in the profile because the studies (L and AB3) ran with it unset. Names the bench launcher
passed that this engine no longer reads (`GLM53_TF_ROCE_*`, `TF_DSV41_PLAN_*`, `TF_DSV41_DRAIN_S` and the like) have
their current names in the profile with the same values, or are fixed engine behaviour; `TF_TP_ROCE_POLLERS=1` and
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

| configuration | result | engine |
| --- | --- | --- |
| lane (arm L; AB3's lane arm), runner off | 78.2% fewer returned tokens than plain in AB3 | `3bf584af1` |
| lane (L vs P), runner off | 187/240 vs 120/240 strictly formatted correct (compute-DL) | `3a429d6f1` |
| lane (L vs P), runner off | 120/120 vs 86/120; on the fresh tasks also run plain, 78% fewer IDs and 77% less response time (format-retest) | `3a429d6f1` |
| lane (D vs L vs P), runner off | identical outputs with execution on or off; 23.7%, 21.9% and 27.4% fewer tokens from the declaration (decl-control) | `3a429d6f1` |
| tool-full (in-thinking: runner, built-in tool, S3, trust guide) | 81.1% fewer returned tokens than plain (CI95 76.6%–84.7%), 174/180 vs 170/180; descriptive secondary | `3bf584af1` |
| tool-full against lane | 13.17% fewer returned tokens than lane, CI95 26.08% fewer to 0.98% more: the interval crosses zero, and this primary comparison failed | `3bf584af1` |

The default profile is the tool-full configuration. No study measured it against arm L with a passing interval; the
lane numbers come from arm L.

## Limits

- These are generated six-template development cohorts and short fixed subsets, not standard benchmark scores.
- Returned IDs are not full processed-token or compute savings: the lane's multi-round inputs were not measured.
- No serving-default claim follows from any of these studies. The in-thinking default is an operator choice: AB3 did
  not meet its frozen spec, and the default's engine (with `TF_DSV41_COMPUTE_LANE_DEFAULT`) has only its smoke boot
  ([Compute in every chat](#compute-in-every-chat)).
- One pair of Sparks, one pack.

## Experimental settings

Configured in the default profile, measured only where stated:

| setting | status |
| --- | --- |
| `TF_DSV41_RUNNER` and the default's `TF_DSV41_RUNNER_*` lines | the in-model runner; measured only in AB3's tool-full arm |
| `TF_DSV41_COMPUTE_LANE_RUNNER` and `_*` | S3, runner-first lane calls; tool-full only |
| `TF_DSV41_COMPUTE_LANE_DEFAULT` | every thinking chat keeps the runner and the trust guide; the built-in tool only when its reply has room for a call; not in any study (AB3's requests sent both) |
| `TF_DSV41_RUNNER_INLINE_ARITH` and `_INLINE_MOD` | not set in either block, so the engine defaults both to `1` and inline arithmetic and modulo cells run (the profile's `TF_DSV41_RUNNER_FORMS` includes `inline`); on in AB3 (mapped source `00b66d9f6`) |

Off in both blocks and never on in an adjudicated arm: `TF_DSV41_RUNNER_TRACE*`, `_SPEC_HOLD`, `_COMMIT`, `_STATED`,
`_INLINE_CHAIN`, `_INLINE_MARK` and `_SYNC_GRAPH`.
The active default explicitly sets `TF_DSV41_RUNNER_COMMIT=0`; listing a setting here does not enable it.

The lane itself (`TF_DSV41_COMPUTE_LANE` with its default limits) and `deadline_s` are the measured path of arm L.

## Compute in every chat

`TF_DSV41_COMPUTE_LANE_DEFAULT=1` (needs `TF_DSV41_COMPUTE_LANE=1`; any other value, or the setting without the lane,
stops the boot) makes the engine read every thinking chat request as the tool-full request, whatever the client
sends: the runner and the boot's guide stay on for all of them, and one whose reply has room for a call also gets the
built-in tool:

- `tools` gets the built-in `{"type": "code_interpreter"}` entry once, when the chat thinks and its reply can run a
  call ([When the built-in tool is offered](#when-the-built-in-tool-is-offered)). A client that lists it, or sends a
  byte-equal copy of the built-in `python` function, still gets one declaration. A client that offers its own, different
  function named `python` keeps it: for that request the built-in tool is left out (the prompt is the client's own and
  its `python` calls go back to it), while the runner's in-thinking cells and guide stay on. A model's call carries
  only the tool's name, so two tools named `python` cannot both be offered. Only a request that lists
  `code_interpreter` next to its own different `python` is refused (HTTP 400), as with the default off. A
  `tool_choice` of `"none"` keeps the built-in tool; `"required"` or a named function leave it out, as they do for a
  request that opts in.
- The runner's guide is the boot's (`TF_DSV41_RUNNER_GUIDE`, or AB3's trust guide when unset, sha256 `53ed328f…`), for
  thinking chats only and whatever their reply's budget, stop strings, logprobs or output format: it was measured
  inside thinking, and in a visible answer it makes the model write `<expr> =` cells where the client expects the
  answer. A thinking-off chat gets no guide. A request's own `runner_guide` is ignored.
- The client's own function tools are kept and their calls go back to the client as usual. Only built-in calls and
  the runner's in-thinking cells run in the engine.
- No request field turns it off.

Each reply's `tensorfold` object then carries `compute_executed`: `calls` (built-in calls the circuit ran to its halt
or the sandbox started; a call with no code, or past `TF_DSV41_COMPUTE_LANE_MAX_CALLS`, stays a report but is not
counted), `runner_cells` (runner cells the engine wrote a value for, all forms) and `total`, 0 when nothing ran. With the setting unset no
reply carries it and requests are rendered exactly as before.

### When the built-in tool is offered

The default offers the built-in tool only to a thinking chat whose reply can hold at a call, run it and still
answer:

- Thinking is on: `chat_template_kwargs.enable_thinking` (or `thinking`, or a `reasoning_effort` other than `none`),
  else the server's default. With thinking off, the prose the model writes before a call would be the visible answer,
  and the answer after the result would run straight on into it (a TP2 benchmark check showed
  `Let me compute.232792560`).
- `max_tokens` is at least 1024 (`default_min_tokens` in the engine's `compute_turns.zig`). A request with no
  `max_tokens`, or 0, counts as the server's default (4096). The call, its result and the answer all count against
  `max_tokens`, so a short reply (a 64-token QA benchmark, for one) has no room for them.
- No `stop` strings, no `logprobs` or `prompt_logprobs`, and no structured output (`response_format`, `guided_json`,
  `guided_regex`, `guided_choice`, `guided_grammar`, `structured_outputs`). These replies cannot hold at a call.

Any other request gets no built-in tool, but it is not otherwise served as with the default off: every thinking chat
still gets the boot's trust guide and keeps the runner, so a thinking chat whose reply cannot hold a call (a short
`max_tokens`, stop strings, logprobs or structured output) has a longer prompt than the default-off one. Only a
thinking-off chat drops the guide and renders as with the default off. The client's own tools are kept as usual. A
request that lists `code_interpreter` itself is not affected by these rules: a thinking-off client that wants the tool
can still opt in.

If a built-in call still cannot finish inside `max_tokens`, it never reaches the client: no `tool_calls`, no DSML
markup in the content. The reply keeps the prose before the call and ends with `finish_reason` `"length"`.

## Runner image

The default needs the runner's frontend (clang-18 and lld-18), circuit and graph bank in the image. Build the image at
the pin, then layer them over it:

```sh
scripts/build-native.sh -t tensorfold-dsv41-recipe:<tag>
scripts/build-runner-image.sh tensorfold-dsv41-recipe:<tag> -t tensorfold-dsv41-recipe:<tag>-runner
```

`build-runner-image.sh` downloads the assets itself: `calc.tfpc` and `graph-v7-powmod/bank/`, the files AB3's image
served, as one archive pinned in [`config/runner-assets.release`](../config/runner-assets.release) (URL and sha256; the
archive is a release asset of this repository). [`config/runner-assets.sha256`](../config/runner-assets.sha256) lists
each file with its sha256, and the build refuses any other set. A failed download, a wrong archive hash or a missing or
changed file stops it with a message naming the file and the URL. `RUNNER_ASSETS_URL` points it at a mirror (the same
sha256 is still required); `scripts/fetch_runner_assets.py --archive FILE DIR` unpacks an archive downloaded elsewhere,
and `scripts/build-runner-image.sh BASE DIR -t TAG` takes an unpacked directory. The circuit is exported from Qwenputer
(MIT; its license ships in the bank and the image) with the engine's `tools/runner_export.py`, and the bank with its
`tools/runner_graph_*` tools. The layer fails to build unless the image's circuit route computes the engine's probe
cells (`tf-runner-frontend-check --require --circuit`). The image `scripts/build-native.sh` builds alone serves arm L
and the rollback profile.

## Reproducing

[`bench/compute-lane/README.md`](../bench/compute-lane/README.md) lists every kit with the commands as they were run;
host paths are placeholders. In short:

1. Build the image at the pin (`scripts/build-native.sh`, [ENGINE](ENGINE.md)) and boot the pair with the flash17
   profile, or with the measured environment in `bench/compute-lane/configs/` for a strict replay. Tool-full and the
   default need the runner image ([Runner image](#runner-image)); AB3's shas are in
   `bench/compute-lane/ab3/AB3-FREEZE.json`. A strict AB3 replay sets `TF_DSV41_COMPUTE_LANE_DEFAULT=0` and sends the
   request fields of `bench/compute-lane/configs/requests-ab3-tool-full.json`.
2. Check the task sets, references and specs against `bench/compute-lane/MANIFEST.sha256`.
3. Run each study's controller for its arms and seeds, then its grader and analysis. Raw receipts are listed by sha256
   in `bench/compute-lane/RAW-RECEIPTS.sha256`.
4. The format-retest and AB3 checks use the r7 owned oracle in `bench/compute-lane/owned-oracle-r7/`.
