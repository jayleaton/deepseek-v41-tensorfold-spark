# STREAM_RB=4: fresh long-context and TP=2 serving regression

2026-10-30. Candidate kernel source: `2be9759fb09da9ced3ee266fc7e2464c6669fba2`.
This receipt supplements the FP64 arithmetic gate in
[the earlier review receipt](../stream-rb4-review-20261030/README.md).
No kernel arithmetic, engine code or launch options changed in this follow-up.

## Tested build and checkpoint

Two NVIDIA DGX Spark GB10 systems, 128 GB unified-memory class each, TP=2 over RoCE.
Native version 1.0.2; engine source `9131e10ad17dde51e94a996da8255227727cd224`
with the existing PR65/PR78 packaging. Exact binary/config/tokenizer fingerprints
and settings are in [environment.json](environment.json).

Stock **DeepSeek-V4.1-Flash-1017 EXL3 flash17, 2.8003 bpw**; original model revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. Both arms read the same original weights
and prepared files; this was not a stock-versus-abliterated model comparison.

All eight rb4 variants were freshly built from the candidate source using Triton
3.7.1 and ptxas 13.3.73. Eight compiled, zero failed. Their SASS matches the eight
previously deployed optimized variants; byte identity is not claimed because
build paths/debug metadata differ. [build-manifest.json](build-manifest.json)
records source and cubin fingerprints. The native six-case `stream-rb --twins 4`
check also passes with the freshly compiled AOT set (five timing repetitions).

Both serving arms use the same complete AOT set, including the fresh variants.
The only difference between the measured arms is `TF_DSV41_STREAM_RB=0` versus `4`.
Shared test overrides: `TF_DSV41_SESSIONS=0`, `--prompt-cache-gib 0`, and read-only
prepared weights with `TF_DSV41_PREPARED_WRITE=0`. Context limit 1,048,576;
parallel limit 4; prefill chunks 4096; FP8 index KV; greedy temperature 0;
thinking disabled in the encoded input; maximum 128 output tokens.
Existing memory floors/admission settings were retained.

## Inputs and checks

[fixtures.zip](fixtures.zip) contains exact public token-ID request bodies and a
SHA-256 manifest. Tokenization used the serving tokenizer and chat template; the
encoded chat prefixes/suffixes are retained in the `/v1/completions` requests.
Seven needle prompts have **exactly 2048, 8192, 32768, 65536, 131072, 300000 and
500000 tokens**. Deterministic synthetic filler surrounds one public dummy fact
near the middle, followed by a retrieval question. Six short prompts exercise
arithmetic, code, structured JSON, prose, sorting and summarization.

Each arm runs 24 requests in the same order:

1. Six short greedy requests and seven cold needle requests with drafts enabled.
2. The six short prompts and 131072-token prompt with `draft: false`.
3. Four simultaneous short requests, compared with their solo counterparts.

Every response must complete without truncation, return generated token IDs,
report the exact prompt length and zero cached tokens, and satisfy its explicit
answer check where applicable. Cross-arm comparison uses full generated token-ID
arrays, including the stop token. Draft/plain and concurrent/solo token IDs are
compared within each arm. The short JSON response is also parsed and checked.
Raw responses and timing statistics are in the `baseline/` and `candidate/`
folders; [comparison.json](comparison.json) records the cross-arm comparisons.

## Results

**The full strict serving gate fails.** All 48 HTTP requests completed normally,
all reported zero cached tokens, and both arms had zero engine errors and zero
error/fatal lines in the active rank logs. However, one inherited short-input
concurrent/solo mismatch prevents an all-pass claim.

| Check | Result |
| --- | --- |
| Seven needle lengths, each arm | 14/14 correct retrievals |
| Single-request cross-arm token identity, including plain decoding | 20/20 pass |
| Cross-arm token identity including concurrent cases | 23/24 pass; 1 fail |
| Draft-enabled versus plain, within both arms | 14/14 pass |
| Concurrent versus solo, within baseline | 3/4 pass; 1 fail |
| Concurrent versus solo, within candidate | 4/4 pass |
| Original service restored and functional | pass |

The failing case is `short-prose-concurrent`, a **16-token prompt**. The baseline
solo and plain replies have 49 tokens; the baseline concurrent reply has 47.
Their first mismatch is generated token index 43 (zero-based): 7574 versus 6877.
The solo response ends with “backend or network operations on cache hits”; the
baseline concurrent response ends with “backend or network round trips”. The
candidate concurrent response matches its 49-token solo response in this run.

This discrepancy already occurs with **RB=0**. A 16-token prompt is below the
long-context stream-index threshold and does not exercise the changed rb4 path.
That localizes the observed failure outside this optimization's active input
range, but does not establish its root cause or justify waiving the contribution
contract. No top-logit margins were available, so no tie-policy exception is
claimed. The strict runner exits nonzero, and the comparison tool also returns
nonzero with `full_gate_pass: false`. The failed outputs and first differing IDs
are preserved, rather than filtered out of the published results.

The long-context optimization and its numerical/device gates pass this tested
scope. **Do not describe the full parallel=4 serving acceptance as passed** until
the existing short-input concurrency discrepancy is triaged by the maintainer.

Engine-reported prefill times, with drafts enabled:

| Prompt tokens | `_stream` prefill (s) | rb4 prefill (s) | rb4 change |
| ---: | ---: | ---: | ---: |
| 2,048 | 0.9286 | 0.9132 | -1.66% |
| 8,192 | 3.4846 | 3.4874 | +0.08% |
| 32,768 | 11.6339 | 12.2412 | +5.22% |
| 65,536 | 23.6679 | 23.8592 | +0.81% |
| 131,072 | 50.8433 | 47.5588 | -6.46% |
| 300,000 | 131.1423 | 118.2301 | -9.85% |
| 500,000 | 252.7722 | 215.8617 | -14.60% |

These are paired observations from one baseline boot followed by one candidate
boot on the same pair, not medians from an alternating multi-run benchmark.
The 32K observation is 5.22% slower with rb4 and is retained here. This
one paired observation does not establish whether that is a reproducible
regression or run-order/thermal variation. Short-input timing changes should
not be treated as robust performance claims. No prefix hits contribute to the reported prefill times. TTFT, wall time,
decode statistics and full outputs remain available in each raw response.

## Reproduction

Configure both ranks for the same stock checkpoint, engine and candidate AOT set.
Use fresh test state for each arm and the shared cold-cache settings above.
Run baseline with `TF_DSV41_STREAM_RB=0`, then restart both ranks with the same
configuration except `TF_DSV41_STREAM_RB=4` and replay the same archive:

```sh
python replay.py --base http://127.0.0.1:18100 --fixtures fixtures.zip --out replay-baseline
python replay.py --base http://127.0.0.1:18100 --fixtures fixtures.zip --out replay-candidate
python compare.py replay-baseline replay-candidate --output replay-comparison.json
```

The second command is run only after switching both ranks to the candidate arm.
For a fast probe of the observed short-input consistency failure, add
`--short-only` to `replay.py`; the replay exits nonzero and preserves all comparisons
when a mismatch occurs. The scripts use only the Python standard library. Setup/orchestration logs stay
private because they contain host paths and transport addresses; the inputs,
responses, numerical comparisons and build fingerprints are published here.

## Window boundaries, failures and restoration

The valid window ran from 09:32 to 09:53 UTC. Each arm completed 24 requests;
both ranks were alive and healthy at the end of each arm. The strict consistency
assertion failed after all inference requests completed; restoration then ran
successfully. [window-summary.json](window-summary.json) preserves setup-failure
counts, strict failure, active-log error counts and restoration status.
[postcheck.json](postcheck.json) verifies the same original container IDs,
environments, commands and mounts, restored watchdog state, gateway health,
zero engine errors, and a successful post-restoration `OK` response.

Two preliminary startup attempts failed before any model request: the peer lacked
the primary node's optional calibration directory, and the test launcher passed
server-only flags to the peer CLI. The launcher now supplies identical calibration
inputs and passes server flags only to rank 0. A later preflight rejected those
attempts' stopped test containers before changing production; only the test
containers created by this run were removed. These are recorded as setup
failures, not suppressed or counted as successful inference cases. They did not
change the kernel. A separate deadline guard and a restore-on-exit path protected
the original service; the existing watchdog was suspended only during the window.

The original persistent serving containers, including their original checkpoint,
were restored without changing their configuration or assets. This stock-model A/B
does not establish quality equivalence for other checkpoint variants. Resumed-session behavior,
other GPUs/backends, 1M-token context, broad model-quality suites and sustained
multi-hour serving are outside this receipt. Retain the opt-in rollout boundary.

## Artifact checks

Fixture manifest hashes and all exact token lengths verified (13 requests).
Ruff on the two replay/analysis scripts: zero findings. Engine Lean remains
6,375 findings versus the pinned base's 6,379, with no additions. The comparison
script intentionally exits 1 for this receipt's consistency failure.
The publication tree and separately unpacked fixture archive pass the public
information scan; `git diff --check` passes.
