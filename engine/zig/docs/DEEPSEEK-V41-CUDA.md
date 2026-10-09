# DeepSeek-V4.1-Flash on CUDA

This port adds a DeepSeek-V4.1 family to TensorFold's Zig engine: EXL3 weight loading, CSA2 attention,
mHC, Engram table reads, tensor-parallel execution, DSpark drafting, keyed sampling, and serving through
the shared lane and HTTP server interfaces. It builds on TensorFold 1.0.2. Existing model families retain
their serving paths. The proposed integration is tracked in [issue #299](https://github.com/ashhart/TensorFold/issues/299).

The source follows TensorFold's [Apache-2.0 license](../LICENSE). Required third-party notices and licenses
are retained. Model weights have their own license.

## Build and host checks

Use the Zig version pinned in `.zig-version` (0.17.0). On a Linux development host:

```sh
zig build
zig build test
```

Without `-Dnvcc` or `-Dfatbins`, the build embeds empty CUDA kernel images. This checks compilation and
host behavior; the resulting binaries cannot perform CUDA inference without qualified kernel assets.
The host suite includes family configuration, drafting, sampling, serving, TP, KV, grammar, and frozen
HTTP corpus checks. Focused steps include `test-dsv41`, `test-dsv41-draft`, `test-dsv41-serve`,
`test-dsv41-sampling`, `test-server`, `test-tp`, `test-kv`, and `test-golden`.

The actual recipe export passed `zig build -j4` (29/29 steps) and `zig build test -j4` (89/89 steps).
The unit suites passed 494 checks, with 11 hardware/checkpoint-dependent skips. The frozen HTTP corpus
matched 312/319 answers, with seven known differences and zero unexpected differences. These checks
ran with Zig 0.17.0 on a Linux development host using a resource-limited wrapper, one job at a time.
Publication build and host-test outcomes are also recorded in the pull request. This guide does not assert
that upstream qualification gates passed. Device and model gates were not rerun for this export.
Required qualification includes greedy reference parity before drafting, drafted/plain equality,
concurrent/solo equality, resumed/fresh equality, prompt seams, cancellation, and TP failure handling.

The lean checker currently reports 512 findings on the pinned upstream base and 5,949 on the export.
This exceeds upstream's zero-added-findings convention. An upstream contribution is deferred until
performance is final; structural cleanup remains necessary before upstream qualification.

## Serving and opt-ins

Serving requires a compatible EXL3 pack, RoPE and Engram assets, rank-specific Triton AOT assets, CUDA
fatbins for the target GPU, and a matching TP fabric configuration. Both ranks must use identical engine
settings. See the public [serving recipe](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/docs/zig-serving-q28-v2/docs/ZIG-SERVE.md)
for asset generation, CUDA builds, rank startup, and the complete environment-switch table.

The export makes the new family's drafting, native large-segment prefill, lookup, session retention, and
Engram prefetch opt-in. Their controls are `TF_DSV41_DRAFTS`, `TF_DSV41_OWN_PREFILL`, `TF_DSV41_LOOKUP`,
`TF_DSV41_SESSIONS`, and `TF_DSV41_ENGRAM_PREFETCH`, each defaulting off. Existing shared server defaults
for other families are unchanged. Decode graphs and additional fused or overlapping paths are also
opt-in. Explicitly enable the desired settings on every rank; `LOOKUP_PLAN` additionally gates the
priced lookup planner.

The measured performance profile is not a strict upstream qualification profile. It enables bounded
replay and other explicitly configured performance choices, including routed-expert pruning. Do not
interpret its token parity with the Python reference at those settings as strict full-prefill fidelity.
Turning native prefill off uses decode windows for prompt processing, so the measured prompt rates do
not describe the default configuration.

## Passed performance measurements

The recorded runs used two DGX Spark systems, TP=2, and q28-v2 EXL3 2.8 bpw. The proposed public pack is
[DeepSeek-V4.1-Flash-EXL3-2.8bpw](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw);
that repository name remains provisional until its publication is confirmed.

Zig measurements are single runs per cell, after one short suite warm-up. Python short-decode reference cells select the best
of three runs. This asymmetric methodology and small sample size limit comparisons. Recorded replies
match the Python reference at the same measured settings. These are historical passed runs, not new
hardware qualification of this sanitized export.

| Workload | Python | Zig all-on |
| --- | ---: | ---: |
| Code, one stream, greedy decode (tok/s) | 86.6 | 87.6 |
| Code, four streams, greedy decode (tok/s) | 145.4 | 147.4 |
| Code, one stream, T=0.7 decode (tok/s) | 92.2 | 90.0 |
| 32K decode (tok/s) | 51.1 | 61.1 |
| 131K decode (tok/s) | 53.4 | 64.1 |
| Mixed long decode, 131K (tok/s) | 69.0 | 67.2 |
| 32K prompt (tok/s) | 2,436 | 2,507; 2,663 with PF_4K |
| 131K prompt (tok/s) | 2,266 | 2,318; 2,486 with PF_4K |

Zig is about 20% ahead on the measured long-context decode cells, roughly at parity on short decode,
and slightly behind on sampled cells and mixed long decode. `PF_4K` improves measured prompt throughput
but needs memory headroom and may fall back to 2K. It does not beat Python in every cell. The public
[benchmark receipt](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/docs/zig-serving-q28-v2/docs/ZIG-RESULTS.md)
contains the complete comparison and its methodology.
