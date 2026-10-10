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

This export passed `zig build -j4` (30/30 steps) and `zig build test -j4` (96/96 steps).
The unit suites passed 534 checks, with 11 hardware/checkpoint-dependent skips. The frozen HTTP corpus
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

The recorded runs used two DGX Spark nodes, TP=2 over RoCE, and q28-v2 EXL3 2.8 bpw. The proposed public pack is
[DeepSeek-V4.1-Flash-EXL3-2.8bpw](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw);
that repository name remains provisional until its publication is confirmed.

Zig ran the serving profile on this engine source; Python is a fresh run of the Python serving path the same night.
Both engines report the best of three warm repetitions. Replies match the Python reference (16/16 short replies,
and the long reply hashes). These are recorded runs, not new hardware qualification of this sanitized export.

| Workload | Python | Zig | Δ |
| --- | ---: | ---: | ---: |
| Code, one stream, greedy decode (tok/s) | 88.0 | 90.3 | +2.6 % |
| Code, four streams, greedy decode (tok/s) | 144.1 | 156.4 | +8.6 % |
| Code, one stream, T=0.7 decode (tok/s) | 91.3 | 92.2 | +1.0 % |
| Code, four streams, T=0.7 decode (tok/s) | 145.0 | 153.0 | +5.6 % |
| 131K decode (tok/s) | 67.0 | 69.3 | +3.3 % |
| Mixed load, four-stream code aggregate (tok/s) | 145.1 | 155.9 | +7.5 % |
| 32K prompt (tok/s) | 2,440 | 2,686 | +10.1 % |
| 131K prompt (tok/s) | 2,264 | 2,496 | +10.3 % |

Zig is ahead in every measured cell; long decode is near parity at 32K (+0.2 %) and +3.3 % at 131K. The public
[benchmark receipt](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/docs/zig-serving-q28-v2/docs/ZIG-RESULTS.md)
contains every cell, medians, first and cold repetitions, and the method.
