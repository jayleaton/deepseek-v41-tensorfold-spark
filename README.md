Follow me on X for more updates: https://x.com/jayleaton

Support me here: https://buymeacoffee.com/jayleaton

# DeepSeek-V4.1-Flash on TensorFold, 2x NVIDIA DGX Spark

Serve DeepSeek-V4.1-Flash (the 2.9 bpw EXL3 pack) across two NVIDIA DGX Sparks, tensor-parallel over the 200 Gb/s
CX7 link, behind an OpenAI-compatible API. The engine is [TensorFold](https://github.com/ashhart/TensorFold) 0.6.0
(pinned, unmodified submodule) plus two patches applied at image build: the two-Spark engine stack from the
[GLM-5.3-Flash recipe](https://github.com/jayleaton/glm53-tensorfold-spark) and a DeepSeek-V4.1-Flash family written
for this model: exact DSpark speculative decoding, CED bounded-replay and exact full prefill, 4 request slots over a
shared FP8 KV pool, NVMe sessions, native image input, structured output and tool calls, and both ranks failing fast
together instead of hanging.

> **Four Sparks:** `patches/0003` runs the same engine TP=4 on four DGX Sparks behind a switch (opt in with
> `--tp 4`, `scripts/serve4.sh`); two-Spark serving is unchanged. Setup, numbers and limits:
> [docs/FOUR_SPARKS.md](docs/FOUR_SPARKS.md).

> **Work in progress.** Measured on one pair of Sparks, against one baseline. Read
> [what is not solved](docs/KNOWN-ISSUES.md) before relying on it.

## Results

Two DGX Sparks (GB10, 128 GB each, one QSFP cable, RoCE), weights
[`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw).
Baseline: [MiaAI-Lab's 2x DGX Spark vLLM kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) on
the same pair and weights. Ours: [`config/prod.env.example`](config/prod.env.example) on engine `7bd2d67` (G19).

### Speed (tok/s)

| | **TensorFold (this recipe)** | MiaAI-Lab vLLM kit | ratio |
| --- | ---: | ---: | ---: |
| Code, 1 stream, greedy | **84.7** | 41.9-45.0 | **1.9-2.0x** |
| Prose (essay), 1 stream, greedy | **46.8** | 32.5 | 1.44x |
| Structured (count 1-200), 1 stream, greedy | **121.3** | 38.0 (50.2 on a JSON task) | **2.4-3.2x** |
| 1 stream, decode aggregate | **87.6** | 32.2 | **2.7x** |
| 2 streams, decode aggregate | **74.1** | 46.7 | 1.59x |
| 4 streams, decode aggregate (steady, all four live) | **101.1** (136.0) | 37.6 | **2.7x** |
| Cold prefill 8K / 32K / 64K / 128K (CED replay, the default) | **2,028 / 2,310 / 2,294 / 2,118** | 1,073 / 1,075 / 1,060 / 1,031 | **1.89-2.16x** |
| Cold prefill, `TF_DSV41_PREFILL=full` (exact: no replay) | **1,561 / 2,025 / 2,197 / 2,152** | same | **1.45-2.09x** |
| Start to ready | **34-44 s** | 378 s | ~9x |

Decode: mean of 2 boots of the shipped build. Prefill: 8K / 32K on the shipped `TF_DSV41_PREFILL_ADAPT_GIB=4.5`,
64K / 128K on the soaked 4.0. Start to ready not re-measured since G11.

### Quality and robustness

| | **TensorFold** | kit |
| --- | --- | --- |
| Teacher-forced top-1 agreement with the kit (8 prompts x 2,048 positions) | **0.9961** (first copy of each prompt: 0.9502); the same with every G13-G17 lever on or off | 1 by definition |
| MMLU-200, 0-shot, greedy, thinking off | **87.5%** | 87.5% |
| MMLU-200 with a 20-question preamble (~2.1K-token prompts), replay vs full prefill | 81.1% / 81.1% (178 of 180 answers equal) | - |
| Needle at 32K / 128K / 299K (replay prefill) | found (19.9 s / 74.0 s / 195 s) | - |
| Multi-step tool chains (`bench/tooleval/chains.py`, 6 scenarios, 12 points), thinking off / high | 11 / 12, 11 / 12 | - |
| tool-eval-bench category C (multi-step), thinking off | 8 / 8 (score 100) | - |
| Structured output: 12 JSON-schema cases + 10 tool-choice cases (drafted == serial == batched, valid, no markup) | 22 / 22 | - |
| Image input (native: DeepSeek's ViT on rank 0): 20 visual questions | 20 / 20 | - |
| 1 h soak (agent sessions to 120K tokens with screenshots, short chats with cancels, 24-200K prompts, image requests) | **906 requests, 0 errors**, 0 capacity refusals; RssAnon falling (-0.4 GiB/h a rank); MemAvailable min head 4.63 / worker 3.99 GiB | - |
| Stress: one 299K prefill + three 64K prompts decoding 2,048 tokens each + 4 image requests | every stream completes; 299K first token **157 s**; MemAvailable min head 5.07 / worker 4.81 GiB | 4.40 GiB at <= 256K (lighter load) |

Every run checks **drafted == serial**, **batched == alone**, and bit-identical outputs across prefill row sizes. How
each cell was measured, what is exact and what is approximate, and strict mode: [`docs/METHOD.md`](docs/METHOD.md).

## Quick start

```bash
git clone --recurse-submodules https://github.com/jayleaton/deepseek-v41-tensorfold-spark.git
cd deepseek-v41-tensorfold-spark
cp config/prod.env.example config/prod.env && $EDITOR config/prod.env
scripts/serve.sh build && scripts/serve.sh preflight && scripts/serve.sh start
```

Weights, Engram shards, requirements and unattended operation: [`docs/INSTALL.md`](docs/INSTALL.md).

## Documentation

| | |
| --- | --- |
| [INSTALL](docs/INSTALL.md) | requirements, weights, Engram shards, build, start, watchdog |
| [API](docs/API.md) / [IMAGES](docs/IMAGES.md) | endpoints, thinking and effort, image input |
| [OPERATIONS](docs/OPERATIONS.md) | settings, memory gates, turning each lever off |
| [METHOD](docs/METHOD.md) / [RESULTS](docs/RESULTS.md) / [BENCHMARKS](docs/BENCHMARKS.md) | how it was measured, every table, how to rerun |
| [CHANGELOG](docs/CHANGELOG.md) | bug fixes and new levers since G13 |
| [KNOWN-ISSUES](docs/KNOWN-ISSUES.md) | what is not solved |
| [PROJECT](docs/PROJECT.md) / [ENGINE](docs/ENGINE.md) / [ARCHITECTURE](docs/ARCHITECTURE.md) / [DECODE](docs/DECODE.md) | patches, layout, model split, decode roofline |
| [campaign/](docs/campaign/README.md) | the development log, windows G1-G19 |

## Credits

- [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash): DeepSeek-V4.1-Flash, its prompt encoding and its
  image path (the engine's image input re-implements `inference/vision.py` and `image_processor.py`, MIT; the
  unmodified files are test oracles in `patches/0002`);
  the ideas this engine implements come from DeepSeek's papers and code: the
  [V4.1-Flash technical report](https://arxiv.org/abs/2609.19969) (CED and bounded replay, CSA2, mHC),
  [DSpark](https://arxiv.org/abs/2607.05147) with [DeepSpec](https://github.com/deepseek-ai/DeepSpec), and
  [Engram](https://arxiv.org/abs/2601.07372) with [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram).
- [Ash Hart (ashhart) / TensorFold](https://github.com/ashhart/TensorFold): the engine, the EXL3 kernels, the drafting
  and the server this recipe builds on.
- Mia / MiaAI-Lab: the [DeepSeek-V4.1-Flash 2x DGX Spark vLLM kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)
  this recipe is measured against (whose packed Engram shards the measured runs used), and the
  [2.9 bpw EXL3 pack](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) on Hugging Face
  ([Mia-AiLab](https://huggingface.co/Mia-AiLab)); their GLM-5.3 kit's fat-expert MoE design also lives on in
  `patches/0001`.
- [dealignai](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw): the uncensored 2.9 bpw
  variant measured here.
- [turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3): the EXL3 format.
- [Cruz (vcruz305)](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe): the finding that DSpark
  verify and plain decode take different EXL3 paths in the vLLM kits (the case for exact speculative decoding), the
  expert-union counts per verify row behind our round model, and the host-side Engram hashing fix
  ([`docs/campaign/LANDSCAPE.md`](docs/campaign/LANDSCAPE.md), [`docs/campaign/TARGETS.md`](docs/campaign/TARGETS.md));
  his [SAGE 1.59 bpw pack](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw) was evaluated
  ([`docs/campaign/PARKED-SAGE-1.59.md`](docs/campaign/PARKED-SAGE-1.59.md)).
- PCTree, [arXiv 2608.02123](https://arxiv.org/abs/2608.02123): the parent-conditioned draft trees implemented as
  `TF_DSV41_TREE_PC` (measured, not adopted: [`docs/DECODE.md`](docs/DECODE.md)).
- [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) (Luke Alonso and contributors): the
  "RoCEnante" one-shot RoCE all-gather that `patches/0001`'s RoCE path (used by this family) adapts, and the prefill
  kernel designs it re-implements.
- [Reederey87](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark): the Apache-2.0 kernel code the GLM
  stack's fat-expert kernels adapt.
- [The vLLM project](https://github.com/vllm-project/vllm): the DeepSeek-V4 / V4.1 implementation our kernels' math
  follows (cited per file; no code copied).
- [bertholomus / BertholomusAI (Albert Lee)](https://github.com/bertholomus/TensorFold), branch `deepseek-v41-tp2`
  (Apache-2.0): ideas we re-implemented (no code copied): geometric context buckets for the CUDA graphs (ours: 1.25x
  past 32K), an allocator ceiling (its `_memory_ceiling`; ours `TF_DSV41_ALLOC_CEIL_GIB`, shipped off), the GIL
  switch-interval knob (`TF_DSV41_SWITCH_MS`) and failing fast across both ranks.
- Issue and pull-request authors on this repository: **WireLLM** (#3, the host memory growth), **ZackO2o** (#4, the
  missing `csa2` kernel sources) and **flashosophy** (#6, the four-Spark work that surfaced the `xgrammar` and
  `text_config` build bugs); see [Changes since G13](docs/CHANGELOG.md).
- [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar): the grammar engine behind structured output.
- [SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) and
  [Weschera/spark-bench](https://github.com/Weschera/spark-bench): the tool-calling benchmarks.
- [MMLU](https://github.com/hendrycks/test) (Hendrycks et al.): the 200 questions in `bench/data/`.
- NVIDIA: the DGX Spark and the PyTorch container.

## Licensing

This project's code, patches, scripts, benchmarks and docs: **Apache License 2.0** ([`LICENSE`](LICENSE),
[`NOTICE`](NOTICE)). TensorFold, the added files, the adapted kernels, the base image and the model weights each keep
their own terms: [`docs/LICENSING.md`](docs/LICENSING.md).
