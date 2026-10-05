Follow me on X for more updates: https://x.com/jayleaton

Support me here: https://buymeacoffee.com/jayleaton

# DeepSeek-V4.1-Flash on TensorFold, 2x NVIDIA DGX Spark

Serve DeepSeek-V4.1-Flash (the 2.9 bpw EXL3 pack) across two NVIDIA DGX Sparks, tensor-parallel over the 200 Gb/s
CX7 link, behind an OpenAI-compatible API. The engine is [TensorFold](https://github.com/ashhart/TensorFold) 0.6.0
(pinned, unmodified submodule) plus two patches applied at image build: the two-Spark engine stack from the
[GLM-5.3-Flash recipe](https://github.com/jayleaton/glm53-tensorfold-spark) and a new DeepSeek-V4.1-Flash family
written for this model: CSA2 attention with FP8 KV rows and the lightning indexer, Single-Pass mHC, Engram rows read
from local NVMe, EXL3 expert and dense kernels tuned for GB10, **exact** DSpark speculative decoding, CED
bounded-replay prefill (and an exact full prefill at 1.45-2.09x the kit), 4 request slots of up to 300K tokens over a
shared FP8 KV pool (614K tokens), sessions with an NVMe tier, prepared per-rank folders for ~40 s restarts, native
image input, structured output and DSML tool calls, and both ranks failing fast together (~1 s) instead of hanging.

> **Work in progress.** Measured on one pair of Sparks, against one baseline. Knobs, defaults and numbers may change.
> Read [What is not solved](#what-is-not-solved) before relying on it.

SPDX-License-Identifier: Apache-2.0 (this project's own code, scripts, benchmarks and docs; see [Licensing](#licensing)).

## Results

Hardware: two DGX Sparks (GB10, 128 GB unified memory each), one QSFP cable between their CX7 ports, RoCE. Weights:
[`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw),
the uncensored variant of [Mia-AiLab's 2.9 bpw EXL3 pack](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw)
(same layout). Baseline: [MiaAI-Lab's 2x DGX Spark vLLM kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)
on the same pair and the same weights (vLLM TP=2, DSpark k=3, its moe_x kernel engaged), measured with our own
clients on 2026-10-01. Our side: the configuration in [`config/prod.env.example`](config/prod.env.example) on the
engine these patches build (development commit `7bd2d67`, what production runs since G19). Raw files:
[`results/`](results/README.md).

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

The decode cells are G19's check of the shipped build (`7bd2d67` with the production words, mean of 2 boots,
[`docs/campaign/G19-RESULTS.md`](docs/campaign/G19-RESULTS.md)); at temperature 0.7: code 78.2, prose 48.0, structured
122.9. The prefill cells are G19's 2,048-row windows: 8K and 32K on the shipped words (`TF_DSV41_PREFILL_ADAPT_GIB=4.5`),
64K and 128K on the soaked 4.0 (the same 32K within 1 tok/s; 4.5 was not run at 64K / 128K). Over HTTP (the stress
server's first-token probe, best of 2) replay reads 2,052 / 2,224 / 2,305 / 2,290 / 2,202 tok/s at 8K / 16K / 32K /
64K / 128K. Start to ready was measured on the G10 / G11 test servers and not re-measured since.

Where it moved since the last update (G13, `767ad9f`): code 82.8 -> 84.7, prose 44.25 -> 46.8, structured 117.8 ->
121.3, C1 / C2 / C4 85.1 / 70.3 / 96.6 -> 87.6 / 74.1 / 101.1 (G14's window levers and calibration VERSION 4); replay
prefill at 32K 2,043 -> 2,310 and full prefill 1,004 -> 2,025 (G17's prefill kernels, the full-mode cone, G19's
2,048-row windows with adaptive rows). Between G16 and G18 production ran 1,024-row windows for memory (replay 32K
1,753 -> 1,990); G19 brought 2,048 back.

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

Exactness, checked in every run: **drafted == serial** (speculative decoding never changes a reply, at T = 0 and
T > 0), **batched == alone** (every concurrent stream equals the same request run alone, serial: `--c-exact`), and the
prefill levers give **bit-identical outputs across prefill row sizes** (2,048 / 1,024 / 512 rows, adaptive or fixed:
first token and a 32-token reply digest equal at every size, mode and boot in G17-G19; the CPU suites compare whole
slot states). Top-1 ran on G17's build (`0bdd276`, which G19 changes only in the prefill row choice), MMLU-200 and the
tool chains (thinking off) on G16's (`da5ae43` with native images), the soak and the stress on `7bd2d67`; structured output, the
needles and the 20-question MMLU are older runs on the same paths (G10 and G7).

### What measures what

- **Decode cells, ours:** `m2bench` (inside the engine, both ranks, no HTTP; `scripts/serve.sh run`), tok/s from the
  first to the last token of a 384-token reply, the median of the repetitions (2 by default), request slots sized
  for 16K tokens. Prompts: an LRU-cache
  class with tests (code), a 400-word essay (prose), counting 1 to 200 (structured). The cells are the mean of G19's
  two boots of the shipped build (`results/campaign/G19-20261005/m2-g19s1-combo-b{1,2}.json`, `g15-depth-speed.txt`).
- **Decode cells, kit:** HTTP clients (`glmbench`, `multiturn` of the GLM recipe): code = a 64-token code reply
  (41.9) and a 512-token one (45.0), prose = a 200-token essay, structured = count 1-200, thinking off.
- **2 / 4 streams:** both sides report decode aggregate = all tokens / (last token - first token). Ours mixes code,
  prose, a JSON task and a copy-heavy edit, every other stream at T = 0.7; the kit's mix is its chat / code prompts,
  256 tokens each. Same metric, different prompts. "Steady" is the aggregate while all four streams are live: the mixed
  cell's wall is mostly the prose stream finishing alone ([`docs/campaign/G17-LEVERS.md`](docs/campaign/G17-LEVERS.md)).
- **Prefill:** one request, cold, after a 2K warm-up. Ours: fresh random text through `m2bench --prefill
  --prefill-reply 32` (the median of 2 boots; the first token and a 32-token reply digest are compared with the
  1,024-row reference at every size). The kit: a repeated filler document over HTTP.
  Repeated fillers can make Engram reads look cheaper, so the kit's cells are not pessimistic.
- **Start to ready:** ours = `docker run` to `/v1/models` answering, with prepared folders and compiled kernels
  cached, page cache dropped first. The kit's = its start script to `/health` on freshly rebooted nodes. The first
  start of a new image is slower (kernel builds, ~80 s) and the very first writes the prepared folders (~95 GB a node).
- **Top-1:** the kit's `prompt_logprobs` (top 5) over 8 built-in prompts repeated to 2,048 tokens
  ([`results/kit-baseline/oracle-kit.json`](results/kit-baseline/oracle-kit.json)), our engine teacher-forced on the
  same token ids. "First copy" counts only each prompt's first pass, before the model can copy itself.
- **Not RigMark.** No RigMark receipt exists for this model yet; every cell comes from the clients above.

[`docs/RESULTS.md`](docs/RESULTS.md) has every table, the lever-by-lever history and the negatives;
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) how to rerun each cell.

### What "exact" means here, and what is approximate

- **Exact:** speculative decoding never changes a reply. Every verify row is the serial step at its position (row-
  invariant kernels) and its token is the request's keyed choice at that absolute position, so drafted == serial at
  T = 0 and at T > 0, and batched == alone. Every benchmark run checks it (`exact_all`).
- **Approximate by design, on in the measured config:**
  - CED decoder bounded replay for prompts (`TF_DSV41_PREFILL=replay`, DeepSeek's own technique: the decoder half
    runs only over a prompt's last 128 tokens). `full` is the exact prefill: with `TF_DSV41_FULL_CONE=1` (on in the
    config, the same bits as plain `full`) the decoder runs only over the ~2,541 rows whose state survives the prompt,
    so `full` is now 1.45-2.09x the kit too.
  - Routed-expert pruning in decode (`TF_DSV41_EXPERT_TOPP=0.85`, at least 3 experts, renormalized): +5% decode,
    MMLU-200 88.5% alone. Delete three lines of the config to serve the unpruned model.
  - mHC mixing weights in bf16 (`TF_DSV41_MHC_FN=bf16`): +2-5% prose, top-1 vs the kit 0.9963.
- **Not bit-identical to the kit.** Different kernels and summation orders; the agreement is the top-1 row above.
- **Every lever added since G13 is exact:** the G14 window levers, the G17 prefill kernels, the full-mode cone and the
  adaptive prefill rows change no bits (on == off, compared on the GPU and in the CPU suites).

### Strict mode: every precision trade off

Measured in G13 and not re-run since. The same engine and the G13 rewrites (which change no bits) with every knob
that trades precision turned off:
`TF_DSV41_EXPERT_TOPP=0` (no expert pruning), `EXPERT_RENORM=orig`, `MHC_FN=fp32`, `KIT_ROUNDING=0`, `LOGITS=fp32`,
`INDEX_KV=bf16`, `PREFILL=full` (no bounded replay). The fast prefill GEMMs and the fused prefill attention stay on.
Measured in G13, same build, same session ([`docs/campaign/G13-RESULTS.md`](docs/campaign/G13-RESULTS.md)):

| | strict | production | kit | strict / kit |
| --- | ---: | ---: | ---: | ---: |
| Code, 1 stream, greedy | **76.9** | 82.8 | 41.9-45.0 | 1.71-1.83x |
| Prose, 1 stream, greedy | **44.2** | 44.25 | 32.5 | 1.36x |
| Structured, 1 stream, greedy | **111.9** | 117.8 | 38.0-50.2 | 2.24-2.95x |
| 1 / 2 / 4 streams, decode aggregate | **79.0 / 64.0 / 89.5** | 85.1 / 70.3 / 96.6 | 32.2 / 46.7 / 37.6 | 2.45x / 1.37x / 2.38x |
| Cold prefill 8K / 32K / 64K / 128K | **915 / 965 / 959 / 923** | (replay) 1,833-2,068 | 1,073 / 1,075 / 1,060 / 1,031 | **0.85-0.90x** |
| Teacher-forced top-1 vs the kit | 0.9963 | 0.9961 | 1 | |

Decode costs 5-9% against production (prose at T = 0 is level only because the strict reply drafts a little better on
that prompt; at T = 0.7 it is -6.9%), and is still 1.4-2.9x the kit. Prefill without replay was below the kit then;
G17's full-mode cone and prefill kernels (all exact) have since taken `full` to 1,561-2,197 tok/s on the production
words, but a strict-mode receipt on the current engine has not been run. Strict MMLU and tool chains were not run.

### Where we are not at 2x

Prose (1.44x), 2 streams (1.59x) and full-mode prefill of short prompts (8K: 1.45x). Prose drafts poorly: DSpark
keeps ~1.7 tokens a round on prose against ~3.7 on code, so prose speed is the verify window's cost. A 1-row window is
~21.8 ms since G14 (23.4 after G13, 26.9 before) against a
bandwidth floor of ~17 ms a rank (the 2.9 bpw weights read once), and the second row costs ~6 ms more because a
second token brings ~5 new experts a layer. That is why G13's rewrites helped prose (+7.2%) far more than code (+1.3%:
code verifies ~4.6 rows a round, where the savings are smaller). 2 streams pair a code stream with a T = 0.7 prose
stream, so the prose stream sets the pace; the same holds for the 4-stream mixed cell (101 aggregate, 136 while all
four are live). Full prefill under ~2.6K tokens has no cone to skip.
[`docs/DECODE.md`](docs/DECODE.md) has the roofline and what was tried.

## Quick start

Requirements:

- two DGX Sparks with their CX7 ports cabled and addressed (one link subnet), Docker with the NVIDIA runtime on both,
  and passwordless ssh from the head to the worker over the link;
- on each node's local NVMe: ~100 GB for the weights, ~95 GB for the Engram shards, ~95 GB for the prepared
  folders, and room for the session tier (`TF_DSV41_SESSION_DISK_GIB`, 128 GB by default);
- nothing else on the GPUs: the stack plans for a 4-5 GiB MemAvailable floor out of 128 GB a node.

**1. Clone and configure** (on the head):

```bash
git clone --recurse-submodules https://github.com/jayleaton/deepseek-v41-tensorfold-spark.git
cd deepseek-v41-tensorfold-spark
cp config/prod.env.example config/prod.env
$EDITOR config/prod.env      # every <placeholder>: WORKER_SSH, HEAD_IP, the paths on each node; check the NIC names
```

**2. Weights** (on both nodes, byte-identical):

```bash
hf download dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw --local-dir <HEAD_MODEL>    # and <WORKER_MODEL>
```

**3. Engram shards.** The EXL3 packs do not carry the Engram tables (layers 1 and 14, ~101 GB each). They come from
DeepSeek's original checkpoint; each rank keeps its half of the hash heads (~47 GiB a layer) on local NVMe:

```bash
mkdir -p <src> && hf download deepseek-ai/DeepSeek-V4.1-Flash model.safetensors.index.json --local-dir <src>
python3 scripts/pack_engram.py --src <src> --list                 # the shard files that hold the tables
hf download deepseek-ai/DeepSeek-V4.1-Flash <those files> --local-dir <src>
python3 scripts/pack_engram.py --src <src> --config <HEAD_MODEL>/config.json --out <HEAD_ENGRAM> --rank 0
python3 scripts/pack_engram.py --src <src> --config <HEAD_MODEL>/config.json --out <dir> --rank 1   # copy to <WORKER_ENGRAM> on the worker
python3 scripts/pack_engram.py --src <src> --config <HEAD_MODEL>/config.json --check <HEAD_ENGRAM> --rank 0
```

`pack_engram.py` writes the format the engine reads (`engram-l{1,14}-r{rank}of2.bin`) from the source tensors'
raw bytes; it is tested against the engine's reader on synthetic tables. The measured runs used shards packed by
the MiaAI-Lab kit (`./start.sh pack`), which writes the same format; `--check` compares either with the source.

**4. Build, check, start:**

```bash
scripts/serve.sh build        # docker/Dockerfile: TensorFold v0.6.0 + patches/, shipped to the worker, then prebuild
scripts/serve.sh preflight    # image on both nodes, weights, Engram shards, RoCE ports, free ports, idle GPUs
scripts/serve.sh start        # memory gate, rank 1 then rank 0, /v1/models, slot check, canary
```

`build` ends with `scripts/serve.sh prebuild`: the CUDA extensions (18, the G13-G17 kernels included) are compiled
into the `CACHE_VOL` volume on both nodes with no weights loaded, so no extension is built beside the weights. It then
copies the measured dense-prefill tuning table (`config/pfdense-table.json`) into the volume and, for native image
input, fetches the image routing bias the EXL3 packs dropped (66 KB of `deepseek-ai/DeepSeek-V4.1-Flash` by HTTP range
requests: both nodes need network access once; `scripts/serve.sh cache` repeats this step alone). Run it again after
clearing the volume. The first start compiles the Triton kernels and writes the prepared rank folders
(`TF_DSV41_PREPARED_WRITE=1`, ~95 GB a node, several minutes); later starts read them back in ~40 s. Then:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "DeepSeek-V4.1-Flash-TF",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}], "reasoning_effort": "low"}'
scripts/serve.sh status | logs [0|1] | canary | stop | restart
```

**5. Run it unattended** (optional): a watchdog tick every minute (heals after 3 bad ticks, at most every 30 min; a
fail-fast exit, code 70, heals on the first tick, at most every 2 min) and a start at boot.

```bash
mkdir -p ~/.config/systemd/user && cp scripts/systemd/dsv41-* ~/.config/systemd/user/
$EDITOR ~/.config/systemd/user/dsv41-*.service        # WorkingDirectory= this checkout
loginctl enable-linger "$USER"
systemctl --user daemon-reload && systemctl --user enable --now dsv41-tf-watchdog.timer && systemctl --user enable dsv41-boot-start.service
```

`scripts/serve.sh` drops the page cache on both nodes around a start (`DROP_CACHES=1`: needs `sudo -n` or root for
`/proc/sys/vm/drop_caches`; otherwise it logs and continues). [`docs/OPERATIONS.md`](docs/OPERATIONS.md) covers the
knobs, the memory gates, the watchdog, and how to turn each lever off.

## API

OpenAI-compatible on `HOST:PORT` (`127.0.0.1:8000` by default: put your own proxy and authentication in front):
`/v1/chat/completions` (streaming, tool calls, `response_format`), `/v1/completions` (text or token ids),
`/tokenize`, `/v1/models` (`max_model_len`), `/health`, `/metrics`.

Thinking follows DeepSeek-V4.1's encoding (`Reasoning Effort: N (range 1-100)`), on by default
(`TF_DSV41_THINKING=0` turns it off). Both the top-level `reasoning_effort` and
`chat_template_kwargs.reasoning_effort` are read (the kwargs win):

| value | thinking | effort |
| --- | --- | ---: |
| `none`, `minimal` | off | - |
| `low` | on | 50 |
| `medium`, `high` (default: `TF_DSV41_DEFAULT_EFFORT`) | on | 75 |
| `xhigh`, `max` | on | 100 |
| an integer 1-100 | on | that |

`chat_template_kwargs.enable_thinking` (or `thinking`) true / false sets the mode directly. Note: the kit's vLLM
path renders `low` as 25; we keep DeepSeek's 50.

Images: see [Images](#images).

## Images

New since the G13 publication (G15a-G16). DeepSeek-V4.1-Flash sees images: the EXL3 packs keep its ViT, and the
engine runs DeepSeek's tower and aligner (BF16, ~0.97 GB) **on rank 0 only**; the image rows reach rank 1 through
the embedding exchange both ranks already do. Image positions route experts with the release's `gate.bias_vl` and get
no Engram contribution, as in DeepSeek's reference (`inference/vision.py`, `image_processor.py`, `model.py`).

```bash
IMG=$(base64 -w0 photo.png)
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "DeepSeek-V4.1-Flash-TF",
  "reasoning_effort": "none", "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What does the sign in this photo say?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG"'"}}]}]}'
```

**Config** (on in `config/prod.env.example`):

```
TF_DSV41_IMAGES=native                    # placeholder (engine default) | reject | native
TF_DSV41_BIAS_VL=/cache/dsv41-bias-vl     # folder holding bias_vl.safetensors, in the cache volume
TF_DSV41_VISION_PREP_MB=64                # preprocessed images cached by data: URL digest
```

**The image routing bias.** Both EXL3 2.9 bpw packs dropped the 43 `*.ffn.gate.bias_vl` tensors (40 layers + 3
DSpark blocks). `scripts/serve.sh prebuild` (or `scripts/serve.sh cache` alone) fetches them into the cache volume on
both nodes, 66 KB read by HTTP range requests from `deepseek-ai/DeepSeek-V4.1-Flash` (each node needs network access
once). By hand, inside the image with the volume mounted:

```bash
python -m tensorfold.families.deepseek_v41.cuda.bias_vl_fetch /cache/dsv41-bias-vl        # or --src <local release dir>
```

The file records the source revision and each tensor's SHA-256. Without it, native mode still runs and image rows
route with the text bias (what the vLLM kit does); the boot log says so. Users who do not want images on the GPU at
all set **`TF_DSV41_IMAGES=placeholder`** (the engine's default, no tower loaded, no bias needed): each image part
becomes a short notice telling the model it cannot see the image, so agent turns with screenshots still work;
`reject` answers HTTP 400 instead.

**Accepted and refused.** Image parts in user, tool and system messages: OpenAI `image_url` (a `data:` URL or a
public `https` URL, fetched with a 10 s / 30 s timeout), `input_image`, `image`, and Anthropic `source` blocks. At
most **8 images a request** (`TF_DSV41_VISION_MAX_IMAGES`; long agent histories should drop old screenshots), 20 MB
and 64 M pixels an image, at most 1,024 positions an image; `http`, private and link-local addresses are refused. A
`<｜deepseek_image｜>` token typed into text is escaped to plain text, never treated as an image.

**Measured** (G15b-G16): visual questions 20 / 20; text-only replies byte-identical with images on, off or
placeholder (4 / 4 HTTP text probes, and the speed suite's replies); hit == cold, batched == alone and turn-2 replay
exact with images in the prompt; 4 image requests during a 299K prefill all answered with the worker at >= 5.1 GiB.
Cost: the head's MemAvailable at boot -0.8 to -1.4 GiB. The duplicate-image bug (the same screenshot twice in one
request killed both ranks) is fixed in this engine (see [Changes since G13](#changes-since-g13)).

## The engine

`vendor/TensorFold` is upstream TensorFold v0.6.0, unmodified. `patches/` holds two patches, applied in order by the
Dockerfile:

| patch | what | licence |
| --- | --- | --- |
| [`0001-spark-stack-060.patch`](patches/0001-spark-stack-060.patch) | the GLM-5.3-Flash two-Spark engine (`families/glm5_next/spark/`) rebased onto 0.6.0, the CUDA communicator interface (`cuda/comm.py`), the family `CUDA_SERVE` hook (`cli.py`, `families/glm5_next/__init__.py`), the server's descriptor fix (`server/cancellation.py`), packaging (`pyproject.toml`), recipes and tests |
| [`0002-deepseek-v41-family.patch`](patches/0002-deepseek-v41-family.patch) | `families/deepseek_v41/` and its tests, the EXL3 linear's device-side skip (`cuda/exl3/linear.*`), fp64 in `cuda/comm.py`, `--kv-dtype fp8` (`cli_args.py`), model aliases in the GLM server, G14's RoCE changes to the GLM stack (the faster all-gather kernel, the host mailbox the round plan uses, the exchange benchmark and split tools, `families/glm5_next/spark/roce*`), packaging (the CUDA sources and headers as package data), NOTICE entries |

Together they are every engine change production runs (development commit `7bd2d67`, plus `66d0dcd`, a packaging-only
fix: 455 files over v0.6.0): applying them to v0.6.0 reproduces that tree except for reworded comments, two
documentation strings, one benchmark argument and the excluded draft-vocabulary files ([`docs/ENGINE.md`](docs/ENGINE.md)
lists each difference).

[`docs/ENGINE.md`](docs/ENGINE.md) explains how the patches were produced, how to get the same tree as a git branch,
and how to run the engine's test suites. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) summarises the model and the
TP=2 split.

## What is not solved

- **4 streams on the mixed cell are below the 120 target.** C4 mixed decode aggregate is ~101 tok/s (136 while all
  four streams are live). The cell's wall is mostly its prose stream finishing alone; 4 distinct code streams are
  estimated at ~125-135 but that cell has not been measured. The built-but-off `TF_DSV41_SPEC_NUCLEUS` measured
  +1.3% on C2 steady and -0.6% on C4 steady: not adopted.
- **Prose drafting.** DSpark keeps ~1.7 tokens a round on prose (3.7 on code), so prose (1.44x) and 2 streams (1.59x)
  are not at 2x. Self-distilling the drafter on our own logs helped prose and hurt code; not shipped.
- **The graph cache still shrinks under repeated memory dips.** Long prefills at 2,048 rows dip the worker's memory;
  G19 keeps the graphs through dips inside 64 decode-only rounds of a prefill, but in the 1 h soak five longer dips
  still took the LRU cap 48 -> 11 (G18: 48 -> 8). Fewer cached graphs cost speed until the next restart.
- **Capacity refusals of streaming requests arrive as in-stream errors.** When the priced admission floor refuses a
  streaming request after its `200` headers went out (it waited up to 30 s for memory first), the client gets an SSE
  `{"error": ...}` event, not an HTTP 503. Sending the headers lazily would fix it. G19's pricing at the smallest
  adaptive step removed the case the soak hit (0 refusals since).
- **The adaptive-row threshold.** Production runs `TF_DSV41_PREFILL_ADAPT_GIB=4.5`; the 1 h soak ran 4.0 (worker
  minimum 3.99 GiB, 9 MiB under our own 4.0 line, no errors). 4.5 has more margin but was not soaked, and not measured
  at 64K / 128K. The engine default (5) makes rows flap on GB10 (replay 32K 1,912 tok/s).
- **The worker is the binding node.** Its MemTotal is 2 GiB below the head's (firmware), and its MemAvailable at
  serving varies 6.2-8.4 GiB between identical boots, outside torch. earlyoom on the nodes (`-m 2,1` here) is the last
  backstop under our 4-5 GiB floors: it kills a rank below ~2.4 GiB, and fail-fast then takes the pair down in ~1 s
  for the watchdog to restart.
- **Two G13 rewrites ship off.** The shortened decode MoE chain (`TF_DSV41_MOE_FUSED`) is exact but measured +0.7 ms
  on a 1-row window in the graph. (`TF_DSV41_BRANCHES`, off in G13, is on since G14 on dedicated priority streams.)
- **Strict mode on the current engine** (every precision trade off, `full` prefill with the cone) has not been
  measured; the strict numbers above are G13's.
- **`/health`** reports `drafted_total` / `accepted_total` as 0 for this family (the counters are not wired).
- **No trimmed draft-head vocabulary is shipped** (`TF_DSV41_DRAFT_HEAD=trim`, off by default and not adopted). The
  development ranking was counted from private chat transcripts and is excluded, so `trim` needs
  `TF_DSV41_DRAFT_VOCAB=<file>` and `tests/test_dsv41_draft_head.py::test_shipped_ranking` fails. A ranking of your
  own traffic (`scripts/campaign/draftvocab.py`) or of public text (the GLM recipe's `bench/draftvocab_public.py`
  method on this tokenizer) can be used.

## Layout

| path | what |
| --- | --- |
| `vendor/TensorFold` | upstream TensorFold v0.6.0 (submodule) |
| `patches/` | the engine changes ([`docs/ENGINE.md`](docs/ENGINE.md)) |
| `docker/Dockerfile` | the image: NVIDIA PyTorch 26.07 + xgrammar + TensorFold with the patches |
| `config/prod.env.example` | the measured configuration, with placeholders for your hosts and paths |
| `config/pfdense-table.json` | the measured tile table of the dense prefill GEMM (`TF_DSV41_PF_DENSE_TABLE`; `scripts/serve.sh prebuild` copies it into the cache volume) |
| `scripts/serve.sh` | build / prebuild / cache / preflight / start / stop / status / watchdog / `run` (engine benchmarks on both ranks) |
| `scripts/prebuild_ext.py` | builds every CUDA extension a rank loads (`scripts/serve.sh prebuild` runs it in the image on both nodes) |
| `scripts/pack_engram.py` | the per-rank Engram shards from DeepSeek's checkpoint |
| `scripts/canary.py`, `scripts/boot-start.sh`, `scripts/systemd/` | post-start canary, start at boot, watchdog units |
| `scripts/check-public.sh` | the sanitizer this repository was checked with |
| `bench/` | HTTP clients: quality (MMLU, needles), structured output, soak, stress, tool calling |
| `docs/` | results, benchmark method, architecture, decode roofline and lessons, operations, the engine |
| `results/` | the raw files behind the tables ([`results/README.md`](results/README.md)); `results/campaign/` the tracked results of the development windows G1-G19 |
| `docs/campaign/` | the development log: plans, targets, the landscape study, the results of every window G1-G19, the vision design ([`docs/campaign/README.md`](docs/campaign/README.md)) |
| `engine/`, `tests/` | the development staging tree: the PyTorch reference model (the correctness oracle), the kernels and serving layer before they were ported into the TensorFold family, and their tests |
| `scripts/campaign/` | analysis tools of the windows (nsys window / idle / skew breakdowns, summaries), the draft-vocabulary study tool, the porting script |

## Licensing

| Part | License |
| --- | --- |
| This project's code, patches, scripts, benchmarks and docs | **Apache License 2.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). Redistributions, modified or not, must keep the copyright line and the NOTICE attributions and state their changes. |
| TensorFold (`vendor/TensorFold`) | Apache License 2.0 from 0.6.0 (code written before 0.6.0 keeps its MIT notice), Copyright 2026 TensorFold contributors; unmodified submodule, the patches are applied at build time. The TensorFold code the patches modify stays under its license. Its third-party notices: `vendor/TensorFold/THIRD_PARTY_NOTICES.md` (the patches extend it). |
| Files the patches add | keep the SPDX notice written in them: the DeepSeek-V4.1-Flash family (`families/deepseek_v41/`, its tests) is MIT, Copyright (c) 2026 Jay Leaton; the GLM Spark engine (`families/glm5_next/spark/`) is MIT ([`NOTICE`](NOTICE)). |
| RoCE all-gather and fast-prefill kernels in `patches/0001` | adapted from / re-implementing [b12x](https://github.com/local-inference-lab/b12x) (Apache-2.0, Luke Alonso and the b12x contributors); details in [`NOTICE`](NOTICE). |
| Fat-expert MoE kernel structure in `patches/0001` | adapted from the Apache-2.0 [Reederey87 kit](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) (code MiaAI-Lab contributed under MIT before 2026-09-07); its NOTICE is reproduced in [`NOTICE`](NOTICE). |
| Ported upstream code in `patches/0001` | from later TensorFold releases (0.3.6.2, 0.5.0), MIT, Copyright (c) 2026 TensorFold contributors; each ported piece names its source commit. |
| xgrammar (structured output) | Apache-2.0 ([mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)), installed into the image by pip, not vendored. |
| Docker base image | NVIDIA Deep Learning Container License (`nvcr.io/nvidia/pytorch:26.07-py3`) |
| Model weights (not included) | `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw` (measured here), its base `Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`, and `deepseek-ai/DeepSeek-V4.1-Flash` (for the Engram tables): each under its model card's terms. The uncensored weights have refusals removed; you are responsible for how you use them. |

Nothing from the MiaAI-Lab DeepSeek kit's AGPL-3.0 code is included: the engine reads the pack and the Engram shard
format (file-format facts), and implements DeepSeek's prompt encoding from DeepSeek's own MIT `encoding.py`.

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
  `text_config` build bugs); see [Changes since G13](#changes-since-g13).
- [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar): the grammar engine behind structured output.
- [SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) and
  [Weschera/spark-bench](https://github.com/Weschera/spark-bench): the tool-calling benchmarks.
- [MMLU](https://github.com/hendrycks/test) (Hendrycks et al.): the 200 questions in `bench/data/`.
- NVIDIA: the DGX Spark and the PyTorch container.

## Changes since G13

The production engine went `767ad9f` (G13) -> `356a188` (G14) -> `7954c1d` (G15a) -> `da5ae43` (G16) -> `cd4245a`
(G18) -> **`7bd2d67`** (G19). Each window's write-up is in [`docs/campaign/`](docs/campaign/README.md).

### Bugs fixed

- **A duplicate image in one request killed both ranks** (G16). The vision row store counted a reference per
  occurrence of an image, so an agent history that resends the same screenshot released it early; the next request
  with that image raised `KeyError` inside rank 0's prefill forward, rank 1 waited in the exchange, and both ranks
  died. The store now counts each image once per request table. Two tests that fail on the old code.
- **The serving memory leak was the CUDA graph cache** (G16). The verify-graph key carries the context bucket (then
  2,048 tokens), so agent sessions growing to 120K tokens kept capturing new graphs (132 held in one soak, ~15 MiB of
  driver host memory each), until MemAvailable sat at the capture floor. Now an LRU cache capped at 48
  (`TF_DSV41_GRAPHS_MAX`) with context buckets 1.25x apart past 32K (26 buckets to 300K instead of 147; replay ==
  eager as before). Host trims also moved off the round thread (they were not the leak).
- **Graph-eviction cascade** (G17-G19). Under the capture floor the cache evicted a quarter of its graphs at every
  refused capture, so the long-prefill dips of one soak walked the cap from 48 to 8. Now one eviction a memory dip (then the freed pool
  is released and 64 capture attempts are skipped), and captures refused within 64 rounds of a prefill round evict
  nothing (`TF_DSV41_GRAPH_DIP_ROUNDS`).
- **NVMe session-index memory drift** (G18). The NVMe session tier's in-RAM index kept every parked entry's token ids as
  a Python `list[int]` (~36 bytes a token): +0.4 GiB/h of anonymous memory a rank under agent traffic, and 0.5-2.5 GiB
  a rank once the 128 GB tier fills (the slow drift seen since G6). The index now keeps (length, page chain, the < 256
  tokens past the last full page). RssAnon slope after the fix: -0.4 GiB/h.
- **Fail fast across both ranks** (G17-G18, `TF_DSV41_FAILFAST=1`, default). An exception on one rank inside a round
  used to leave the other waiting in an exchange: 120 s (RoCE), forever (NCCL), then the plan link's 300 s, then three
  watchdog ticks. Now both ranks fail the in-flight requests with a 500, tell each other over the plan link's TCP
  connection, and exit with code 70 within ~1-3 s (a CUDA out-of-memory on rank 1: both gone in 3.1 s, the client's
  500 names the error). `scripts/serve.sh watch` heals an exit 70 on its first tick (at most every 2 min).
- **A two-rank, priced memory floor** (G17, `TF_DSV41_FLOOR_PRICE=1`, default). Admission ran on rank 0 only with a
  zero price, so a 299K prompt was admitted at 6.2 GiB on the worker (never consulted) and its own prefill took it to
  3.5. Now rank 1 reports its usable memory every 0.5 s and a prompt is admitted only if its priced prefill transient
  fits on the tighter rank; otherwise it waits.
- **The "empty reply" was a capacity refusal priced too high** (G19). The floor priced an 18K prompt at the full
  2,048-row transient and refused it after 30 s (as an in-stream error: the 200 had gone out). Admission now prices at
  the smallest adaptive step; the soak client records in-stream errors.
- **Adaptive prefill rows** (G19, `TF_DSV41_PREFILL_ADAPT=1`, default). Rank 0 picks each round's prompt rows (2,048 /
  1,024 / 512) from the tighter rank's memory, so 2,048-row windows (+13-16% prefill) no longer push the worker under
  the floor in long prompts. Bit-identical outputs at any row count.
- **Fail-fast exit code race** (G18): rank 1 could exit 1 instead of 70 when rank 0 died first, which the watchdog's
  fast heal did not recognise.
- **Image input on CUDA** (G15): the ViT's attention handed 3-D tensors to the fused SDPA kernels (every image -> HTTP
  500); a typed `<｜deepseek_image｜>` in text is escaped instead of refused.
- **Build and packaging** (this update): the image build failed on `xgrammar.__version__` (xgrammar 0.2.8 has none;
  the Dockerfile now reads the package metadata); an installed engine lacked `csa2/*.cu` / `*.cpp` (ATTN_CUDA could not
  build) and the `pfdense` headers (PF_DENSE=fused could not build): packaging fix `66d0dcd`, and the Dockerfile checks
  the files; `scripts/pack_engram.py` failed with `KeyError: 'engram_layer_ids'` on the EXL3 packs' configs, which
  nest the text model's keys under `text_config`.
- earlyoom note: with earlyoom running on the nodes (`-m 2,1`), building CUDA extensions beside the loaded weights
  once pushed MemAvailable under its line and it killed both ranks (G16). `scripts/serve.sh prebuild` builds every
  extension before any weights are loaded; keep it that way.

Thanks to **WireLLM** for issue #3 (host memory growth on rank 0 until admission stalled: the graph cache and the
session index above), **ZackO2o** for PR #4 (the missing `csa2` kernel sources in an installed engine) and
**flashosophy** for PR #6, whose four-Spark work surfaced the `xgrammar.__version__` and `text_config` bugs. The fixes
here are our own; the PRs themselves are under review.

### New levers

| lever | window | what | measured | in the config |
| --- | --- | --- | --- | --- |
| `TF_DSV41_PLAN_LINK=rdma`, `PLAN_PIN` | G14 | the round plan through a RoCE host mailbox, pinned plan threads | with the three below: 1-row window 23.4 -> 21.8 ms; code +3.7%, prose +3.4%, structured +3.3%, C2 +5.4%, C4 +3.3% | on |
| `TF_DSV41_BRANCHES=1` (`_PRIO=side`) | G14 | CSA2 indexer / compressor on a dedicated high-priority side stream | 1-row -0.53 ms over 3 boots | on |
| `TF_DSV41_L2PF_PACE_GBPS=150` | G14 | the L2 prefetch paced beside the exchanges | 1-row -1.09 ms | on |
| `GLM53_TF_ROCE_FAST=1` | G14 | a shorter critical path in the RoCE all-gather kernel | 1-row -0.16, 16-row -0.51 ms | on |
| calibration VERSION 4 | G15 | every verify-table row measured (no fitted lines) | prose +2.8%, the rest flat | default |
| `TF_DSV41_IMAGES=native` | G15-G16 | DeepSeek's ViT + aligner on rank 0, `bias_vl` routing, Engram shut at image positions | VQA 20 / 20; text replies byte-identical to text-only builds | on |
| `TF_DSV41_PF_COPIES=1` | G16-G17 | one prefill weight copy | replay +1.7-3.3%, full +0.2-6.3% | on |
| `TF_DSV41_MHC_PF=1` | G16-G17 | the mHC prefill kernel | replay +3.8-4.7%, full -0.1 to +11.3% | on |
| `TF_DSV41_PF_DENSE=fused` + table | G16-G17 | the dense EXL3 prefill GEMM with the trellis decoded inside it (mma.sync, no fp16 weight workspace), tile table from a sweep | replay +5.4-6.9% with the swept table (+1.5-3.4% with the built-in one); the three together +12.8-13.6% | on |
| `TF_DSV41_FULL_CONE=1` | G17 | full-mode prefill: the decoder runs only over its dependency cone (the last ~2,541 rows), the same bits as plain full | full 32K 853 -> 1,619 alone, 1,828 with the three above (1,024 rows); 2,025 at 2,048 rows | on |
| adaptive 2,048-row prefill | G19 | `PREFILL_CHUNK` / `_ROWS` 2,048 with rows chosen a round from memory (`_ADAPT_GIB=4.5`) | replay 32K 1,996 -> 2,310 | on |
| `TF_DSV41_PF_XOVL`, `PREFILL_TILES`, 4,096-row windows, `SPEC_NUCLEUS`, `XMAP`, `DEPTH_JOINT=2`, `ALLOC_CEIL_GIB`, `SWITCH_MS` | G15-G17 | (built, exact or opt-in) | slower, no gain, or not needed: [`docs/RESULTS.md`](docs/RESULTS.md) section 4 | off |

The dense-prefill tile table (G17's sweep on GB10, 10 shapes): best 68-78 TFLOP/s at 2,048 rows on the wide layers
(Engram 78, `wo_b` 75, shared gate 70, `wq_a` 68), 54-57 on the narrow ones; the whole table is
[`config/pfdense-table.json`](config/pfdense-table.json) and `results/campaign/G17-20261004/sweep.txt`.

**Still open:** the items under [What is not solved](#what-is-not-solved).
