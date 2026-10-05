# Results

Everything measured on one pair of DGX Sparks (GB10, 128 GB each, CX7 link, RoCE), 2026-10-01 to 10-05, on
`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`. Development ran in numbered test windows (G1-G19); the window
names are kept so the raw files in [`../results/`](../results/README.md) can be matched to a row. How each cell is
measured: [BENCHMARKS.md](BENCHMARKS.md).

## 1. The baseline: MiaAI-Lab's vLLM kit on the same pair (2026-10-01)

| | |
| --- | --- |
| Kit | MiaAI-Lab 2x DGX Spark kit, vLLM `0.1.dev20904` TP=2, DSpark k=3, moe_x `2,12,8` engaged, max model length 600,000, 4 sequences, prefix caching on, `fp8_ds_mla` KV, `MAX_NUM_BATCHED_TOKENS=2048` |
| KV pool | 773,163 tokens (2.5 GiB pinned) |
| Nodes | both rebooted inside the window; GPU clocks 2,223 MHz |

| metric | kit |
| --- | ---: |
| Code, 1 stream (64-token reply, T=0 / T=1; 512-token reply) | 41.9 / 40.6; 45.0 |
| Prose: 200-token essay; short chat T=0 / T=1 | 32.5; 19.2 / 17.7 |
| Structured: count 1-200; JSON; primes list | 38.0; 50.2; 47.4 |
| Copy-heavy edit (1,024 tokens) | 55.1-56.2 |
| C1 / C2 / C4 aggregate | 32.2 / 46.7 / 37.6 (C4 per stream 9.2-10.2) |
| Cold prefill 8K / 32K / 64K / 128K / 256K | 1,073 / 1,075 / 1,060 / 1,031 / 983 tok/s |
| MemAvailable floor (worst node, every phase) | head 4.94, worker 4.40 GiB |
| Start to `/health` (freshly rebooted nodes) | 378 s |
| DSpark tokens a round, mixed | 2.27 (MMLU 1.46, multiturn 1.83, code+chat ~1.8, structured ~3.1, copy-heavy ~3.8) |
| MMLU-200, 0-shot, thinking off, greedy | 87.5% |

Two findings from the baseline that shaped the work: the kit's C4 is *below* its C2 (4 x 4 verify rows run slower
than 2 x 4), and the kit renders `reasoning_effort: "low"` as effort 25 (vLLM's mapping), not DeepSeek's 50.

## 2. The production configuration (engine = this repository's patches; G19, `7bd2d67`)

| tok/s (T=0 / T=0.7; C = decode aggregate) | code | prose | structured | C1 | C2 | C4 | 1-row window |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| **G19 production** (`7bd2d67`, the config's words, mean of 2 boots) | **84.74 / 78.15** | **46.75 / 47.98** | **121.27 / 122.91** | **87.62** | **74.07** | **101.09** | ~21.8 ms |
| G18 production (`cd4245a`, 1,024-row prefill; G19's same-session reference) | 85.37 / 78.06 | 47.09 / 48.18 | 121.74 / 123.09 | 87.50 | 74.03 | 100.29 | |
| G14 combo (`356a188`, mean of 3 boots) | 86.58 / 78.87 | 46.24 / 48.78 | 122.21 / 123.67 | 87.69 | 74.63 | 100.66 | 21.7-21.8 ms |
| **G13 production** (`767ad9f`, MHC_CUDA + ATTN_CUDA + DENSE_V3) | **82.83 / 75.02** | **44.25 / 45.30** | **117.84 / 117.59** | **85.05** | **70.27** | **96.61** | **~23.4-24.8 ms** |
| G13, the same engine with the three off (same session) | 81.74 / 71.94 | 41.28 / 42.22 | 116.79 / 116.61 | 82.46 | 67.64 | 93.95 | 26.8-27.0 ms |
| G10 / G11 final (`38f6500`) | 79.0-81.5 / 72 | 40.9-41.2 / 42 | 116.6 / 117.5 | 82-83 | 67.1 | 93.5 | 26.9 ms |
| kit | 41.9-45 | 32.5 | 38-50 | 32.2 | 46.7 | 37.6 | |
| G13 production / kit | 1.8-2.0x | 1.36x | 2.3-3.1x | 2.6x | 1.5x | 2.6x | |
| **G19 production / kit** | **1.9-2.0x** | **1.44x** | **2.4-3.2x** | **2.7x** | **1.59x** | **2.7x** | |

Steady aggregates (all streams live, G19): C2 88.7, C4 136.0 (code-only C4 192.5, prose-only C4 120.2). G19 vs its
G18 reference: every cell within 1.5%, exact on both boots (drafted == serial, every concurrent stream == the same
request alone, replies == the reference). G14's own boots read a little higher than the later windows' on the same
levers (boot-to-boot spread ~1-1.5 tok/s on code); later windows changed prefill, memory and images, not decode.

G13 rows below for history. `exact_all True` (drafted == serial) in every run. DSpark tokens a round (G10 / G11): code 3.88, prose 1.57-1.59,
structured 5.91; in G13 code read 3.80 (the depth policy reads the cheaper calibrated windows and drafts a little
differently; the replies are the same). G13 gates on the production set: top-1 vs the kit 0.9961 (first copy 0.9502),
equal to the rewrites off; MMLU-200 0-shot 87.5% (175 / 200); tool chains thinking off 11 / 12; a tool call
`get_weather {"city":"Hanoi"}`. One boot a configuration; the 1-row window range is the spread over the G13 combo
boots (`results/campaign/G13-20261003/combo-window.txt`, `combo-speed.txt`).

Ship gates on the final configuration (a test server started by `scripts/serve.sh` from the production config):

| gate | result |
| --- | --- |
| teacher-forced top-1 vs the kit | 0.9963 (first copy 0.9502) |
| MMLU-200 0-shot | 87.5% (kit 87.5%) |
| structured output (json_schema x thinking off / on, tool choice) | 12 + 10 cases pass |
| tool chains (`chains.py`), thinking off / high | 11 / 12, 11 / 12 (pass line 10) |
| tool-eval-bench category C | 8 / 8, score 100 |
| 30-min soak | 529 requests, 0 errors, 77 intentional cancels, drained, 17*23 = 391; MemAvailable min head 5.11 / worker 4.07 GiB |
| stress 4 x 300K (one 299K prefill + three 64K prompts decoding 2,048 tokens, ignore_eos) | every stream complete; first token of the 299K prompt at 217 s; **worker MemAvailable min 4.01 GiB: under the 5 GiB target** |

Earlier gates with the same prefill path (G7 engine commit): MMLU-200 replay / full / kit 88.5 / 88.0 / 87.5% (199 of
200 answers equal); MMLU with a 20-question preamble, replay / full 81.1 / 81.1% (178 of 180 equal); needles at 32K /
128K / 299K found in 19.9 / 74.0 / 195 s (replay) and 32K / 128K in 34.7 / 137.3 s (full).

### Prefill (one slot, cold, fresh random text after a 2K warm-up; tok/s)

G19, engine `7bd2d67`, `m2bench --prefill --prefill-reply 32`, median of 2 boots, every cell exact (the first token
and a 32-token reply digest equal the 1,024-row reference's):

| config | full 8K | 32K | 64K | 128K | replay 8K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **2,048 rows, adaptive, keep 4.5 GiB (production)** | **1,561** | **2,025** | - | - | **2,028** | **2,310** | - | - |
| 2,048 rows, adaptive, keep 4.0 (the soaked setting) | 1,568 | 2,126 | 2,197 | 2,152 | 2,026 | 2,311 | 2,294 | 2,118 |
| 2,048 rows, adaptive, keep 5 (the engine default) | 1,069 | 2,122 | 1,904 | 1,808 | 1,084 | 1,912 | 1,774 | 1,753 |
| 1,024 rows (G16-G18 production) | 1,256 | 1,838 | 1,898 | 1,859 | 1,249 | 1,996 | 1,926 | 1,747 |
| kit | 1,073 | 1,075 | 1,060 | 1,031 | 1,073 | 1,075 | 1,060 | 1,031 |

At keep 5 the worker's mid-prefill usable memory less the priced 2,048-row transient sits just under 5 GiB, so rows
flapped 2,048 <-> 1,024 with an allocator release at each step down (up to -46% at 8K). The 1,024-row 8K cells
(~1,250) are low against G17's 1,842 for the same words; not investigated. Over HTTP (the G19 stress server's
first-token probe, best of 2) replay reads 2,052 / 2,224 / 2,305 / 2,290 / 2,202 tok/s at 8K / 16K / 32K / 64K / 128K.

G17, engine `0bdd276` (the same prefill kernels), the levers one at a time at 1,024 rows (2 boots):

| full / replay at 32K | tok/s | vs off |
| --- | ---: | ---: |
| off (G16 production words) | 853 / 1,763 | |
| `PF_COPIES` + `MHC_PF` + `PF_DENSE=fused` with the swept table ("final") | 952 / 1,990 | +11.6% / +12.9% |
| final at 2,048 rows | 1,109 / 2,285 | +30.0% / +29.7% |
| `FULL_CONE` alone (full) | 1,619 | +89.9% |
| final + `FULL_CONE` (full) | 1,828 | +114.4% |

The G7 table (before G16's memory cap and G17's kernels), for history:

| config | 8K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: |
| **replay + every prefill lever (G7-G15 production)** | **1,833** | **2,043** | **2,068** | **1,953** |
| replay, base kernels | 1,343 | 1,421 | 1,407 | 1,379 |
| full (no replay), every lever | 969 | 1,004 | 1,003 | 876 |
| full, base | 756 | 772 | 764 | 739 |
| kit | 1,073 | 1,075 | 1,060 | 1,031 |

TTFT at 128K: 67 s. One run of the production prefill was anomalous (854 tok/s at 128K: both GPUs at ~36 W instead of
~52 W at normal clocks) and did not reproduce in two later runs.

### Strict mode (G13: every precision trade off; not re-run on the current engine)

Production's set with `TF_DSV41_EXPERT_TOPP=0`, `EXPERT_RENORM=orig`, `MHC_FN=fp32`, `KIT_ROUNDING=0`, `LOGITS=fp32`,
`INDEX_KV=bf16` and `PREFILL=full`; the fast prefill GEMMs and the fused prefill attention stay on. Same build and
session as the G13 production row.

| | strict | production | strict vs production | kit | strict / kit |
| --- | ---: | ---: | ---: | ---: | ---: |
| code, T0 / T0.7 | **76.88** / 74.94 | 82.83 / 75.02 | -7.2% / -0.1% | 42-45 | 1.71-1.83x |
| prose, T0 / T0.7 | **44.21** / 42.18 | 44.25 / 45.30 | -0.1% / -6.9% | 32.5 | 1.36x |
| structured, T0 / T0.7 | **111.92** / 109.59 | 117.84 / 117.59 | -5.0% / -6.8% | 38-50 | 2.24-2.95x |
| C1 / C2 / C4 | **78.96 / 64.00 / 89.53** | 85.05 / 70.27 / 96.61 | -7.2% / -8.9% / -7.3% | 32.2 / 46.7 / 37.6 | 2.45x / 1.37x / 2.38x |
| 1-row / 16-row window | 24.7 / 81.3 ms | 23.5 / 72.1 ms | +1.2 / +9.2 ms | | |
| cold prefill 8K / 32K / 64K / 128K (`full`, in-engine) | **915 / 965 / 959 / 923** | | | 1,073 / 1,075 / 1,060 / 1,031 | **0.85 / 0.90 / 0.90 / 0.89x** |
| top-1 vs the kit (first copy) | 0.9963 (0.9601) | 0.9961 (0.9502) | | | |

Prose at T0 is level only because the strict reply differs on that prompt and drafts better (1.607 tokens a round
against 1.567). The 16-row cost is mostly the unpruned experts. Strict MMLU and tool chains were not run. Upstream
TensorFold's own clients against a strict test server: `bench_concurrent` 0 failures, every stream == alone == serial;
`prefill_cold` over HTTP 853-920 tok/s at 8K-64K ([campaign/G13-RESULTS.md](campaign/G13-RESULTS.md)).

### Start

34-44 s from `docker run` to `/v1/models` on the G10 / G11 test servers (4 slots x 300K pool, prepared folders,
compiled kernels cached, page cache dropped); `m2bench` boots in 37-39 s. The weights are read back from the
prepared folders with parallel O_DIRECT readers; M1 measured 18 s for the weights alone.

## 3. How it got there

| step | code | prose | structured | C1 | C2 | C4 | what changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| G4 | 57.8 | 31.3 | 86.0 | 60.5 | 46.1 | 70.7 | the family running with CUDA graphs and RoCE; expert load path; router fix |
| G7 | 69.1 | 37.4 | 102.2 | 70.0 | 48.9 | 67.0 | the router as a CUDA GEMV (1-row window 33.1 -> 29.5 ms); bf16 exchanges, mHC split; prefill levers (1.7-1.95x the kit) |
| G8 | 72.7 | 39.0 | 109.1 | 69.1 | 58.9 | 77.5 | the GPU waits for Engram rows instead of the host; slot-agnostic row graphs (C2 +34%, C4 +12%); round plan over TCP; speculative DSpark pass |
| G9 | 80.2 | 39.9 | 114.5 | 76.0 | 63.0 | 92.8 | decode glue (bit for bit) + bf16 mHC weights; routed-expert pruning p85k (x1.05) |
| G10 (final) | 79.0-81.5 | 40.9-41.2 | 116.6 | 82-83 | 67.1 | 93.5 | cross-op L2 prefetch, 12 MiB a site (code +2.5%, prose +3.5%) |
| G13 | 82.8 | 44.25 | 117.8 | 85.1 | 70.3 | 96.6 | three CUDA rewrites, bit for bit: the mHC boundary in one launch, the decode attention core + indexer top-k, dense EXL3 over a coalesced repack (1-row window 26.8 -> ~23.4 ms) |
| G14 | 86.6 | 46.2 | 122.2 | 87.7 | 74.6 | 100.7 | the round plan through a RoCE host mailbox, BRANCHES on priority streams, the paced L2 prefetch, the faster RoCE kernel (1-row window 23.4 -> 21.8 ms) |
| G15-G16 | 85.3-86.0 | 46.0-47.0 | 121.6 | 87.6-87.8 | 73.6-74.5 | 99.3-102.3 | calibration VERSION 4 (prose +2.8%); native images; the memory cap (1,024-row prefill: replay 32K 2,015 -> 1,753); the graph-cache and vision fixes |
| G17-G18 | 84.9 | 47.0 | 122.0 | 87.6 | 74.2 | 100.3 | prefill kernels and the full-mode cone (replay 32K 1,990, full 1,828 at 1,024 rows); fail-fast; the priced two-rank floor; the session-index fix |
| G19 | 84.7 | 46.8 | 121.3 | 87.6 | 74.1 | 101.1 | adaptive 2,048-row prefill (replay 32K 2,310, full 2,025) |

## 4. Measured and not adopted

| lever | result |
| --- | --- |
| **Lossy verify budget** (`TF_DSV41_VERIFY_BUDGET`: verify drafts with fewer experts) | B=4: code 89.3 / prose 41.1 / C4 100.7, but only 12% of greedy tokens equal the exact reply (first divergence at token ~13), MMLU-gen 84.5% vs 86.0%, tool chains 4 / 12. B=2 failed the server's own canary. B=0: code 184.6 from degenerate loops the drafter accepts, MMLU-gen 79.5%, chains 1 / 12. **Dropped**: even as an opt-in it breaks tool calls. |
| Draft trees (parent-conditioned) | the target's 2nd choice is the draft's 2nd only 17% of the time at position 1 (line 35%); tree replays +0.1%. The drafter is wrong, not near. Not run on the GPU. |
| PDL (programmatic dependent launch) | 1-row window within +-0.1 ms in every part (segments, singles, experts, Triton): no gain |
| x3dn dense kernels (v1 persistent, v2 upstream-grid) | v1 9-24% slower than upstream on every shape in the engine; v2 wins some 1-row shapes, loses at 2 rows (prose -9%) |
| Joint multi-slot draft depth | C2 -2.1%: the windows do not get cheaper enough |
| Long projection plan | no 1-row gain |
| 4-bit / trimmed draft head | -0.27 ms a pass but code acceptance -0.11 tokens a round; trim: pass unchanged |
| More DSpark candidates (K 256 / 1,024 a rank) | prose +0.6% / +0.1% |
| DSpark self-distillation (LoRA deltas on our own drafting logs) | delta A: prose **+5.5%** (43.3) but code -4.4%; delta B: +17.6% tokens a round offline, only +11% in the engine, code -5.7%; a balanced re-capture: prose +4.1%, code -2.9%. The training port disagrees with the engine's drafter after position 1 (agreement 0.39); that comes first. |
| x3pf prefill experts, x3tc tensor-core experts | slower at every size (x3tc 3-5x) |
| Streaming top-k from 4K keys | -2% |
| Shortened decode MoE chain (`TF_DSV41_MOE_FUSED`, G13) | exact (bit for bit R 1..16 on real layers), 9-46 us a layer faster in isolation, but **+0.7 ms** on a 1-row window in the graph and 0 at 2 rows. Off; where the isolated gain goes is not measured yet (nsys) |
| CSA2 indexer / compressor on a side stream (`TF_DSV41_BRANCHES`, G13) | exact (on == off over 24 windows), but the 1-row window was 26.2 ms in one boot and 31.5 ms in another (+2.25 ms mean); 16 rows -0.55 ms. A fork / join per layer costs more than the overlap returns. Off |
| `TF_DSV41_PF_XOVL` (a prefill segment's exchanges by row halves on a side stream, G17) | exact; -1.2 to -2.4% alone, nothing added to the final set. Off |
| `TF_DSV41_SPEC_NUCLEUS` (the speculative DSpark pass after top-p windows, G17) | exact (tokens == serial); C2 steady +1.3%, C4 steady -0.6%, single streams -0.2 to -1.2%: below its pass line. Off |
| 4,096-row prefill windows (G16) | 4K == 2K bit for bit; +0.66 GiB on the worker; refused at boot without `TF_DSV41_PF_4K_MEMORY_OK=1`. Off |
| `TF_DSV41_INDEX_BUDGET_MIB=32` at 2,048 rows (G19) | no gain (128K replay -9%). 64 stays |
| Allocator ceiling (`TF_DSV41_ALLOC_CEIL_GIB`, G16) | at 1.5 GiB it held the worker >= 4.54 GiB in a 1 h soak, but at 1.0 with 2,048 rows a 299K prefill hit an out-of-memory error. Off (fail-fast now ends such a failure in ~1 s) |
| Joint depth mode 2 (`TF_DSV41_DEPTH_JOINT=2`, G15) | C2 steady +1.7%, but C4-prose steady -8% (bimodal by boot). Off |
| Expert map L2 prefetch (`TF_DSV41_XMAP`, G15) | +0.22 ms on a 1-row window. Off |
| Lower `L2PF_PACE_GBPS` (125 / 100, G15) | slower at 1, 2 and 4 rows. 150 stays |
| Hiding the window graph's submission (G13 d1) | nothing to hide: `graph.replay` takes 0.02 ms of host time a round without a profiler (the 1.3 ms seen earlier was nsys overhead) |

## 5. Memory

| run | head | worker |
| --- | ---: | ---: |
| **G19: 1 h soak, 2,048 rows, adaptive keep 4.0 (906 requests, 0 errors)** | **4.63** | **3.99** |
| **G19: stress 299K + 3 x 64K + 4 image requests, 2,048 rows** | **5.07** | **4.81** |
| G18: 1 h soak on the session-index fix, 1,024 rows (760 requests, 0 errors) | 5.49 | 5.02 |
| G18: 1 h soak, 2,048 rows, fixed rows (900 requests, 1 refusal) | 4.07 | 3.17 |
| G17: 1 h soak, 2,048 rows, before the session-index fix (828 requests, 4 errors) | 3.69 | 3.27 |
| G16: 1 h soak with the LRU graph cache and a 1.5 GiB allocator ceiling | 5.02 | 4.54 |
| G16: stress with the memory cap (1,024 rows, native images) | 5.30 | 5.10 |
| decode benchmarks (1-4 streams) | >= 10.3 | >= 7.3 |
| prefill to 128K | 7.6 | 5.66 |
| soak, 30 min | 5.11 | 4.07 |
| stress 4 x 300K, current engine (G12, `a6f5792` + `MIMALLOC_ALLOW_THP=0`) | 4.31 | 3.60 |
| stress 4 x 300K, G12 runs on the way to the fixes (six runs) | 3.78-6.00 | 3.00-3.69 |
| stress 4 x 300K (six runs on the G10 engine `38f6500`) | 5.5 | 3.6-4.4 |
| stress, the G8 engine | | 4.07 |
| stress, the G4 engine | 5.46 | 4.38 |

MemAvailable minimum, GiB, 0.5-1 s samplers.

**Fixed in G12** ([campaign/G12-RESULTS.md](campaign/G12-RESULTS.md)):

- **Host memory growth.** Up to G10 each rank's anonymous RSS grew 4-5.6 GiB during the 299K prefill and was not
  returned. The cause was torch's CPU allocator: in NVIDIA's PyTorch build it is an embedded mimalloc. A prompt
  segment's Engram rows (~25 MB) were made on a prefetch thread and freed on the round thread, and mimalloc keeps
  such cross-thread frees of large blocks until the owning thread collects. Live CPU tensors stayed under 100 MiB
  while mimalloc's arenas held ~1.9 GiB. The rows now live in reused NumPy-owned buffers (same values, bit for bit).
  Growth is now 0.67-0.69 GiB a rank, flat from ~18K rows to the end. The 299K prefill under the stress takes 175 s,
  against 181-185 s in the same window's runs before the fix.
- **The stall** (19 minutes once in G10). It was not an NCCL hang. Transparent-huge-page faults under ~103 GiB of
  weights and KV each triggered a synchronous memory compaction (GPUs spinning at 17 W, PSI memory "full" 64-71%,
  one stress prefill at ~150 tok/s). NumPy's huge-page request is now off in the engine, and
  `MIMALLOC_ALLOW_THP=0` in the config turns off mimalloc's. No stall in the stress runs after the fix, one of
  them with `nvidia-smi` polling every second. A stall watchdog and deadlines stay as a safety net.
- **Fast-prefill segmentation dependence** (~1 ulp in a few rows when one prompt is prefilled in different segment
  sizes). The RoPE tables were built at a length that followed the segment size, and one torch `cos` / `sin` call on
  the Sparks' CPU gives other bits at some positions for different lengths. The tables are now built in fixed
  blocks, so an entry depends only on its position. The two red tests pass. At 302K positions, 0.08-0.15% of
  entries change, by at most 1 fp32 ulp. The engine's code digest changes too, so session entries an older engine saved
  on NVMe are not resumed.

**Fixed in G16-G19** (details: the README's "Changes since G13" and [campaign/](campaign/README.md)): the CUDA graph
cache's growth with agent sessions (the serving leak), the graph-eviction cascade, the NVMe session index's
`list[int]` per entry (the slow drift), a single-rank and unpriced admission floor, failures on one rank stranding the
other, the vision store's double count. RssAnon now falls over a 1 h soak (-0.4 GiB/h a rank) instead of growing
(+0.4 to +1.0 GiB/h before).

**Was open after G12 (now covered by the G16 cap, the priced floor and adaptive rows):** the worker's minimum, 3.0-3.7 GiB in every G12 run. The worker is at 4.5-7.8 GiB when the server is ready,
and the minimum comes in the first ~20K prompt rows (+1.3 GiB of CUDA reservations as the first segments run). It
comes from the boot budget (4 x 300K pool) and that device-side part, not from growth: from 80K rows on the worker
stays at 3.7-5.0 GiB. The minimum is under the 5 GiB target and the 4 GiB admission floor.
