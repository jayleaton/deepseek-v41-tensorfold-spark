# Four DGX Sparks (TP=4)

The same engine on four GB10s instead of two: `patches/0003-four-sparks.patch` on top of 0001 and 0002,
`scripts/serve4.sh` to run the four ranks, `config/tp4.env.example` for the overrides. Same EXL3 2.9 bpw pack, same
exact speculative decoding (drafted == serial), same OpenAI / DSML / structured-output server.

Measured 2026-10-04 (engine G13 with the first version of this patch) on four Sparks behind one switch, against the SGLang TP4 / EP2 deployment the same cluster served
the day before (a different checkpoint precision: see "Comparing" below).

## 1. What changes at TP=4

| part | TP=2 (rank 0 / 1) | TP=4 (a rank) |
| --- | --- | --- |
| query heads, `wq_b`, sparse attention core | 32 | 16 |
| `wo_a` groups / `wo_b` K | 4 / 4,096 | 2 / 2,048 |
| routed + shared experts (intermediate 2,304) | 1,152 | **640 / 640 / 512 / 512** |
| vocabulary (head columns, embedding rows) | 64,640 | **32,384 / 32,384 / 32,256 / 32,256** |
| Engram hash heads | 12 | 6 (`engram-l{1,14}-r{rank}of4.bin`) |
| resident weights a rank | ~99 GiB | ~56 GiB (rank 0, the widest) |

Every split stays on whole 128-wide Hadamard blocks (an EXL3 matrix's rotations are per block, so a cut inside a
block cannot be undone on either side). 2,304 = 18 blocks and 129,280 = 1,010 blocks do not divide by four, so the
first ranks take one block more (`weights.block_bounds`); every even split, all of TP=2 included, is unchanged.
Exchanges are the same rank-ordered all-gathers (sums in rank order on every rank), now over four ranks; the RoCE
one-shot all-gather was already N-way and each proxy now starts its posts at the next rank.

At TP=2 nothing changes: the vocabulary (1,010 blocks) and the experts (18 blocks) split evenly, every rank sends
the same candidate count it sent before, and the session tier keeps its TP=2 key.

Since the first version of this patch (2026-10-04, on G13), it carries the fixes from the review of PR #6: candidates
past the narrowest vocabulary slice (every rank sends the widest slice's k, the narrower ones padded), the trimmed
draft head on uneven slices, the TCP plan link's rank handshake, `--tp 4` only for families that declare it, the BMQ
clamp to the heads a rank holds, the session tier's world size, and the G14-G19 subsystems at four ranks (fail-fast
with a channel per follower, the memory floor and adaptive prefill reading the tightest follower, the RDMA plan link
at any world size, image budgets on the followers), plus candidate kernels warmed at boot.

Tests: `tests/test_dsv41_tp4.py` runs four ranks as threads on the CPU with a checkpoint shaped to exercise the
uneven splits (four ranks == one rank in exact numerics, the ranks agree bit for bit, row invariance, greedy over
uneven vocabulary slices, Engram from `of4` shards, the partition rules); `tests/test_dsv41_four_sparks.py` pins each
bug the review found.

## 2. Setup

Prerequisites beyond the two-Spark recipe: four nodes with one switched IPv4 subnet per CX7 port function, passwordless
ssh from the head to the three workers over the link, the pack and the base checkpoint's Engram source on each node.

```bash
# 1. weights: the same EXL3 pack on all four nodes (~197 GiB), same path
hf download dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw --local-dir <MODEL>

# 2. Engram shards for four ranks, on each node (rank r on node r; ~47 GiB a node, a couple of minutes)
python3 scripts/pack_engram.py --src <deepseek-ai/DeepSeek-V4.1-Flash dir> --config <MODEL>/config.json \
    --out <ENGRAM> --world 4 --rank <r>

# 3. the image (0001 + 0002 + 0003 from patches/, as for two Sparks), then copy it to the workers
docker build -f docker/Dockerfile -t dsv41-tensorfold:tp4 .
cp config/prod.env.example config/prod.env        # the knobs; the 2-node keys are not read by serve4.sh
cp config/tp4.env.example config/tp4.env          # workers, link, paths
bash scripts/serve4.sh ship
bash scripts/serve4.sh prebuild                        # CUDA extensions on all four nodes, nothing else on the GPUs

# 4. serve (the first start writes ~50 GB prepared rank folders a node; later starts load in ~35 s)
bash scripts/serve4.sh start
bash scripts/serve4.sh status | logs [R] | stop
```

Optional: `scripts/keeper4.sh` from cron restarts the four ranks after a reboot or three failed health checks.

Lessons from the first boot: `TF_DSV41_PREFILL_ATTN_BMQ` is clamped to the 16 heads a rank holds; list both CX7
functions in `NCCL_IB_HCA` (prompt segments over NCCL: 2.9 ms instead of 5.6 ms per 2,048-row all-gather); a node with
an unplugged port holding an address on the link subnet can drop TCP to it (use the other subnet for ssh / NCCL
sockets).

## 3. Results

One boot, production config of `config/prod.env.example` (expert pruning on, `TF_DSV41_EXPERT_TOPP=0.85`), thinking
off, temperature 0, 512-token replies, three trials a cell (first one cold), isolated from other traffic. The depth
sweep is spark-bench's `bench-v41-depth-ab.py` with its published `fixtures.json`, changed only to read `/health` for
idleness (TensorFold has no `/v1/loads`) and to skip `/flush_cache` (none; every fixture is a new prompt, so trial 0 is
cold on both servers). Decode rate: (completion tokens - 1) / (last content event - first content event).

### Decode tok/s (median of 3) and cold time to first token

| prompt | TP=4 prose | SGLang prose | TP=4 code | SGLang code | TP=4 cold TTFT | SGLang cold TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1k | **63.6** | 37.7 | **104.9** | 62.2 | 0.3-0.8 s | 0.5 s |
| 20k | **65.7** | 38.0 | **97.0** | 56.9 | 12.9-14.2 s | **5.3-5.5 s** |
| 40k | **64.0** | 38.1 | **96.4** | 53.8 | 24.5-24.7 s | **10.7-10.8 s** |
| 80k | **62.8** | 37.1 | **103.1** | 59.9 | 45.9-46.8 s | **22.0 s** |
| 160k | **61.0** | 38.3 | **99.8** | 54.6 | 97.4-99.8 s | **48.1-52.1 s** |
| geometric mean | **63.4** | 37.8 | **100.2** | 57.4 | | |

Decode: 1.68x (prose) and 1.75x (code). Cold prompt reading is about 2.1x slower than SGLang's (below). Warm repeats
reuse the prefix: 0.26-0.64 s to the first token at every depth.

### Short prompts, concurrency, gates

| cell | TP=4 | SGLang TP4/EP2 |
| --- | ---: | ---: |
| story / code / copy-edit / retrieval, same prompt token ids, 256 tokens (median of 3) | 57.5 / 101.5 / 208 / ~100 | 36.9 / 63.9 / 70.1 / 51.4 |
| `dsbench` single-stream median / 4-stream aggregate | **62.6 / 119.2** | 37.6 / 75.7 |
| verify window 1 / 2 / 4 / 8 / 16 rows (boot calibration) | 17.3 / 20.8 / 26.0 / 36.3 / 48.9 ms | |

Gates on the running server: arithmetic, forced tool call (`tool_choice=required`), tool continuation, strict JSON
schema at T = 0 / 0.7 / 1.0, reasoning, and `scripts/canary.py` (chat, thinking, tool, json, tokenize): all pass.

For scale: the two-Spark G13 production row is 1-row window ~23.4-24.8 ms and code 82.8 tok/s on this repository's
own prompts (different prompts: not a like-for-like row).

### Where prompt reading goes

A profiled 2,048-row prompt segment (rank 0) is GPU-bound: ~1.19 s, the GPU busy 95% of it, of which ~0.40 s is 43
NCCL all-gathers of bf16 partials (~3.1 ms each: each rank receives three peers' 21 MB) and ~0.73 s compute. At two
ranks a segment moves a third of those bytes but computes twice as much, which is why TP=4 prefill is only slightly
faster than TP=2. The next lever is overlapping the segment exchanges with compute (two micro-batches a segment);
a reduce-scatter that keeps the rank-order sum exactly saves only ~25% of the bytes.

## 4. Comparing

- The SGLang rows run `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8` (FP8 dense, MXFP4 experts); TP=4 rows run
  `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`: fewer bytes a token is most of the decode gain. This is a
  serving comparison on one cluster, not a same-weights engine comparison.
- Expert pruning (`TF_DSV41_EXPERT_TOPP`) is lossy (README: top-1 0.9944 vs 0.9961 unpruned); delete its three lines
  for the unpruned model.
- One boot of TP=4; SGLang's rows are its published EP2 sweep.
