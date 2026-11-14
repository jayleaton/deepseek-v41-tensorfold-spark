# Eight slots of 192K (C8) against four of 1M (c4) — 2026-11-14

Engine `6c4eeccda` (the pin this recipe builds), flash17 K6 MTP pack, TP2 on two DGX Sparks, the flash17 profile with
two lines changed on both ranks: `TF_DSV41_SLOTS=8`, `TF_DSV41_CONTEXT=196608`. Pool (2,100,000 tokens), floors
(5 / 4 GiB, serve dip 3, PF workspace floor 4.75), KV split/compact, mirror 4096, rows cap 64, slot drafts on: unchanged.

## AOT

`tf-dsv41 aot-needs` with the C8 environment, both ranks: 402,334 programs, 129 Triton variants, every one already in
the 1M profile's AOT set (144) and in its `needs.json`; `scripts/check_native_assets.py --needs` passes on both ranks.
No new assets. The 1M RoPE table covers 196,608 + 2,048 rows.

## Memory per rank (engine's own end-of-boot measurement)

| | rank 0 C8 | rank 0 c4 | rank 1 C8 | rank 1 c4 |
|---|---:|---:|---:|---:|
| weights GiB | 95.2 | 95.2 | 95.2 | 95.2 |
| pool | 2,100,224 tokens = 2.07 GiB (1,060 B/token) | same | same | same |
| MemAvailable at the KV check | 16.91 | 17.10 | 16.28 | 17.18 |
| boot growth | 4.21 | 4.88 | 2.41 | 3.75 |
| idle at end of boot | 10.62 | 10.15 | 11.79 | 11.36 |
| worst serving point (idle − 3 GiB dip) | **7.62** | 7.15 | **8.79** | 8.36 |

Sampled every 2 s from boot through every test below, the lowest MemAvailable was 6.27 GiB (rank 0) and 8.06 GiB
(rank 1), including eight distinct cold 20,480-token prompts at once and one 196,544-token prompt. The recipe's hard
floor is 4 GiB. Eight slots at 192K use no more memory than four at 1M on this engine.

## Correctness

- Boot: `slots: 8 live`, DSpark passes of up to 8 slots (40 rows), no PoolDoesNotFit / OOM / missing variant.
- Smoke (391), strict tool schema (required + auto), greedy 2K and the 8K prefill reference: identical to the
  reference token IDs. Single-stream decode (coding / prose / ceiling cells): token hashes identical to c4's.
- Context: a 196,544-token prompt + 64 output tokens is served; 197,120 tokens get HTTP 400 with the limit.
- Concurrent vs solo (`tools/bench_concurrent.py --alone --serial`, fixed seeds): chat prompts are identical under
  concurrency on both shapes (8/8). Raw-completion and some distinct prompts differ from their solo replies **on c4
  as well** (c4: code 0/4, distinct 1–2/4; C8: code 1–2/8, distinct 4–5/8), and on both a lone request with drafts can
  differ from `draft: false`. This is the production profile's precision-trading settings (expert top-p pruning per
  verify window), not the slot count; strict mode was exact at four streams (docs/campaign/G13-RESULTS.md).

## Throughput (greedy, 256 tokens, 3 reps, same tool and prompts, same day)

| cell | requests | C8 tok/s | c4 tok/s | C8 TTFT p50 / p95 s | c4 TTFT p50 / p95 s |
|---|---:|---:|---:|---|---|
| code (same prompt) | 1 | 84.5 | 84.5 | 0.15 / 0.16 | 0.16 / 0.16 |
| | 4 | 160.9 | 170.1 | 0.52 / 0.57 | 0.52 / 0.57 |
| | 8 | **208.9** | 161.8 | **1.07 / 1.08** | 2.88 / 7.59 |
| chat (same prompt) | 1 | 61.3 | 61.5 | 0.19 / 0.20 | 0.20 / 0.20 |
| | 4 | 168.6 | 153.9 | 0.67 / 0.71 | 0.65 / 0.76 |
| | 8 | **284.3** | 149.2 | **1.29 / 1.36** | 4.02 / 7.84 |
| distinct (8 different prompts) | 1 | 84.5 | 81.9 | 0.18 / 0.18 | 0.18 / 0.18 |
| | 4 | 116.4 | 114.9 | 0.55 / 0.69 | 0.61 / 0.70 |
| | 8 | **127.3** | 113.0 | **1.29 / 1.32** | 3.94 / 9.70 |

Aggregate = tokens after each stream's first / span from first to last token of any stream (median of 3 reps). Same-
prompt cells flatter mixture-of-experts batching; the distinct cell is the realistic one (+13% at 8 requests). At 8
live streams each runs at 21–35 tok/s. The gain is mainly latency under bursts: c4 runs four and queues four.

## Trade-off

A request's prompt + output is capped at 196,608 tokens. Long single-context work (agent sessions past 192K) needs
`TF_DSV41_CONTEXT=1048576` with `TF_DSV41_SLOTS=4`, the previous default, which uses the same assets.
