# Decode: the roofline, where the time went, and what worked

Single-stream decode on this model is a sequence of rounds: a DSpark draft pass (~3.5 ms) proposes up to 5 tokens,
one verify window runs the target over the pending token plus the kept drafts (R rows), and the round emits the
accepted drafts plus one. Speed = tokens a round / round time.

## 1. The floor

What one rank reads for a 1-row window (TP = 2, the 2.9 bpw pack):

| | MB | ms at ~220 GB/s |
| --- | ---: | ---: |
| routed + shared experts (7 x 40 layers) | ~1,610 | 7.3 |
| attention dense matrices (5 bits) | ~1,760 | 8.0 |
| routers (bf16, replicated) | ~170 | 0.8 |
| head (6 bits, vocabulary half) | ~248 | 1.1 |
| **total** | **~3,790** | **~17** |

Each extra row adds ~5-6 new experts a layer (a second token routes elsewhere), ~3.5-5 ms a row. With 2.9 bpw on two
Sparks, 2x the kit's *best published* prose cells is above this ceiling; only fewer bytes a token (a smaller pack,
cheaper attention reads) beats it.

## 2. Where it went

| 1-row window | ms |
| --- | ---: |
| first working version (G4) | 35.0 |
| router as a CUDA GEMV (G7) | 29.5 |
| decode glue + bf16 mHC weights (G9) | 28.7 |
| cross-op L2 prefetch (G10) | 26.9-27.4 |
| three CUDA rewrites: mHC boundary, attention core + top-k, dense over a coalesced repack (G13) | ~23.4-24.8 |
| round plan through a RoCE host mailbox, BRANCHES on priority streams, paced L2 prefetch, faster RoCE kernel (G14) | **~21.8** |

The final 1-row window under nsys (G9, before L2 prefetch), per rank: routed + shared experts 9.4 ms, dense EXL3
matrices 11.5 (155-200 GB/s), mHC 3.0, exchanges 1.7 (0.6 transfer + 1.1 waiting for the other rank), attention 1.3,
router 1.3, torch copies 1.0, top-k sorts 0.9, norms 0.4; GPU idle inside the window 0.3. Outside the window: the
DSpark pass 3.5 ms and ~2.4 ms of GPU idle a round.

Multi-stream rounds had a different problem (G7 -> G8): only single-slot windows were graph-captured, so a round
mixing slots launched thousands of kernels from Python (23-34 ms of host time), and the host blocked twice a round on
Engram reads. Slot-agnostic row graphs and a GPU-side wait for the Engram rows fixed both: C2 +34%, C4 +12%.

## 3. What worked

| lever | effect | bits |
| --- | --- | --- |
| CUDA graphs for windows and draft passes; RoCE one-shot all-gather instead of NCCL | the baseline of every later row (RoCE: +11% 1 stream, +17% C4) | same |
| Router as a warp-per-expert CUDA GEMV with the top-6 fused (was a Triton fp32 kernel at ~36 GB/s) | 112.6 -> 24.6 us a layer; code +4%, prose +7% | top-1 0.9958 |
| bf16 exchanges (the partials are bf16-rounded already), mHC split / unroll, fused norm | prose +11% together | same |
| Engram gate: the GPU waits on an event before layers 1 / 14, the host never blocks | C2 +3.5%, C4 +2.6% | same |
| Slot-agnostic row graphs | C2 +34%, C4 +12% | same |
| Round plan over host TCP, speculative DSpark pass during sample / commit (99.2% of passes used) | code +1.7%, prose +2.8% | same |
| Decode glue (mHC finish, paired RMSNorm, bf16 partials, narrow router, device top-k, fused Engram) | -0.8 ms at 1 row, -1.2 ms at 2 rows | same |
| mHC mixing weights in bf16 | prose +2.4% on top of the glue | top-1 0.9963 |
| Routed-expert pruning in decode, top-p 0.85, at least 3, renormalized | x1.05 (code 77.3, C4 92.1 at the time) | lossy: top-1 0.9944, MMLU-200 88.5% |
| Cross-op L2 prefetch of the next dense group, 12 MiB a site (GB10: 24 MiB L2) | code +2.5%, prose +3.5%, C2 +2.4% | same |
| G13: the mHC boundary as one CUDA launch, CSA2's decode attention core + indexer top-k in CUDA, dense EXL3 over a 16-byte-coalesced repack (`TF_DSV41_MHC_CUDA`, `ATTN_CUDA`, `DENSE_V3`) | 1-row window 26.8 -> ~23.4 ms; prose +7.2%, code +1.3%, C2 +3.9%, C4 +2.8% | same (top-1 0.9961 on and off) |
| G14: the round plan as an RDMA write + flag through a RoCE host mailbox with pinned plan threads (rank 1's window-entry lag +130 -> -9 us), the CSA2 indexer / compressor on dedicated high-priority streams, the L2 prefetch paced at 150 GB/s beside the exchanges, a shorter RoCE all-gather critical path | 1-row window 23.4 -> 21.8 ms; code +3.7%, prose +3.4%, structured +3.3%, C2 +5.4%, C4 +3.3% | same (top-1 0.9961) |
| G15: calibration VERSION 4 (every verify-table row measured; the 2nd row's price 5.88 -> 4.62 ms, measured 4.66) | prose +2.8% (2-row rounds 33% -> 46%), the rest flat | same |

## 4. What did not

- **Lossy verify budget.** Verifying drafts with fewer experts makes windows much cheaper (B=0: 1 / 4 / 16 rows 28.6 /
  32.3 / 40.1 ms) and the numbers look spectacular (code 184.6), but approximate verification changes 88-95% of greedy
  tokens after the first few, tool chains collapse (4 / 12 at B=4, 1 / 12 at B=0), B=2 fails the server's own
  arithmetic canary, and B=0's code speed is the drafter accepting degenerate repetition loops. Dropped, including as
  an opt-in: a speed number that breaks tool calls is not a speed number.
- **Draft trees.** Parent-conditioned trees only pay if the target's second choice is often the draft's second. On
  34,753 logged DSpark passes it was 17% at position 1 (line: 35%), and the serial token was outside the drafter's
  top-4 in 28% of first positions. Replays put trees at +0.1%. The drafter is wrong, not near: width does not help.
- **PDL.** Programmatic dependent launch on segments, single matrices, experts and Triton kernels: within +-0.1 ms in
  every part; the expected -0.35 to -0.68 ms a window did not show.
- **x3dn dense kernels.** v1 (persistent CTAs, perfectly balanced SMs) was designed from an L2-hot microbenchmark and
  lost 9-24% in the engine at cold DRAM: its 16-20 partials a strip each paid a fence, two atomics and an epilogue.
  v2 (upstream's grid, the x3ld load ring, few splits) won some 1-row shapes and lost at 2 rows (prose -9%). Lesson:
  benchmark kernels cold (a 256 MB write between calls inside the graph), at the row counts decode actually uses.
- **Joint multi-slot depth.** Choosing all slots' draft depths together trimmed rows (prose 2.48 -> 2.38) at nearly
  the same tokens a round; C2 -2.1%.
- **More draft candidates, 4-bit / trimmed draft heads, the long projection plan:** within noise or a net loss.
- **G13's other two rewrites.** A shortened decode MoE chain (`TF_DSV41_MOE_FUSED`) saved 9-46 us a layer in
  isolation and measured +0.7 ms on a 1-row window in the graph (G14's variants: no gain either; off); the CSA2
  indexer / compressor on a side stream (`TF_DSV41_BRANCHES`) left the 1-row window unstable between boots (26.2 /
  31.5 ms) until G14 moved it onto dedicated priority streams with one capture stream (-0.53 ms, on). Hiding the
  graph's submission had nothing to take: 0.02 ms of host time a round without a profiler.
- **G15-G17:** an expert map that L2-prefetches each routed expert's first gate split (`TF_DSV41_XMAP`: +0.22 ms at 1
  row), joint depth mode 2 (C4-prose -8%, bimodal by boot), and the speculative DSpark pass after top-p windows
  (`TF_DSV41_SPEC_NUCLEUS`: C2 steady +1.3%, C4 steady -0.6%). All exact, all off.
- **Why C4 mixed sits at ~101 against ~136 steady** (G17's study, [campaign/G17-LEVERS.md](campaign/G17-LEVERS.md)):
  the cell is one prose stream's serial chain. Prose is in every round; 31% of the wall is prose alone at ~36 ms a
  round, and a 4-live window costs ~114 ms because four distinct streams touch ~57 distinct experts a layer. Only
  fewer bytes an expert or a better prose drafter move it much.
- **Drafter self-distillation** is the one lever that moved prose acceptance (+5.5% prose with a LoRA delta trained
  on our own drafting logs) but it cost code acceptance (-4.4%), and the training port disagrees with the engine's
  drafter after position 1. Not adopted; it needs balanced data and that fidelity fixed first.

## 5. Lessons

1. Profile both ranks. A rank waiting on the other shows up as exchange time on the fast rank.
2. Count host time per round, not just kernel time: the largest multi-stream gain (row graphs) was a host problem.
3. Keep exactness as a gate on every lever (drafted == serial, batched == alone, graphs == eager); it catches bugs
   that speed numbers hide (a broken import made every decode MoE call raise; a test stand-in missed a new keyword).
4. Report lossy levers separately, with quality next to speed, and gate them on tool calls as well as MMLU.
5. Prose speed is set by drafter acceptance and the 1- and 2-row window cost; neither width nor depth fixes a drafter
   that is wrong.
