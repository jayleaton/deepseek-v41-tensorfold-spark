# G17: the next levers for C4 / C2 aggregate decode and full-mode prefill (offline, 2026-10-04)

This is a CPU-only study plus a build. No Spark was touched. Engine branch **`dsv41-agg`**, from dsv41-060 `253fcd7`.
It was built in a separate clone and fetched into the development engine repository. Its commits:

- `bc99080`: `TF_DSV41_FULL_CONE`;
- `eb7d0b9`: `TF_DSV41_SPEC_NUCLEUS`;
- `c02e6ff`: `m2bench --c-distinct`.

All three are off by default. GPU plan: `scripts/windows/g17agg.sh` (not run).

Sources:

- G15b's speed boots: `results/G15depth-20261004/m2-g15d-prod-b{1,2,3}.json`, i.e. prod 7954c1d, its per-cell phases and per-stream drafting stats;
- G9's C2 nsys (DECODE-ROOFLINE section 6, G9-RESULTS section 3) and G15b's expert map (U(R));
- G6's gmbench, G7's prefill table, G16's TTFT table;
- the prefill study numbers in the pfdense, mhc_pf and xovl module notes.

Rule used throughout:

- **exact** = drafted == serial, batched == alone, and the replies / state bit-identical to the reference.
- **precision trade** = needs an explicit sign-off (top-1 vs the kit ≥ 96%, MMLU within 1).

## 0. The ranked list (top 8)

| # | lever | target | expected | exactness | effort | risk | status |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | **Full-mode prefill with the decoder over its dependency cone** (`TF_DSV41_FULL_CONE`) | full prefill | 32K: **1,004 → ~1,880** (2K chunks) / **~1,630** (prod's 1K chunks); 8K 969 → ~1,440; 64K 1,003 → ~1,990; 128K 876 → ~1,920 | exact (bit-identical state and replies to full mode; same session tag) | S | low | **built** `bc99080` |
| 2 | **Measure the C4 target cell honestly**: same-workload cells currently run one prompt 4x (`--c-distinct`) | C4 metric | C4-code "steady 194" is 2 distinct sequences. 4 distinct code streams estimated **~135 steady / ~125-130 aggregate**: the 120 target is probably met on its own definition (RigMark short code). The mixed cell (~102) is a different, harder cell | n/a | S | none | **built** `c02e6ff` |
| 3 | **Speculative DSpark pass after top-p windows** (`TF_DSV41_SPEC_NUCLEUS`) | C2 / C4 mixed, any T > 0 top-p stream | C4 mixed decode agg **+3-5%** (102 → ~106); C2 mixed **+4-5%** (74.5 → ~78); code / prose T0.7 single stream +1.5-3% | exact by construction (the host still decides every token) | M | low-med (the hit rate) | **built** `eb7d0b9` |
| 4 | Prose / low-acceptance drafting (DSpark refit to the abliterated target, STS recalibration, BF16 drafter embedding) | C4 / C2 mixed, prose | Elasticity **0.56** on C4 mixed and ~0.4 on C2 mixed: +10% prose tokens a round → C4 mixed +5.5%, C2 mixed +4% | exact (drafts only propose) | L | med | not started |
| 5 | Back to 2,048-row prefill chunks once the two-rank floor / transient reservation lands (G16's cap cost) | prefill, both modes | **+13-16%** (G16: 2,015 → 1,753 at 32K replay from the cap alone); with #1: full 32K ~1,630 → ~1,880 | exact (segmentation-invariant) | M (memory work) | med (memory) | blocked on the memory task |
| 6 | Batch the per-slot round work: one drafter-ingest launch for all slots, one Engram call, graph the commit | C4 / C2 | `commit` grows 0.56 → 1.35 (C4 mixed) → 2.3 ms (C4-code) a round; −1 to −1.5 ms a round → **+1.5-2%** C4 | exact | M | low | not started |
| 7 | Slot-aware row pricing in the depth policy: a row from another slot reads ~1.4x the new experts of a draft row | C4 steady | Rejected draft rows in 4-live rounds ≈ 0.7 s of 14.95 s; pruning a third of them **+1.5-2%** | exact | M | med (joint mode 2 failed in G15b: C4-prose bimodal) | not started |
| 8 | Lower-bit routed experts (2.0-2.5 bpw pack, 0.69x expert bytes) | everything decode, C4 most | 4-live window ~114 → ~95 ms; C4 mixed **+8-10%**, single stream +10-15% | **precision trade: sign-off** | L | quality | decision |

Also considered, and smaller:

- the L2PF / gather overlap: G14a, −0.2 to −0.4 ms a window, ~+0.5-1%;
- the row-graph buckets (24 / 32 / 48 pad rows that mirror the last row: no extra expert bytes, <1%);
- graph coverage of 4-stream mixes: row graphs serve any mix up to 64 rows (6 eager windows in a whole G15b boot), so this is not a lever;
- the 0.73 s ramp of the 4-stream cell (≤ 1.5%).

**Combined.**

- **Full prefill 32K: 1,004 → ~1,630** at prod's chunk with #1 alone (target 1,500 met). It reaches ~1,880 with #5 (stretch 1,800). With the separately built prefill levers (expected +5-15% where they pass) it reaches ~1,750-2,100.
- **C4 mixed decode aggregate: ~102 → ~106** (#3) → ~108 (+#6) → ~114 (+#4 at +10% prose). Reaching 120 *on the mixed cell* needs #8 or a large prose drafter gain.
- **The target's own cell** (4 code streams) should be measured first with `--c-distinct`. The estimate below says it is already ~125-130.

## 1. Why C4 aggregate (~100) sits far below steady (~138)

C4 mixed in m2bench is 4 streams at once: code T0, prose T0.7, tweet T0 and edit T0.7, 384 tokens each. Prod 7954c1d, boot 1 of G15b:

- 1,455 tokens over a 14.95 s wall;
- decode aggregate 102.3, steady 138.1;
- all four streams live 43% of the time.

Per stream:

| stream | rounds | rows / round | tokens / round | DSpark kept / drafted |
| --- | ---: | ---: | ---: | --- |
| tweet | 49 (ends at 304 tokens, EOS) | 6.6 | 6.2 (6 lookup rounds of ~15 rows) | 175 / 185 |
| edit | 61 | 6.5 | 6.3 (5 lookup rounds) | 273 / 277 |
| code | 101 | 4.5 | 3.8 | 283 / 353 |
| prose | **228** | 2.4 | 1.68 | 155 / 309 |

**The cell is one prose stream's serial chain.** Prose is in every one of the 229 rounds, so the wall is the sum of those rounds. Timeline:

| phase | rounds | time | ms / round | share of wall |
| --- | ---: | ---: | ---: | ---: |
| ramp: one prefill round for all 4 prompts (292 ms), first token at 0.73 s | 1-2 | 0.73 s | | 5% |
| 4 streams live (~20 rows a round) | ~47 | 6.07 s | **~129** | 41% |
| 3 live | ~12 | ~1.0 s | ~85 | 7% |
| 2 live (code + prose, ~7 rows) | ~40 | ~2.4 s | ~60 | 16% |
| **prose alone** | **~127** | **~4.6 s** | 36.5 | **31%** |

**Round composition** (all 229 rounds, ms a round):

- window 54.0 (forward + candidates), 83%;
- DSpark pass 5.2 (draft.pass 4.8), 8%, **serial: speculation is off in every round that holds a top-p row**;
- commit 1.35, prefill 1.28, nucleus 0.46, prefetch 0.46, sample 0.23, plan / share 0.08, the rest ~1.9.

C4-code's round is 79 ms with a 63.5 ms window: ~15 ms a round outside the window.

**Why a 4-live round is ~129 ms, not the table's ~80.**

- The calibration's 16-row window is 71.9 ms, and C4-code runs ~18 rows in a 62.5 ms window.
- The candidates' time over the run gives the 4-live window by subtraction: 12,227 ms in total, less 127 prose-alone windows at ~30 ms, 40 two-stream windows at ~50 ms and 12 three-stream windows at ~68 ms. That leaves **~114 ms a 4-live window**.
- The reason is the distinct experts. G15b's map: U(16) = 50.3 at c4 against 36.4 for one slot's draft rows, and U(20) ≈ 57 at c4.
- At 6.38 MB an expert a layer a rank, 57 × 40 layers is 14.5 GB a rank a window: ~63 ms at 230 GB/s for the experts alone.
- C4-code is cheap because it is not four streams (§2).

**Model of the wall:**

| term | arithmetic | time |
| --- | --- | ---: |
| prose's chain | 228 rounds × 36.5 ms (prose-alone round) | 8.3 s |
| the other streams' rows | ~1,175 rows × ~3.5 ms marginal | 4.1 s |
| ramp | | 0.73 s |
| multi-slot overheads (bigger DSpark pass, per-slot commit, exchange waits that grow with rows) | | ~1.8 s |
| **total** | | **14.95 s** |

Elasticities on the C4 mixed aggregate:

- prose's round count × its round time: **0.56**;
- the other streams' marginal row cost: 0.27;
- the overheads: 0.12;
- the ramp: 0.05.

The same split explains C2 mixed (74.5 vs 89.5 steady). There, code finishes in 101 rounds and prose runs 127 more alone (prose@C2 26.5 tok/s).

Answers to the specific questions:

| question | answer |
| --- | --- |
| Ramp / tail | The 4-stream ramp is 5%. The tail (prose alone) is 31%. It is a property of the cell's mix, not of admission. |
| Admission | All 4 prompts are admitted and prefilled in round 1 (`prefill` 292 ms, n=1). Nothing waits. |
| Rows a round | ~20 at 4-live: code 4.5, prose 2.4, tweet / edit ~6.5 with lookup rounds of ~15. Graphs pad to the 24 / 32 bucket with mirror rows: no new experts, so the cost is small. |
| Drafting depth with 4 streams | `DEPTH_JOINT=0` prices each slot's rows from the single-slot table. At 4-live a row costs ~4-5 ms (≈3.5 new experts × 1.1-1.4 ms), close to the table's 4-4.6 ms, so depth is roughly right. The waste is prose's 154 rejected drafts (~0.7 s in shared rounds; lever #7). |
| Graph coverage | Complete (row mode). |
| MoE expert reads | ~57 distinct experts a layer at 4-live: the dominant cost. Only fewer bytes per expert (#8) or fewer rows move it. |
| Exchanges | They grow with rows: 1.4 → 3.4 ms a window from 1 to 12 rows (G9). They are inside the window time above. G14a's L2PF / gather overlap is the remaining exchange lever (−0.2 to −0.4 ms a window). |

## 2. The C4 metric itself (lever #2, built: `m2bench --c-distinct`)

`concurrent(order=("code",))` sends the same CODE prompt to all four streams. Under `--c-sampling mixed`, streams 0 / 2 are T0 and streams 1 / 3 are T0.7 with one seed (1234), so the cell is **two pairs of identical sequences**. Evidence: per-stream rates [51.1, 49.0, 51.1, 49.0] and identical drafting stats in pairs.

Their rows share experts as one stream's draft rows do, so U(18) ≈ 36 instead of ≈ 53.

**Estimate for 4 distinct code streams:**

- window = 62.5 ms + 17 extra experts × 40 layers × 6.38 MB / ~200 GB/s ≈ **84 ms**;
- round ≈ 84 + DSpark pass 6.1 + ~10 ms other (C4-code's measured round minus its window, less the pass) ≈ 100-106 ms;
- 4 × 3.7 tokens = 14.6 tokens a round;
- → **~138-146 steady, ~125-135 aggregate** (96% live).

TARGETS defines the C4 target on RigMark short code with a 256-token cap, so the 120 target is likely met already. The ~100 on the mixed cell is the prose-bound cell of §1.

`--c-distinct` runs code / code2 / code3 / code4 (prose likewise; the new prompts are workloads covered by `--c-exact`). Default behaviour is unchanged. Tests: `tests/test_dsv41_m2bench_steady.py`, 7 passed.

## 3. Lever #3, built: `TF_DSV41_SPEC_NUCLEUS` (eb7d0b9)

**The problem.** G8's speculative DSpark pass (`spec.py`, prod `TF_DSV41_SPEC_DRAFT=1`) launches the next pass on the GPU right after the window, while the hosts sample, plan and commit. `decode.window` refused it whenever any row samples top-p without top-k:

- `nucleus.finish`'s statistics exchange came after the candidates and would have queued behind the pass;
- the device choice had no full-vocabulary normalizer.

Every m2bench mixed cell has such rows: the T0.7 streams use top_k 0 and top_p 0.95. G15b shows `spec` 0.001 ms a round in C2 / C4, against 1.19 in C1. The 4.8-5.7 ms pass therefore sat on the critical path in every multi-stream round.

**The change** (with `SPEC_DRAFT=1` and `SPEC_NUCLEUS=1`):

- `nucleus.Pre` computes the rows' (max, Σexp) and gathers them **before** the pass is queued: same exchange, same order on both ranks.
- It copies them to pinned memory with an event, and computes each row's full-vocabulary Z on the device (`device_z`).
- `vsample.choose` uses that Z, which is `keep_count`'s rule, for the device's guess.
- `nucleus.finish` reuses the statistics: no second exchange, the same values.
- The host's `choose_rows` / `nucleus.choose` still decide every token. A wrong guess only drops the speculation (`ingested` / `take`), as before.
- Verify graphs and draft graphs have separate memory pools, so the rare full-row fallback still reads valid logits after the pass.

**Expected gain.**

- At C4 mixed the host has ~2.7 ms a round between the window and the next pass to hide the pass under (commit 1.35 + nucleus 0.46 + prefetch 0.46 + sample 0.23 + plan / share / stage ~0.2). That is about −2.5 ms × 229 rounds = −0.57 s of 14.95 s, **+4%**.
- C2 mixed: ~−2 ms × 229 rounds of 10.8 s, **+4.5%**.
- A single stream at T0.7 now speculates too. G8 measured spec at +1.7 / +2.8% on code / prose.

**Exactness and tests** (`tests/test_dsv41_host_round.py`, 26 passed):

- drafted == serial with a top-p stream (1 and 2 slots), passes launched and used, and the same drafts as with speculation off;
- `vsample.choose` with Z == `nucleus.choose` on every row the candidates decide;
- two ranks with a top-p stream: the same drafts and the same speculation decisions.

Also passed: `test_dsv41_nucleus.py` (14) and `test_dsv41_calib_knobs.py` (42; the knob is in `KNOBS`, beside SPEC_DRAFT).

## 4. Full-mode prefill: where a chunk's time goes, and lever #1

### Per 2,048-row chunk at 32K, one rank, prod's knobs at 2K chunks

- Full: 32,768 / 1,004 = 32.6 s = **2.04 s a chunk**.
- Replay: 32,768 / 2,015-2,043 = 16.0 s = **~1.0 s a chunk**.

So **the decoder half (layers 20-39 over every row) is ~1.04 s a chunk, half of full mode**. Reconstruction by class, per half:

| class | source | ~s a chunk-half |
| --- | --- | ---: |
| routed + shared experts (x3gm) | gmbench 16.1-18.1 ms a layer at 2,048 rows (1.16-1.35x its floor) | 0.36 |
| router / group / rot_in / epilogues / combine | G5 nsys classes, scaled | 0.05 |
| dense EXL3 GEMMs (wq_a / wkv / wq_b / wo_a / wo_b; indexer wq_b) | pfdense note: ~400 ms a full chunk | 0.20 |
| mHC sites + finish | mhc_pf note: 3.0-3.2 ms a boundary × 40 | 0.125 |
| TP exchanges | xovl note: ~1.1 ms × 40 | 0.045 |
| sparse attention + SWA, the 4 decoder reindex scans (24 / 28 / 32 / 36) over ~8K compressed keys, top-k | remainder | ~0.25 |
| **total** | | **~1.03** |

`g17agg.sh nsys` replaces this with a measured budget.

The full-mode-specific work is the whole decoder half:

- its 20 MoE passes;
- 20 attention layers;
- the 4 reindex scans;
- 40 mHC boundaries.

All of it runs over rows whose decoder outputs, as it turns out, nothing reads.

### Lever #1: the decoder's dependency cone (`TF_DSV41_FULL_CONE`, bc99080)

**Why rows can be skipped.** Under CED the decoder's global KV and index keys are layer 20's projection of H_19, which the encoder pass already stores for every row (ARCH-LEVERAGE section 2.1). A decoder layer's only other per-position state is its SWA ring (w = 128).

So a decoder row depends on earlier rows only through SWA windows, and each decoder layer widens that dependence by w − 1 rows. Everything a full-mode prompt keeps depends on decoder rows ≥ S − CONE only:

- the decoder rings at the snapshot point S and at the end;
- the DSpark context rows (taps entering layers 37-39);
- the first verify row.

CONE = w + (w − 1)·jmax, with jmax = max(decoder layers − 1, tap layer − 20) = 19, so CONE = 128 + 127·19 = **2,541**.

**What `cone.py` does.**

- At admission, the executor sets r0 = grid_floor(snapshot_point(n) − CONE) on both ranks. It uses None (plain full) when r0 ≤ the cached prefix; at 32K, r0 = 30,208.
- Segments that end before r0 run the encoder pass only: layers 0-19 plus layer 20's attention site and compressor, i.e. CED's encoder pass.
- Segments that cross r0, or start within 127 rows after it, run the encoder pass and then layers 20-39 over their rows ≥ r0, with SWA windows from r0 (replay's decoder pass, chunked, with rings carried).
- Later segments run the plain 40-layer run.
- `finish_prompt` drops the cone.

**Exactness.** Every kernel is row-independent, and encoder pass + decoder pass equals the 40-layer run row for row (`test_dsv41_replay`'s short-prompt equality). By the bound, every value that survives the prompt comes from rows the cone computes with untruncated windows.

So the state and replies are **full mode's bit for bit**. The session tag stays full mode's, prompt and turn snapshots stay valid, and resumed == fresh, batched == alone and drafted == serial hold. No precision question.

**Tests** (`tests/test_dsv41_full_cone.py`, 16 passed; twin with w = 16, CONE = 61). Bit-identical:

- logits plus the whole slot state (every ring, compressed rows, keys, carries) at 5 chunkings, which exercise enc / split / full segments;
- prompt-snapshot arrays;
- fast prefill kernels;
- two ranks;
- serving: fresh, an identical resend, turn 2 and batched (2 / 3 slots) replies == full mode's;
- drafted == serial, with the drafter's snapshot arrays == full's.

Two further checks show the cone is real and tight:

- decoder rings filled with noise beforehand change nothing (the decoder rows before r0 are really skipped);
- a cone shifted 16 rows short **does** change the logits.

Regressions pass: replay 11, slots 6, serving_batch 17, calib_knobs 42.

**Expected.**

- Cone TTFT ≈ replay TTFT + a decoder pass over ~2,433 more rows than replay's 127. At 0.505 ms a row (the full − replay difference over 32.6K rows) that is +1.23 s at 2K chunks, or ~+1.4 s at 1K.
- 32K: 16.0 + 1.23 = 17.3 s → **~1,890**. At prod's 1K chunks: 18.7 + 1.4 s → **~1,630**.
- Other sizes (2K chunks): 8K 4.47 + 1.23 = 5.7 s → ~1,440; 64K → ~1,990; 128K → ~1,920.
- Prompts under ~2.6K tokens are unchanged (no cone).
- Memory: one clone of a split segment's rows entering layer 20 (~50 MB at 1K rows), freed after its pass.

Prod runs `PREFILL=replay`, so this lever matters for:

- the full-mode receipt;
- any deployment that wants vLLM-equivalent (non-CED) outputs at replay-like speed.

Only the prompt-logprobs / M1-oracle path (logits at every prompt position) still needs the plain decoder over every row.

### Beyond the cone

Full mode ≈ replay + 1.2 s, so the remaining prefill levers are the encoder's, shared by both modes:

- the separately built set: PF_DENSE −75-100 ms an encoder chunk, MHC_PF −60-75, XOVL −30-40, PF_COPIES, 4K chunks;
- **#5: the 2K chunk back** once the memory work lands. That is +13-16% by G16's own A/B. x3gm at 1,024 rows takes 13.9 ms a layer against 18.1 at 2,048: 1.54x per row, because the expert weights stream once per chunk.

x3gm at 2K is already 1.16-1.35x its floor (G6), so a new MoE kernel is not the next prefill lever.

## 5. GPU plan (`scripts/windows/g17agg.sh`, inside a held campaign window)

```
g17agg.sh stage            # workstation: dsv41-agg (c02e6ff) -> both nodes
g17agg.sh prebuild         # 16 / 16
g17agg.sh tests            # CPU suites one file a run in the image, + prod words with both levers; GPU host / multistream with SPN
g17agg.sh pf               # off-full cone-full off-2k-full cone-2k-full off-replay, 2K-128K, 3 boots, reply digests
g17agg.sh nsys             # one cold 32K full prefill under cone: the measured remaining budget
g17agg.sh speed            # off vs spn, g15depth's suite + --c-distinct, 3 boots interleaved, --c-exact
g17agg.sh table
```

**Pass lines** (also in the script header).

Cone:

- every (size, boot) first token and 32-token reply digest == off-full's;
- 32K / 64K ≥ +60% and 8K ≥ +35% vs off-full;
- MemAvailable ≥ 4 GiB on both nodes.

SPEC_NUCLEUS:

- exact on every boot, with replies == off's;
- speculative passes launched and used in the mixed cells;
- C2 / C4 mixed ≥ +2%;
- the steady-cell mean ≥ +1%;
- no single-stream cell below −1.5%.

**Adoption.** `TF_DSV41_SPEC_NUCLEUS=1` goes in prod.env on a pass. `TF_DSV41_FULL_CONE=1` goes wherever `PREFILL=full` runs.

**Budget:** pf ~3 h (use `PF_SIZES` without 128K to halve it), speed ~2 h, tests ~25 min.
