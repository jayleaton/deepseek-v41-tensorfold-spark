# G16-4K: 4,096-row prefill chunks + the x3gm retune (offline, NOT RUN on the Sparks)

Status 2026-10-04: built and tested offline (CPU, Triton interpreter, sm_121 compile) on tf `4358ef7` (code) + `9bbc991`
(exactness / GPU tests): stage dsv41-060 at `9bbc991` or later. **DISABLED by default; the boot
REFUSES it without `TF_DSV41_PF_4K_MEMORY_OK=1`.** It must not ship until the release cap and the leak check in
[G16-RELEASE.md](G16-RELEASE.md) are validated. The production config must not set any 4K knob
(`g16chunk4k.sh check` fails if it does). Harness: `scripts/windows/g16chunk4k.sh` (not run).

## 1. Knobs and the guard

| knob | meaning |
| --- | --- |
| `TF_DSV41_PF_4K=1` | 4,096-row segments and rounds (`PREFILL_CHUNK` / `PREFILL_ROWS` default 4,096; any other explicit value is a ValueError) |
| `TF_DSV41_PF_4K_MEMORY_OK=1` | the acknowledgement. Without it, any prefill window above 2,048 rows refuses at boot |
| `TF_DSV41_PREFILL_CHUNK`, `TF_DSV41_PREFILL_ROWS` > 2048 | the same refusal (a window holds max(chunk, rows) rows). Above 4,096 always refuses |
| `TF_DSV41_GM_CFG=gu,dn` | x3gm tiles: `a` = the tuning table, gu 0..5, dn 0..4 |
| `TF_DSV41_GM_TUNE=file` | a measured tuning table (`gmbench --sweep --tune-out`) |
| `TF_DSV41_CED_STASH_TRIM`, `TF_DSV41_ENGRAM_BULK_SUB` | the two memory savings. Default: on under 4K, off otherwise |

`pf4k.check()` runs in `Dsv41Engine.__init__` right after the PDL check, before the config or weights load. It also runs
in `forward.prefill_chunk()` and in the batcher's env path. The message reads: "4096-row prefill windows ... need about
0.66 GiB more worker memory (0.52 GiB device activations and gathers + 0.14 GiB Engram host buffers, with 4K's savings;
x3gm's block reuses the expert scratch); enable only after the memory cap/leak work (see docs/G16-RELEASE.md ...) is
validated: set TF_DSV41_PF_4K_MEMORY_OK=1 to acknowledge."

## 2. Exactness: 4K == 2K, bit for bit

The math allows it, and the tests show it. Every kernel computes a row from that row alone. Every reduction has a
fixed order per row: the k16 chains, the mHC column order, the per-row indexer selection over the keys that row can
see. A segment's compressed rows and index keys are stored before its rows select or attend. Segment starts stay even,
so a ratio-2 group never straddles a boundary through the carry (and the carry is the exact fp32 projection row
anyway). The replay tail is 127 rows whatever the chunk.

Some paths are chosen per window rather than per row: the streaming top-k (`STREAM_MIN`), the index budget's row blocks,
the mHC prefill tile, the router's 2,048-row launches, and x3gm's blocks and tiles. At 4K these switch for different
rows. Each pair of paths gives the same bits; this was already relied on at 2K and is now tested at the boundaries 4K
moves.

- CPU twin, literal sizes (`test_dsv41_chunk4k_exact.py`, 4 passed, ~11 min at -n 4): 4,161 tokens, 4,096-row vs 2,048-row segments, full and
  replay. Logits and the whole slot state (rings, compressed rows, keys, carries, lookback, stash) are equal. Also:
  resumed (2K snapshot -> 4K) == fresh, and two slots in one 4K window == each alone.
- Triton interpreter, scaled (`test_dsv41_chunk4k_interp.py`, 2 passed): 2P vs P segments, with each window-size threshold set
  so the 2P windows cross it. Covers the exact path and fast kernels + fused attention.
- x3gm emulator (`test_dsv41_chunk4k_x3gm.py`): every new tile == G6's bits, with 128-member passes and 3-pass experts.
- GPU, for the window (`tests/cuda/test_dsv41_chunk4k_gpu.py`): the 4K tiles at the target widths, the arena, router
  halves, and the whole forward with 4K == 2K == 1K (fast + gm, full and replay).

drafted == serial is unaffected: decode rows are full rows, and the prefill state they start from is identical.
**No precision question to sign off.**

## 3. x3gm retune

Why 4K did not pay before: G6 measured gm at 35.3 ms / 4,096 rows vs 18.1 / 2,048. At 4,096 rows the x3gm buffers
(Y = 503 MB) don't fit the shared scratch's `z` (264 MB), so `Buffers` silently halved to two 2,048-row blocks, and the
trellis was streamed and decoded twice. On top of that, ~64 members an expert split most experts into two 64-member
passes.

Changes:

1. **One 4,096-row block, in an arena.** Under 4K, `moe.arena_scratch` lays upstream's scratch buffers (z, y, xg, xu,
   xd) out as views of one allocation of 574 MB at 1,024 rows. x3gm puts X/Y at its start and Xd at its end:
   503 + 57 = 560 MB fits. **Zero new bytes**; fresh buffers would have cost 534 MiB.
2. **128-member tiles** (`GM_GU2..5`, `GM_DN2..4`, mirrored in `x3gm.GU_TILES`/`DN_TILES`): NG 16 (BM 128) with
   16 warps x 64 accumulators or 8 warps x 128 accumulators, 2 or 4 k tiles a stage, rings of 3-6. Uniform routing at
   4,096 rows needs 2/3 of the 64-member passes. Only data movement changes: each element is still one ascending-k mma
   chain from +0.0, followed by the same epilogue. Plans are computed per BM.
3. **Launch bounds:** `MINB` is 2 only for ≤8 warps and ≤64 accumulators. The G6 tiles compile to identical PTX
   (checked against HEAD).
4. **Tuning table** (`x3gm.TUNE`): ≤48 members an expert → G6's (0|1, 0), unchanged for 2,048-row blocks. Above that →
   gu 3 (one rotated input) / 2 (two), dn 2. These entries are provisional until `gmbench --sweep` measures them;
   `TF_DSV41_GM_TUNE` loads the measured table.
5. **sm_121 compile:** all 51 gm_kernel instances, registers 124-242, no spills, no stack.

## 4. Memory: extra peak per node for 4K (worker; `pf4k.terms()`)

| term | +2,048 rows |
| --- | ---: |
| persistent through a run (streams 80, mHC scratch 60, gathered partials 40, Engram rows 48, xh 24, out 20, stale hidden 20, cand / sel 20) | 312 MiB |
| peak sublayer: attention locals (q 64, o 64, wo_b partial 40, qi 16, z 16, ...) | 223 MiB |
| TP all-gathers: the 21 MB a gather becomes 42 MB; receive [2, n, D] 40 + partial 40 + bf16 copy 20 | inside the peak above (exchange 100 < attention 223) |
| x3gm workspace | **0** (arena) |
| Engram host: kept fp32 row buffers (6 x 24 MiB) | 144 MiB |
| **total, with the savings** | **679 MiB = 0.66 GiB** |
| without the savings: + CED stash (4 slots x 100 MiB) + Engram O_DIRECT read buffers (4 jobs x 96 MiB) | 1.43 GiB |

Savings found (same bits, on under 4K):

- **CED stash trim.** `torch.cat(...)[cut:].contiguous()` is a view, so each slot's stash keeps KEEP + segment rows of
  storage (~51 KB a row) until the slot's reset. Trimmed, it holds 127 rows. Saves 401 MiB at 4K with 4 slots. **This
  also applies to production at 2K/1K** (~425 / ~225 MiB at 4 slots), so it is a candidate for the release cap work:
  `TF_DSV41_CED_STASH_TRIM=1` alone.
- **Engram bulk sub-reads.** Reads of at most 24,576 records into one fp32 buffer keep the O_DIRECT buffers (a 4 KiB
  sector a record) at the 2K size. Saves 384 MiB at 4K.
- **x3gm arena.** Avoids 534 MiB.

Not done (would cut ~80 MiB more): dropping attention locals (qa/kva/q/qi) before `_out` in `blocks.attention`, which
is a hot shared file. `TF_DSV41_PF_COPIES=1` takes another 20 MiB.

RoCE: `GLM53_TF_ROCE_MAX_KB=1024` routes shards of ≤1 MiB over RoCE. Prefill gathers (21 MB at 2K, 42 MB at 4K; 21 MB
halves under XOVL) go through NCCL either way, and decode is unchanged, so nothing changes. Graph buckets are context
positions, not rows: unaffected.

`memory.load_check` subtracts `pf4k.extra_gib` under 4K. `chunk_gib` scales by the window.

## 5. Window plan (g16chunk4k.sh)

`check`, `prebuild`, `refuse`, `tests`, `sweep`, then `pf` with 2k / 4k / 4k-best x full / replay x 3 boots
(2K-128K, reply digests, both nodes' minima), then `stress` 2k vs 4k, `table` and `gate`. Pass lines are in the
script header. Expected: +10-16% prefill (−200 to −320 ms a 2,048-row equivalent).
