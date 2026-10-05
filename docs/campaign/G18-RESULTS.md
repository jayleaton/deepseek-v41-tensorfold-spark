# G18: the serving drift found and fixed; G17's prefill words shipped at 1,024 rows (2026-10-05)

One held campaign window (`DEADMAN_MIN=420 campaign.sh open` at 08:47, deadline moved to 17:17 at 12:20 for the 2,048-row
soak), engine branch `dsv41-g17`. Raw files: `results/G18-20261005/` on the head. Tools: `soak.sh` (now with a 10 s
`/proc/meminfo` node sampler per node, `soak-TAG-msamp-r{0,1}.log`), the engine's memprobe (`TF_DSV41_MEMPROBE_SMAPS=1`,
`_TENSORS=8`, `_TRACE=12`), `nvidia-smi` per-process device memory every 60 s (`nv-n{0,1}.log`), and an attribution
script that fits every series after the 30 min warm-up (`scripts/windows/g18attr.py OUT TAG [WARM_MIN]`; its output is quoted below).

## Verdict

**SHIPPED 2026-10-05 15:25**: prod runs **`cd4245a0fe82440347c6349a87c5e934f80457e0`** (`dsv41-g17`) with G17's prefill
words + `TF_DSV41_FULL_CONE=1` at PREFILL_CHUNK / ROWS = 1,024 (config/prod.env's G18 block). The serving drift was the NVMe
session tier's in-RAM index keeping every entry's token ids as a Python `list[int]` (fixed in `cd4245a`); the 1 h soak on
the fix passes (RssAnon slope -0.39 / -0.32 GiB/h, MemAvailable min 5.49 / 5.02 GiB, 0 errors, 0 CapacityError). The
2,048-row soak fails on the long-prefill transient (worker min 3.17 GiB, 1 error), so rows stay 1,024. Decode == prod
(+-1.2%), exact. Rollback: `TF_COMMIT=da5ae43cf61abeb8f40eb149146bd0fbe05fe7af` (still staged under dsv41-prod/) and
delete the G18 words block; `scripts/serve.sh stop && scripts/serve.sh start`.

## G17 recap (results/G17-20261004, G17b-20261004)

- FINAL_WORDS = `TF_DSV41_PF_COPIES=1 TF_DSV41_MHC_PF=1 TF_DSV41_PF_DENSE=fused TF_DSV41_PF_DENSE_TABLE=/cache/pfdense-table.json`:
  exact everywhere, gate top-1 0.9961 == off, decode neutral.
- Prefill at 32K (tok/s): replay off 1,763 -> final 1,990 (1,024 rows) -> final-2k 2,285 (2,048 rows); full off 853 ->
  final 952 -> final-2k 1,109; `TF_DSV41_FULL_CONE=1` full 1,619 (cone) / 1,828 (fcone = cone + final words), exact.
- Fail-fast GPU tests A1/A2/B1/B3/C/D passed; B2 (rank 0 killed, rank 1 idle) took both down but rank 1 exited 1, not 70.
- 299K stress at 2,048 rows: worker MemAvailable min 5.05 GiB (pass); 1,024 rows: 5.41.
- 1 h soak at 2,048 rows FAILED: MemAvailable slope -1.1 GiB/h head, worker min 3.27; head RssAnon +0.95 GiB/h; two
  CapacityErrors (the priced floor refused 144K / 67K prompts while memory sat under the 4 GiB hard floor); the graph
  cache cascaded 48 -> 8 (fixed in `04296a5`).

## 1. Fail-fast B2 (fixed: `6cc918b`)

`ff-B2-r1.log`: rank 1 idled in `Batcher.follow()`. Its fate thread saw rank 0 hang up first, set `dead` and slept its
1 s grace; then `link.recv()` raised `ConnectionError`, and `follow()` re-raised because `fate.dead` was already set. The
main thread then returned from `cli._serve_cuda`, and the process exited 1 before the fate thread's `os._exit(70)`.
Fix: `follow()` always hands the error to `Fate.fatal`, which for a second caller waits for the first caller's exit.
CPU test `test_follower_link_error_while_dying_exits_70_not_1` (tests/test_dsv41_failfast.py, 19 passed). Not re-run on
GPU (the B-series harness is g17ff.sh; the change only touches the path that already went down).

## 2. The drift: attribution

### Soak d1k (`6cc918b`, 1,024 rows, FINAL_WORDS + FULL_CONE + probes), 1 h

| rank | RssAnon warm -> end | slope | MemAvailable min | slope |
|---|---|---|---|---|
| r0 | 3.55 -> 3.96 GiB | +0.44 GiB/h | 5.03 | -1.73 |
| r1 | 3.10 -> 3.40 | +0.40 | 4.77 | -1.37 |

soak.py PASS (791 requests, 0 errors), 0 CapacityError. The line failed on the RssAnon slope only.

What did **not** grow (slopes after the warm-up, GiB/h; warm -> end):
- device memory: nvidia-smi per process 106,055 -> 105,955 MiB (n0), 105,035 -> 104,885 (n1); torch reserved
  102.49 -> 102.49 (its +0.27 "slope" is prefill transients, the endpoints are flat); cuda alloc flat;
- kernel: SUnreclaim +0.005, PageTables, KReclaimable, Shmem, Mapped, Mlocked flat; pinned host 0.0-0.03 GiB;
- CPU tensors: 231.8 MiB / 460 storages at +9 min -> 159.3 MiB / 445 at +57 min (falling);
- glibc mmap'd chunks 0.71 flat; the c10 mimalloc arenas -0.26.

What grew: AnonPages +0.45 (n0) / +0.41 (n1), all of it the rank process. By mapping (smaps): **"other" anonymous
mappings +0.52 / +0.49**, glibc [heap] +0.09 / +0.15, thread arenas +0.10 / 0.00; glibc in use +0.17 / +0.16. "Other"
excludes [heap], the 64 MiB arenas, the 1 GiB mimalloc reservations and is flat in glibc's mmap'd chunks, which
leaves CPython's own obmalloc arenas (mmap'd directly): small Python objects. Rank 1 (no HTTP server, tokenizer, request
log or vision cache) grew as fast as rank 0, so the leak sits in code both ranks run. Page cache (Cached +0.57 head)
is the NVMe session tier's writes: reclaimable.

### Soak "trace" (tracemalloc, 45 min): the allocation site

`TF_DSV41_MEMPROBE=4000 TF_DSV41_MEMPROBE_TRACE=12`, one diff interval (~20 min), largest growth on each rank:

```
r1  +59.9 MiB (+1,962,446 blocks) planlink.py:177 <- planlink.py:232 <- batch.py:607        (token ints from the plan)
r1  +19.7 MiB (+128 blocks)       sessdisk.py:187 <- sessions.py:239 (Store.park -> DiskTier.write)
r0  +59.9 MiB (+1,962,553 blocks) prompt_tokens.py:203 <- app.py:279                        (token ints from the tokenizer)
r0  +19.7 MiB (+129 blocks)       sessdisk.py:187 <- sessions.py:239
    +0.4 MiB  sessions.py:315 (chain digests)   +4.7 / +8.4 MiB prefetch.py:145 (the Engram LRU filling to its cap)
```

**Cause**: `sessdisk.DiskEntry` (the NVMe session tier's in-RAM index) kept every parked entry's whole token list as a
Python `list[int]` (8 B a pointer + a 28-32 B int object a token, shared with the request that made it). Entries pile up
until the tier's budget fills (`TF_DSV41_SESSION_DISK_GIB=128`): after 28 min of soak traffic the tier held 346 entries /
25.1M tokens (45 GB). Steady state at a full tier is ~70M tokens, i.e. 0.5-2.5 GiB of anonymous RSS on each rank; in
production it fills over days, which is the slow MemAvailable drift seen since G6. Ruled out: the RAM session tier
(bounded by `TF_DSV41_SESSION_RAM_MIB=256`; parked entries leave it), pinned host buffers, the Engram prefetch LRU
(65,536 rows x 2 layers, bounded), the vision cache (64 MB), request log / tokenizer (rank 1 has neither), CUDA driver
allocations for graphs (device memory flat), NCCL / RoCE buffers (flat), CPU tensors.

**Fix** (`cd4245a`, `dsv41-g17`): the index keeps `(n, chain, tail)`: the length, the page chain (16 B digests a
256-token page) and only the < 256 tokens past the last full page as int32. `find()` matches exactly as
`sessions.is_prefix` (property test over 400 random cases), `load()` reads the full ids from the file as before, and the
boot scan no longer builds a `list[int]` per file (a multi-GiB transient at a full tier). CPU tests one file at a time:
sessdisk_index 2, serving_batch 17, serving_memory 10, slots 6, replay 11, full_cone 16, failfast 19: all passed.

## 3. Soaks on the fix (`cd4245a`), 1 h each, SOAK_FRESH=1

| soak | rank | RssAnon warm -> end | slope | MemAvailable min | CapacityError | soak.py | verdict (your line) |
|---|---|---|---|---|---|---|---|
| fix1k (1,024 rows) | r0 | 3.44 -> 3.13 | -0.39 | 5.49 | 0 | PASS, 760 req, 0 errors | **PASS** |
| | r1 | 2.94 -> 2.80 | -0.32 | 5.02 | 0 | | |
| fix2k (2,048 rows) | r0 | 3.77 -> 3.24 | -0.37 | 4.07 | 0 | FAIL, 900 req, 1 error | **FAIL** |
| | r1 | 3.14 -> 2.78 | -0.40 | **3.17** | 0 | ("long" at 3,160 s: empty reply) | |

The smaps "other" slope fell from +0.52 / +0.49 to +0.04 / +0.06 GiB/h (fix1k) and -0.02 / +0.00 (fix2k). The drift is
gone. soak.sh's own stricter gates (reserved <= 0.10, MemAvailable slope >= -0.15) still flag fix1k, but the endpoints are
flat (reserved, MemAvailable warm -> end 6.81 -> 6.17 / 6.67 -> 6.66): those slopes are the long-prefill transients
sampled at random phases. 2,048 rows fails on the transient itself, not on drift: the worker dipped to 3.17 GiB
during long prefills, and the graph cache shrank 7 times on each rank (48 -> 36 -> 27 -> 20 -> 15 ...: one eviction per
separate dip, as `04296a5` intends, but repeated dips still walk the cap down). fix1k: one eviction (48 -> 36). So prod
keeps PREFILL_CHUNK / ROWS = 1,024.

## 4. Speed (decode check before shipping)

`SPEED_BOOTS=2 REPORT_BASE=prod REPORT_PREFIX=g18s2 g15depth.sh speed prod combo` (prod = da5ae43 sources
`dsv41-ctl-da5ae43` with prod.env's words; combo = staged `cd4245a` + `combo.env` = the G18 words; rows 1,024 both).
(`g18s1` produced nothing: a relative OUT is not a valid docker volume; g15depth.sh needs an absolute OUT.)

| tok/s, mean of 2 boots | prod | combo | delta |
|---|---|---|---|
| code T0 / T.7 | 84.88 / 77.75 | 84.93 / 77.77 | +0.1 / +0.0% |
| prose T0 / T.7 | 46.92 / 48.34 | 46.98 / 48.24 | +0.1 / -0.2% |
| C1 / C2 / C4 decode | 87.75 / 74.48 / 100.06 | 87.62 / 74.17 / 100.32 | -0.2 / -0.4 / +0.3% |
| C2 steady (all / code / prose) | 89.29 / 118.03 / 70.00 | 88.73 / 117.81 / 69.80 | -0.6 / -0.2 / -0.3% |
| C4 steady (all / code / prose) | 136.12 / 190.71 / 120.42 | 136.60 / 190.80 / 118.97 | +0.3 / +0.0 / -1.2% |

Exact (all / concurrent) True / True on every boot; combo replies == base. Every cell is within 1.5% (worst: C4-prose
steady -1.2%). The report's own "FAIL" is g15depth's improvement gate (steady mean >= +0.5%), which a neutrality
check doesn't need.

## 5. Production

- `scripts/serve.sh stage` (workstation) -> the pinned `cd4245a0fe82440347c6349a87c5e934f80457e0` sources on both nodes;
  prod.env + scripts copied to the head; `campaign.sh close` 15:25 (rc 0; window 398 min).
- Verified: both containers mount `dsv41-prod/cd4245a...`; `docker exec dsv41-tf-r0 env` shows TF_DSV41_PF_COPIES=1,
  MHC_PF=1, PF_DENSE=fused, PF_DENSE_TABLE, FULL_CONE=1, PREFILL_ROWS=1024; `dsv41-tf-watchdog.timer` active; :8000
  answers 17*23 = 391.
- Engine defaults now on in prod (no words needed): TF_DSV41_FAILFAST=1 (exit 70 on both ranks within ~1 s, healed on
  the watchdog's first tick by serve.sh f2250a0) and the priced two-rank admission floor (TF_DSV41_FLOOR_GIB=5 /
  _HARD_GIB=4 from prod.env).
- Rollback: `TF_COMMIT=da5ae43cf61abeb8f40eb149146bd0fbe05fe7af` and delete the G18 words block in config/prod.env, copy
  it to the head, `scripts/serve.sh stop && scripts/serve.sh start` (da5ae43 is still staged).

## Prefill vs targets

| | target | shipped (1,024 rows) | measured |
|---|---|---|---|
| replay 32K | 2,200 tok/s | 1,990 | G17 (final); 2,285 at 2,048 rows, which fails the soak |
| full 32K | 1,500 tok/s | 1,828 (fcone) | G17 |

Prod serves `TF_DSV41_PREFILL=replay`, so the replay number is the one users see; the full-cone gain applies to
`full`-mode prefills. Not re-measured in G18 (the words and kernels are G17's, unchanged by the two fixes).

## Not run / open

- fail-fast B2 not re-run on GPU (CPU test only).
- The replay 32K target (2,200) needs 2,048 rows, and those need the long-prefill transient cut by ~1 GiB on the worker
  (or a lower graph cap at boot so evictions don't cascade) before they pass the soak.
- The one fix2k empty reply ("long" request at 3,160 s) was not root-caused; it came while the worker was under the
  hard floor.
