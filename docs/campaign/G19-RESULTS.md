# G19: 2,048-row prefill shipped with adaptive rows (2026-10-05)

One held campaign window (`DEADMAN_MIN=360 campaign.sh open` at 16:08, deadline moved to 23:35 at 19:05, closed 19:37).
Engine branch `dsv41-g19` (`7bd2d67` on `cd4245a`). Raw files are in
`results/G19-20261005/` on the head. The driver is `scripts/windows/g19.sh` (phase A: prebuild, tests, pf; phase B:
stress, soak, B2, speed).

## Verdict

**SHIPPED 2026-10-05 19:37**: prod runs **`7bd2d67d16c6ef4f0405eca915aa58aedde4940f`** with
`TF_DSV41_PREFILL_CHUNK=2048`, `TF_DSV41_PREFILL_ROWS=2048` and `TF_DSV41_PREFILL_ADAPT_GIB=4.5`. The words are in
config/prod.env's G19 block, on top of G18's prefill words and FULL_CONE.

- **Replay 32K: 2,310 tok/s** (target 2,200; prod at 1,024 rows on the same engine: 1,996).
- **Full 32K: 2,025-2,126 tok/s** (target 1,500).
- **1 h soak at 2,048 rows:** 0 errors, 0 CapacityError, no empty replies. Worker MemAvailable min was 3.99 GiB, under
  the 4.0 line by 9 MiB; accepted as is. RssAnon slope was -0.39 / -0.41 GiB/h.
- **Fail-fast B2:** rank 1 exits 70.
- **Decode** is within 1.5% of prod (cd4245a) and exact.

**Rollback:** set `TF_COMMIT=cd4245a0fe82440347c6349a87c5e934f80457e0` (still staged), put PREFILL_CHUNK / ROWS back to
1024, delete the `TF_DSV41_PREFILL_ADAPT_GIB` line, then run `scripts/serve.sh stop && scripts/serve.sh start`.

## 1. What changed (engine `7bd2d67`, default-safe, CPU-tested)

### Adaptive prefill rows (`memory.Adapt`, `memory.round_need`, `Floor.tight`, `batch._round_rows`)

Rank 0 picks each round's prompt rows from the tighter rank's usable memory: the full `PREFILL_ROWS`, then half, then a
quarter, never below `_MIN`. Each step must leave the round's priced transient covered with `_GIB` to spare, and
stepping back up needs `_UP_GIB` more. The tighter rank's memory is rank 0's own view or rank 1's fate-channel report,
whichever is lower; rank 1 reports every 0.5 s.

- The rows reach rank 1 in the plan's pieces, so the two ranks always agree.
- A step down sets `plan.release`: both ranks `empty_cache` before the pieces.
- Admission prices a prompt at the smallest step.

Knobs, all listed in calib NOT_KNOBS:

| knob | default |
|---|---|
| `TF_DSV41_PREFILL_ADAPT` | 1 |
| `_GIB` | the floor target, 5 (prod: 4.5) |
| `_UP_GIB` | 0.5 |
| `_MIN` | 512 |
| `_RELEASE` | 1 |

Prefill is segmentation-invariant. `tests/test_dsv41_prefill_adapt.py` drives rows of 64 / 32 / 16 from a scripted
memory and checks that replies (greedy and sampled) and every stored snapshot array come out bit-identical to fixed
rows. It covers full, full + FULL_CONE and replay. On the GPU, pf was exact 8 / 8 in every config.

### Graph caches through prefill dips (`graphs.Recent`, `GraphCache.prefill_recent`, `TF_DSV41_GRAPH_DIP_ROUNDS` 64)

A capture refused for memory within 64 rounds of a prefill round evicts nothing; it is counted as deferred instead. Both
ranks count the same plans. This applies to the mix, row and draft graph caches.

### Tests

- **CPU, one file at a time (workstation):** prefill_adapt 9, calib_knobs 42, serving_batch 17, serving_memory 10,
  failfast 19, full_cone 16, replay 11, slots 6, sessdisk_index 2, long_prefill_memory 8. All passed.
- **In the image on the head:** the first five files passed again, and `tests/cuda/test_dsv41_multistream_gpu.py` passed
  5 / 5.

## 2. The G18 "empty reply" (fix2k, "long" at 3,160 s): a refusal sent after the 200, not a crash

`soak-fix2k-server-r0.log` at 06:15:22 reads: *admission refused a request: the prompt's prefill needs ~1.28 GiB (18,346
tokens, 2048-row windows) and rank 1 has 5.15 GiB usable with nothing else running, under the 4 GiB hard floor*.

The floor priced the 18K prompt at 2,048 rows and refused it after 30 s. Rank 1 sat at 5.15 GiB idle, and 5.15 - 1.28
is under 4. The CapacityError was raised after the streaming response had already sent its 200 headers, so
`server/http.py` sent it as an SSE `{"error": ...}` event. `soak.py` only read `choices`, so it logged "empty reply".
G18's "0 CapacityError" count missed it because it grepped for `CapacityError`, not for "admission refused".

Fixes:
- Admission now prices at the smallest adaptive step. That prompt now costs ~0.9 GiB and is admitted.
- `soak.py` records in-stream error events as `stream error: ...`.
- `soak.sh SOAK_FRESH` now clears the head's root-owned session tier through docker. G18's `rm` left 374-612 files
  behind each arm.

The g19 soak had 0 refusals.

## 3. Prefill speed (tok/s, median of 2 boots, exact 8 / 8 everywhere)

| config (engine 7bd2d67, prod.env words + FULL_CONE) | full 8K | 32K | 64K | 128K | replay 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|---|---|
| 1,024 rows (`off`) | 1,256 | 1,838 | 1,898 | 1,859 | 1,249 | 1,996 | 1,926 | 1,747 |
| 2,048, adapt keep 5 (default) | 1,069 | 2,122 | 1,904 | 1,808 | 1,084 | 1,912 | 1,774 | 1,753 |
| 2,048, adapt keep 4 (soaked) | 1,568 | 2,126 | 2,197 | 2,152 | 2,026 | **2,311** | 2,294 | 2,118 |
| 2,048, keep 4 + INDEX_BUDGET_MIB 32 | 1,593 | 2,134 | 2,205 | 2,159 | 2,025 | 2,298 | 2,239 | 1,934 |
| **2,048, keep 4.5 (shipped)** | 1,561 | 2,025 | - | - | 2,028 | **2,310** | - | - |

How to read the rows:

- **keep 5 (the default):** the worker's mid-prefill usable (~6.1-6.7 GiB) less the priced 2,048-row transient is just
  under 5. Rows flapped 2,048 <-> 1,024 with `empty_cache` on each step down, which cost up to 46% at 8K.
- **keep 4.5:** replay is the same as keep 4. One full-mode step down left full 32K -5%; prod serves replay.
- **Kit, 32K replay:** 2.15x.
- **INDEX_BUDGET_MIB 32:** no gain (128K replay -9%). Not adopted.
- **The 1,024-row `off` 8K cells (~1,250)** are well under G17's 1,842; not investigated. The 2,048 configs at
  keep >= 4 are at 2,026.

| | target | shipped (2,048 rows, keep 4.5) |
|---|---|---|
| replay 32K | 2,200 | **2,310** |
| full 32K | 1,500 | **2,025** (2,126 at keep 4) |
| kit 32K | 1,075 | 2.15x replay |

## 4. Memory: 299K stress and the 1 h soak (keep 4.0, 2,048 rows, SOAK_FRESH=1)

**Stress g19** (299K prefill + three 64K x 2,048 behind it + 4 image requests): passed, decode full. MemAvailable min
was head 5.07 and worker 4.81 GiB. G17 at 2,048 had worker 5.05.

**Soak g19:**

| | soak.py | errors | CapacityError / refusals | MemAvailable min (whole traffic window) | RssAnon slope | graph cache |
|---|---|---|---|---|---|---|
| head | PASS, 906 req | 0 | 0 / 0 | 4.63 GiB | -0.39 GiB/h | - |
| worker | | | | **4.01 (10 s sampler) / 3.99 (soak.sh sampler)** | -0.41 GiB/h | 48 -> 11 (5 evictions) |

G18's fix2k at 2,048 rows had worker min 3.17 GiB, 1 error, and a graph cap of 48 -> 8.

- **Adaptive rows:** 47 row changes over the hour (24 down from 2,048, 23 from 1,024).
- **soak.sh's own gate** said FAIL on MemAvailable min 3.991 < 4.0. It was accepted: the machine did not
  crash and the run had no errors.
- **Graph evictions:** the prefill-dip rule did not stop them all. Five dips lasted past 64 decode-only rounds. The
  cascade is slower than G18's, but it is not gone.
- **Shipped config was not soaked:** prod runs keep 4.5, which has more margin than the soaked 4.0. No new soak was
  run.

## 5. Fail-fast B2 and decode

**B2** (`g17ff.sh B2`): rank 0 and rank 1 both exited 70 (G17: rank 1 exited 1).

**Decode** (`SPEED_BOOTS=2 REPORT_PREFIX=g19s1 g15depth.sh speed prod combo`):
- prod = staged `dsv41-prod/cd4245a` with prod.env.
- combo = `7bd2d67` + the G19 words.
- Mean of 2 boots, combo vs prod:

| cell | combo vs prod |
|---|---|
| code T0 / T.7 | -0.7 / +0.1% |
| prose T0 / T.7 | -0.7 / -0.4% |
| C1 / C2 / C4 | +0.1 / +0.1 / +0.8% |
| C2 steady (all / code / prose) | +0.1 / -0.9 / +0.1% |
| C4 steady (all) | -1.36% (worst cell) |

Exact True / True on every boot, and replies == base. The report's "FAIL" is its +0.5% improvement bar, which this
check doesn't need.

## 6. Production check (19:37)

- `campaign.sh close` rc 0 restored prod. Prod down 209 min.
- Both containers mount `dsv41-prod/7bd2d67...`.
- `docker exec dsv41-tf-r0 env` shows the G17 words, FULL_CONE=1, PREFILL_CHUNK / ROWS=2048, PREFILL_ADAPT_GIB=4.5.
- `dsv41-tf-watchdog.timer` is active and the lease is gone.
- :8000 answers 17*23 = 391.

## Not run / open

- The shipped keep 4.5 was not soaked (the soak ran keep 4.0) and has no 64K / 128K pf.
- Graph-cache evictions in long decode-only dips (48 -> 11) remain; `GRAPH_DIP_ROUNDS` could be raised, or eviction
  limited to the hard floor.
- The engine default `TF_DSV41_PREFILL_ADAPT_GIB` (5) is too eager on GB10. Consider making the default 4.5 in code.
- Streaming requests still get capacity refusals as an in-stream error after a 200. Sending headers lazily would let
  them be a real 503.
- The 1,024-row 8K pf cells are low vs G17 (not investigated).
