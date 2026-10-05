# G14a: MOE_FUSED timeline, BRANCHES boots, the exchanges (2026-10-03, two windows, 17 min of prod down)

Diagnostic only: no rewrite and no engine change. Prod's staged commit `767ad9f` with prod's knobs
(`config/prod.env`: MHC_CUDA, ATTN_CUDA, DENSE_V3), plus the lever under test.

- **Harness.** `scripts/windows/G14a.sh` drives the run (`open_window`, lease, deadman, `restore_prod`, the
  `SECTIONS` knob). `scripts/windows/g14a_tap.py` wraps m2bench on both ranks and only records: per-rank calibration
  reps, window GPU time from CUDA events, the RoCE trace ring and exchange sizes. `scripts/windows/g14a_report.py`
  builds the tables. `nsys_moe.py` gained `--timeline` and stream ids.
- **Windows.** Window 1 ran 19:12:27-19:24:15 (sections moe, branches, exch). Window 2 ran 19:34:04-19:39:34
  (exch2: an L2PF A/B and one more off boot).
- **Raw files.** Everything is in `results/G14a-20261003/`: `window.log`, `g14a-report.txt` / `.json`,
  `moe-nsys.txt`, `moe-timeline*.txt`, `window-*.txt`, `branches-segments.txt`, `branches-summary.txt`,
  `exch-nsys-check.txt`, the `g14tap-*.json[.gz]` tap files, `m2-*.json` and the rank logs.
- **Not committed.** The nsys SQLite exports (165 MB, untracked in the same folder) and the `.nsys-rep` files (on
  the head).

**Tracing mode.** All three traces used `--trace=cuda,nvtx --cuda-graph-trace=node` without `cuda-sw`. nsys
2026.3 reported "Hardware tracing used for CUDA tracing". The trace is light:

- code ran at 79.7 / 82.5 tok/s under it, against 82.8 unprofiled;
- the traced 1-row window span (25.33 ms) matches the unprofiled serving window GPU time (25.7 ms).

Inter-kernel gaps inside the graph are **0.19-0.22 us**. G13's `cuda-sw` node mode is the software CUPTI path, and
it inflated exactly these short gaps.

## 1. MOE_FUSED

**Why G13's nsys step gave no numbers.** It never ran. The G13 section driver (a section script on the head) ran
`window gate verdict summary` for moe, with no `nsys` step. `moe_verdict` then read the missing `moe-nsys.json` as
None ("FAIL or not run"). G13's mhc nsys step was skipped the same way: its verdict printed PASS because an empty
file passes.

The capture itself works on prod's config. It found the windows (MHC_CUDA's boundary rows from dynamic shared
memory) and every chain (40 a window). The fix:

- run the step;
- use the hardware trace;
- add stream ids and a per-kernel timeline (`nsys_moe.py --timeline`).

**Chain a layer** (rank 0, `--nsys single`: code + prose; layers 2-17 and 23-39, medians over 858 / 825 chains at
1 row and 297 / 297 at 4 rows):

| | 1 row off | 1 row on | 4 rows off | 4 rows on |
| --- | ---: | ---: | ---: | ---: |
| span (us) | 211.2 | 213.5 | 472.8 | 472.5 |
| span, all 40 layers, mean (us) | 207.2 | 210.8 (**+3.6**) | 466.4 | 463.8 (-2.6) |
| launches | 10 | 9 | 10 | 9 |
| device idle in the chain (us) | 1.9 | 0.6 | 1.8 | 0.6 |
| 2+ kernels at once (us) | 0 | 73.6 | 0 | 74.5 |
| DRAM-idle above the floor (us, mean) | 11.4 | 15.0 | | |

**1-row timeline, off** (one stream; gaps 0.19-0.22 us; start / duration in us):

```
narrow router 0/16.7 > prune 16.9/4.4 > 2 casts 21.4/0.9, 22.6/1.2 > group 24.0/3.2 > rot_in 27.6/3.2
> x3ld gate/up (routed + shared) 31.1/102.4 > epilogue 134.7/3.7 > x3ld down 139.2/67.6 > down_combine 207.4/3.9
```

**1-row timeline, on** (HW stream ids 145 = side, 146 = main):

```
side: rot_in 0/4.4 > shared x3ld gate/up 8.8/21.6 > epilogue 34.0/5.9 ......... shared x3ld down 142.2/38.7 > combine2 209.5/4.1
main:   fused_kernel 3.0/36.4 > routed x3ld gate/up 40.6/102.4 > epilogue 142.3/5.0 > routed x3ld down 147.8/60.7
```

**The shared expert does overlap the router.** 29.5 of the fused kernel's 36.4 us run concurrently with the side
stream's rot_in, gate/up and epilogue. Nothing serialises on the fork. But it buys nothing, for two reasons:

1. **The shared expert was already free in the old grouped launch.** The routed-only x3ld gate/up takes 102.4 us,
   the same as off's routed + shared launch. The grouped kernel's time is set by the routed experts, and the
   shared expert's blocks fill SMs that would otherwise idle. In the down projection it saved only 6.9 us
   (67.6 -> 60.7), and the side's 38.7 us shared down then makes combine2 wait.
2. **The fused kernel is slower than the chain it replaces.** It takes 36.4 us against 29.6 us for router +
   prune + casts + group + rot_in, and it runs while the side stream's 21.6 us shared gate/up streams DRAM beside
   it. The launches it removed cost 0.2 us each in the graph, not the ~2 us a launch the plan assumed from
   software traces.

**Net:** +2.3 us (median) / +3.6 us (mean) a layer at 1 row, about **+0.1 to +0.14 ms a 1-row window**, and
-2.6 to -4.1 us a layer at 2-4 rows. G13's +0.7 ms came from a single calibration and sits inside calibration
spread (section 2).

**For engineers:**
- Keep `TF_DSV41_MOE_FUSED=0`.
- Do not start b2 on the plan's numbers. The real MoE chain idle at 1 row is ~10-11 us a layer (~0.45 ms a
  window), not 46-58.
- What remains is the ~31 us serial head of the chain: router 16.7 + prune 4.4 + casts + group + rot_in.
- Re-derive REWRITE-PLAN's class split (sections 0 and 1.1) from hardware traces. Its "small kernels and gaps"
  rows came from `cuda-sw` node traces.

## 2. BRANCHES

**Boots.** Calibration rows are the table (1 row raw best of 3, 2+ fitted). Serving is rank 0's window GPU time
in the prose run (CUDA events, median over ~110-140 windows). Rank 1 is within 0.15 ms in every boot.

| boot | calib 1 / 2 / 4 rows (ms) | 1-row calibration runs, r0 (warm-up first) | serving 1 / 2 / 4 rows (ms) | prose tok/s |
| --- | --- | --- | --- | ---: |
| off | 23.3 / 29.1 / 37.3 | 24.22, 24.42, 23.94, 23.26 | 25.74 / 29.87 / 38.09 | 43.64 |
| off2 (window 2) | 23.5 / 29.4 / 37.6 | 25.85, 24.31, 24.05, 23.45 | 25.81 / 29.92 / 38.12 | 43.41 |
| prod + trace (g14x, g14x2) | 23.6 / 29.4 / 37.6; 23.3 / 29.1 / 37.2 | | 25.67 / 29.61 / 38.11; 25.86 / 29.77 / 38.67 | 43.59; 43.69 |
| **on1** | 23.2 / 28.9 / 37.2 | 24.43, 24.82, 24.13, 23.19 | 25.55 / 29.58 / 37.90 | 43.81 |
| **on2** | 23.1 / 29.2 / 37.3 | 24.14, 23.24, 23.08, 23.75 | 25.35 / 29.21 / 37.85 | 43.95 |
| **on3** | 23.1 / 29.6 / 37.4 | 25.64, 24.53, 23.09, 23.09 | 25.37 / 29.33 / 38.16 | 43.25 |
| **on (nsys boot)** | 23.1 / 29.0 / 37.2 | 24.00, 23.19, 24.39, 23.12 | 25.64 / 29.61 / 37.69 (under nsys) | |

**No slow boot in four BRANCHES boots.** The 1-row table reads 23.1-23.2 ms against off's 23.3-23.6. In serving,
on averages **25.42 / 29.37 / 37.97 ms** at 1 / 2 / 4 rows, and off averages 25.77 / 29.79 / 38.25 (four boots).
That is **-0.35 / -0.42 / -0.28 ms**. Prose tok/s is level (+0.2%).

**Per layer** (hardware trace, compute between consecutive exchanges, 1-row windows, `branches-segments.txt`):

- **The 8 index-layer segments get faster.** Segments 6, 18, 31, 43, 51, 59, 67 and 75 (kv sources 2 / 8 / 14 /
  20, reindex 24 / 28 / 32 / 36) are each 21-42 us shorter, -253 us in total.
- **The other 73 segments, each with its SWA-store fork / join, sum to +5 us.** The fork / join is effectively
  free.
- The trace window is -0.2 ms (21.45 vs 21.65 ms of compute between exchanges). The side branch runs on its own
  hardware stream (3 streams a window against 2).

**Answer:**
- There is no steady per-layer cost from the fork and join; the lever is a small, consistent gain.
- G13's 31.5 ms was not a per-boot property of BRANCHES that this window could reproduce: 0 slow boots in 4.
- It was most likely a calibration event. In G13's slow boot, prose ran only -3.4% against its sister boot. A real
  +5.3 ms 1-row window would have cost ~12-15%.
- Calibration is fragile at 1 row in every config. The first timed reps run 0.5-2.5 ms slow (24-25.9 ms) and settle
  at 23.1-23.5 only by the third or fourth run, and the table keeps the best of only 3 (after 1 warm-up). A boot
  that is still warming takes a high 1-row entry, and the depth policy then prices 1-row rounds wrong.
- Note: G13 ran BRANCHES on the older base (MHC_CUDA etc. off). This window tested it on prod's current base.

**For engineers:**
- BRANCHES is worth about -0.3 to -0.4 ms a window at 1-4 rows on prod's base, below its section's -0.5 ms line.
  Adopt it only after a speed run (m2bench code / prose / C2) shows a gain.
- Separately, make the calibration robust:
  - more reps at 1 row (`calib.REPS` 3 -> 6), or a median after 2 warm-ups;
  - or re-measure 1 row at the end of the sweep.

## 3. The exchanges (RoCE, prod config)

**What crosses.** At 1 row, each window graph holds **83 exchanges**: 81 x 10,240 B a rank (the bf16 partials,
2 a layer, +1) and 2 x 6,144 B. One more exchange, the candidates (8-120 B), runs outside the graph: **84 a
window**. All are on RoCE (`fast_ops`), and decode never falls back to NCCL. The 664 NCCL ops are prefill
segments above 1 MiB.

There is no eager / rendezvous switch. Every exchange is one RDMA write of the staged slot plus a 4-byte flag on
pinned host slots, striped over both CX7 functions (`ROCE_STRIPE_KB=0`). Load-time probe: 13.5-14.8 us at 16 KiB.

**Method.** The runtime's own trace ring (`GLM53_TF_ROCE_TRACE=65536`). The kernel stamps start, doorbell, flags
seen and end on the GPU's globaltimer, on both ranks. The tap reads the completed counter after each calibration
sync, so every 1-row calibration window's 84 sequence numbers are exact (11 reps a boot).

**Split a rank:**
- **stage** = start -> doorbell (shard into the pinned slot, fences);
- **wait** = doorbell -> the peer's flag seen;
- **copy** = flag -> end (both shards out of the pinned slots with system-scope loads, plus the tail's fence,
  host read and host write).
- Pairing both ranks by sequence: the late rank barely waits (its poll, **0.90 us**). When both ranks waited,
  (wait0 + wait1) / 2 is the **one-way transfer x**.

Notes:
- The proxy's posts take 0.24-0.35 us.
- globaltimer is not the host clock here (offset ~1.79e9 s, drift ~9 ppm). Serving windows are matched with a
  local offset.

**1-row windows, sums a window (us):**

| | g14x calib | g14x2 calib | **L2PF=0 calib** | g14x serving | g14x2 serving | L2PF=0 serving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exchange kernels, r0 / r1 | 1,399 / 1,418 | 1,421 / 1,357 | **959 / 1,065** | 1,646 / 1,799 | 2,014 / 1,932 | 1,306 / 1,462 |
| stage, r0 | 184 | 187 | 144 | 181 | 183 | 154 |
| copy, r0 | 571 | 568 | **324** | 533 | 539 | 328 |
| wait, r0 / r1 | 611 / 706 | 658 / 658 | 491 / 595 | 908 / 1,112 | 1,310 / 1,209 | 808 / 978 |
| late rank's exchange (critical path) | 829 | 863 | **622** | 792 | 835 | 641 |
| one-way x, med [p10-p90] | 5.66 [5.0-10.4] | 5.66 [5.1-9.9] | **5.10 [4.9-5.3]** | 5.53 | 5.62 | 5.07 |
| r0 waited longer (of 84) | 40 | 45 | 39 | 39.5 | 43 | 37 |
| first exchange: rank 1 arrives late by | | | | 126 | 102 | 174 |

**Per exchange (calibration, g14x, p10 / med / p90 us):**

- stage 1.8 / 1.9 / 3.7;
- copy 3.8 / 5.1 / 13.0;
- the late rank's whole kernel 6.5 / 7.5 / 17.6.

With L2PF=0: stage 1.7 / 1.7 / 1.8, copy 3.8 / 3.9 / 4.0, late rank 6.4 / 6.6 / 9.7.

**Where G13's extra ~0.7 ms (1.3-1.4 ms against 0.61) goes:**

1. **The 0.61 ms "floor" is one rank's uncontended view.** 84 x (stage 1.7 + poll 0.9 + copy 3.9 + tail) is
   measured at **0.62 ms** with L2PF off. It leaves out the protocol's one-way latency.
2. **The one-way transfer, ~0.24 ms a rank.** x = 5.1-5.7 us: doorbell -> proxy -> RDMA write + flag -> visible to
   the peer's poll. The rank that arrives first pays it at every exchange. Because it then leaves ~x later, it
   tends to be late at the next exchange, so the role alternates. r0 waits longer at 37-45 of 84: neither rank is
   systematically slow.
3. **L2PF contention, ~0.35-0.45 ms a rank** (1.36-1.42 -> 0.96-1.06 ms with `TF_DSV41_L2PF=0`; ~0.2 ms of it on the
   critical path, 0.83-0.86 -> 0.62).
   - The L2 prefetch kernel (`tfl2pf::segments_kernel`, 64 us, 119 a window) is launched 0.1 us before the gather
     and overlaps 2,079 of 2,158 gathers in the trace.
   - Its DRAM / C2C stream fattens the copy-out tail (p90 13 -> 4 us), the stage tail (3.7 -> 1.8) and x's tail
     (10.4 -> 5.3).
   - In nsys, a gather with L2PF beside it runs 26.3 us median; without it, 9.0 us.
   - **But L2PF=0 makes the whole window +1.7 ms slower** (calibration 25.0 vs 23.3, serving 26.9 vs 25.7-25.9).
     Keep L2PF and move it.
4. **The rest is real imbalance between the ranks' compute**, ~0.1-0.2 ms a rank.
5. **Serving only:** each window's first exchange waits **100-175 us** for rank 1, which gets the round plan over
   TCP. Serving skew is also higher overall: exchange kernels 1.65-2.0 ms a rank against 1.40 in calibration.

**For engineers:**
- **The wire is not the lever.** x is 5.1 us uncontended for 10 KiB, the post is 0.25 us and the poll 0.9 us.
- **Biggest lever: stop the L2 prefetch overlapping the gather's copy-out and stage.** Options: launch it after the
  gather in stream order, or at a point in the layer that is not an exchange; use fewer CTAs; or prefetch less a
  site. Expected **-0.2 to -0.4 ms a 1-row window** while keeping L2PF's ~1.7 ms.
- **Copy-out is 3.9 us even uncontended.** Two cheap A/Bs:
  - `GLM53_TF_ROCE_LEAN=1` (exists, prod off): drops one system fence and the tail's host read of the failure word;
  - not copying the local shard (consumers read it in place).
- **Serving's 100-175 us rank-1 start lag a window** is the next target after that. Send the plan earlier, or let
  rank 1 start on a speculative plan.

## Production status

- **Both windows restored prod through `prod-switch.sh restore`** and passed the close checks: `767ad9f`,
  MHC_CUDA / ATTN_CUDA / DENSE_V3.
  - Window 1 verified at 19:24:15; window 2 at 19:39:34.
  - `:8000` lists DeepSeek-V4.1-Flash-TF (+ aliases), 17*23 -> 391, `dsv41-tf-watchdog.timer` active, lease gone.
  - Both windows left no deadman, refresher or `sleep` behind.
- **Downtime:** 11.8 + 5.5 min.
- **Deadman fix** (`common.sh`; `campaign.sh` too):
  - the deadman and the lease refresher start under `setsid`, so each leads its own process group;
  - the new `kill_group` kills the whole group, `sleep` included. It never kills the caller's own group, so a
    deadman's own restore does not kill itself.
  - Tested on the workstation (bash + `sleep` gone) and used by both windows.
- **Left alone:** pid 796755 on the head, a G8-era `until grep ...; do sleep 20` watcher, running for a day. It is not
  this harness's. Kill it if nobody owns it.
