# G14: the MoE, BRANCHES and comm variants, the new calibration, and the combo (2026-10-03, 20:14-23:21)

One held campaign window (`campaign.sh open` at 20:14:41 with `DEADMAN_MIN=270`: lease + refresher, deadman at
00:44:39, watchdog timer stopped, DeepSeek prod stopped with 0 requests in flight). Engine: branch `dsv41-060` at
**`356a188`** (G14's commits 35a51d2, 1a64981, 3b7b0b0, f70de83, b74b565, 1e432b6, c465b6f, 311c5f2, 356a188 on top of
prod's `767ad9f`), staged once (`G14.sh stage`) and prebuilt once on both nodes before any boot (16 / 16 extensions
each, roce v4 and l2pace included; `prebuild_ext.py` gained the `l2pf:_pace_ext` entry). Every config ran prod's
knobs (`config/prod.env`) plus its own words, with a fresh calibration directory a boot. Caches were dropped before
every boot and every step ran under `timeout`. Raw files: `results/G14-20261003/` (`window.log` = everything,
`step-*.out` = each step's log, `g14-window.txt` / `.json`, `combo-speed.txt` / `.json`, `combo-window.txt`,
`comm-entry.txt`, `tests-g14-*.log`, `wt-*.json`, `m2-*.json`, `g5-gate-g14-*.json`, rank logs).

- **Driver.** `scripts/windows/G14.sh` (new). It runs the three sections' window steps as ONE interleaved loop with off
  shared, on `g14wintime.py` (both ranks: the boot's own table, then 5 more calibration passes with every rank's raw
  ms, then replay == eager and a SHA-256 digest of the candidates at 1-16 rows). `g14_report.py` (new) builds the
  table. The section scripts' own `tests` / `entry` steps ran unchanged as separate processes (the three scripts share
  variable names such as `WINDOW_CFGS`, so they cannot be sourced into one shell and driven from it).
- **Harness bugs found and worked around** (not in the engine):
  - `g14branches.sh`'s window table strips trailing digits to merge repeats, so `pace150` / `pace200` /
    `pace250` and `inlinefast` / `inlinefast2` would have merged into one row. `g14_report.py` keys on the full name.
  - `prebuild_ext.py` did not build `l2pace` (G14's paced prefetch); added.
  - `moe_fused_v2` changed sources under the same extension name; `G14.sh prebuild` clears its build dir first.

## Verdict

"1 / 2 / 4 / 16" = the median over 3 boots of each boot's median over 5 calibration passes (the slower rank's raw
ms), minus off's (off: 23.43 / 28.03 / 37.52 / 73.18 ms over 3 boots). "exact" = replay == eager in every boot, ranks
agree, and the candidates' digest equals off's (`095a3459a7cc7909`, the same for all 45 boots of all 15 configs).

| section | variant | tests | exact | 1 / 2 / 4 / 16 rows vs off (ms) | expected | adopted |
| --- | --- | --- | --- | --- | --- | --- |
| moe | `inlinefast` (MOE_FUSED=1 SHARED=inline TAIL=fast) | pass (98 GPU, 203 CPU) | yes | +0.05 / +0.07 / -0.00 / +0.12 | -0.1 to -0.2 | **no** (no gain; 1-row spread 0.43) |
| moe | `inlinefast2` (the same, MOE_FUSED=rows2) | pass | yes | -0.12 / -0.06 / -0.14 / **+0.11** | 2-4 rows -0.1 to -0.25 | **no** (16 rows over the +0.05 line; gains at 1-4 rows are inside off's spread) |
| moe | `side2` (b1 side stream, rows2) | pass | yes | **+0.11** / -0.19 / -0.39 / -0.24 | 1 row 0, 2-4 rows -0.1 to -0.16 | **no** (1 row slower, spread 0.38) |
| branches | `on` (BRANCHES=1, PRIO main) | see below | yes | -0.12 / -0.35 / -0.32 / -0.82 | -0.2 to -0.5 | no (1-row line -0.3) |
| branches | **`side`** (BRANCHES=1, PRIO side) | see below | yes | **-0.53 / -0.50 / -0.50 / -0.76** | -0.4 to -0.7 | **yes** (spread 0.17) |
| branches | `equal` (BRANCHES=1, PRIO equal) | see below | yes | -0.43 / -0.50 / -0.59 / -1.09 | -0.3 to -0.6 | no (passes; side is lower at 1 row) |
| branches | `serial` (copy removal only) | see below | yes | -0.26 / -0.25 / -0.23 / -0.55 | -0.1 to -0.2 | no (side passes) |
| comm | `fast` (GLM53_TF_ROCE_FAST=1) | pass (39 GPU, 30 CPU) | yes | -0.16 / -0.06 / -0.09 / -0.51 | with pace -0.3 to -0.45 | **yes** (not slower; in the combo) |
| comm | **`pace150`** (TF_DSV41_L2PF_PACE_GBPS=150) | pass | yes | **-1.09 / -1.06 / -1.01 / -0.70** | -0.1 to -0.25 | **yes** |
| comm | `pace200` | pass | yes | -0.72 / -0.67 / -0.58 / -0.39 | -0.1 to -0.25 | no (pace150 lower) |
| comm | `pace250` | pass | yes | -0.73 / -0.77 / -0.61 / -0.84 | | no (pace150 lower) |
| comm | `after` (L2PF_AT=after, control) | pass | yes | +0.02 / +0.10 / +0.00 / -0.12 | +0.3 to +0.8 | no (control) |
| comm | **`planrdma_planpin`** (PLAN_LINK=rdma, PLAN_PIN=auto) | pass | yes (drafted == serial) | serving only: rank 1's entry lag **+130 -> -9 us** a window | -0.03 to -0.08 ms a round | **yes** (in the combo: combo beats combo-without-it on every speed cell) |

**Tests.** moe: 98 passed / 2 skipped (GPU), 203 passed (CPU / compile). comm: 39 passed (GPU), 30 passed / 1
skipped (CPU). branches: the G14 GPU suite 12 / 12 with and without prod's levers, CPU / interpreter 84 / 84; the
engine suites with BRANCHES=1 and =serial 15 / 16 each. The one failure, `test_dsv41_pdl_gpu.py::
test_engine_windows_pdl_l2pf_on_equals_off`, is **not BRANCHES's**:
- it fails with BRANCHES=0 too, and passes with BRANCHES=1 when prod's levers are off;
- it fails on prod's own `767ad9f` with the same levers (pre-existing);
- of the levers, DENSE_V3 alone makes it fail (MHC_CUDA alone, ATTN_CUDA alone pass);
- split into its two knobs (a copy of the test, `results/G14-20261003/tests-g14-ctl-*.log`): **L2PF alone with all
  three levers passes; PDL alone with DENSE_V3 fails.**
So **PDL (TF_DSV41_PDL=1) is not bit-exact with DENSE_V3.** Prod runs PDL=0 (G10: no gain), so production is not
affected. It must stay 0 while DENSE_V3 is on. The test should pin DENSE_V3 off or mark the combination.

**Prod: restored on `356a188`** with prod's G13 set (MHC_CUDA, ATTN_CUDA, DENSE_V3) plus `TF_DSV41_BRANCHES=1`,
`TF_DSV41_BRANCHES_PRIO=side`, `TF_DSV41_L2PF_PACE_GBPS=150`, `GLM53_TF_ROCE_FAST=1`, `TF_DSV41_PLAN_LINK=rdma`,
`TF_DSV41_PLAN_PIN=auto` (`config/prod.env`, workstation and head). **(verified 23:21-23:23: 356a188 mounted on both ranks
with all nine words; 17*23 -> 391; watchdog and rearm timers active; lease gone; no deadman, refresher or sampler
left)**

## The combo against off

Combo = prod's set + BRANCHES side + PACE 150 + ROCE_FAST + PLAN_LINK rdma + PLAN_PIN auto. `combonp` = the same
without the plan-link words (prod's TCP link), to price the plan link inside the combo. Boots interleaved off / combo /
combonp, 3 each, each with a fresh calibration (G14's method).

Window (`combo-window.txt`, g14wintime, 3 boots each, median of boot medians over 5 passes, slower rank):

| config | 1 row | 2 rows | 4 rows | 16 rows | 1-row spread | exact (digest == off) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| off | 23.57 | 28.17 | 37.68 | 72.92 | 0.11 | yes |
| **combo** | **21.71** | **26.42** | **35.87** | **71.72** | 0.10 | **yes** |
| combo - off | **-1.86** | **-1.75** | **-1.82** | **-1.20** | | |

The speed boots' own boot tables agree: 1 / 2 / 4 / 16 rows 23.4 / 29.2 / 37.5 / 72.2 -> 21.8 / 27.7 / 36.1 / 71.8
(-1.6 / -1.5 / -1.4 / -0.4; the 2-row entries are the fit's, see the calibration section).

Speed (`combo-speed.txt`, m2bench code / prose / structured at T0 / T0.7 with reps 2, then C1 / C2 / C4 decode
aggregates with mixed sampling; means over 3 boots):

| config | code T0 | code T0.7 | prose T0 | prose T0.7 | struct T0 | struct T0.7 | C1 | C2 | C4 | exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| off | 83.53 | 75.07 | 44.71 | 45.56 | 118.34 | 118.70 | 85.53 | 70.80 | 97.40 | yes (3 / 3) |
| **combo** | **86.58** | 78.87 | **46.24** | 48.78 | **122.21** | 123.67 | 87.69 | **74.63** | **100.66** | yes (3 / 3) |
| combonp | 86.13 | 77.22 | 46.12 | 47.58 | 121.21 | 122.09 | 87.49 | 73.76 | 99.78 | yes (3 / 3) |
| combo vs off | **+3.7%** | +5.1% | **+3.4%** | +7.1% | **+3.3%** | +4.2% | +2.5% | **+5.4%** | **+3.3%** | |
| combonp vs off | +3.1% | +2.9% | +3.1% | +4.4% | +2.4% | +2.9% | +2.3% | +4.2% | +2.4% | |

- Per boot, every combo cell is above every off cell (`combo-speed.txt`): the gain is well outside boot-to-boot
  noise (off's spread is ~1-1.5 tok/s on code, ~0.5 on prose).
- Replies == off in every boot (same-boot pairs), drafted == serial in every boot.
- **The plan link pays inside the combo:** combo beats combonp on all nine cells, most at T0.7 (+2-3 points), where
  rounds are shorter and the per-round entry lag weighs more.
- Prose at T0 gains +3.4%, less than the 1-row window's -7.9% would give alone: prose verifies ~1.6 tokens a round
  and the round has fixed host / sample / draft costs beside the window.

**Quality (combo):**
- gate top-1 **0.9961 == off's 0.9961** (first copy 0.9502 both), >= 0.996;
- MMLU-200 0-shot **0.875** (175 / 200, 0 errors; G13's prod 0.875);
- tool chains **11 / 12** (pass line 10);
- one tool call `get_weather {"city":"Hanoi"}`;
- drafted == serial in every speed boot.

## The calibration (calib.py VERSION 3 against G13's method)

G14's method: 2 warm-ups, the median of 5, rows 1-2 timed again after the sweep. G13's: 1 warm-up, the best of 3
(`*-g13cal` configs: `TF_DSV41_CALIB_REPS=3 WARM=1 STAT=min RECHECK=0`). For each boot, the boot's own 1-row table
entry is compared with the median of the 5 calibration passes run after it in the same process:

| method | boots | median abs(table - passes) at 1 row | max | boots > +0.5 ms | outliers > +1 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| G14 (VERSION 3) | 39 (+ 6 combo) | 0.04 ms | 0.71 ms | 2 (br-side b1 +0.71, moe-side2 b1 +0.58) | **0** |
| G13 | 6 | 0.04 ms | 0.21 ms | 0 | **0** |

- **G13's 31.5 ms did not come back with either method:** 0 outliers in 45 + 6 boots. The fragility G14a saw (first
  timed runs 0.5-2.5 ms slow) did not show up as a table outlier here under either method.
- The new method does not remove 0.5-0.7 ms 1-row deviations completely (2 of 39 boots, both side-stream variants'
  first boots). Neither came near the +1 ms line.
- **New finding: the table's 2-row entry is the fit's, and it reads ~1.1 ms above the measured 2-row window in every
  config** (off: table 29.15-29.91 against passes 27.97-28.16; cx-fast b1 31.39 against 28.09). The depth policy
  therefore prices 2-row rounds ~4% high in every boot. This is a bias of the fit, not noise, and is the next
  calibration fix (keep the measured raw values for rows 1-2 in the table).
- The passes themselves are tight: the 5 passes of one boot agree within ~0.05-0.1 ms, the two ranks within 0.02 ms,
  and off's 3 boots within 0.15 ms at 1 row. The configs' comparisons above are on pass medians, not on boot tables.

## Run notes

- **the worker hard reset at 21:19:56** (during boot `off-b2`, its rank had built the forward: 99.4 GiB allocated,
  MemAvailable 10.6 GiB). The journal stops mid-session with no shutdown lines; no pstore record, no MCE / Xid lines;
  the worker was back at 21:21-21:22. The memory guard then saw "worker 0 GiB" (an ssh read failing, not memory)
  and stopped the ranks. The loop continued on its own (RoCE came up normally on the next boot) and `off-b2` was rerun
  at 22:21. Cause unknown; worth watching.
- The plan link is not inside a calibrated window, so its step (`g14comm.sh entry`, under `g14a_tap.py`, code + prose
  1 stream) measures rank 1's window-entry lag over 738 serving windows a config; one boot each.
- 7954c1d (image parts as a text note, TF_DSV41_IMAGES=placeholder) landed on `dsv41-060` after this window staged
  356a188; it is not in this deploy and ships in the next one.

## Open

- PDL x DENSE_V3 is not bit-exact (above). Keep TF_DSV41_PDL=0; fix or pin the test.
- The fitted 2-row table entry (~+1.1 ms over measured in every config).
- `pace150` was the slowest pace tested and the best; 100 GB/s and below were not tried.
- MOE_FUSED: no variant gains on prod's base; b1 / b2 stay parked.
- The worker's hard reset at 21:19:56.

## Close and verification

- **23:19:57-23:21:03.** `campaign.sh close` -> `prod-switch.sh restore` on the new `config/prod.env`. It started
  DeepSeek prod and verified it: models, 17*23 = 391, canary ok, https ok. It enabled dsv41-boot-start,
  dsv41-tf-watchdog.timer and dsv41-watchdog-rearm.timer, and wrote the marker. **Prod was down 186 min**
  (20:14:41-23:21:03), inside the 3 h aim + 6 min and well before the deadman (00:44:39, killed by the close).
- **Checked after (23:23):**
  - `:8000` lists DeepSeek-V4.1-Flash-TF, deepseek-v4.1-flash, GLM-5.3-Flash-EXL3;
  - 17*23 -> 391;
  - both timers active, lease gone, no deadman / refresher / sampler (the one `sleep` on the head is the kit's
    `mem-watch.sh`, up 33 h, not this harness's);
  - both ranks mount `dsv41-prod/356a188...`, with the nine lever words, and PDL=0;
  - rank 0: `roce:` on both CX7 functions, `branches: streams (dedicated), priority side`, `plan link: rdma`,
    plan thread pinned to cpu 19, `serving: 4 slot(s)`.
- The first boot of the new commit recalibrates: calib VERSION 3 is a new cache key, so prod's old table is not reused.
- **Rollback.** In `config/prod.env`:
  - set TF_COMMIT back to `767ad9ff024a39663fb6dccff133d6e39d0f338d` (still staged on both nodes);
  - set `TF_DSV41_PLAN_LINK=tcp`;
  - drop `TF_DSV41_PLAN_PIN`, `TF_DSV41_BRANCHES`, `TF_DSV41_BRANCHES_PRIO`, `TF_DSV41_L2PF_PACE_GBPS` and
    `GLM53_TF_ROCE_FAST`.

  Then run `scripts/serve.sh stop && scripts/serve.sh start` on the head. One lever at a time can also be turned off
  on 356a188: each one's default is G13's behaviour.
