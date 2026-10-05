# G15b: native vision, calibration VERSION 4, joint depth mode 2, the expert map (2026-10-04 02:59 to 06:19)

One held campaign window (`campaign.sh open` at 02:59:10 with `DEADMAN_MIN=375`: lease and refresher, deadman at
09:14, watchdog timers stopped, DeepSeek prod stopped with 0 requests in flight). Engine: **`6e03615`** (`dsv41-060`
HEAD: clean, contains 8f10167, 6e03615, 9a22907, 3ea96ea and c3caf0e). It was staged twice:

- with `stage_tf` into the windows' staged source folder;
- with `TF_COMMIT=6e03615621c833ec54bcc43d4330b345eb8ffd11 TF_REF=6e03615 scripts/serve.sh stage` into
  `dsv41-prod/6e03615...` on both nodes.

Both nodes were prebuilt before any boot (16 / 16 extensions each, plus xmap).

**The control config `prod`** ran prod's own engine files (`dsv41-prod/7954c1d...`) with 6e03615's `m2bench.py`
overlaid (a control source folder, `.commit` = `7954c1d...+m2bench-6e03615`). m2bench is a bench-only,
self-contained module, and `Batch.generate` is identical in the two commits. The overlay gives prod the new fair
multi-stream cells and `--c-exact`, so the combo is compared against true prod on the same suite.

**Where things are:**

- Raw files:
  - `results/G15v-20261004/`: vision.
  - `results/G15calib-20261004/`: calibration.
  - `results/G15depth-20261004/`: depth, the combo against prod, and the top-1 gates.
  - `results/G15x-20261004/`: the expert map. The raw routing traces, `xtrace-*` (8-9 MB each), stay on the head.
  - `results/G15b-20261004/`: the campaign log, the thermal samplers, the boot IDs and the combo's quality runs.
- Drivers: `scripts/windows/g15vision.sh`, `g15calib.sh`, `g15depth.sh` and `g15experts.sh`. This was the first run of
  each. They were chained on the head (chain scripts).

**Prod was down 199 min** (02:59:10 to 06:18:35). That is inside the 5 h aim, and the deadman was never needed.

## Verdict

| section | result | production |
| --- | --- | --- |
| vision, `native` on 6e03615 | **FAIL on memory only.** Every functional check passes: battery s1 / s2 / s3, typed-token escape, harmony, agent flow, concurrency, text == prod, and tower rows bit-identical to G15a's fix copy. The 299K stress completes, but head MemAvailable falls to **3.30 GiB** (gate 5). The GPU test's one failure is its own threshold (below) | **not shipped**: your call on a lower context limit |
| vision, `placeholder` with the firmer notice | the model no longer guesses what the image shows | in 6e03615; not deployed (see Production) |
| calib VERSION 4 vs 3 | **window: pass**: table − measured 0.41 vs 1.13 ms mean; rows 1-4 never ≥ 0.5 ms off (old: 17 of 18 tables); no 1-row outlier in either. **Speed: neutral** (replies == old) | already the engine default since 9a22907 |
| joint depth mode 2 vs 0 | **FAIL** (the report's rule): steady cells mean −0.84%, C4-prose steady **−8.0%**, C4-code −3.2%; C2 cells +0.4 to +2.3%. Exact on every boot | stays `TF_DSV41_DEPTH_JOINT=0` |
| expert map (`TF_DSV41_XMAP`) | **FAIL**: `post` is +0.22 ms at 1 row and −0.16 / −0.43 ms at 8 / 16 rows; `prev` / `both` are +0.3 to +1.0 ms everywhere. All exact (one digest) | stays off |
| combo vs prod 7954c1d, 3 interleaved boots | combo = 6e03615 + prod's words, since nothing else passed. Exact; top-1 0.9961 == prod; MMLU-200 87.5%; chains 11 / 12; tool call ok. **Speed neutral**: steady cells mean +0.01% | **not faster: prod restored on 7954c1d** |

## 1. Vision (`g15vision.sh all`, then `placeholder` and `stressctl`)

### Checks (verdict file `G15v-20261004/vision-verdict.txt`)

| check | result |
| --- | --- |
| check / images | sources 6e03615 on both nodes; `bias_vl` sha256 `bb6f9bdd` on both nodes; 39 images (25 MB PNG regenerated) |
| CPU suites (`test_dsv41_vision`, `_images`, `_serving_app`, `_serving_template`) | 75 passed |
| GPU test `tests/cuda/test_dsv41_vision_gpu.py` | 2 passed, **1 failed: a threshold, not the engine** (below) |
| tower x2 (two processes) | rows **bit-identical across processes and to G15a's tested fix copy** on all 4 images (carrots `9782909387f4`, corn `a6dfadb9361c`, screenshot `9e171a78540b`, vqa `1fdd66049780`); warm encode 0.06-0.47 s |
| s1 native + `bias_vl` (main) | smoke 3 / 3 (verbatim banner and button); harmony 672 prompt tokens == G15a, carrot / corn; exactness (hit == cold, batched == cold, turn 2, replay tail); agent flow t1 / t2 (1,344 cached) / t3 / thinking; concurrency ok (2 text streams 18.6 / 19.0 → 11.7 / 11.9 chunks/s while 4 images run; images 200 in 6.3-11.7 s); limits all pass; VQA 20 / 20; text probes **byte-identical to prod** (4 / 4) |
| typed `<｜deepseek_image｜>` (8f10167) | **escaped**: alone, in history and beside a real image → 200. `/tokenize`: 0 image ids for the typed token, 189 beside the real corn (== the corn alone) |
| s2 `PREFILL_CHUNK=128` | cold == s1 cold |
| s3 without `bias_vl` | VQA 20 / 20 (s1 also 20 / 20: the set still cannot price `bias_vl`) |
| rank logs | 0 tracebacks in 6 / 6 logs; s1 cold == G15a's fix-copy s1 cold |
| 299K stress, native (`stnat`) | **completes**: 299K first token 173.6 s; three 64K streams decode 2,048 each (110-196 tok/s); 4 / 4 image requests 200 during the long prefill (screenshot encode 1.44 s); 0 tracebacks, 0 stalls |
| **stress head MemAvailable ≥ 5 GiB** | **FAIL: head 3.30 GiB, worker 3.14 GiB** (minima at +75 s and +43 s) |

**The GPU test's failure is a threshold bug, not an engine fault.** `test_checkpoint_tower_matches_reference` requires
the minimum per-row cosine against DeepSeek's **bf16** reference to exceed 0.8. On the screenshot it is 0.7637 (mean
0.99865). G15a showed this bar cannot hold row by row for two bf16 runs. Against an **fp32** reference, this run's
engine is better than the reference's own bf16 run on the same image: worst row 0.860 against 0.809 (means 0.99903 /
0.99900). The tower's bits equal the battery-tested G15a copy. The test should compare against fp32, or require
"engine worst row ≥ reference-bf16 worst row − ε"; it is not changed here (it is in the engine branch). Under the rule
"ship only if native passes everything" the memory failure alone would block native anyway.

### Memory: native against placeholder (`stressctl`)

| run | head MemAvailable min | worker MemAvailable min | outcome |
| --- | ---: | ---: | --- |
| `stnat` native, whole run (301 s) | **3.30** | **3.14** | completes |
| `stnat`, first ~35 s (4 images in flight) | 3.51 | 3.23 | |
| `stctl` placeholder, first 34 s | 5.75 | **2.93** | **the guard stopped our ranks** (`KILL_GIB` 3.0, worker 2.9) at +33 s; no tokens |
| s1 battery (native, no stress) | 5.79 | 6.42 | |
| at serving start, native / placeholder | 7.6-8.6 / 8.9-9.2 | 6.6-7.7 | |

- **The 4×300K stress breaches the 5 GiB floor without the tower too.** The worker, which has no tower, fell below 3.0
  GiB on placeholder within 34 s; on native it reached 3.14 GiB. G12 saw the head at 4.31 GiB on this stress. So the
  floor problem at `CONTEXT=300000 x PARALLEL=4` exists in production today, independent of vision.
- **The tower's share on the head** is about 2.2 GiB at the early peak (3.51 native with 4 image requests in flight,
  against 5.75 placeholder over the same span). It is about 1 GiB at rest (serving start).
- The `stctl` run was not repeated with a lower guard, because both the worker resets came under heavy memory pressure. Its
  head numbers cover only the first 34 s.
- **Decision for you:** a lower context limit, or fewer pool pages (`TF_DSV41_POOL_TOKENS`), for the worker as well as
  for the head's tower.

### Placeholder with the firmer notice (optional step)

The server ran 6e03615 with `TF_DSV41_IMAGES=placeholder`. **The model no longer guesses.**

- Screenshot as a user image: "I can't determine what the red banner says or what the blue button's label is from the
  screenshot, because the image wasn't actually passed through to me ... If you paste the text ... I can help."
- In a tool result: "the screenshot couldn't be passed through to me as an image, so I can't see what it shows."
- G15a's notice on 7954c1d produced "likely says something like 'Preview mode'".

The text probes are equal to prod's. Prod now runs 7954c1d's older notice. On the corn photo at 06:18 it said it
could not see the image and asked for a description, with no guess, but the firmer wording is not in prod.

## 2. Calibration VERSION 4 (`g15calib.sh window` old / new, `speed` old / new)

**Configs:**

- `old` = prod's words + `TF_DSV41_CALIB_SHAPE=fit` (VERSION 3's method and fitted table).
- `new` = VERSION 4, the engine default.

Both ran on one commit, 3 interleaved boots each. g15wintime measured REF windows (the same instrument in every
config) alternating with the method's own passes: 18 tables a config. `WIN_CFGS` was cut to `old new`; `rawg14`,
`cycw2` and `w3` did not run.

**Window** (`G15calib-20261004/g15-calib-window.txt`):

| config | mean \|table − measured\| rows 1-16 | 1 row off ≥ 0.5 / ≥ 1 ms | rows 1-4 off ≥ 0.5 / ≥ 1 ms | 1-row dev (3 boots) | 2-row dev (3 boots) | 2nd row's price, table / measured | calib s |
| --- | ---: | --- | --- | --- | --- | --- | ---: |
| old (V3) | 1.13 | 0 / 0 of 18 | **17 / 15 of 18** | −0.17 −0.48 −0.15 | **+0.98 +1.14 +1.06** | 5.88 / 4.65 ms | 14.7 |
| new (V4) | **0.41** | 0 / 0 of 18 | **0 / 0 of 18** | −0.01 +0.06 +0.03 | +0.01 −0.01 −0.01 | 4.62 / 4.66 ms | 16.6 |

Median table − measured at each row (ms), old then new:

- rows 1-8: old −0.17 +1.06 −0.04 +0.04 −0.52 −0.10 +1.59 +1.66; new +0.03 −0.01 +0.01 −0.03 −0.03 +0.11 +0.02 −0.04.
- rows 9-16: old −1.98 −1.70 −0.84 −0.29 −2.74 −2.35 −2.00 −0.76; new +0.74 +0.48 −0.06 −0.06 **+2.78 +1.70** +0.15 +0.49.

Findings:

- **Table − measured shrinks:** 1.13 → 0.41 ms mean, and to within 0.11 ms at rows 1-8 on every boot.
- **No row-1 outlier:** none of 36 tables reads the 1-row entry ≥ 0.5 ms off, in either method. G14's ~4% did not
  recur in 18 old tables either.
- **The 2nd row's price is now right:** 4.62 against 4.66 ms measured (old 5.88).
- Rows 9-16 still read up to +2.8 ms high at 13 / 14 rows. They have 5 runs a row, and those rows are rarely used.
- All digests are equal (`095a3459a7cc7909` in every config). Calibration costs +1.9 s a boot.

**Speed** (`g15-calib-speed.txt`; 3 interleaved boots each; m2bench code / prose / structured at T0 / T0.7, C1 / C2 /
C4 mixed):

| config | code T0 | code T.7 | prose T0 | prose T.7 | struct T0 | struct T.7 | C1 | C2 | C4 | 1-row / 2-row rounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| old | 85.97 | 78.19 | 45.72 | 48.03 | 121.67 | 123.14 | 87.63 | 74.04 | 99.89 | 17.2% / 33.4% |
| new | 85.29 | 77.90 | 47.02 | 48.16 | 121.60 | 123.09 | 87.57 | 73.57 | 99.32 | 9.9% / 46.2% |
| delta | −0.8% | −0.4% | **+2.8%** | +0.3% | −0.1% | −0.0% | −0.1% | −0.6% | −0.6% | |

- Exact on all 6 boots. Replies == old in every boot.
- With the right 2nd-row price the policy picks 2-row rounds far more often (46% against 33%). Prose gains; code and
  the concurrent cells are flat to slightly down. The code T0 delta is driven by one boot (84.3).
- Net: speed-neutral; a better-founded table.

## 3. Joint depth mode 2 (`g15depth.sh tests`, `speed prod off on`)

**Tests:**

- `test_dsv41_joint2`, `_joint`, `_m2bench_steady` and `_calib_knobs`: 78 passed, and pass again under prod's words +
  mode 2.
- jointsim (200 seeds) predicted on vs off, steady: C2 +1.1%, C2-code +0.8%, C2-prose +1.4%, C4 +1.6%, C4-code +1.3%,
  C4-prose +3.9%.

**Speed:** 3 interleaved boots of `prod`, `off` and `on`. m2bench ran code / prose / structured (reps 2), streams
1,2,4, mixed, `--c-same code,prose`, `--c-reps 2` and `--c-exact`. A boot took 6.3 min.

**Exactness:** drafted == serial, and every concurrent stream == the same request run alone and serial, on all 9 boots
in every cell (`concurrent_exact_all` true; prod included). `on`'s replies == off's in every boot.

Means over 3 boots (tok/s; `st` = steady aggregate while all streams are live; `prose@` = the prose stream's own
steady rate):

| config | C2 st | C2-code st | C2-prose st | C4 st | C4-code st | C4-prose st | prose@C2 | prose@C4 | C2 dec | C4 dec | code T0 | prose T0 | C1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prod (7954c1d) | 89.46 | 119.38 | 68.40 | 138.96 | 193.73 | 117.91 | 26.47 | 12.72 | 74.51 | 102.25 | 85.99 | 45.99 | 87.84 |
| off (6e03615, mode 0) | 88.64 | 118.27 | 69.89 | 136.82 | 191.49 | 120.80 | 27.41 | 12.69 | 74.29 | 101.61 | 85.60 | 47.17 | 87.65 |
| on (mode 2) | 90.10 | 121.04 | 70.17 | 139.25 | 185.36 | 111.12 | 27.80 | 12.81 | 74.91 | 102.76 | 84.77 | 46.70 | 87.97 |
| **on vs off** | +1.7% | +2.3% | +0.4% | +1.8% | **−3.2%** | **−8.0%** | +1.4% | +0.9% | +0.8% | +1.1% | −1.0% | −1.0% | +0.4% |
| jointsim | +1.1 | +0.8 | +1.4 | +1.6 | +1.3 | +3.9 | +0.4 | −5.3 | +0.7 | −0.4 | 0 | 0 | 0 |

**Verdict: FAIL.** The steady-cell mean is −0.84% (needs ≥ +0.5) and the minimum is −8.0% (needs ≥ −1.5).

**C4-prose is bimodal by boot.** Under `on` it ran at 105.5 / 106.1 / 121.7 tok/s on boots 1 / 2 / 3; off ran
120.6-120.9 on every boot. The two reps inside each boot agree to within 1 tok/s, so a boot's own state decides it.
C4-code dropped on one boot too (172.8 against ~191).

A hypothesis, not tested: it may be the calibration table steering the policy. In the slow boots the 8-row entry read
51.1-51.7 ms; in the fast one it read 52.5. With the measured cost step from 8 to 9 rows (51 → 58 ms), a policy that
packs C4-prose's shared rounds just past 8 rows would lose about that much.

Mode 2's C2 gains match the sim and are real but small. As built it cannot ship: it would make C4-prose 8% slower in
two boots of three.

**The new fair multi-stream metrics (prod, 7954c1d, means of 3 boots):**

- C2 mixed: steady 89.5 tok/s against a decode aggregate of 74.5. All streams are live 59% of the time.
- C4 mixed: steady 139.0 against 102.3, all live 43%.
- Same-workload cells: C2-code 119.4, C2-prose 68.4, C4-code 193.7, C4-prose 117.9. Their decode aggregates are within
  0.5% of these: all streams are live 94-100% of the time.
- The prose stream's own rate: 26.5 tok/s in C2 mixed, 12.7 in C4 mixed.

## 4. The expert map (`g15experts.sh tests`, `trace c1 c4`, `window` x 6 configs, `nsys off post`)

**Tests:**

- GPU: 9 passed (kernels read-only, ring exact, engine windows == off for post / prev / both-f-paced / trace +
  pregate).
- CPU / compile: 77 passed.
- The slab probe: 6 x 540 KiB gate splits take 114.4 µs cold and 105.1 µs after the map's prefetch.

### Routing traces (prod words + `TF_DSV41_XMAP_TRACE`, `_PREGATE=1`)

The traces ran m2bench code + prose: `c1` = 1 stream; `c4` = the 4-stream mixed cell. There were 2,474 / 2,552 decode
windows (deduplicated), all 40 MoE layers. Rows in a window come from DSpark drafts (1-6 rows a stream) and from
concurrent streams.

**Distinct routed experts a layer, U(R)**, measured against independent uniform picks (k = 6):

| R rows | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c1 | 4.94 | 8.24 | 11.30 | 13.86 | 15.62 | 18.96 | 18.5* | 27.9* | 30.6* | 36.4* |
| c4 | 4.94 | 8.25 | 11.40 | 14.08 | 16.64 | 19.02 | 25.29 | 34.09 | 36.38* | 50.28 |
| uniform | 6.00 | 11.91 | 17.72 | 23.44 | 29.08 | 34.62 | 45.46 | 55.95 | 66.12 | 85.53 |
| c4 / uniform | 0.82 | 0.69 | 0.64 | 0.60 | 0.57 | 0.55 | 0.56 | 0.61 | 0.55 | 0.59 |

\* 1-3 windows only (40-120 layer samples); every other cell has 7 or more windows (c4 at 16 rows: 7).

How to read U(R):

- 1 row reads 4.94 experts, not 6, because of prod's top-p 0.85 prune (min 3).
- Rows overlap strongly. 4 rows read 14 distinct experts (uniform: 23.4), and 16 rows ~50 at C4 (uniform: 85.5). One
  stream's draft rows share even more (c1 at R = 16: 36.4).
- This is the multi-row slope. Each extra distinct expert is about 6.4 MB a layer a rank.

**Prediction hit rates** (recall = the share of the window's distinct experts that the map names; precision = the
share of named experts the window reads). c1 and c4 agree to 0.01-0.03:

| map | recall | precision | what it is |
| --- | ---: | ---: | --- |
| exact (`post`, the router's own picks) | 1.00 | 1.00 | known only after the router (TF_DSV41_XMAP=post) |
| pre_next / row | 0.77 | **0.95** | layer L's gate applied to layer L−1's MoE input, per row |
| pre_next @6 | 0.66 | 0.67 | the same, top-6 a row over the window |
| pre_attn @6 | 0.45 | 0.46 | layer L's gate on its own attention input |
| prev-any @6 / prev-same @6 | 0.31 | **0.38** | the previous window's union (what `prev` / `both` use) |
| affinity @6 | 0.30-0.33 | 0.32-0.35 | co-occurrence table fitted on half the windows |

- **The previous window predicts poorly** (38% precision). That is why `prev` / `both` lose.
- **The pre-gate on the previous layer's MoE input predicts well**: 95% precision per row, 67% at top-6. No engine mode
  uses it yet; it is the only predictor worth building.
- Bytes model for the ceiling (every first gate split from L2), µs a layer / ms a window:
  - exact: 7.4 / 0.30 at 1 row, 13.7 / 0.55 at 4 rows, 19.6 / 0.78 at 16 rows;
  - pre_next@6: 5.0 / 0.20 at 1 row, 6.0 / 0.24 at 4 rows.

### Window (3 interleaved boots a config, g14wintime, slower rank's median ms)

| config | 1 row | 2 rows | 4 rows | 8 rows | 16 rows | Δ vs off (1 / 2 / 4 / 8 / 16) | exact, digest == off |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| off | 22.03 | 26.63 | 36.16 | 51.25 | 71.90 | (1-row spread 0.12) | yes |
| post | 22.25 | 26.67 | 36.11 | 51.09 | 71.48 | +0.22 +0.04 −0.06 −0.16 −0.43 | yes |
| post8 (2 splits) | 22.20 | 26.68 | 36.10 | 51.16 | 71.70 | +0.17 +0.05 −0.06 −0.09 −0.20 | yes |
| prev (site b) | 22.69 | 27.09 | 36.54 | 51.65 | 72.36 | +0.66 +0.46 +0.38 +0.40 +0.46 | yes |
| both (site b) | 22.71 | 27.05 | 36.70 | 51.55 | 72.21 | +0.68 +0.42 +0.54 +0.30 +0.31 | yes |
| bothf150 | 23.06 | 27.41 | 36.91 | 51.75 | 72.08 | +1.03 +0.78 +0.75 +0.50 +0.17 | yes |

**Verdict: nothing adopted** (no config is ≤ off + 0.05 at every row count), so there was no speed step.
`TF_DSV41_XMAP` stays 0. The expected −0.1 to −0.25 ms at 1 row became +0.22 ms.

**Why `post` loses at 1 row** (`nsys off post`, prose, 1 stream, graph nodes; `xmap-nsys.txt`):

- The prefetch kernel (33 µs a layer at 1 row) runs on a side stream beside the MoE chain's head and **slows it**.
- `_prune` takes 13.1 µs instead of 4.4 (the casts, `group` and `rot_in` slow down too). x3ld's gate/up launch starts
  12.6 µs later.
- The prefetch does hit: x3ld gate/up takes 91.1 µs instead of 102.3 (−11.2 µs).
- Net, the layer's chain is 217.4 µs against 213.2 (+4.2 µs, about +0.17 ms a window).
- At 4 rows the gate/up launch saves 18 µs (272 against 290) and the chain is flat (517.2 against 516.7).

A map that lands without stalling the chain head would be worth the 11-18 µs a layer measured here: about 0.45-0.7 ms
a window. That needs either a lower-priority prefetch or one issued earlier (the pre-gate).

## 5. Combo against prod and the quality gate

Native vision did not ship, depth mode 2 failed and no xmap mode won. So **the combo is 6e03615 with prod's words
exactly** (`results/G15b-20261004/combo.env` empty; `off` above). It ran against `prod` (7954c1d) in the same 3
interleaved boots (`G15depth-20261004/combo-vs-prod.txt`):

| combo vs prod (means of 3 boots) | code T0 | prose T0 | struct T0 | C1 | C2 | C4 | C2 st | C2-code st | C2-prose st | C4 st | C4-code st | C4-prose st | prose@C2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| delta | −0.4% | +2.6% | −0.3% | −0.2% | −0.3% | −0.6% | −0.9% | −0.9% | +2.2% | −1.5% | −1.2% | +2.5% | +3.5% |

The six steady cells' mean is +0.01%. The gains are in prose (the 2-row rounds of calib V4) and the small losses in
code and the C4 aggregate. **Not faster.**

**Quality gate (the combo):**

| gate | result |
| --- | --- |
| top-1 vs the kit oracle (G5 gate, 8 prompts x 16, 4K) | **0.9961** for prod, off (= combo) and on: equal and ≥ 0.996 (first copy 0.9502 each) |
| MMLU-200 0-shot (test server on :8001, config/prod.env on 6e03615) | **87.5%** (175 / 200), equal to G14's combo |
| tool chains (`chains.py`, off) | **11 / 12**, PASS (line 10) |
| one tool call | `get_weather {"city": "Hanoi"}` |
| drafted == serial | yes, every m2bench run (calib 6 boots, depth 9 boots, traces) |
| concurrent == serial (`--c-exact`) | yes, all 9 depth / combo boots, every cell (C1, C2, C2-code, C2-prose, C4, C4-code, C4-prose, both reps) |

**Deploy decision:** the rule was "the combo exact and faster, or vision passes". The combo is exact but not faster,
and vision failed its memory gate. **`config/prod.env` was not changed; prod was restored on 7954c1d.**

6e03615 is staged in `dsv41-prod/` on both nodes and passes every quality check. Deploying it on placeholder would
bring the typed-token escape (8f10167) and the firmer notice at neutral speed. That is your call; it was not done here.

## Harness bugs found and fixed (first runs of all four scripts)

1. **`g15experts.sh trace`**: `g15_xmap_report.py trace` needs numpy, and the head's host `python3` has none. Fixed:
   the analysis runs in the image.
2. **`g15experts.sh` / `xm_prebuild` hung forever.** A bare `wait` also waited on the memory samplers that the script
   had started itself (`mem_start`), so `tests` never returned. It cost 20 min before it was noticed. Fixed: `wait` on
   the docker pid.
3. **`g15experts.sh` wrote into `results/G14-20261003/` on the head.** The sourced `g14branches.sh` sets
   `OUT=G14-<UTC date>`, and the UTC date was still 10-03. Fixed by keeping the caller's `OUT` before sourcing. The 73
   g15x files were moved to `G15x-20261004`. Note: `G14-20261003/{window,mem-r0,mem-r1}.log` on the head now carry
   G15x lines appended.
4. **`G15a.sh`** had lost its exec bit on the workstation (rsync carried it to the head). Fixed.
5. **Additions for this window** (not bugs):
   - `g15depth.sh`: configs `prod` (CTL sources), `combo` (`$OUT/combo.env`) and `xpost`; a `gate` step on each
     config's own sources.
   - `g15depth_report.py --base / --prefix`.
6. **Not fixed (engine test):** `test_dsv41_vision_gpu.py::test_checkpoint_tower_matches_reference`'s `min > 0.8`
   against bf16 (section 1).

Two operator slips during the window had no effect on any measurement: `pkill -f` patterns matched the operator's own ssh command
lines twice. In both cases the queued chain was restarted by hand.

## the worker

**No reset during G15b.**

- Boot ID `eb4ba2e5-c252-44fa-9301-6301684ebb62` was the same at 02:59 and 06:19, with uptime 5 h 37 min → 8 h 57 min.
  The head's boot ID `b3028acf...` was also unchanged.

**5 s samplers on both nodes for the whole window** (`therm-r{0,1}.log`, 2,376 samples each; the worker's streamed to
the head), plus the worker's `dmesg -w`:

| node | GPU temp max / mean | power max | hottest ACPI zone max |
| --- | --- | --- | --- |
| The head | 84 / 66.3 C | 66.3 W | 92.0 C |
| The worker | 82 / 65.0 C | 64.1 W | 94.4 C |

- Both nodes ran warmer than in G15a (76 C), because the speed runs were longer and denser. The twins are within 2 C.
- The live kernel log shows nothing during the window: only docker / apparmor noise and drop_caches.
- **A lead from its boot-time history:** the boot after the 2026-10-03 21:19 reset logged
  `BERT: [Hardware Error]: Skipped 1 error records` at 21:21:47. The firmware's Boot Error Record Table held a
  hardware error from before that boot, which points to a firmware-level or hardware event (not an OS crash), as G15a
  suspected. Next: dump the BERT record (`/sys/firmware/acpi/tables/BERT`, or `rasdaemon` / `cper` decoding) and check
  whether the 10-02 13:17 reset left one too.
- Memory pressure did not reset it this time, even when the worker went below 3 GiB MemAvailable (`stctl`, stopped by
  the guard).

## Production (verified 06:18-06:20)

`campaign.sh close` ran `prod-switch.sh restore` on the head's unchanged `config/prod.env`, then verified the result.

- `:8000` lists `DeepSeek-V4.1-Flash-TF`, `deepseek-v4.1-flash` and `GLM-5.3-Flash-EXL3`; https lists them too.
- 17*23 → **391**.
- An image request (corn photo, user message) → 200 with `images_omitted` 1 (placeholder, 7954c1d's notice).
- Both ranks mount `dsv41-prod/7954c1def8c821335f10f5a5463745b90ad71d70` and carry `TF_DSV41_IMAGES=placeholder`,
  `TF_DSV41_DEPTH_JOINT=0`, `TF_DSV41_L2PF_PACE_GBPS=150`, `TF_DSV41_BRANCHES=1`, `TF_DSV41_PLAN_LINK=rdma` and
  `GLM53_TF_ROCE_FAST=1`. No `TF_DSV41_XMAP` and no `TF_DSV41_BIAS_VL`.
- Rank 0 shows `serving: 4 slot(s)` and `images placeholder (TF_DSV41_IMAGES)`.
- `dsv41-tf-watchdog.timer` and `dsv41-watchdog-rearm.timer` are active, `dsv41-boot-start` is enabled, and the stack
  marker is `dsv41`.
- The lease is gone. No deadman, refresher, chain, memory sampler, thermal sampler or worker `dmesg -w` is left. The
  one `sleep` on the head belongs to the kit's `mem-watch.sh`, which predates the window; the worker has none.

## Next

1. **Memory at 4 x 300K, with or without native:** the worker breaches 3 GiB on the G12 stress today. Choose a lower
   `CONTEXT` or `TF_DSV41_POOL_TOKENS` (the worker needs ~2 GiB more, the head ~2.2 GiB more with the tower at its
   peak), then rerun `g15vision.sh stress` / `stressctl` and ship native on the same verdict.
2. Fix the vision GPU test's bar (compare against fp32, as `g15a_tower.py` does).
3. Depth mode 2: find what makes C4-prose lose 13% in some boots (log the policy's chosen rows per round in C4-prose;
   the 8→9-row step is the first suspect) before another speed run.
4. Expert map: a pre-gate predictor (95% per-row precision) issued during the previous layer, or the `post` prefetch at
   lower priority so that it does not slow `_prune` / `group` / `rot_in`.
5. The worker: decode the BERT record.
6. Optional deploy of 6e03615 on placeholder (neutral speed, all gates pass): the typed-token escape and the firmer
   notice. Rollback: `TF_COMMIT=7954c1def8c821335f10f5a5463745b90ad71d70`.
