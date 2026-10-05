# G17: the G16 prefill levers on the GPU, the full-mode cone, fail-fast, and what shipped (2026-10-04)

One held campaign window (`campaign.sh open` 18:21 with `DEADMAN_MIN=480`, re-armed at 20:04 to 07:04 when the window
grew: downtime approved for the performance work). Two engine commits, both on `dsv41-060` lineage:

- **253fcd7** (`dsv41-060`): every G16 prefill lever (PF_COPIES, MHC_PF, PREFILL_TILES, PF_XOVL, PF_DENSE) + today's
  prod fixes. Results: `results/G17-20261004/`.
- **0bdd276** (`dsv41-g17` = 253fcd7 + `dsv41-agg` FULL_CONE / SPEC_NUCLEUS / m2bench `--c-distinct` + `dsv41-failfast`
  fail-fast and the priced two-rank floor, both default on). Results: `results/G17b-20261004/`.

Drivers: `scripts/windows/g16prefill.sh` (pf / tests / dsweep / gate; G17 adds the `cd`, `cdx`, `final`, `cone`,
`fcone` configs, the `-2k` suffix, and `prod_knobs` now carries `MALLOC_ARENA_MAX` / `MIMALLOC_*` so `off` runs
exactly prod's words), `scripts/windows/g17agg.sh` (tests, speed with a `prod` config on prod's own sources),
`scripts/windows/g17ff.sh` (new: the fail-fast / floor GPU checks), `g16.sh stress`, `soak.sh`.

Every speed number is m2bench cold prefill (`--prefill 8192,32768,65536,131072 --prefill-reply 32`, one slot, a 2K
warm-up, a fresh random prompt a size, one process a (config, mode, boot), configs interleaved within a boot) unless
it says HTTP. "Exact" = the first token and the SHA-256 of a 32-token greedy reply equal `off`'s at every size, mode
and boot.

## 1. Tests (step 1)

| suite | 253fcd7 | 0bdd276 |
| --- | --- | --- |
| prebuild (16 shared + mhc_pf; pfdense) | 17 / 17 both nodes; pfdense built, 56 configurations described | same |
| GPU new (mhc_pf, pf16, xovl) | 27 passed, **1 failed**: `test_xovl_one_gpu_loopback_same_bits` | |
| GPU regressions (prefill2, mhc_cuda) | 51 passed | |
| CPU (pf16, mhc_pf, xovl, prefill2, prefill_fast) | 36 passed | |
| pfdense GPU (minus sweep) | 44 passed, **1 failed**: `test_out_view_bias_and_row_independence` | |
| pfdense CPU + pf16 + calib knobs | 123 passed, 7 skipped | |
| g17 CPU, one file a run (full_cone 16, host_round 26, m2bench_steady 7, calib_knobs 42, replay 11, nucleus 14, slots 6, serving_batch 17) | | all passed |
| g17 CPU under prod's words + CONE + SPEC_NUCLEUS | | 40 passed, **2 failed** |
| g17 GPU regressions with SPEC_NUCLEUS=1 (host, multistream) | | 18 passed, **1 failed** |

None of the five failures is a lever producing other bits:

- `test_xovl_one_gpu_loopback_same_bits` fails in its **off** reference run, before xovl is compared: the synthetic
  checkpoint with `TF_DSV41_PREFILL_ATTN=fused` overflows the CSA2 attention scratch (`attention: q bf16 [R, 32, 512]
  contiguous, within the scratch`). A test-setup bug; xovl's bits are covered end to end below (exact 8 / 8) and it
  is not adopted anyway (slower).
- `test_out_view_bias_and_row_independence`: the out-view and row-independence asserts pass; the test then builds a
  biased matrix (`ix_wq_b`) and calls `dense3.to_lanes`, which refuses biased matrices by design (`layout / bias`).
  Test bug.
- `test_spec_drafted_equals_serial[nucleus]` (CPU, prod words + SPN): `got == want` passes (drafted == serial); the
  test then asserts `launched == 0` for a top-p stream, the old rule SPEC_NUCLEUS exists to remove. Stale assertion.
- `test_plan_link_mode` (CPU, prod words): prod's `TF_DSV41_PLAN_LINK=rdma` in the environment vs the test's tcp
  expectation. Environment artefact.
- `test_spec_drafted_equals_serial_gpu[4]` (GPU, SPEC_NUCLEUS=1, 4 slots): the tokens equal serial (line 206 passes),
  but `seen == seen_off` fails: the per-round grouping of accepted drafts differs from speculation off (round 22:
  `[[31], [39], [206]]` vs `[[31], [206]]`). Outputs exact; the "same drafts as speculation off" invariant is not.

## 2. The pfdense sweep (step 2)

`dsweep` wrote `/cache/pfdense-table.json` on both nodes (`results/G17-20261004/pfdense-table.json`). Best at 2,048
rows: Engram 78, wo_b 75, sh_gate 70, wq_a 68 TFLOP/s (the >= 80 line on wq_b / wo_b / Engram / shared gate is not
met: 57 / 75 / 78 / 70); narrow ones 54-57 (ix_wk 16). The chunk budget measured by the test (2,048 rows, one rank,
the dense GEMMs alone): off 386 ms, fused 375 ms, **-10 ms a chunk** (the doc's -150 ms line is far off), yet the
whole-prefill gain of dense-t is +4-6% (below: the fused path also removes the unpack launches and their traffic).

## 3. The levers one by one (253fcd7, step 3)

Boot 1 every config; boot 2 only off and the candidate `cd` (cut to make room for 0bdd276's steps). Prod's words
(`off`): 1,024-row chunks, the G16 memory cap, native images, MALLOC_ARENA_MAX=4. tok/s:

| full | 8K | 32K | 64K | 128K | | replay | 8K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| off (2 boots) | 832 | 851 | 839 | 777 | | off (2 boots) | 1,621 | 1,757 | 1,743 | 1,691 |
| copies | +2.9% | +1.0% | +0.2% | +6.3% | | copies | +2.1% | +1.7% | +3.3% | +3.0% |
| mhcpf | +3.8% | -0.1% | +2.9% | +11.3% | | mhcpf | +4.7% | +3.8% | +4.5% | +4.3% |
| xovl | **-2.0%** | **-2.4%** | **-1.3%** | +5.5% | | xovl | **-1.2%** | **-2.0%** | **-1.7%** | **-1.6%** |
| combo (copies + mhcpf) | +6.6% | +5.1% | +2.6% | +14.6% | | combo | +7.8% | +7.2% | +7.5% | +7.1% |
| dense (built-in table) | +0.9% | +0.9% | +2.0% | +7.2% | | dense | +3.4% | +1.5% | +1.8% | +1.6% |
| dense-t (swept table) | +4.4% | +4.1% | +5.6% | +12.0% | | dense-t | +6.9% | +5.7% | +6.1% | +5.4% |
| **cd** (copies + mhcpf + dense-t, 2 boots) | **924** | **950** | **943** | **895** | | **cd** | **1,832** | **1,993** | **1,981** | **1,907** |
| cd vs off | +11.1% | +11.7% | +12.5% | +15.2% | | cd vs off | +13.1% | +13.4% | +13.6% | +12.8% |
| cdx (cd + xovl) | +10.8% | +11.9% | +13.6% | +20.2% | | cdx | +12.9% | +12.9% | +13.5% | +12.6% |

(off's 128K in boot 1, 749, is the low outlier; 1-boot cells are vs off's 2-boot median.) **Every config exact** in
both modes (4 / 4 a boot, cd and off 8 / 8). xovl is slower on its own and adds nothing to cd: dropped. `PREFILL_TILES`
was not run (G16's kernel lines show no tile beating upstream at the shapes that matter).
Memory (pf, MemAvailable minima): off head >= 7.6 / worker >= 6.1, cd head >= 7.8 / worker >= 6.5 GiB.

## 4. The final set on 0bdd276, 1,024 vs 2,048 rows, and the cone (steps 3-4)

`final` = `TF_DSV41_PF_COPIES=1 TF_DSV41_MHC_PF=1 TF_DSV41_PF_DENSE=fused TF_DSV41_PF_DENSE_TABLE=/cache/pfdense-table.json`.
`cone` = `TF_DSV41_FULL_CONE=1` (full mode only), `fcone` = final + cone. 2 boots each, interleaved; cone / fcone at
8K-64K only:

| full | 8K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: |
| off | 836 | 853 | 844 | 808 |
| final | 926 (+10.7%) | 952 (+11.6%) | 945 (+11.9%) | 897 (+10.9%) |
| final-2k | 1,064 (+27.2%) | 1,109 (+30.0%) | 1,093 (+29.5%) | 1,032 (+27.6%) |
| cone | 1,249 (+49.4%) | 1,619 (+89.9%) | 1,677 (+98.7%) | - |
| **fcone** | **1,394 (+66.7%)** | **1,828 (+114.4%)** | **1,893 (+124.2%)** | - |

| replay | 8K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: |
| off | 1,631 | 1,763 | 1,752 | 1,696 |
| final | 1,842 (+13.0%) | 1,990 (+12.9%) | 1,980 (+13.0%) | 1,906 (+12.4%) |
| **final-2k** | **2,025 (+24.2%)** | **2,285 (+29.7%)** | **2,273 (+29.8%)** | **2,017 (+19.0%)** |

vs the kit (vLLM 1,073 / 1,075 / 1,060 / 1,031): final-2k replay 1.89 / 2.13 / 2.14 / 1.96x; fcone full (1K rows)
1.30 / 1.70 / 1.79x; off full 0.78 / 0.79 / 0.80 / 0.78x.

**Exact everywhere:** off, final, final-2k 8 / 8 in each mode; cone and fcone 6 / 6 (== off-full's digests).
Memory (pf minima, MemAvailable head / worker GiB): final 7.5 / 6.3, fcone 7.8 / 6.4, cone 7.6 / 6.8, **final-2k
replay 6.6 / 5.0**, final-2k full 7.3 / 6.0.

HTTP (the stress server's TTFT probe, fresh random prompts, best of 2, tok/s): final-2k 2,046 / 2,233 / 2,314 /
2,283 / 2,203 at 8K / 16K / 32K / 64K / 128K; final at 1,024 rows 1,861 / 1,966 / 2,002 / 1,981 / 1,905 (G16's d4,
prod's words: 1,652 / 1,723 / 1,753 / 1,740 / 1,690).

Targets: **replay 32K / 64K >= 2,200: met at 2,048 rows (2,285 / 2,273), not at 1,024 (1,990 / 1,980). Full >= 1,500:
met by fcone (1,828 / 1,893 at 1,024 rows).**

## 5. Memory at 2,048 rows (step 4)

g16's stress (299K prefill + 3 x 64K x 2,048 decode, 4 image requests during the long prefill, guard 3.0 GiB), on
0bdd276 (priced two-rank floor on), MemAvailable minima over the whole run (stress + TTFT probes):

| run | words | head min | worker min | 299K TTFT | Tracebacks / stalls |
| --- | --- | ---: | ---: | ---: | --- |
| G16 d4 (prod today, 6e03615) | cap, 1,024 rows | 5.30 | 5.10 | 199.2 s | 0 / 0 |
| fin1k | final, 1,024 rows | 6.01 | 5.41 | 179.4 s | 0 / 0 |
| **fin2k** | **final, 2,048 rows** | **5.25** | **5.05** | **158.2 s** | 0 / 0 |

1 h soak at 2,048 rows (fin2k, `soak.sh`, results/campaign/G17b-20261004/summary-soak.txt): **FAIL**. soak.py 828
requests, 4 errors; MemAvailable min head 3.69 / worker 3.27 GiB; RssAnon slope +0.95 (head) / +0.46 (worker) GiB/h.
The causes were found and fixed in G18 (the NVMe session index's `list[int]` per entry, the graph-eviction cascade):
[G18-RESULTS.md](G18-RESULTS.md).

## 6. Fail-fast and the priced floor (0bdd276; `g17ff.sh`, test server on :8001, WATCH_HEAL=0, CANARY=off)

| check | expected | measured |
| --- | --- | --- |
| A1: SIGKILL rank 1 15 s into a 64K prompt | survivor exits 70 in ~2 s, request 500 | rank 0 saw it in 0.45 s, **exit 70** 3.5 s after the kill; the client got **500** "rank 1 closed the fate channel ...; both ranks are restarting" |
| A2: SIGKILL rank 0 | rank 1 exits 70 | rank 1 saw it in 0.08 s, **exit 70** 3.1 s after the kill; client: connection closed (its server died) |
| B1: inject 1:200:prefill | both exit 70, 500 | **both 70** (r1 14:47:07.548, r0 .595); one 500 naming the injected error |
| B2: inject 0:200:window:key | both exit 70 | rank 0 **70**; **rank 1 exit 1**, not 70: its round thread's `ConnectionError: plan link: rank 0 closed the connection` won the race against the fate thread's `os._exit(70)` (the fate thread logged "exiting (70) in 1 s" 0.1 s before). Both gone within 3 s, so the pair still ends fast; serve.sh's exit-70 fast heal would not fire for the worker's code 1 |
| B3: inject 0:200:sample | one 500, health ok, later requests served, no exit | the injected error raised on rank 0 at round 200; the following requests answered **200** and both ranks kept running (no exit) |
| C: real OOM (2,048 rows, ALLOC_CEIL 0.3, FLOOR_PRICE 0, 128K) | both exit 70 in ~2 s, 500 names OOM | rank 1 `torch.OutOfMemoryError` in round 1, **both 70** within 3.1 s, the client's **500** names OutOfMemoryError (G16: 300 s+) |
| D: floor (2,048 rows, INDEX_BUDGET 256, 299K) | waits on rank 1's memory, worker >= 4 GiB | "rank 1 reports memory every 0.5 s" on both ranks; admitted at once (worker MemAvailable 8.3 GiB, enough for the 2.2 GiB price: **the wait path was not exercised**); worker min **5.80**, head 6.42 GiB |

## 7. Gate (step 5)

M1 top-1 vs the kit's oracle (G5 gate, 0bdd276): **off 0.996091842 / first copy 0.950166113, final the same to the
last digit** (== prod and G15b / G16: 0.9961 / 0.9502).

## 8. Decode check (step 6)

`g17agg.sh speed`, 2 interleaved boots (`results/campaign/G17b-20261004/speed-table.txt`): `prod` (0bdd276 with the
final words) vs `off`: every cell within -1.6% / +1.1% (code T0 85.2, prose T0 46.9, struct T0 120.0, C1 / C2 / C4
87.4 / 74.2 / 99.8), exact (drafted == serial, every concurrent stream == alone, replies == off) on every boot.
`spn` (`TF_DSV41_SPEC_NUCLEUS=1`): C2 steady +1.3%, C4 steady -0.6%, single streams -0.2 to -1.2%: below its pass
line, not adopted.

## 9. Ship

Not shipped in this window: the 2,048-row soak failed (section 5). G18 found and fixed the drift and shipped the
final words + FULL_CONE at 1,024 rows; G19 went back to 2,048 rows with adaptive rows
([G18-RESULTS.md](G18-RESULTS.md), [G19-RESULTS.md](G19-RESULTS.md)).
