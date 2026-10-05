# G15a: native image input (tf 7954c1d) and the L2 prefetch pace below 150 GB/s (2026-10-03 23:40 to 2026-10-04 00:45)

One held campaign window (`campaign.sh open` at 23:40:00 with `DEADMAN_MIN=210`: lease + refresher, deadman at
03:10:00, watchdog timers stopped, DeepSeek prod stopped with 0 requests in flight). Engine: **exactly `7954c1d`**
(`dsv41-060`, the image commit, parent `356a188` = prod). It was staged with `TF_REF=7954c1d` and prebuilt on both
nodes before any boot (16 / 16 extensions each). The 63 KB `bias_vl` sidecar went into the cache volume on both nodes
(`/cache/dsv41-bias-vl/bias_vl.safetensors`; file sha256 `bb6f9bdd...cfd36` on both nodes; all 43 per-tensor SHA-256
values in its metadata check). Driver: `scripts/windows/G15a.sh` (new), with `g15a_tower.py` (the tower on the GPU
against DeepSeek's reference, no serving) and `g15a_vision.py` (the HTTP battery and the test images). Raw files:
`results/G15a-20261003/` (`window.log`, `step-*.out`, `tower-*.json`, `vision-*.json`, `server-g15v-*-r{0,1}.log`,
`vision-speed.txt`, `pace-window.txt`, `therm-r{0,1}.log`, `dmesg-r1-live.log`, `worker-pre.txt`; the 27 MB
limit-test PNG stays on the head).

**Prod was down 65 min** (23:40:00 to 00:45:24): inside the 2.5 h aim, with the deadman never needed.

## Verdict

| part | result | production |
| --- | --- | --- |
| vision, `native` on 7954c1d | **FAIL**: every image request returns HTTP 500 `RuntimeError: No available kernel` (an engine bug, below) | **not enabled** |
| vision, `placeholder` (default) on 7954c1d | pass: image parts in user and tool messages return 200 with `stats.images_omitted`; text replies == 356a188; speed equal | **shipped** |
| vision, `native` + a one-line 4-D fix (TEST-ONLY copy, not committed) | passes every check except one limit item (a typed `<｜deepseek_image｜>` is not refused) | next: commit the fix, run a short window, then ship native |
| L2PF pace 100 / 125 vs 150 | both **slower** at 1, 2 and 4 rows, well outside the boot spread; all exact | **150 stays** |

**Production now:** `7954c1d` with `TF_DSV41_IMAGES=placeholder` (written out in `config/prod.env`) and every G14 word
unchanged (pace 150). This fixes the coding-agent preview-browser turns that failed with HTTP 400 (`messages[N].content: images
are not served ...`).

## Part 1: vision

### The bug in 7954c1d (why native fails)

`vision._sdpa` limits SDPA to `[FLASH_ATTENTION, EFFICIENT_ATTENTION]` and passes q / k / v as 3-D `[heads, n, 64]`.
PyTorch's fused kernels need 4-D inputs ("All fused kernels requires query, key and value to be 4 dimensional, but got
Query dim: 3"), and the math fallback is excluded, so every image fails:

- `G15a.sh tower p1` / `p2` (the tower alone): `RuntimeError: No available kernel. Aborting execution.`;
- server `s0` (7954c1d, native, bias_vl): the screenshot as a user image -> **500**; the same screenshot in a tool
  result -> **500**. Text requests on the same server -> 200 and byte-identical to prod's 356a188 (4 / 4 probes). The
  server survived (no stall; the rank 0 log has the two tracebacks).

The CPU tests could not catch this: on the CPU, `_sdpa` calls plain `F.scaled_dot_product_attention`. The fix
(tested below as a copy staged in a separate source folder on both nodes, `.commit` = `7954c1d+g15a-fix4d`):

```python
with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
    return F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)).squeeze(0)
```

It was not committed to `dsv41-060` (other work was in progress on that branch). It needs a GPU test that runs `Tower.span` on
CUDA.

### The tower with the fix (VISION.md section 6 step 2)

The tower is BF16, 0.971 GB, and loads in 1.0 s. Four images were encoded: the release's `carrots.jpeg` (1024x701) and
`corn.jpeg` (450x308), a 1920x1080 dashboard screenshot, and a text image. Each was compared against DeepSeek's
`inference/vision.py` (`tests/fixtures/dsv41_vision_ref.py`), run with the same weights in bf16 and in fp32 on the GPU.

| image | ViT patches | span | warm encode | rows SHA-256 (f1 = f2 = f4 = no-flash) | engine vs ref-bf16 | cos vs fp32 (mean / min): engine | ref bf16 |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| carrots | 3,774 | 444 | 0.16 s | `9782909387f4` | 8.5 ulps of row max | 0.99907 / 0.959 | 0.99923 / 0.964 |
| corn | 1,551 | 189 | 0.05 s | `a6dfadb9361c` | 6.2 | 0.99901 / 0.960 | 0.99914 / 0.960 |
| screenshot 1920x1080 | 8,418 | 968 | 0.47 s | `9e171a78540b` | 6.3 | 0.99903 / 0.860 | 0.99900 / 0.809 |
| text image | 1,584 | 202 | 0.06 s | `1fdd66049780` | 9.3 | 0.99837 / 0.880 | 0.99884 / 0.901 |

- **Repeatable across processes: yes.** Three separate processes gave the same SHA-256 for every image's rows, and
  so did a fourth with `enable_flash_sdp(False)` (the memory-efficient kernel is the one selected either way). Two
  calls in one process are equal too.
- **"<= 1 bf16 ulp of the reference" (VISION.md's line) does not hold, and cannot hold for bf16.** The engine and the
  reference differ by 6-9 ulps of the row max after 32 bf16 layers. Against an **fp32** reference, though, the engine
  is as accurate as DeepSeek's own bf16 run: the mean row cosine is 0.999 for both, and the engine's worst row is
  better on 2 of 4 images. The difference is kernel rounding, not a math error. The bar should be "as close to fp32 as
  the reference bf16 is".
- The first encode in a process takes ~0.75 s (kernel selection). After that, ~5.6 us a patch.

### The HTTP battery (fix copy, 3 servers on :8001; config/prod.env + the vision words)

Server `s1` ran with `TF_DSV41_IMAGES=native TF_DSV41_BIAS_VL=/cache/dsv41-bias-vl TF_DSV41_VISION_PREP_MB=64`. `s2`
was the same plus `TF_DSV41_PREFILL_CHUNK=128`, and `s3` the same without `TF_DSV41_BIAS_VL`. Each server had a fresh
state dir. Every request ran at T=0 with thinking off unless noted. Boot logs: `bias_vl: 40 routers`, `vision tower:
0.97 GB`, `images (TF_DSV41_IMAGES=native): tower 0.97 GB on rank 0; image routing bias loaded` (s3: `MISSING`).

| check | result |
| --- | --- |
| smoke (step 3) | carrots: "a stack of five fresh orange carrots with green stems against a white background"; corn: "three ears of fresh sweet corn, one partially husked"; screenshot: banner "Error 503: Service Unavailable - the orders API did not respond", button "Export CSV" (both verbatim) |
| release two-image example (`example_harmony.json` item 0) | 第一张图中是**胡萝卜**，第二张图中是**玉米** (carrot: the taproot; corn: the kernels). `prompt_tokens` 672 == `/tokenize` count 672; image tokens 633 = 444 + 189, the spans of our processor (bit for bit with the reference's in the CPU tests) |
| `prompt_tokens` vs the vLLM kit | **not compared: the kit's numbers for these images are not recorded** (kit README / logs only record pass/fail probes on `files/ds.png`). Running the kit takes the pair (only one stack fits), so this was left out |
| exactness (step 4) | cold == session hit (992 of 995 cached) == beside 3 decoding text streams == cold on the `PREFILL_CHUNK=128` server (spans cut by chunks); turn 2 (`$12,480`) equal on s1 (hit) and s2; the replay tail (3,664-token prompt, image inside the last 127 rows): "Carrot", equal on s1 and s2 |
| `bias_vl` A/B (step 5) | live: the screenshot description diverges at character 311 ("The title ... in bold white text" vs "The text ... in bold white letters"), and the carrots sentence differs. **Not measurably better**: our 20-item VQA set (words, colours, counts, order numbers) scores 20 / 20 with and without it, with the same answers. The set is too easy to show a quality gap |
| agent flow (step 6) | tools + a screenshot inside a `tool` message: 200, "Yes - the dashboard is showing an error ... Error 503 ..."; turn 2: "$12,480 ... Export CSV" (1,344 cached tokens); turn 3 adds a user photo: corn, correct (1,440 cached); thinking on: 200, same answer |
| concurrency (step 7) | 2 text streams alone 19.8 / 20.2 chunks/s; with 4 distinct image requests at once 13.6 / 13.7 (-32% while the 4 image prefills run); the 4 images 200 in 4.6-9.2 s; no capture errors, no tracebacks |
| limits (step 8) | 9 images -> 400; 27 MB PNG -> 400; `http://` -> 400; https to a private address -> 400; a public https URL (HF) -> 200 (fetch on by default); **a typed `<｜deepseek_image｜>` -> 200, not 400** (VISION.md expects 400; the model reads it as an ordinary token) |
| text unchanged | 4 HTTP probes (17*23, code, prose, thinking) byte-identical to prod 356a188 on s0 and s1 and on the new prod; speed suite below |

Minor finding: the `tensorfold.vision` stats block is `Encoder.last`, one shared dict. Concurrent image requests all
reported the last encode (`encode_s 1.0299` on all 4), not their own.

### Memory (the head, rank 0)

| | prod 356a188 (no tower) | native (s0-s3) |
| --- | ---: | ---: |
| `allocated` at serving start | 101.71 GiB | **102.65 GiB (+0.94)** |
| MemAvailable at serving start | 9.43 GiB | 8.0-8.7 GiB (**-0.75 to -1.4**) |
| KV pool | 4,692 pages | 4,692 pages (the tower does not shrink the pool) |
| MemAvailable minimum during the s1 battery | | head 5.60, worker 5.89 GiB (floor 5, hard 4) |

The worker went as low as the head in s1, so most of that minimum is KV / session load, not the tower. With the tower
the head has ~1 GiB less headroom over `TF_DSV41_FLOOR_GIB=5`. Before native ships, re-run the G12 299K stress with
native on, or lower `TF_DSV41_POOL_TOKENS` by ~1.1 GiB of pages (VISION.md section 6 step 1). The floor was not
breached in G15a.

### Text speed (7954c1d placeholder vs prod 356a188)

G14's speed suite (m2bench code / prose / structured at T0 and T0.7, reps 2; C1 / C2 / C4, mixed sampling): 3
interleaved boots each, prod knobs, fresh calibration each boot (`vision-speed.txt`).

| config | code T0 | code T.7 | prose T0 | prose T.7 | struct T0 | struct T.7 | C1 | C2 | C4 | exact | replies == ctl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ctl (356a188) | 86.49 | 78.48 | 46.06 | 48.42 | 121.84 | 123.04 | 87.69 | 74.07 | 99.98 | 3 / 3 | |
| 7954c1d | 85.93 | 78.44 | 46.01 | 48.32 | 122.08 | 123.15 | 87.22 | 74.11 | 99.94 | 3 / 3 | **3 / 3** |
| delta | -0.6% | -0.1% | -0.1% | -0.2% | +0.2% | +0.1% | -0.5% | +0.1% | -0.0% | | |

All deltas are inside the boot-to-boot spread (code T0 ranges 85.7-86.8 within each config).

## Part 2: the L2 prefetch pace below 150

`g14wintime` boots (G14's method: the boot's table, then 5 calibration passes with the slower rank's raw ms, replay ==
eager, the candidates' digest). There were 3 interleaved boots a config on prod's knobs (7954c1d, text only); off =
prod = pace 150 (`pace-window.txt`).

| config | 1 row | 2 rows | 4 rows | 16 rows | 1-row spread | exact (digest == off) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| off (pace 150) | 21.78 | 26.58 | 36.14 | 72.04 | 0.10 | yes (`095a3459a7`) |
| pace125 | 22.22 (**+0.44**) | 26.81 (+0.23) | 36.21 (+0.07) | 71.51 (-0.52) | 0.12 | yes |
| pace100 | 22.91 (**+1.13**) | 27.41 (+0.83) | 36.35 (+0.21) | 71.42 (-0.62) | 0.03 | yes |

- Below 150 the 1-row window gets slower monotonically: the prefetch no longer finishes inside its window. Only the
  16-row window gains, by 0.5-0.6 ms, and it is not the serving hot path. **No pace beats 150 at 1, 2 and 4 rows, so
  there was no speed run and 150 stays.** G14 (150 < 200 / 250) and G15a (150 < 125 < 100) together make 150 a
  minimum near 150, not an edge of the tested range.
- The calibration was tight: |boot table - pass median| at 1 row has a median of 0.025 ms and a maximum of 0.073 ms
  over 9 boots, with no outliers. Off reproduces G14's combo (21.71 / 26.42 / 35.87 / 71.72) within 0.1-0.3 ms.

## the worker

**No reset during G15a.** The boot ID `eb4ba2e5...` was the same before and after, and the uptime was 3 h 24 min at
00:46. A 30 s boot-ID watch ran from the workstation for the whole window and saw no change and no unreachable period.

Before the window (`worker-pre.txt`):

- **The 21:19:56 reset (G14) was not the first.** `journalctl --list-boots` and `last -x` show **two** unclean ends:
  - boot -2 ends at **2026-10-02 13:17:46** with no shutdown record, back at 13:30:30. Its last lines are earlyoom
    reporting `mem avail: 6402 of 122540 MiB (5.22%)` at 13:17:01 and network-daemon chatter, so it was under load;
  - boot -1 ends at **2026-10-03 21:19:56**, mid-session: ssh sessions from the head's memory guard every ~1 s, and
    a rank-1 container started at 21:19:25 (G14 `off-b2`, building its forward).
- The other ends on 10-02 (02:43, 06:27, 11:16) have clean `shutdown` records (planned reboots).
- Neither unclean end has a panic, an MCE (rasdaemon "Can't register mce handler" on this platform), an Xid / NVRM
  fault, a thermal line or a hung-task line. pstore is empty, and the hard watchdog is "permanently disabled". There
  were 8 x `NVRM: refcntRequestReference_IMPL: Failed to enter state 1` on 10-02, hours before the 13:17 end and not
  near it.
- Idle at 23:31: GPU 45 C, 10 W, SM 2236 MHz (`dgx-gpu-clock-cap.service`: `nvidia-smi -lgc 300,2250`); ACPI zones
  45-49 C. No `sensors` binary.

During the window (`therm-r{0,1}.log`: 5 s samples, 783 a node; the worker's streamed to the head so a reset would
keep them; `dmesg-r1-live.log`: the worker's kernel log streamed live):

| node | GPU temp max / mean | power max | hottest ACPI zone max |
| --- | --- | --- | --- |
| The head | 76 / 60.3 C | 50.1 W | 89.7 C |
| The worker | 76 / 59.5 C | 49.8 W | 90.3 C |

The two nodes run the same temperatures, so the worker is not hotter than its twin under this load. Its live kernel log
has nothing but docker / apparmor noise. Both unclean ends fell inside a boot with ~5-10 GiB MemAvailable (heavy
memory pressure, a fresh rank building or a stack loaded), with no kernel trace. That pattern fits a firmware / power
level reset more than an OS crash. Next time:

- keep the streamed `dmesg -w` and the samplers;
- consider enabling `ramoops` / `efi-pstore` and a netconsole to the head;
- check the BMC / SEL if the Spark exposes one.

## Production (verified 00:45-00:47)

`campaign.sh close` -> `prod-switch.sh restore` on the new `config/prod.env` -> DeepSeek prod verified (models,
17*23, canary, https); watchdog and rearm timers on, boot-start enabled, marker `dsv41`.

- `:8000` lists `DeepSeek-V4.1-Flash-TF`, `deepseek-v4.1-flash`, `GLM-5.3-Flash-EXL3`; https lists them too.
- 17*23 -> **391**.
- Both ranks mount `dsv41-prod/7954c1def8c821335f10f5a5463745b90ad71d70` and carry `TF_DSV41_IMAGES=placeholder`,
  `TF_DSV41_L2PF_PACE_GBPS=150`, BRANCHES 1 / side, PLAN_LINK rdma, PLAN_PIN auto, ROCE_FAST 1.
- Rank 0 shows `images placeholder (TF_DSV41_IMAGES)`, `plan link: rdma`, the plan thread pinned to cpu 19,
  `branches: streams (dedicated), priority side`, and `serving: 4 slot(s)`.
- **The image path (placeholder):**
  - a screenshot in a user message -> 200, `images_omitted` 1, and the log line `images: 1 image part(s) replaced by
    the placeholder notice`;
  - the same screenshot in a tool result -> 200, `images_omitted` 1.
- Text probes are byte-identical to 356a188 (4 / 4).
- `dsv41-tf-watchdog.timer` and `dsv41-watchdog-rearm.timer` are active; `dsv41-boot-start` is enabled.
- The lease is gone, and no deadman, refresher, memory sampler or G15a thermal sampler is left on either node. The one
  `sleep` on the head belongs to the kit's `mem-watch.sh`, which predates the window.

Caveat of placeholder mode: on the screenshot question the model says it cannot see the image, but then **guesses** a
plausible banner ("likely says something like 'Preview mode'"). Agents get no wrong-image hallucination, but a
user-facing answer may still speculate. Shipping native removes this.

**Rollback** (`config/prod.env` comments): `TF_COMMIT=356a188224bf2daec19bffb885eedd38f52b6926` (still staged on both
nodes), drop `TF_DSV41_IMAGES`, then `scripts/serve.sh stop && scripts/serve.sh start`. Image parts are then refused
with HTTP 400 again.

## Next

1. Commit the 4-D `_sdpa` fix with a CUDA test of `Tower.span`. Refuse a typed `<｜deepseek_image｜>` (or document that
   it passes). Make `stats.vision` per request.
2. A short window: the fix commit, the G15a battery (`G15a.sh vserve` s1 / s2 / s3, ~10 min), a 299K stress with native
   for the head floor, then `TF_DSV41_IMAGES=native` + `TF_DSV41_BIAS_VL=/cache/dsv41-bias-vl` +
   `TF_DSV41_VISION_PREP_MB=64` in prod.
3. A harder VQA set (dense screenshots, small text, charts) to price `bias_vl`; ours saturates at 20 / 20.
4. The pace is settled at 150 (100-250 tested).
