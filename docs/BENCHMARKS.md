# HTTP benchmark clients

Historical engine benchmarks and their measurements remain in release 60b6688. Start the native server using
[ZIG-SERVE](ZIG-SERVE.md) before running these clients.

## 2. HTTP clients: `bench/` (standard library only)

| client | what | gate used here |
| --- | --- | --- |
| `bench/quality.py mmlu --data bench/data/mmlu200.jsonl` | MMLU-200, 0-shot, thinking off, greedy, max_tokens 8, first A-D letter | within 1 point of the kit (87.5%) |
| `bench/quality.py mmlu ... --shots 20` | the same after a 20-question preamble: ~2.1K-token prompts, so CED replay really runs | replay vs full within 1 point |
| `bench/quality.py needle --sizes 32768,131072,299000` | a passkey at 1/3 depth of a filler document | found |
| `bench/quality.py compare --replay R.json --full F.json --kit 0.875` | the two runs side by side | |
| `bench/structured.py` | 6 JSON schemas x thinking off / on, each drafted, serial (`"draft": false`) and 4 at a time; tool_choice required / named / strict / none / streamed | drafted == serial == batched, valid, no markup |
| `bench/tooleval/chains.py --mode off\|low\|high` | 6 multi-step tool scenarios with simulated tools, history sent back as agent clients do; 12 points | >= 10 |
| `bench/tooleval/run.sh setup`, then `teb\|sb\|both\|chains\|own off\|low\|high` | tool-eval-bench (69 scenarios; category C = multi-step) and spark-bench TrueScore at pinned commits | |
| `bench/soak.py --minutes 30` | 1-4 streams of mixed kinds (code, prose, structured, tools, thinking, 8K-48K documents), ~15% streamed cancels, ~5% client disconnects; `/health` every 10 s | 0 errors, inflight back to 0, 17*23 = 391 |
| `bench/soak.py --minutes 60 --kinds g16` | the same plus growing agent sessions (40 tools, to 120K tokens, screenshots), image requests and idle gaps: the G16-G19 serving-memory soaks. In-stream error events (a refusal after the 200) count as errors | 0 errors; sample RssAnon and MemAvailable on both nodes (G19: RssAnon slope < 0) |
| `bench/stress.py stress --long 299000 --dctx 65536 --decode 2048` | one 299K prefill while three 64K prompts decode 2,048 tokens (ignore_eos) | every stream complete; watch MemAvailable on both nodes |
| `bench/stress.py admit` | seconds to the first token of one request after a start (admission) | |

Common flags: `--base http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-TF --out R.json`. Sample MemAvailable on
both nodes while they run (`awk '/MemAvailable/' /proc/meminfo` every second; avoid polling `nvidia-smi` every second,
see the README).

## 3. The kit baseline

The kit's cells come from the GLM recipe's HTTP clients against the kit's server: `bench/glmbench.py` (suites tf,
tweet, kit, edit), `bench/multiturn.py concurrent --streams 1,2,4`, `bench/quality.py` (MMLU-200) and a long-prefill
client (an exact token-id prompt of a repeated filler document, 8K-256K). Raw files: `results/kit-baseline/`.

## Speed suites and preflight (#61)

`python3 bench/speed.py --help` exposes prefill, decode, multi, needle, interleave, exact, reuse and Anthropic suites,
plus saved-run comparison. Multi reports a common wall-window aggregate and first-token spread. Server-derived
error text is credential-masked before stdout and JSON output. Unknown native health fields remain unmeasured.

`python3 scripts/preflight.py --config config/prod-zig.env --rank 0` checks local Engram/memory inputs.
Set `PREFLIGHT=1` for the optional launcher hook: warnings continue, failures stop the start. The helper does not
certify the checkpoint or justify a pool increase; native HTTP/SSE qualification remains required.

## Release candidate G

These numbers come from the paired bench boots behind the pin to G (`58bece1d3`). They used the flash17 K6 MTP variant
([FLASH-1017](FLASH-1017.md#the-k6-mtp-variant)), TP2 on two Sparks, greedy decoding, and the nine settings in
`config/prod-zig-flash17.env.example`. Each boot ran two cells: coding (200 output tokens) and set-b code (384). Each
row is one boot and one run per cell. These are speed ratios between engines on the same pack and settings, not
quality results.

**G vs F** (one window, alternating F, G, G, F):

| Boot | coding tok/s | set-b tok/s |
| --- | ---: | ---: |
| F | 105.17 | 90.28 |
| G | 105.75 | 90.39 |
| G | 107.25 | 89.49 |
| F | 105.31 | 89.96 |

The G/F pairs are 1.006 and 1.018 on coding, and 1.001 and 0.995 on set-b. Their geometric mean is **1.005**. All
rows produced the same tokens, including a fifth G boot with the compute lane on.

**F vs H** (one window): coding **1.051** (F 107.3 and 106.3 tok/s, H 102.4 and 100.7), set-b **1.000**, same tokens.
Turning `TF_DSV41_KV_MIRROR=4096` off and on in the same window (the P1 spot) measured **+3.3 %** on coding (45.69 to
44.30 ms a round) and **+3.9 %** on set-b.

**Compute lane on and off** (2026-11-01, one window on G, ABBA; the only change is `TF_DSV41_COMPUTE_LANE`; median of
three runs per cell, no tool calls in these prompts):

| Boot | Lane | coding tok/s | set-b tok/s |
| --- | --- | ---: | ---: |
| L01 | off | 107.20 | 90.47 |
| L02 | on | 104.27 | 89.44 |
| L03 | on | 103.62 | 90.35 |
| L04 | off | 104.89 | 89.44 |

On/off is 0.973 and 0.988 on coding, and 0.989 and 1.010 on set-b. The geometric means are **0.980** and **0.999**:
plain coding costs about 2 % with the lane on. The two lane-off boots differ by 2.2 %, so expect about ±1.5 % on these
ratios. Greedy 2K and 8K outputs (draft off and on) were identical in all five boots of the window, the four above
plus L05 (lane on), which ran MBPP.

**Compute lane on MBPP** (2026-11-01, L05, lane on). 40 MBPP-sanitized tasks (ids 11-92), greedy, at most 8,192
tokens, client off-node, each cell in its own quiet window. Arms: (a) no tools, (b) the client runs the same
`python` tool itself (one HTTP round trip per call), (c) the lane.

| Arm | Streams | Solved | Wall s / solved | Completion tokens / task | Requests / task |
| --- | ---: | ---: | ---: | ---: | ---: |
| (a) no tools | 1 | 38 / 40 | 9.94 | 636 | 1.00 |
| (b) client round trip | 1 | 38 / 40 | 11.24 | 715 | 1.53 |
| (c) lane | 1 | 38 / 40 | **9.67** | 645 | 1.00 |
| (a) no tools | 4 | 38 / 40 | 6.35 | 636 | 1.00 |
| (b) client round trip | 4 | 38 / 40 | 8.01 | 715 | 1.53 |
| (c) lane | 4 | 38 / 40 | **6.70** | 645 | 1.00 |

Against the client round trip, the lane cut wall time per solved task by **14 %** at one stream and **16 %** at four,
with 10 % fewer completion tokens, one request per task instead of 1.53 and about a fifth of the request bytes. Against
no tools it was 2.7 % faster at one stream and 5.5 % slower at four, where only two sandboxes run at a time. All arms
solved 38 of 40. These numbers replace the earlier MBPP run, which used the lane commit over an earlier candidate
and was not in a quiet window.

## Credential-safe output

Host launchers and recipe clients sanitize stdout, stderr, snapshots and report
strings before output or persistence. Credential-named native environment variables
are excluded from Docker forwarding; arguments containing a known credential are
refused. Configuration-source diagnostics are withheld, including keys loaded from
files during shell evaluation. Keep operator configuration and credential inputs
private; diagnostic filters do not make the supplied credentials public data.

`bench/tooleval/run.sh own` uses the recipe's sanitized client. The pinned external
`teb`, `sb`, `both` and `chains` evaluator modes now exit 78 before contacting the
server or invoking a writer: their direct report/raw-dump writes lack an audited
pre-write adapter. A post-run scrub is insufficient. Setup remains available;
restore those modes only with an adapter and tests that observe the first write.
Historical third-party results remain historical and unchanged.
