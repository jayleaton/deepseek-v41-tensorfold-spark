# Raw results

The files behind [`../README.md`](../README.md) and [`../docs/RESULTS.md`](../docs/RESULTS.md), copied from the
development log with machine names and local paths replaced. Engine logs, nsys traces, drafting logs and training
data are not included.

| directory | what | produced by |
| --- | --- | --- |
| [`zig-q28-v2/`](zig-q28-v2/README.md) | q28-v2 Python reference, single-run Zig all-on and optional 4K prompts; sanitized numeric fields and reply hashes (2026-10-09) | recorded HTTP benchmark suites |
| `kit-baseline/` | MiaAI-Lab's vLLM kit on the same pair and weights (2026-10-01): `summary.txt` (DSpark acceptance per phase, memory per phase), `glmbench.json` (single-stream cells), `multiturn.json` (C1 / C2 / C4), `long-*.json` (cold prefill 8K-256K), `quality.json` (MMLU-200), `oracle-kit.json` (the kit's top-5 prompt logprobs for 8 prompts x 2,048 tokens: the top-1 gate's reference) | the GLM recipe's HTTP clients |
| `best-config/` | the final configuration: `m2-ab-best.json`, `m2-g10s-l2pf-mb12.json`, `m2-ab-dk64.json` (decode speed, three runs), `g5-gate-best.json` (top-1 vs the kit), `mmlu0-best.json` (MMLU-200), `chains-best.json`, `best-knobs.txt` (every knob) | `m2bench`, `gate`, `bench/` |
| `best-config/ship/` | the ship gates on a test server started from the production config: `summary-ship.txt`, `structured.json`, `chains-off.json` / `chains-high.json`, `soak.json`, `stress.json` (the 4 x 300K stress whose worker memory failed the 5 GiB target) | `bench/` |
| `best-config/tool-eval-bench-C/` | tool-eval-bench category C, thinking off: `teb.json`, the command line | `bench/tooleval/run.sh` |
| `prefill/` | cold prefill, one slot: `m2-pf-g7x-s-all4.json` (production prefill, 8K-128K), `m2-pf-g7-full4.json` (no replay, every lever), `m2-pf-g7-basefull.json` (no replay, base kernels) | `m2bench --prefill` |
| `not-adopted/verify-budget/` | the lossy verify budget: speed (`m2-g10b-*.json`), greedy replies vs exact (`greedy-*.json`), generation-based MMLU (`mmlugen-*.json`), the pick (`budget-pick.txt`) | `m2bench`, test servers |
| `not-adopted/pdl-l2pf/` | PDL and L2-prefetch A/B (`g10-speed.md`) and the L2 probe on GB10 (`l2probe.json`) | `m2bench`, `l2probe` |
| `campaign/G14-*` ... `campaign/G19-*` | the summary files of the G14-G19 windows: speed tables (`g15-depth-speed.txt`, `speed-table.txt`, `m2-*.json`), prefill tables (`pf-table.txt`, `m2-g16pf-*.json`), soak reports (`summary-soak.txt`, `soak-*.json`), stress reports, fail-fast checks (`ff-*`), gate reports, the pfdense sweep (`G17-20261004/sweep.txt`, `pfdense-table.json`). Engine and driver logs and memory samplers are not included | `m2bench`, `gate`, `bench/` |
| `not-adopted/drafter/` | draft acceptance analysis of 34,753 DSpark passes (`draftsim.json`); drafter self-distillation: offline evaluation (`trainA.json`, `trainB.json`) and engine runs without / with the deltas (`m2-g11-off.json`, `m2-g11-A.json`, `m2-g11-B.json`) | `draftsim`, `dsparktrain`, `m2bench` |

In `m2bench` reports: `workloads.<name>.t0` / `.t07` are the single-stream runs (tok/s, tokens a round, the reply's
token ids), `concurrent.C<n>.decode_aggregate_tok_s` the multi-stream cells, `exact_all` the drafted == serial check,
`knobs` the configuration, `m2` the engine's state (graphs, Engram prefetch hits, the calibrated window costs).
