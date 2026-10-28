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
