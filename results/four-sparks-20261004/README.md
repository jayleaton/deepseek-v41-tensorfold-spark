# Four DGX Sparks (TP=4), 2026-10-04

One boot of `scripts/serve4.sh` with `config/prod.env.example`'s knobs (expert pruning on). Method and tables:
[docs/FOUR_SPARKS.md](../../docs/FOUR_SPARKS.md).

| file | what |
| --- | --- |
| `tf4-pruned-1.jsonl` | spark-bench depth sweep (prose / code at 1k, 20k, 40k, 80k, 160k; 3 trials, the first cold; 512 tokens), one row a request with its output and stream events |
| `summary-tf4-vs-ep2.json` | medians a cell beside the SGLang TP4/EP2 sweep of the same fixtures |
| `short-prompts.json` | four prompts sent as token ids to `/v1/completions` (the SGLang `/generate` twin), 256 tokens, 3 trials, with the engine's per-request stats |
| `c4-tf4-isolated-1.txt` | `dsbench` single-stream median and 4-stream aggregate |
| `campaign-gates.jsonl` | arithmetic, forced tool call, tool continuation, strict JSON at three temperatures, reasoning |
| `prefill-profile-r0.txt` | `TF_DSV41_PREFILL_PROFILE`: one 2,048-row prompt segment on rank 0 (by CUDA time, then by CPU time) |
