# Sanitized q28-v2 measurement export (earlier comparison)

These records are from the earlier 2026-10-09 comparison (single Zig runs, 2K prefill). The current comparison,
best of three for both engines against a fresh same-night Python run, is in [ZIG-RESULTS](../../docs/ZIG-RESULTS.md);
its raw records are not exported.

These files contain numeric measurement fields, repetition IDs and reply hashes only. Prompts, responses, paths,
server addresses, logs and infrastructure metadata were not exported.

| Public prefix | Recorded input | Repetitions |
| --- | --- | --- |
| `python-*` | Python reference suite | 3 for `cells`, 1 for `long` and `mix-*` |
| `zig-*` (excluding `zig-4k`) | earlier all-on suite | 1 |
| `zig-4k-long` | earlier optional 4K prompt suite | 1 |

Short cells select the largest `aggregate_tok_s` (multi-stream) or `mean_tok_s` (one stream) per workload,
temperature and stream count. Long and mix cells use their single record. Prompt cells use `prompt_tok_s`.
Reply parity compares `sha` arrays for the matching workloads. 
