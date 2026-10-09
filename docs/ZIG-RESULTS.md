# q28-v2: CUDA Zig vs Python

Measured on two DGX Sparks (GB10, 128 GB each), TP=2, RoCE, q28-v2 EXL3 2.8 bpw, 2026-10-09.
The recommended [all-on profile](../config/prod-zig.env.example) uses 2K prefill; PF_4K is optional.

**These Zig numbers are single-run measurements.** The recorded Python short-decode references are the best of
three repetitions, not one run each. Long-context and mixed-load cells have one measurement per engine.
The suite starts with one short code warm-up (T0, one stream, 32 tokens); it does not independently warm every cell.
There are no confidence intervals, and small differences should not be treated as repeatable gains.

Decode tok/s unless marked prompt. Four-stream figures are aggregate throughput. T0 is greedy; T0.7 is sampled.
All paired reply hashes match, including sampled, long-context, and mixed-load replies; the separate reply check
also passed 16/16. This is parity with this Python reference, rather than a claim about model quality generally.

| Cell | Python reference | Zig (all-on) |
| --- | ---: | ---: |
| code ×1 | 86.6 | 87.6 |
| code ×4 | 145.4 | 147.4 |
| code T0.7 ×1 | 92.2 | 90.0 |
| code T0.7 ×4 | 147.0 | 144.8 |
| prose ×1 | 54.2 | 54.4 |
| prose ×4 | 99.0 | 98.3 |
| prose T0.7 ×1 | 50.4 | 49.2 |
| prose T0.7 ×4 | 98.2 | 99.0 |
| mix: 4 streams | 140.0 | 141.8 |
| mix: 131K decode | 69.0 | 67.2 |
| 32K decode | 51.1 | 61.1 |
| 131K decode | 53.4 | 64.1 |
| 32K prompt (tok/s) | 2,436 | 2,507 (2,663 with PF_4K) |
| 131K prompt (tok/s) | 2,266 | 2,318 (2,486 with PF_4K) |

Zig is ahead on 32K/131K decode by 19.6%/20.0%, and optional 4K prompts are ahead by 9.3%/9.7%.
Short decode is roughly at parity. Sampled T0.7 cells are slightly behind overall, with prose ×4 ahead;
mixed long decode is 2.7% behind. Zig does not beat Python everywhere.

The 32K/131K labels identify the workload sizes before prompt encoding. Actual prompt lengths in the files are
37,974 and 154,183 tokens. Prompt throughput is prompt tokens divided by time to first token.

## Evidence

[Sanitized measurement files](../results/zig-q28-v2/README.md) preserve numeric fields, repetition IDs, and reply
hashes from the recorded Python reference, the c12 all-on run, and c8's optional 4K prompt run. The optional figures
come from a separate run, not an extra mode measured in c12. All displayed numbers match those files after rounding.
The underlying files do not include a complete served environment; attribution of c8 to PF_4K follows the campaign
record. They are not enough to reconstruct every build input or to claim a new local hardware validation.

## Historical vLLM comparison

The [existing Python vs vLLM table](../README.md#historical-29-bpw-results-python-vs-vllm) remains unchanged:
code greedy ×1 41.9–45.0, prose greedy ×1 32.5, aggregate ×1/×2/×4 32.2/46.7/37.6, and cold prefill
8K/32K/64K/128K 1,073/1,075/1,060/1,031 tok/s for vLLM. Those runs use the older 2.9 bpw pack and different
workloads. There is no matched q28-v2 Zig vs vLLM run here, so no new ratio is claimed.
