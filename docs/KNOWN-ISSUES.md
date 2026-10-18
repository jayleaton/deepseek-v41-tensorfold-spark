# What is not solved

- **The mixed C4 cell is prose-bound, and 120 is the wrong yardstick for it.** Measured over HTTP on the flash17
  pair at the recommended profile (pool 2.1M, 2184 MHz; `tools/tensorfold-dsv41/dsv41_bench/c4bench.py`, medians of
  6 reps, ranges in brackets): the mixed cell (code, prose, tweet, edit; 384-token caps; every other stream at
  T 0.7) decodes 98.0 tok/s [96.7-102.1], with 47% of the span all four streams live and 32% the prose stream
  running alone; steady 139.4 [137.9-141.2]. The prose stream ends last in every rep: 246 rounds at 1.57 tokens a
  round to its 384-token cap, while the other three streams end by 10.2 s. The 120 target is defined for short code
  at four streams (docs/campaign/TARGETS.md) and is met on that definition: four distinct code streams decode 137.9
  [136.9-140.2], inside the old 125-135 estimate. Four distinct prose streams decode 87.6 [86.7-90.2], so at 1.57
  tokens a round no prose-heavy cell reaches 120. The lever on the mixed cell is prose acceptance, the drafting item
  below: 1.57 to ~1.92 accepted tokens a round shortens the span to 12.3 s and reads 120 (arithmetic from the
  measured chain, round times held). Four distinct code streams run 17% under the same-prompt cell
  (166.8 [165.7-184.3]): four streams touch more distinct experts a round. `TF_DSV41_SPEC_NUCLEUS` stays off
  (G17: C2 steady +1.3%, C4 steady -0.6%).
- **Prose drafting.** DSpark keeps ~1.7 tokens a round on prose (3.7 on code), so prose (1.44x) and 2 streams (1.59x)
  are not at 2x. Self-distilling the drafter on our own logs helped prose and hurt code; not shipped.
- **The graph cache still shrinks under repeated memory dips.** Long prefills at 2,048 rows dip the worker's memory;
  G19 keeps the graphs through dips inside 64 decode-only rounds of a prefill, but in the 1 h soak five longer dips
  still took the LRU cap 48 -> 11 (G18: 48 -> 8). Fewer cached graphs cost speed until the next restart.
- **Capacity refusals of streaming requests arrive as in-stream errors.** When the priced admission floor refuses a
  streaming request after its `200` headers went out (it waited up to 30 s for memory first), the client gets an SSE
  `{"error": ...}` event, not an HTTP 503. Sending the headers lazily would fix it. G19's pricing at the smallest
  adaptive step removed the case the soak hit (0 refusals since).
- **The adaptive-row threshold.** Production runs `TF_DSV41_PREFILL_ADAPT_GIB=4.5`; the 1 h soak ran 4.0 (worker
  minimum 3.99 GiB, 9 MiB under our own 4.0 line, no errors). 4.5 has more margin but was not soaked, and not measured
  at 64K / 128K. The engine default (5) makes rows flap on GB10 (replay 32K 1,912 tok/s).
- **The worker is the binding node.** Its MemTotal is 2 GiB below the head's (firmware), and its MemAvailable at
  serving varies 6.2-8.4 GiB between identical boots, outside torch. earlyoom on the nodes (`-m 2,1` here) is the last
  backstop under our 4-5 GiB floors: it kills a rank below ~2.4 GiB, and fail-fast then takes the pair down in ~1 s
  for the watchdog to restart.
- **Two G13 rewrites ship off.** The shortened decode MoE chain (`TF_DSV41_MOE_FUSED`) is exact but measured +0.7 ms
  on a 1-row window in the graph. (`TF_DSV41_BRANCHES`, off in G13, is on since G14 on dedicated priority streams.)
- **Strict mode on the current engine** (every precision trade off, `full` prefill with the cone) has not been
  measured; the strict numbers above are G13's.
- **`/health`** reports `drafted_total` / `accepted_total` as 0 for this family (the counters are not wired).
- **No trimmed draft-head vocabulary is shipped** (`TF_DSV41_DRAFT_HEAD=trim`, off by default and not adopted). The
  development ranking was counted from private chat transcripts and is excluded, so `trim` needs
  `TF_DSV41_DRAFT_VOCAB=<file>` and `tests/test_dsv41_draft_head.py::test_shipped_ranking` fails. A ranking of your
  own traffic (`scripts/campaign/draftvocab.py`) or of public text (the GLM recipe's `bench/draftvocab_public.py`
  method on this tokenizer) can be used.
