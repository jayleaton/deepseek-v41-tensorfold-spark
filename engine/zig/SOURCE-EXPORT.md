# Source export provenance

This is a sanitized source snapshot based on TensorFold 1.0.2, upstream revision
`f8fe17d24629aedabf90bbf78279dd776e6d62e7`, from
[ashhart/TensorFold](https://github.com/ashhart/TensorFold). The DeepSeek-V4.1 source snapshot revision is
`8e96f99cc27743bb80f4827ea78b30481e3b4bcc`, the engine source of the current serving kit: the previous snapshot's
engine plus the serving fixes `71fbcd316f840f353bd0898edf0cf8424ec52a02`, `3d88039d9105eabf12677d3c82e00a5f6ec6e366`,
`4bb2c952c7ab5e4038a2c58041c4d457ef6bf9b8` with its test `5dbd1a0bbfaa8ab5045f60e9701e0b3d222dd38b`,
`db7a5559dad5def07291ed2229bd1f75117f329d` and `8e96f99cc27743bb80f4827ea78b30481e3b4bcc`. The previous snapshot was
`55c96c65bf97b8c30455f171e01aec8c59544aeb`; the one before it `0d723a8275f5b7879a083e8042a7048242ee624c`.

## Changes since 55c96c65

- **Serving:** SIGTERM / SIGINT drains: new requests get 503 with `Retry-After` and `/health` reports `draining`,
  requests in progress get `TF_DSV41_DRAIN_S` seconds (default 20), then both ranks halt at a round boundary; a second
  signal cuts the drain short. A stream that has not sent its first token gets a keepalive every
  `TF_DSV41_SSE_KEEPALIVE_S` seconds (default 15): `: keepalive` comments on OpenAI streams, `ping` events on
  Anthropic streams. The API key store and per-key reply counts are freed when the server stops.
- **CUDA:** a failed kernel launch names its kernel and launch configuration; the server logs the signal that stopped it.
- **Engram:** the row cache's record map is rehashed in place once removal markers reach a quarter of its capacity.
  Before, the markers were never cleaned up, so after a long uncached prefill every lookup scanned the whole map.
- **TP runtime:** `TF_DSV41_KEEP_BATCH`, `TF_DSV41_PT_PINNED` and `TF_DSV41_PIN_ISOLATE` are removed (default off,
  measured no gain); `TF_DSV41_WIN_PROF` times the Engram gate's arm in parts.
- **Bench tools:** per-stream content-chunk arrival times (`bench_http.py --times`), a side-by-side comparison of two
  such cells (`timescmp.py`), and a per-round kernel comparison of two profiler exports (`kcmp.py`).
- New host tests for the drain (including a stop between a request's count and its drain check), the keepalive and
  the rehash.

## Changes since 0d723a82

Every new engine path below is default-off; `config/prod-zig.env.example` opts into the ones the measured serving
profile uses.

- **Prefill:** the 4K workspace shares the index-stream scratch (`TF_DSV41_PF_WS_SHARE`); piece all-gathers write into
  the layer rows, with optional side-stream exchange pieces (`TF_DSV41_PF_OVERLAP_SITE`); row-blocked and prefetching
  stream top-k twins (`TF_DSV41_STREAM_RB`); x3gm gate/up at 4K segments (`TF_DSV41_GM_GU2`) and bounded prefill mHC
  sites (`TF_DSV41_MHC_SITE_ROWS`); CED replay keeps its stash ring in place across piece and replay runs' slot switches.
- **Decode and drafting:** one-pass dense linears for 17-64-row windows (`TF_DSV41_LIN2`); row-bounded index scores and
  decode top-k (`TF_DSV41_INDEX_BOUND`); fused MoE expert epilogues (`TF_DSV41_X3LD_EPI`); mHC coefficient launches
  deferred to just before their exchange (`TF_DSV41_MHC_DEFER_AT=exchange`, `TF_DSV41_COEF_LATE`); draft graph capture
  with the round (`TF_DSV41_DRAFT_CAPTURE`); graphs kept under the memory floor (`TF_DSV41_GRAPH_FLOOR=hold`); the
  engine's own boot calibration of draft costs (`TF_DSV41_CALIB=zig` / `measure`, `TF_DSV41_CALIB_FILE`).
- **TP runtime:** host step profiling (`TF_DSV41_WIN_PROF`). (This snapshot also added `TF_DSV41_KEEP_BATCH`,
  `TF_DSV41_PT_PINNED` and `TF_DSV41_PIN_ISOLATE`; they are removed since 55c96c65.)
- **Serving:** tool-call keys, invoke names and argument values, and queued Anthropic text, are copied instead of
  borrowed from a growing buffer; the JSON writer is bounds-safe on a UTF-8 sequence cut off at a string's end; nested
  completion prompt ids are range-checked; calls to tools the request did not offer are surfaced as `tool_calls`
  (`Tools.unknown_calls`, default true); a reply with DSML markup but no call is logged once.
- **Bench tools:** a mixed-window cell (`bench_http.py --mixwin`) whose long-stream rate counts tokens
  (`long_live_tok_s`), not decode rounds.
- New host tests for each of the above, GPU micro-benches for the new kernels, and their AOT fill inputs.

The source derives from the public upstream tree and is added through logical source commits. Development
history is not imported. Source changes include removal of infrastructure references, conversion of
machine-specific benchmark configuration to environment inputs, and default-off controls for the new
family's drafting, native prefill, lookup, session retention, and Engram prefetch.

Apache-2.0 license headers, attribution, `LICENSE`, `NOTICE`, third-party notices, and applicable
third-party license texts are retained. Weight and dataset licenses remain separate.

Excluded material includes performance diaries and infrastructure notes; private benchmark results
and logs; profiler captures; databases; packaged kits; caches; scratch files; generated build outputs;
and large generated binary goldens. Small checked-in source fixtures remain; no huge binary fixtures are included. The source tools generate
checkpoint-dependent fixtures and AOT assets locally; the public serving recipe documents asset
generation. No checkpoint weights or generated AOT bundle are shipped in this source export.

The export is intended to make the source inspectable and buildable. Historical device receipts are
identified separately from checks on the exported tree. Host compilation does not qualify hardware,
models, arithmetic, or precision choices. See [the CUDA guide](docs/DEEPSEEK-V41-CUDA.md) and the pull
request for build/test outcomes, unrun gates, and the outstanding upstream lean-rule cleanup.
