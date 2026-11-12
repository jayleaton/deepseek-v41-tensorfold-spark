# Undeclared tool calls: the outside report behind the 3ca06dd4b pin (2026-11-12)

The report is an outside OpenCode benchmark of this recipe, run by evanwtf/local-llm on their own two Sparks. It is not a
measurement of ours.

- Report: [evanwtf/local-llm#1329, comment 6814669413](https://github.com/evanwtf/local-llm/issues/1329#issuecomment-6814669413)
- Rows: [evanwtf/local-llm#1399](https://github.com/evanwtf/local-llm/pull/1399), merged as `5057de3d70eb90cc5e88711cd92e9db85b1e4197`, file
  [`hardware/Cortex-X925-128GB-GB10-x2/results.jsonl`](https://github.com/evanwtf/local-llm/blob/5057de3d70eb90cc5e88711cd92e9db85b1e4197/hardware/Cortex-X925-128GB-GB10-x2/results.jsonl)
  (sha256 `7888852aacd8df57fcb1db5ca7f0f6cf7f0653bc5aba279b545778fed1df3717`), backend `dsv41f1017tfzigjay176024bdual2xrc`
- Versions, as the report gives them: recipe `176024b`, TensorFold `1ea71240a` (image `tensorfold-dsv41:jay-176024b`),
  pack `9e7c02ec06`, OpenCode 2.2.7, client harness `1bb6e0b`
- Hardware: two NVIDIA GB10 (DGX Spark), GPU clock capped at 1,900 MHz
- Server: `tensorfold-dsv41 serve /model --parallel 4 --context 1048576 --max-tokens 32768 --reasoning-effort low`
- Request set: their standard set, 10 tasks x 3 trials, 2026-11-12 04:01-04:29 UTC

[`trials.tsv`](trials.tsv) holds the 30 rows from that file: start time, task, trial, pass or fail, model steps, wall
seconds, exclusion (its label only) and pytest summary. Their client log paths are left out. 26 of 30 passed. The four failures are
`parser-mbox-quoting-nodoc` trial 1, `mbox-scan` trials 2 and 3, and `mbox-quoting-both-halves` trial 2. Each ended
after 0 or 1 model steps, with 5.5 to 7.4 s of wall time.

## What each source shows

- **The rows** show the four failures and how short they were.
- **The report's transcript reading** gives the cause. Each failing reply is a well-formed DSML call to `bash` while
  OpenCode 2 declares `shell`, and the server returned it as content with `finish_reason: "stop"`. The rows do not
  carry the replies, so the cause rests on that reading.
- **Our reproduction on the Sparks** is one reply, not a frequency. A request that declares only `shell`, after a
  history whose calls used `bash`, got the same DSML `bash` call back as content with `finish_reason: "stop"`. Those
  bytes are the engine's host-test fixture (`zig/src/server/stream_dsml_test.zig`).
- **vLLM's behaviour** comes from the report: vLLM returns such a call, OpenCode answers that the tool is unknown, and
  the model retries. We did not measure that recovery, and the benchmark has not been re-run at the new pin.
