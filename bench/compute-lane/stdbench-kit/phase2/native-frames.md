# Native frame authentication

`native_frames.authenticate(record, decoder)` returns `qualified`, `reason`,
`calls`, `delivered_successes`, `native_frame_count`, `inserted_tokens_bound`,
and authenticated frame spans. This module is offline and does not execute
tokenizers, code, endpoints, or tools. It changes no client/accounting files.

Standards: standalone not-git artifact with no applicable ancestry
AGENTS/assistant/CONTRIBUTING/build/lint configuration, as recorded in
`../LEDGER.md`. Diagnose/principle-prove-it-works applied by source inspection
and explicit regression fixtures; no Python execution or tests on <dev-host>.
Parent must run scoped <rank1-host> py_compile, selftest and actual native-record
fixtures before qualification/final freeze. No claim of runtime verification
is made by authoring this module.

```python
from native_frames import authenticate, selftest
proof = authenticate(record, decoder)
# Parent's authorized CPU scope only:
receipt = selftest("/absolute/new-native-frame-receipt.json")
```

Record schema follows phase2/client.py: tensorfold.token_ids/token_sha256,
usage.completion_tokens, native_accounting.complete. Decoder exposes
pieces[id] -> bytes and decode(ids) -> bytes. Native accounting must already
be complete under the parent's frozen cap/framing/usage policy. This module
also rechecks the full comma-separated-ID SHA256, decoder consistency and
completion-plus-inserted reconciliation; it does not replace the parent's
aggregate-cap, SSE completion or error validation.

EOS/USER/ASSISTANT marker IDs are discovered by reverse matching exact byte
pieces, without hardcoded vocabulary IDs. Every EOS-ID immediately followed
by USER-ID outside an already authenticated inserted region is a candidate
frame and must authenticate. Native source comes from the preceding generated
assistant segment, after its last </think>, with one final frozen DSML calls
block. Only Python invokes with one exact code parameter are supported.
string=true binds its literal UTF-8 payload unchanged. string=false must hold
a JSON string literal; the decoded string must equal the reported code.
No escape guessing, substring search, whitespace normalization or source repair
is performed. Unsupported/ambiguous DSML fails closed. Ordinary prose before
the final call block is allowed, but not an unclosed thought/another DSML block.

For each frame, the number/order of Python invokes selects the next inserted
reports in compute.calls. The complete expected USER region is built from
those report outputs as:

```text
<tool_result>OUT0</tool_result>\n\n<tool_result>OUT1</tool_result>
```

A real ASSISTANT marker ID must begin exactly after that **whole** region,
followed by exact <think> bytes ending at a token boundary (one or several
IDs). Outputs may contain result-closing tags, role text, special role IDs,
or even EOS+USER IDs: these are payload bytes inside the authenticated region,
not boundaries selected by a non-greedy tag parser. A reported323 cannot match
1323 or text in a later assistant answer. Embedded frames are ignored only
after their enclosing whole region is authenticated.

Each inserted span counts USER through the end of the inserted <think>,
including result delimiters, multi-call separators and ASSISTANT, excluding
the preceding generated EOS. Sum of spans must equal compute.inserted_tokens;
compute.prefix_tokens must equal the final span's exclusive end token index.
All inserted reports must be consumed exactly once; any missing or foreign
frame fails. Noninserted reports remain delivered=false/native_bound=false
and do not count as delivered_successes, even when their status is ok.

If compute exists, calls (list), inserted_tokens and prefix_tokens must all be
explicit; missing fields never become zero. Absent compute is accepted only
with completion-only reconciliation and zero EOS/USER frames. Explicit empty
compute requires both counters zero. Runner/S3/trace/guide runtime activity,
optional compute execution counters, external tool calls and a non-code-tool
executor fail. Missing executor means code tool only under the frozen server's
S3-off contract. The parent still verifies effective environments and frozen
source/image identity; native bytes cannot establish those independently.

Per-call proofs contain id, delivered/status, code-tool executor, native_bound,
source/output SHA256, and byte source_at/source_end/output_at/output_end in
the complete decoded continuation. For JSON-string code, the source span is
the exact encoded JSON parameter and source_encoding=json_string; its hash
is of the decoded code, so raw span hash equality applies only to literal
source. Output spans always hash directly to output_sha256. Frames also expose
half-open token spans and report indices. Failure may retain partial proofs;
only qualified=true authorizes their use as complete authentication evidence.

Authentication proves exact attribution, not Python correctness. Parent must
independently reference-run successful delivered code, verify native output
cap/token accounting and compare normalized result bytes. Native execution
errors remain separately reported failures, never fabricated successful values.

Selftest uses fabricated byte/marker IDs: multiple frames/calls, JSON source,
Unicode, one/multiple-ID think openers, substring1323/later-assistant323 forgery,
wrong/reordered source, missing counters, hidden frames, zero plain/compute,
exact newline mismatch, embedded role/result tags, foreign executor, positive
runner cells and noninserted calls. It writes an optional exclusive-create
receipt. It never loads an actual tokenizer or invokes another process.
