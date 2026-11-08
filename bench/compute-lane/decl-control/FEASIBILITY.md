# Declaration-control feasibility

Historical pre-A1 report. JJ subsequently authorized the bounded substitutes in `T5-DECL-CONTROL-SPEC-A1.md`, SHA256 `e7308621c645aef4402924a1b47c1a52e05baabe00239f915011f8431bbd0e89`. Its CPU-matched prompts/wire/count contract supersedes the served-array and unconditional input-ledger blockers below. No engine edits are needed for the amended configuration study; its remaining qualification gates still apply.

2026-11-05 Bangkok. **BLOCKED before collection.** Frozen specification SHA256: `adb05167fff21469ee7251555cb243f1f109c4d8e3de237fdcaa59ada8eb55e6`.

The pinned engine has a candidate execution-off mode with a matching declaration. It does not expose the served evidence required to qualify this protocol. No dataset request, activation request, GPU turn, ORDER mutation, engine edit or serving change was made for this feasibility investigation.

## Mode and prompt construction

Read-only engine: `<rank0-work>/imr-te/build-3a429d6f122d`, revision `3a429d6f122d9eb2900c43e622933d311e4be44f`; image `tensorfold-dsv41-imr:r-3a429d6f122d`.

`zig/src/server/compute_turns.zig:44` rejects `code_interpreter` when the server has no compute pool. However, an explicit ordinary function declaration passes through `tool_specs.active`. The same file's `isBuiltIn` recognizes Python by its exact name and built-in description, and `arms` returns false when the pool is absent. Thus both D and L can send the same explicit native Python function declaration, with `TF_DSV41_COMPUTE_LANE=0` for D and `=1` for L. P sends no tools with execution off. D does not use client cancellation to disable dispatch.

The declaration must preserve the native JSON object's insertion order. The old client's sorted-key canonical serialization cannot simply be reused: native template serialization includes schema key order. The CPU harness expands the original built-in declaration using unchanged native code, serializes/parses it, then renders/tokenizes that exact explicit declaration in both declared arms. It also compares against the original built-in lane prompt.

**CPU equality PASS:** all184 dataset task/seed pairs and both diagnostic pairs have identical full D/L arrays, also identical to the original built-in lane prompt. Full arrays are retained in `prompt-arrays.jsonl` (SHA256 `7887963d2c6b3f4de450792ba5b89655b3ab80f64e9cc1f4a5300d12ac7187db`); source/compiler/tokenizer hashes and step verdicts are in `cpu-proof.json` and `cpu-steps.jsonl`. The harness uses the native tools/arming, template and tokenizer implementations, with inert API/server/chat type stubs; it is **not** a full server or a dispatch-boundary negative control. No native token-ID arrays were printed to the terminal. Draft 552-request layouts have identical D/L wire bodies and are explicitly marked **NOT QUALIFIED**; they cannot be used as a collection seal. `verification.json` records the independent full-array/draft/syntax artifact checks. The new staging script passed turn-preflight with12literal paths, zero refusals and zero notes.

Expected source-level D behavior is an ordinary client-visible tool-call response if the model attempts a call: no compute pool, no held-EOS lane continuation, no external execution or fabricated result. This remains an unqualified candidate until the independent forced-call dispatch-boundary negative control and frozen error/termination semantics are implemented and CPU-verified. Arming false alone is insufficient under the spec.

## Hard blocker: actual served initial prompt IDs

The spec states: “CPU expansion may qualify the full manifest initially, but actual served prompt arrays must reconcile during collection.”

The pinned chat path cannot supply these arrays:

- `server/openai.zig:155` forwards `return_token_ids`, without a chat `return_prompt_token_ids` option.
- `server/chat.zig:200` retains initial prompt length and the returned-ID flag; it does not retain/export the initial array as response evidence.
- `server/generation.zig:363` exports the generated continuation IDs, not the serialized initial prompt IDs.
- `server/request_log.zig:1` logs request bodies, not expanded/tokenized prompts.
- `server/decisions.zig:18` has `return_prompt_token_ids` for a different typed-decision route. That route cannot replace the specified tool-bearing chat experiment.

Copies of these read-only sources are in `source-evidence/`. CPU-reconstructed arrays, hashes and matching prompt counts cannot substitute for actual served arrays. Consequently even a successful CPU equality check does not qualify this image for the frozen collection protocol.

## Additional accounting prerequisite

The spec also requires actual prompt-input accounting across inference rounds and CPU multi-round qualification. `server/compute_segments.zig:23` merges continuation IDs, inserted results and call reports; `finish` at line88 passes the original `p.prompt_len` to every segment. No per-round processed-input ledger is emitted. The engine resumes held generation with inserted suffixes, so one must not assume entire histories were replayed, multiply the repeated API prompt count, or invent zero additional input. Actual resume/reprocessing behavior needs explicit evidence and a qualified accounting contract.

## Required next change, subject to JJ's authorization

Expose actual serialized initial prompt IDs from the real chat preparation boundary, and an authenticated per-round input ledger that distinguishes initial, resumed/forced, cached and genuinely reprocessed input without double counting returned IDs. CPU-qualify multi-round accounting and the execution-off forced-call dispatch negative control. Review and pin a new per-SHA image identically for all three arms, then finish the prospective client/grader/statistics/mapping seal and fake-response/preflight gates before collecting any of the552 fresh replies. These remaining gates have **not** passed; no activation pilot or measured reply has been collected for this protocol.

No engine change is authorized yet. The existing source-level execution-off candidate does not itself require a new execution-policy flag; the missing served telemetry cannot be repaired solely in the external client. No approximate matched-control substitute has been collected. Historical results and graders remain unchanged.
