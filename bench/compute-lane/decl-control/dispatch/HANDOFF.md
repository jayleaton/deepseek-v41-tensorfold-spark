Standalone dispatch probe; not executed or compiled by the author.

Read standards: artifact LEDGER.md (no repository, AGENTS/assistant/CONTRIBUTING or existing lint/build configuration on artifact ancestry). Engine AGENTS/CONTRIBUTING read only. Documented engine checks are native, test/test-golden and test-cuda-host; none are invoked here. Only the unique `declaration control:` test is selected. No imported upstream test entry is copied: only Script/Reply/fixture helpers from the provided reference's first 91 lines, extended locally. No S3/runner-engine fixture is referenced by the probe. The native Python sandbox's own runner is necessary for actual Python execution.

A1 read and SHA checked: e7308621c645aef4402924a1b47c1a52e05baabe00239f915011f8431bbd0e89.

Overlay the two owned files zig/src/server/decl_control_probe.zig and decl_control_tools.json onto the parent's pinned zig snapshot; copy build.zig, dispatch_spy.c and this handoff alongside zig/. Do not overwrite any production source. Build uses unchanged zig/build/server.zig plain and sys.wire, no inert routing stubs, ReleaseSafe, Linux glibc, Zig 0.17.

Commands FOR THE PARENT'S approved Spark2 constrained scope only (2GiB, Swap0, nice19, ionice3, outer timeout7200, -j1; existing admission/QUIET/boot rules). CWD must be the owned dispatch directory. No local execution authorized here.

    ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-global-cache" zig build compile -j1
    ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-global-cache" zig build check -j1 > observed.jsonl 2> check.log

Before build, parent retains `sha256sum` manifest of every production zig/src and zig/build file (exclude only the two owned additions). Recheck `sha256sum -c` after all checks. Hash the fixture, declaration, spy, build file, actual compiler, copied source and response artifact. The author did not have the full pinned snapshot locally; API signatures were inspected read-only in tf-compute-lane, so the parent must resolve any revision-specific compilation finding in the owned harness only.

Expected baseline: D-whole and D-stream each one engine submit, held=false, zero spawn attempts/forwards, zero resume/input events, HTTP200 and exposed python call with finish_reason tool_calls; no compute result. L-whole and L-stream each held=true, exactly one intercepted AND forwarded native posix_spawn, actual sandbox CPython output323, one exact template-derived inserted result span and one final empty resume, HTTP200/stop and final answer. D/L submitted prompt arrays and serialized request bodies are compared in full. Noncall variants with pool enabled and spawn forbidden dispatch zero times. Missing Python or failed confinement is a failure, never skip. Each record retains the full native response/SSE payloads, request bytes, submit IDs and all resumed input IDs; capture stdout intact.

The in-harness wrong-configuration controls deliberately give D a pool and require the D verifier to reject the resulting dispatch; they supplement, not replace, the A1 guard mutant below.

For A1's actual absent-pool guard mutation, parent runs preparation on Spark (does not modify baseline):

    bash prepare_guard_mutant.sh
    cd guard-mutant
    ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-global-cache" zig build compile -j1 > compile.log 2>&1
    ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-global-cache" zig build check -j1 > observed.jsonl 2> check.log

The compile command MUST succeed or the mutant is VOID. The run MUST fail behaviorally: disabled native pool guard wrongly holds D, then the native generation path catches an invalid absent-compute transition (ReleaseSafe null unwrap), or the D assertions reject it. Inspect actual runtime failure/backtrace; timeout/compiler failure is not a caught mutant. Retain source.patch, compiler log, runtime log and hashes. This intentional separate source-copy mutation is not a baseline or engine-repository edit. Keep baseline and mutant hash manifests separate. Parent removes exact owned cache/mutant scratch after retaining receipts.

Limits: OpenAI run/Out is the native HTTP response abstraction; this does not open TCP sockets or exercise socket framing. The glibc symbol interposer counts the unchanged pinned pool's posix_spawn boundary, including failed attempts; D rejects any attempt. L's observed count plus real output/insertion proves the interposer was reached. It does not observe arbitrary future syscall/exec paths bypassing posix_spawn, instrument the deployed binary, prove GPU/model behavior or replace actual rank pool-absence evidence. This fixture uses mini native tokenizer/template bytes for route behavior; dataset-array qualification remains the parent's separate pinned-tokenizer gate. Collection remains blocked until parent execution and guard-mutant behavior pass.
