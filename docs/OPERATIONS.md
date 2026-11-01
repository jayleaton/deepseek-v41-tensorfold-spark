# Native operations

Use [ZIG-SERVE](ZIG-SERVE.md) for the current rank commands, memory profile, assets and drain behavior.
The native containers are `tensorfold-zig-r0` and `tensorfold-zig-r1`, created by `scripts/run-zig.sh`.
The old Python launcher and its incompatible unattended units have been removed; use the native lifecycle below. Do not assume the retired Python stall-exit controls protect the native server.

Historical operational measurements remain in the campaign and result records under their original engine labels.

## Optional native guard (#48)

The guard is adapted from @jayleaton’s #48. It is a separate reverse proxy, bounded admission queue, sampler and
STOP/ownership controller. The generalized upstream-native port remains **unqualified**. The example is disabled,
profiles are empty, and persistent clock changes are disabled. Previous deployed-soak results do not qualify this pin.

Keep operator configuration and state outside the checkout, for example `~/.config/tf-guard/guard.json`. Start with
`config/guard.example.json`, review every node path, container prefix, port and ownership field, and qualify the exact
running profile before enablement. Set the engine's head port separately (for example 8001), with the guard on 8000.
Set `worker_repo` to the absolute worker recipe checkout containing `scripts/safe_exec.py`.
This optional guard JSON path defaults to the head checkout's absolute path when both nodes use the same layout.
Remote diagnostics and telemetry start with an empty environment and a fixed system PATH, then run through
the worker supervisor. Worker SSH profiles must keep startup diagnostics quiet; configured `RemoteCommand`
is overridden. The shell watchdog uses `WORKER_REPO` for the same worker boundary.

Do not expose an unguarded alternate port to clients. `scripts/guard/profile.py` snapshots exact image, mounts,
engine environment and optional CPU sets; `control.py` prepares/authorizes an approved profile. A snapshot alone is
not qualification. The candidate profile's workspace floor must meet the guard's minimum before it can be approved.

The default owner window is pause 92°C, resume 86°C and hard STOP at 93°C held for 30 seconds or at once on a fresh
94°C reading, with a 2100 MHz starting cap. Either STOP is durable, cancels active requests and stops the owned
ranks. When `/etc/spark/thermal.env` exists, its `GUARD_PAUSE_C`, `GUARD_RESUME_C`, `GUARD_KILL_C`,
`GUARD_KILL_HOLD_S` and optional `GUARD_IMMEDIATE_C` replace these defaults, and the config's `policy` must equal the
result; a malformed or disordered file is refused. Preserve the actual owner policy and existing cap on each host.
Review the optional clock-service template and its drop-in before any installation; this recipe update installs or
enables no service and changes no host clock.

When configured, `GUARD_CONFIG=<local JSON>` makes the native launcher obey `scripts/guard/lifecycle.py`: exit 75 is
a hold and 78 is durable STOP/invalid state. It must never be overridden by automatic recovery. STOP is durable before
cancellation or container actions; maintenance leases relinquish ownership, and profile/image/mount/environment
changes invalidate ownership. `CPUSET_CPUS` is optional and rank-local; no host CPU topology is assumed.

`sampler.py --config <local JSON>` and `daemon.py --config <local JSON>` run the guard. The provided user-service
files are templates for an explicitly approved deployment. `guard-watch-zig.sh` is the optional guarded watchdog;
its container prefix must match the launcher, and it respects both its own stop marker and the native supervisor's
`STATE_DIR/native-ops/STOP` (or `STOP_FILE`). Keep one configured supervisor in charge of recovery.

CPU tests use isolated fake runtimes, HTTP servers and clocks. Actual native health/tokenize, two-node recovery,
thermal behavior, memory floors and mixed-concurrency gates remain unrun. No guard default or performance promotion
is implied by including the source.

### Short-prompt admission, sampler period and request bodies (#48)

- **Short-prompt admission (off by default).** Set `decode_admission_max_prompt_tokens` in `limits` (1..2048; absent
  or 0 is off) to let short text prompts start together. The guard then counts each generation request's prompt with
  the engine's `/tokenize` (3 s timeout) before admission. A prompt at or under the limit does not take the prefill
  lane and does not make later requests wait for a post-token sample. FIFO order, the in-flight limit, the admission
  cooldown, fresh telemetry, the memory margin, its own reservation, the health check and STOP all still apply.
  Images, other routes, a failed or slow count and watchdog probes take the normal path. Keep the limit at or below
  the smaller of `TF_DSV41_PREFILL_CHUNK` and `TF_DSV41_PREFILL_ROWS`, so such a prompt is one prefill segment.
- **Sampler period.** `sampler_period_s` (2.0; 0.25..2.0) sets the sampler's interval. A shorter period shortens the
  post-token wait between long requests; each sample still opens one SSH connection to rank 1. The 6 s freshness
  window and the per-node sample timeout do not change.
- **Request bodies.** A chunked body is read into one buffer bounded by its payload bytes (`max_body_bytes`), so many
  tiny chunks cost their bytes, not one object each.

### Dispatch-cost options (#66)

The follow-up from @jayleaton adds `generation_verify_cache_s` (0–5 seconds) and
`small_body_release_bytes` (at most 1 MiB and the configured maximum body size).
Both default to **0/off**. Ownership, lease, profile and readiness checks remain
per request. The cache invalidates on native doubt, unhealthy reads, incomplete
requests, server errors and STOP; an otherwise healthy change within the same
container can remain undetected until its TTL expires. Early buffer release holds
small bodies within the pending-plus-inflight bound rather than the reader-buffer
bound. Qualify the resulting memory bound and cancellation before enabling it.

CPU tests establish those contracts, not the proposed performance benefit. The
required one-boot alternating G-off/G1v2 Spark comparison remains unrun: start spread
must be at most 0.3 seconds, and memory minima no more than 0.25 GiB below both
neighboring off blocks. Keep both options off until those device gates pass.

## Compute lane (`TF_DSV41_COMPUTE_LANE`, on in the flash17 profile)

The flash17 profile turns the compute lane on (`TF_DSV41_COMPUTE_LANE=1`); the engine's own default is off, and the
q28-v2 rollback profile leaves it off. A request that lists `{"type": "code_interpreter"}` in `tools` gets a
`python(code)` tool that the engine runs itself and answers inside the same reply, so the client makes no tool round
trip. Requests without that entry are unchanged, and the lane is not armed for requests with stop strings. The lane's
calls are not returned as `tool_calls`: they are reported in a non-standard `compute` field (`message.compute` in a
chat reply). Clients that do not read it see only the answer.

Each call runs `/usr/bin/python3` (the runtime image installs it) in a sandbox: it drops to `nobody`, sets rlimits and
nice 10, sets `no_new_privs`, applies Landlock (ABI 6 or later: read and execute only under the interpreter's roots, no
network or signals) and seccomp (no fork, no sockets). Docker's default seccomp profile allows this setup inside the
container. At boot the engine runs `print(1+1)` in the sandbox; if that fails (for example on a kernel without Landlock
ABI 6, Linux 6.12 or later), the boot stops with `compute lane: the sandbox self-test failed`. The log line
`compute lane: /usr/bin/python3 in a sandbox, 2 at a time, ...` confirms it is on.

| Setting | Default | Range |
| --- | --- | --- |
| `TF_DSV41_COMPUTE_LANE` | off (`1` in the flash17 profile) | `1` on; unset, empty or `0` off; anything else stops the boot |
| `TF_DSV41_COMPUTE_LANE_WORKERS` | 2 | 1-64 concurrent sandboxes |
| `TF_DSV41_COMPUTE_LANE_MAX_MS` | 10000 | wall limit per call, 1 ms to 1 h |
| `TF_DSV41_COMPUTE_LANE_MAX_CALLS` | 8 | built-in calls per reply |
| `TF_DSV41_COMPUTE_LANE_MAX_RESULT_TOKENS` | 512 | result size inserted into the reply |
| `TF_DSV41_COMPUTE_LANE_MEM_MB` | 1024 | address-space limit, at least 64 |
| `TF_DSV41_COMPUTE_LANE_CPUS` | unset | CPU affinity list such as `16-19` |
| `TF_DSV41_COMPUTE_LANE_PYTHON` | `/usr/bin/python3` | an absolute path |

The profile keeps every limit at its default: two sandboxes at a time, 10 s and 1 GiB per call, 8 calls and 512
inserted tokens per reply.

**Turning it off.** Set `TF_DSV41_COMPUTE_LANE=0` (or delete the line) in `config/prod-zig.env` on both nodes and
restart the pair. Nothing else changes: the lane-off rows in the measurement below served the same tokens. The lane
runs model-written code on the serving host inside the sandbox above; turn it off if no client sends
`code_interpreter` or if that is not acceptable for your host.

**Measured on G (`58bece1d3`), 2026-11-01** (TP2, flash17 K6 MTP, the bench settings of
[BENCHMARKS](BENCHMARKS.md#release-candidate-g); the only change between arms is `TF_DSV41_COMPUTE_LANE`):
- Tokens: greedy 2K and 8K outputs, draft off and on, were identical in all five boots (two lane off, three on).
- Plain serving without tool calls, lane on vs off over two ABBA pairs: coding **0.980** (about 2 % slower),
  set-b **0.999**. The two lane-off rows differed by 2.2 % from each other, so treat these as about ±1.5 %.
- MBPP-40 (40 MBPP-sanitized tasks, greedy, at most 8,192 tokens, each cell in a quiet window), lane vs the client
  running the same `python` tool itself: **14 %** less wall time per solved task at one stream (9.67 vs 11.24 s) and
  **16 %** less at four (6.70 vs 8.01 s); 645 vs 715 completion tokens per task; 1 request per task instead of 1.53.
  All arms solved 38 of 40. Against no tools at all, the lane was 2.7 % faster at one stream and 5.5 % slower at four,
  where only two sandboxes run at a time (`TF_DSV41_COMPUTE_LANE_WORKERS`).

## Stalled-engine exit (`TF_DSV41_STALL_EXIT_S`, off)

The pinned engine watches its own work: a round, a session snapshot, the KV union sync and each operation a follower
runs from the leader. Every issued GPU call counts as progress. If work is in flight and nothing has progressed for
`TF_DSV41_STALL_EXIT_S` seconds, that rank logs `no progress in <phase>`, withdraws readiness and exits. The other
rank then fails fast through the pair's fate channel, so both containers stop instead of wedging with requests
queued (#38, #59, #64). An idle engine, or a rank waiting for its next operation, never trips it.

| Setting | Default | Values |
|---|---|---|
| `TF_DSV41_STALL_EXIT_S` | off | whole seconds; unset or `0` off; anything else stops the boot |

It is off by default, and the example profiles leave it unset. For unattended TP2 serving, set
`TF_DSV41_STALL_EXIT_S=300` in both ranks' config, and run a supervisor that restarts the pair: the `serve.sh` watch
with `WATCH_HEAL=1` (below) or your own. Without a supervisor the pair stays down after the exit.
Five minutes is far above the longest single GPU wait in normal serving (a 4K-row prefill segment takes seconds, even
at deep context), and the wedges reported in #64 lasted hours. Lower values save little and risk exiting during a
slow first-use kernel load. This exit has not yet been triggered on the pair. Its tests block the actual
synchronization calls on the host (engine commits `53033f7ee`, `aee501160`, `d6dbc870a`).

## Native pair lifecycle and watchdog (#58)

For authenticated native serving, put the keys in a file with mode `0600`, owned by
you or by root (root-only is stronger; see [SECURITY.md](SECURITY.md)), outside the
checkout and the model, engram, assets, prepared, state and cache directories. Set
`NATIVE_AUTH_FILE=/absolute/path/to/keys` in the head's config. Name the file plainly,
like `keys`. A credential-like name such as `api-key`, `secret` or `auth-keys` makes the
mount argument read as a `name: value` pair, so the launcher refuses it. Keep key contents out
of shell config and environment variables. The launcher validates the file's type,
owner, mode and location, mounts it
read-only at `/run/tensorfold/auth-keys` on rank 0, and passes that path with
`--api-key-file`. Host supervisors read its keys into memory only to mask and
refuse them (see [SECURITY.md](SECURITY.md)). Rank 1 and AOT diagnostic containers receive no key mount.
The file may contain one key per line or upstream's labelled/scoped keys; the
native server validates the contents and refuses an invalid or empty key set.
The pinned engine supports this through `zig/src/server/cli.zig` and
`zig/src/server/auth.zig`; no credential environment exception is required.
Its authentication environment name is `TENSORFOLD_API_KEY`, rather than generic
`API_KEY` or `TF_DSV41_API_KEY`. A head config containing that supported name
without `NATIVE_AUTH_FILE` fails with a migration error. The file becomes the
authentication source when configured; remove the old environment setting.

The mount pins a single file inode. To rotate keys using an atomic file
replacement, explicitly restart through the lifecycle after replacing the file;
do not assume the existing mount follows the replacement. Treat file ownership,
permissions and authentication checks as part of deployment qualification.
With `NATIVE_AUTH_FILE`, lifecycle readiness uses the public `/health` response
plus owned/running rank checks and the existing preflight model/asset identity
comparison. The pinned server checks engine health before returning
`{"status":"ok"}` and deliberately hides model details with authentication on
(`zig/src/server/status_routes.zig:13–23`). Lifecycle never authenticates with
the keys or calls authenticated `/v1/models`, so model-list availability is not
checked in this mode. Check it separately with an authorized client during qualification.
The optional completion watchdog probe requires `WATCH_PROBE=0` when it addresses
an authenticated native endpoint directly; it receives no serving credential.

Use `CONFIG=config/prod-zig.env bash scripts/serve.sh preflight` on the head before
`start`. The config is exported before invoking the native supervisor; set absolute
`STATE_DIR`, `WORKER_REPO`, `WORKER_CONFIG` and a single `WORKER_SSH` target. Both
checkouts must have the same engine pin and both ranks must report the same image,
model, common assets and profile identity. Rank-specific Engram headers are checked
locally before any worker starts. `serve.sh rank N --identity` checks a single rank;
`serve.sh build` invokes the pinned native builder.

`start` launches worker then head and waits for both owned containers plus healthy
`/health` and, without file authentication, `/v1/models` responses. Names are `tensorfold-zig-r0` and
`tensorfold-zig-r1`, or `${CONTAINER_PREFIX}-r0/r1` when overridden on both nodes.
`stop` writes a durable STOP before signalling, then stops only containers with the
expected name, rank and revision labels, using their inspected IDs. Its timeout
allows at least `TF_DRAIN_S + 60` seconds (minimum 80); size service timeouts
for any larger configured budget. A failed boot also leaves STOP set.

Only an explicit `start`/`restart` can clear an unchanged operator STOP after
preflight. `watch --once`, reboot timers and the optional guard watcher respect it;
raw `run-zig.sh` also refuses a launch while it exists. A maintenance lease or guard
hold prevents recovery. An unavailable Docker daemon/SSH connection is an error,
never evidence of an absent rank or permission to kill an unverified container.

`status` reports readiness. `watch --once` records persistent failure counts;
`WATCH_HEAL=0` is observation only. Opting into `WATCH_HEAL=1` permits recovery after
`WATCH_FAILS` failures and `WATCH_MIN_HEAL` seconds since the last attempt. Defaults
are 3 failures, 300 seconds between attempts and 10 seconds between ticks when run
continuously. The example timer ticks once per minute. Recovery time additionally
includes bounded runtime calls, drain time and up to `READY_TIMEOUT` (900 seconds)
for model loading. This is launcher supervision, not an engine progress proof.

The `scripts/systemd/tf-zig-watchdog.{service,timer}` templates are **not installed**.
Review the serving user, checkout, environment and timeouts before installation.
Keep healing off until actual rank-death, stalled-health, cold-start and operator
STOP scenarios pass on your pair. CPU tests use fake runtimes and HTTP; no new
hardware or unattended-recovery qualification is claimed. The guard watchdog is an
alternative opt-in observer, not a second independently enabled healer. When using
it, route recovery through `serve.sh watch --once` so automatic recovery cannot
clear STOP; never set its restart command to explicit `start` or `restart`.

Launcher subprocesses share a supervised process group. A timeout or termination
signals the group, escalates after a bounded grace period, and reaps children
before lifecycle cleanup. Worker commands carry a deadline based on the worker's
monotonic clock; SSH failure waits through that deadline before STOP cleanup,
preventing a delayed remote launch from racing the stop. This behavior has CPU
regression coverage; hardware qualification is still required as described above.

Host watchdog and launcher commands share `safe_shell.sh` / `safe_exec.py`: argv
validation happens before external commands spawn, and shell-operation diagnostics
are filtered as well as child output. Watchdog numeric controls are validated
before arithmetic. A heal holds ownership until its command finishes; it does not
detach an unowned restart after the observing tick exits. Image verification runs
the image's `/opt/tensorfold/share/libproof.sh` through the host wrapper.

The complete credential model, including what the output filter does and does
not promise, is in [SECURITY.md](SECURITY.md).

Trusted shell config can temporarily export values while it is sourced. The
wrapper immediately removes credentials and their aliases from its exported
environment before launching diagnostic commands. Known values remain unexported
in the wrapper for validation and filtering; only the checked supervisor receives
that registry over a private pipe, which it closes before starting a child.
Diagnostic children receive neither that pipe nor credential environment values.
A credential in an operational control fails with exit 78 rather than selecting
a default. The initial caller and trusted config-loading shell are outside this
child-environment guarantee; keep serving keys in `NATIVE_AUTH_FILE` as above.

The optional guard watchdog accepts `WATCH_RESTART_ARGV` as a JSON argv array:

```bash
WATCH_HEAL=1
WATCH_RESTART_ARGV='["bash","scripts/serve.sh","watch","--once"]'
```

Strings are literal arguments, with no shell quoting, interpolation, command
substitution or heredoc evaluation. An entry such as `{"env":"RECIPE_LAUNCHER"}`
explicitly reads one exported variable as one complete argument; resolved values
pass the shared credential check before any child starts. Missing references fail
closed. Only this checkout's reviewed `bash scripts/serve.sh watch --once` recovery
entry is admitted, including its equivalent absolute script path. Arbitrary
programs/interpreter payloads and explicit `start`/`restart` are refused, preserving
durable STOP. The lock remains held through synchronous completion, and command
failures propagate. `WATCH_RESTART_CMD` is removed: a nonempty old setting exits
with a clear error directing operators to the JSON form; it is never evaluated.
