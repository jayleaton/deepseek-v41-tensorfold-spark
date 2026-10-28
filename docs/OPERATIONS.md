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
Do not expose an unguarded alternate port to clients. `scripts/guard/profile.py` snapshots exact image, mounts,
engine environment and optional CPU sets; `control.py` prepares/authorizes an approved profile. A snapshot alone is
not qualification. The candidate profile's workspace floor must meet the guard's minimum before it can be approved.

The default owner window is pause 92°C, resume 86°C and hard STOP at 93°C held for 30 seconds, with a 2100 MHz starting
cap. Preserve the actual owner policy and existing cap on each host. Review the optional clock-service template and
its drop-in before any installation; this recipe update installs or enables no service and changes no host clock.

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

## Native pair lifecycle and watchdog (#58)

Use `CONFIG=config/prod-zig.env bash scripts/serve.sh preflight` on the head before
`start`. The config is exported before invoking the native supervisor; set absolute
`STATE_DIR`, `WORKER_REPO`, `WORKER_CONFIG` and a single `WORKER_SSH` target. Both
checkouts must have the same engine pin and both ranks must report the same image,
model, common assets and profile identity. Rank-specific Engram headers are checked
locally before any worker starts. `serve.sh rank N --identity` checks a single rank;
`serve.sh build` invokes the pinned native builder.

`start` launches worker then head and waits for both owned containers plus healthy
`/health` and `/v1/models` responses. Names are `tensorfold-zig-r0` and
`tensorfold-zig-r1`, or `${CONTAINER_PREFIX}-r0/r1` when overridden on both nodes.
`stop` writes a durable STOP before signalling, then stops only containers with the
expected name, rank and revision labels, using their inspected IDs. Its timeout
allows at least `TF_DSV41_DRAIN_S + 60` seconds (minimum 80); size service timeouts
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
