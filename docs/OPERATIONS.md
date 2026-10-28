# Native operations

Use [ZIG-SERVE](ZIG-SERVE.md) for the current rank commands, memory profile, assets and drain behavior.
The native containers are `tensorfold-zig-r0` and `tensorfold-zig-r1`, created by `scripts/run-zig.sh`.
The old Python launcher and its incompatible unattended units have been removed; native supervisor wiring follows
separately. Do not assume the retired Python stall-exit controls protect the native server.

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
