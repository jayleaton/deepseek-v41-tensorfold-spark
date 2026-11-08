# Phase-2 rank operations

These stdlib-only helpers are local artifacts. No Python execution, SSH, package
installation, GPU launch or node modification was performed while writing them.
The ancestry standards search found no AGENTS.md / LOCAL-AGENT.md / CONTRIBUTING or
build/lint configuration; Projects is not a Git repository. See `../LEDGER.md`.
The parent performs constrained <rank1-host> CPU checks (MemoryMax=2G, swap disabled,
nice 19, ionice class 3, outer timeout 7200, one worker). This helper performs no
queue, lock, thermal-control or restoration operations. The production memory
guard is called only for a real start.

Run generation on <rank1-host> CPU from the supplied reference snapshots:

```sh
python3 -B generate_envs.py --references /absolute/phase2/references --out /absolute/new-stage
```

`--out` must not exist, even as an empty directory or dangling symlink. Generation
never connects to either rank. It writes 14 boot manifests and 28 env files:
`pilot-plain`, `pilot-lane`, then `t01-plain`/`t01-lane` through
`t06-plain`/`t06-lane`, with alternating plain/lane order matching freeze.py.
Each boot is one two-rank serving boot. Main turn order follows the frozen
schedule; the freeze/controller decides requests within each boot. No request
selection or inference happens here.

The top-level `manifest.json` is an index, with `boots[].manifest` relative to the
stage directory, per-file SHA256s, and reference hashes. Each indexed boot manifest
is a standalone original-verifier-compatible `arms.D` manifest plus `phase2`
metadata. Do not pass the index to rank_ops. `phase2.ranks` lists the remote root,
config, stage-relative config, fresh state/cache paths, exact container name,
rank-local frozen tuning-table source, filtered configured env and thermal source.
The copied D record retains the original recipe, adapter and asset provenance,
with fresh config paths, pinned images and updated effective TF settings.
`arms.D.settings` contains settings shared across ranks;
`arms.D.ranks[RANK].settings` contains that rank's complete TF/GLM53_TF settings.
The wrapper validates the complete per-rank env. Only critical feature controls
must match across ranks; transport/HCA/master differences are preserved.

The parent installs `rank0/configs/*` under
`<rank0-work>/stdbench/phase2/stdbench-20261105-v1/configs/`, and similarly
`rank1/configs/*` under <rank1-host>'s root. Install each boot manifest under each rank's
owned root (for example `boots/pilot-plain/manifest.json`). Copy both Python
helpers together into an owned phase2 location. Installation must refuse existing
destination files: this generator deliberately does not overwrite or install.
`--prefix` permits another fresh `stdbench-*` namespace after an aborted run.
Never resume a boot whose `started/BOOT-rRANK` marker already exists.

Rank-local CLI (the parent runs the appropriate rank on each host):

```sh
python3 -B rank_ops.py verify /absolute/BOOT/manifest.json 0 --dry-run
python3 -B rank_ops.py start /absolute/BOOT/manifest.json 0 --dry-run
python3 -B rank_ops.py start /absolute/BOOT/manifest.json 0
python3 -B rank_ops.py snapshot /absolute/BOOT/manifest.json 0
python3 -B rank_ops.py stop /absolute/BOOT/manifest.json 0
```

Use rank 1 on <rank1-host>. `verify` and `start --dry-run` may create only fresh owned
state/sessions/cache/proof directories. They copy only `pfdense-table.json` from
the rank's original frozen recipe, checking its sealed hash. Repeated verification
accepts an identical table without rewriting it; any cache extras or changed table
fail. Existing prepared assets must already exist and are never created or changed.
`stop --dry-run` and `snapshot --dry-run` only record the planned Docker operation.
No dry run starts Docker containers or calls the boot memory guard. Real Docker
image inspection by the original verifier is required by verify/start dry runs.

rank_ops imports original `<home>/work/dsv41-bench/ops/rank.py` for config_env
with DSV41ROOT and DSV41_BENCH_ROOT set to that benchmark root, then invokes its
`verify D RANK` with DSV41_MANIFEST pointing at the boot manifest. That preserves
its image-ID/revision, asset-size, frozen recipe/adapter hash and tuning checks.
The Docker argv is parsed from the original verifier's dry-run launcher output.
The supplied D manifest selects its pinned `ops/native-rank.sh` adapter rather
than recipe `scripts/run-zig.sh`; respecting that selection preserves the native
TF transport flags and executable. The container `--name`, `/prepared` mount
mode (`:ro`) and image token (rank-specific immutable ID) are changed. Any sealed
TF/GLM53_TF/NCCL flags missing from the captured argv are inserted as explicit
`-e KEY=value` pairs immediately before the image token. Existing conflicting
values fail. The original recipe/adapter remains unchanged. Unexpected Docker
options, ambiguous commands or unresolved env names fail.
The adapter's NCCL_CUMEM_ENABLE and NCCL_CUMEM_HOST_ENABLE defaults are made
explicit as 0 in generated envs. All other normal G values remain from references.

Generation strips inherited RUNNER / TRACE / GUIDE / COMPUTE_LANE settings, then
sets TF_DSV41_COMPUTE_LANE to 0 (plain) or 1 (lane), and explicitly disables RUNNER,
COMPUTE_LANE_RUNNER, RUNNER_COMMIT, RUNNER_STATED, RUNNER_INLINE_CHAIN,
RUNNER_INLINE_MARK, RUNNER_SYNC_GRAPH and PREPARED_WRITE. RUNNER_FORMS is
`fence,repl,inline`. Forbidden split-dedup, split-early and chunked-RoCE flags fail
validation. Each env sources `/etc/spark/thermal.env`; generated files contain no
literal GUARD/SPARK overrides. Effective guard values are checked against that
node's thermal file before any operation. Host launch flags are not inherited.

Every operation writes a new exclusive JSON proof under the owned
`proof/BOOT/` directory and emits a small JSON summary on stdout with exactly
`command`, `rank`, `boot`, `dry_run` and `proof` (the absolute proof-file path).
Environment and launch details are recorded in that artifact, not terminal JSON.
Start/verify proofs include the
complete shell-quoted safe launch command, argv, filtered sourced host env and
effective container launch env. Snapshots read actual Docker Config.Env and
validate effective TF/GLM53_TF/NCCL values. Proof env dictionaries contain only
TF_, GLM53_TF, NCCL, launch paths/identity, GUARD and SPARK keys. The original
launcher's TENSORFOLD_REQUEST_LOG is retained in the argv but excluded from env
snapshots under this whitelist.

Real start checks that state/sessions are empty, refuses an existing exact
container and previously attempted boot, sets `DSV41_BOOT_MIN_GIB=114` in the
rank launch and guard environment (also recorded as `boot_min_gib`), calls production
`mem_guard.before_boot(str(rank), cancellation_callback)`, writes proof, then runs
the derived Docker argv in the foreground. The callback returns false because
the parent owns rank process cancellation. Parent must preserve its rank process
PID and issue exact-container stop before draining/joining that process. Stop
checks exact container name and image ID, uses `docker stop --time 120`, then
waits/drains, allowing automatic `--rm` removal. No daemon mode, arbitrary kill,
process scanning or service restoration exists.

Parent CPU verification should cover py_compile for both files; stage generation
against the actual snapshots; 14 boot/28 config counts, inventory hashes,
plain/lane controls and fresh paths; overwrite rejection; fake original-verifier
and Docker preflight (including inherited-flag stripping, image mismatch,
prepared read-only argv transformation and stop identity rejection). Then run
rank-local verify/start --dry-run against installed files before any authorized
serving turn. No execution check has been claimed by this implementation worker.
