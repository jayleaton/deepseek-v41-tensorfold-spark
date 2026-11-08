# Parent handoff: A1 rank helpers

Source-only, not executed locally. Original parent and A1 specs were read.
A1 SHA256 is e7308621c645aef4402924a1b47c1a52e05baabe00239f915011f8431bbd0e89;
parent SHA256 is adb05167fff21469ee7251555cb243f1f109c4d8e3de237fdcaa59ada8eb55e6.
No repo/standards/config exists on this artifact ancestry as recorded in LEDGER.
Old phase2 files remain untouched. No SSH, tests, engines, GPU, API, queue or
lock operation was performed. Only the requested new helpers and staging files
were authored. Parent owns controller3/client3/freeze3 and actual CPU evidence.

`generate_envs3.py` adapts phase2/generate_envs.py: nine boots/eighteen envs,
orders plain/declared_off/lane, declared_off/lane/plain, lane/plain/declared_off;
no pilot. Default fresh prefix `stdbench-decl-a1-20261105-v1`; output must
not exist. Immutable per-rank 3a image IDs, tuning, transport and assets stay
pinned. Both rank roots are `<rank{rank}-work>/stdbench/decl-control/PREFIX`.
`arms.D` remains the generic engine record, `phase2` schema1 remains unchanged
in shape, and `phase2.arm` now also accepts `declared_off`. P/D envs differ only
in owned paths/identities; normalized L differs solely in COMPUTE_LANE 0/1.
Declaration presence and exact D/L wire equality belong to client3/freeze3.
Runner/trace/guide/S3 inheritance is removed; existing explicit off controls
remain. Thermal values come only from the node's thermal policy source.

`rank_ops3.py` adapts phase2/rank_ops.py with only its module import, namespace,
boot regex and explanatory text changed. Commands stay `verify|start|stop|snapshot
MANIFEST RANK [--dry-run]`. Roots point to decl-control; accepted boots are
`t0[1-3]-(plain|declared_off|lane)`. Original frozen rank verifier/native adapter,
exact image/name checks, prepared read-only, owned state/cache, cancellation,
114GiB boot guard, actual native-process environment snapshot and proof schema
are preserved. **Even verify/start --dry-run invokes original verifier/Docker
inspection; the fake selfcheck does not invoke those operations.** No lock or
queue code was added. Parent must retain root-<rank1-host> read-only fd9 lease and
exact-owned cleanup/G restore in controller3; t03 is the last scheduled turn.

GPU becomes authorized after the actual CPU gates pass; these files themselves
are marked STAGED_CPU_GATES_PENDING. Pending gates include A1 source/hash freeze,
all-arm CPU/count matching, identical D/L wire bytes, mandatory whole/streamed
real-route forced-call control with L positive twin/non-call/behavioral mutant,
native ID accounting, conditional input missingness/claim withholding, graders,
120 blind MATH map, jail/statistics, real rank preflights, controller admission,
reserve-only-unmeasured and cleanup/restore qualification. An unqualified
multi-input ledger is not a blanket collection blocker under A1: input-total
claims must instead be withheld as A1 requires. No served-ID-equality claim.

Parent-only commands below are payloads for an already admitted Spark CPU scope:
MemoryMax2G/MemorySwapMax0, nice19/ionice3, single worker/-j1, timeout>=7200,
QUIET/queue-boot checks and both-node memory floors. No local execution or Zig.

```sh
cd <rank1-work>/stdbench/decl-control
python3 -B -c 'import ast,pathlib; [ast.parse(pathlib.Path(p).read_text(), filename=p) for p in ("generate_envs3.py", "rank_ops3.py", "rank_check3.py")]; print("syntax PASS")'
python3 -B rank_check3.py
python3 -B generate_envs3.py --references <rank1-work>/stdbench/phase2/references --out <rank1-work>/stdbench/decl-control/stdbench-decl-a1-20261105-v1
```

`rank_check3.py` uses automatically removed exact scratch, creates no bytecode,
and prints no env values. It checks actual references, inventories/counts/fresh
paths, P/D and L env differences, guard source/no overrides, fake context for
all18 ranks, bad boot/overwrite refusals and AST equality of inherited native
snapshot/Docker/operate implementations. It is not a live rank or CPU gate pass.
After parent installs manifests/configs exclusively on both ranks, run original
rank adapter verify/start dry-run qualification with rank_ops3 under the parent's
authorized CPU/admission procedure, retaining actual receipts. No artifacts
were generated locally. Early stage-turns shell scaffolding stays unbound and
is not used by the actual controller3 campaign.
