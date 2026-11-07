# P preparation: SPEC_PENDING

Source only. Nothing here has been executed on the <dev-host> or Sparks by this
task. Parent owns all bounded Spark CPU execution, client/native/clipping
verification and turn launchers. No new task generation, answers, model requests,
GPU work, SSH, endpoint/ORDER/lock/engine changes, <other-host>, or Lead actions.

`staging_only=true`, `collection_authorized=false`, `A1sha=null`,
`specpending=true`. JJ must steer the prospective T5-COMPUTE-DL-SPEC-A1 SHA
before collection can be considered. These artifacts cannot authorize it.
No final campaign, GO, controller layout, budget or dispatch qualification is
created. The existing D/L-only controller is incompatible with this new context.

The original c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141
spec is provenance, not authorization for P. The original 3723 sealed files,
observations, failing receipt, grader and statistics remain untouched. The 138
remaining original requests are read-only pinned provenance, not a P schedule.

## API for the parent

- `p_stage.plan(root=compute_dl_root, prefix=PREFIX)` returns `(index, payloads)`;
  payloads maps relative paths to bytes. It reads existing pinned public data only.
- `p_stage.stage(out=None, *, root=compute_dl_root, prefix=PREFIX)` returns the
  same index and writes new files exclusively. Default output is
  `a1-prep/staged-P`. Output must be a fresh non-symlink descendant of a1-prep;
  an existing or partial output is never reused. No installation API is supplied.
- `p_check.check(root=compute_dl_root)` returns bounded in-memory fake-check
  results; `p_check.check_stage(stage_path, root=compute_dl_root)` reads and checks
  the actual staged artifact. Exceptions fail the check. Neither runs a client.

Parent must make `a1-prep` importable, with bytecode disabled, inside its existing
Spark 2GiB/no-swap CPU scope. `python -B p_check.py` is the fake-check entrypoint
for that scope only. No <dev-host> check execution is authorized. AST parsing of both
files, bounded fake/client/config checks, actual native prompt/clipping gates,
and applicable turn-preflight checks remain for the parent. No checks are claimed
passed by this source delivery.

## Staged outputs

`requests-P.json` contains exactly 240 request templates, 120 public tasks times
seeds 101/102, mapped to the original matched L boot/position/wire SHA. Candidate
order is original t01-L 120 then t02-L 120, labelled **ORDER_NOT_FROZEN**; T5
overrides must later bind the actual order. `p-all` is a candidate boot only.

P copies the original matched L body and deletes the `tools` key entirely.
Everything else, including key order, user prompt, model/high/thinking, T1,
top_p .95, top_k 20, seed, 8192 cap, 270-second deadline, streaming and returned
token IDs, is identical. Policy and provenance are outside the wire body. The
public schema rejects answers/hidden fields; no private answer file is read.

`rank0/configs/p-all-r0.env`, `rank1/configs/p-all-r1.env` and each rank's
`effective-env.json` derive from pinned base-r0/r1 and base-arms references using
the pinned generate_envs3 literal-reader/filter/off/env helpers. They preserve
the original serving/calibration/transport settings and set lane=0, runner,
runner variants/commit/stated/chain/mark/sync and prepared-write=0. Inherited
TRACE/GUIDE/S3 values are stripped. Native runner forms remain fence,repl,inline
as in D, with execution disabled. Node thermal policy remains node-sourced.

Node targets use new `compute-dl/a1-prep/` descendants under prefix
`stdbench-compute-a1-prep-20261105-v1`. State/cache/container/config names
are fresh. Manifests record both full effective envs, off assertions, source
hashes, revision 3a429d6f122d9eb2900c43e622933d311e4be44f, and distinct rank
image IDs 5be7d7d4… / 80beaba7…. They are preparation manifests, not launchable
native-controller manifests. `P-stage-index.json` records output hashes and all
source pins. Old whole/SSE dispatch proof SHA77e43ebc… is named for prospective
reuse only after the parent rechecks production source/image identity; this code
does not claim that check or a fresh dispatch proof.

Fake controls cover exact pair coverage/order, sampling, tools omission,
public-only inputs, hidden wire metadata, corruption, policy, every off flag on
both ranks, namespace escapes, approval promotion, and drift of every pinned
source/data input. The stage reader additionally verifies exact file inventory
and actual file bytes against an independently rebuilt plan.

## Standards

Read the current runtime_stage.py, frozen public dataset identity, freeze manifest,
matched L requests, protocol, remaining-collection inventory, reference envs and
LEDGER. The ledger records ancestry AGENTS/assistant/CONTRIBUTING absence and no
applicable lint/type/format/build configuration. This is an artifact directory,
not a Git repository; no branch/commit/export/PR is created. Only p_stage.py,
p_check.py and this README belong to this task. No additional helper agent was
used because this tightly coupled three-file source change has no independent
slice without overlapping ownership.
