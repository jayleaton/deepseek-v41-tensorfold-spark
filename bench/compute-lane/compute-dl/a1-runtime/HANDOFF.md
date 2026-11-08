# A1 runtime v1: source-only handoff

No dev Python/tests, SSH, model requests, engine edits, queue actions, GO creation or original-file mutations were performed by this implementation agent. The artifact tree is not Git. Original LEDGER.md documents the absent ancestry AGENTS/assistant/CONTRIBUTING and absence of lint/build configuration. Existing standards/checks apply: parent Spark CPU scopes (2GiB/no swap/nice19/ionice-c3/-j1/timeout>=7200), AST, fake-client/config/protocol, real native clipping/reference/provenance/prompt arrays, both actual rank env/source/image checks and shell queue preflight. No <other-host>.

New copies: controller.py, rank_ops.py, request_adapter.py, budget.py, runtime_stage.py, render_prompts.py, fake_client_check.py and fence.sh derive from the sealed original compute-dl files. New protocol.py, clipping.py, cpu_check.py, cpu_gate.sh, launch.sh and turn.sh supply A1 boundaries. Original protocol/controller/budget/GO remain untouched. Final execution qualification is parent-owned and PENDING; source inspection is not qualification.

## Parent CPU commands / API

Run under existing phase2 CPU guards on the Sparks, never dev Python:

- `python3 -B a1-runtime/cpu_check.py`: AST + actual raw mask/stage + negative protocol/config/budget/policy + fake native-client controls; temporary outputs cleaned, no model access. Requires copied A1 runtime and spec on the Spark. Fake controls are not native or lifecycle qualification.
- `python3 -B a1-runtime/runtime_stage.py stage --out a1-runtime/runtime-stage-v1` (fresh output only); Python API `stage(rootout=None, *, root=HERE.parent, observed_sources=None, references=None, spec_file=None)`.
- `python3 -B a1-runtime/runtime_stage.py materialize a1-runtime/runtime-stage-v1 0` and rank1 on <rank1-host>; Python API `materialize(stage_root, rank, destination=None)`; <rank0-host> gets <rank1-host> shadow manifests too. Never writes GO/final campaign/budget.
- `python3 -B a1-runtime/render_prompts.py`: pinned native renderer; output under new prefix. Exactly242 unique task/seed+diagnostic records, each with plain_ids/declared_off_ids/declared_lane_ids; identical messages deduplicated across arms. Controller retains original tuple(key) lookup.
- Parent actual `rank_ops.py verify MANIFEST RANK --dry-run`, actual source/image identity checks, fresh namespace clash on both nodes, queue turn-preflight for launch.sh/turn.sh/cpu_gate.sh/fence.sh and dependencies, and parent SIGPIPE inherited-ignore regression.
- Qualified launch only: `bash a1-runtime/launch.sh FINAL_CAMPAIGN A1-resume`, then A1-plain. The latter requires prior QUALIFIED resume verdict. Every turn retains original queue/locks/request guards/drain/G restore and step verdict logs. Old GO has no authority.

Stage layout is original compatible: frozen-v1/manifest.json and boots, configs, both rank manifests, retention.json, source-hashes.json, prompt-inputs.json, stage-campaign.json, stage-manifest.json. Fixed boots are A1-resume-L(18), A1-resume-D(120), A1-plain-P(240). P is exact original t01-D then t02-D order with only tools removed. Remaining D/L templates are copied unchanged. retention.json seals all342 keys plus their source-line hashes and six raw/response source-file hashes, against all480 original keys and the frozen remaining inventory. Broader revalidation source pins must contain these six and may include diagnostics/safety/logs.

## Final campaign / CPU proof contract

Parent takes stage-campaign.json and supplies final hashes, code seals and evidence. Copy the same final campaign bytes to both rank-local campaign-v1.json paths; protocol resolves <rank0-host> evidence paths on <rank1-host> without changing the digest. Required additions beyond original campaign:

- `base_spec`, `base_spec_sha256` original frozen c3e6...; `spec`, `spec_sha256` amended 0dca... (full constants in protocol.py).
- `original_budget`, `original_budget_sha256`: original sealed budget.json at3413.519101628801s, no active interval. New ledger is schema2 and preserves original turns, both spec hashes, original ledger hash and carry; no reset. Limit18000/admission15900 unchanged.
- `revalidation_receipt`, `revalidation_receipt_sha256`: configurable final receipt (stage default a1-prep/revalidation-v2/qualification.json). Requires passed:true, spec_sha256:A1, parent_spec_sha256:base, observed342, remaining138, counts D120/L222, activation_distinct_L>=30, preserved_original_seal:true, raw_source_sha256 mapping. All source files hash-checked; must include retention's raw/response inventory.
- `cap_policy` keys binary,tokenizer,source,receipt (defaults parent NativeCap paths). Adapter loads a1-prep/reference_policy.py, calls NativeCap and compare_pairs and presentation. Preserve all pinned native/Jail/source/status/byte/time/frame/ID checks. Every new inserted call retains original independent result plus presentation metadata in safety.jsonl; no first-line/prefix shortcut. Policy exceptions stop collection.
- `cpu_receipt`, `cpu_receipt_sha256`, `cpu_prompt_arrays`, `cpu_prompt_arrays_sha256`, `code_sha256`, `freeze_sha256` and original rank seals populated after real gates.

CPU qualification JSON must include passed:true, spec_sha256:A1, parent_spec_sha256:base, exact `gates == protocol.GATES`, identical code_sha256, and freeze_sha256/cpu_prompt_arrays_sha256/revalidation_receipt_sha256/original_budget_sha256 identical to campaign. `preparation_file_sha256` pins actual evidence/dependencies, including **a1-prep/a1-policy-check-receipt.json** (passed:true), reference_policy.py, NativeCap build/harness/binary/tokenizer/source pins, actual env/dispatch proofs and analysis/secondary source pins. Code seals should cover every new runtime source and all inherited imported dependencies (bench/client3/reconciliation/Jail/generate_envs3/native provenance code).

Final proof also includes `campaign_sha256 = protocol.campaign_digest(final_campaign)`: canonical sorted compact JSON excluding only cpu_receipt_sha256 to avoid a circular receipt hash. Then hash proof into campaign.cpu_receipt_sha256. Final campaign and CPU proof must be bound before the first request; new prefix GO contents exactly A1 SHA, only after actual CPU qualification. `require_collection_ready` verifies exact720-key manifest and all378 new request templates against original immutable sources every gate.

Systemd uses IgnoreSIGPIPE=no. cpu_gate.sh explicitly restores SIGPIPE default before exec of the unchanged CPU admission guard; shell pipelines settle producer and guard before command substitution returns. Both launch and turn gates complete before queue append, which remains exactly once inside controller. Request attempt markers are exclusive and written before transport; an interrupted/error attempt cannot be retried through this runtime.

## Qualification remaining

Parent must execute all checks above, including actual CPU reference/native count/environment/provenance and SIGPIPE evidence, finish current342 revalidation, and seal final campaign. This handoff does not claim those passed. No user re-approval is built into staging APIs; qualification evidence and the newly authorized A1 GO are the collection gate.
