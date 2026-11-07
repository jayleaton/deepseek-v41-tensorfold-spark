# Implementation ledger

Scope: source only; write only `stdbench/format-retest/analysis/`. No execution,
git, parent prep/runtime edits, children, reviewer, CPU checks or collection.

Read and SHA256-verified frozen inputs:

- `in-model-runner/T5-FORMAT-RETEST-SPEC.md`: `b47ede697eed3a38d244ad5ff7447690732bd12945ff4592554d45bc2d8527ec`.
- `stdbench/compute-dl/grade.py`: `1a2249a1b5e0b06f03a4300995d0c320e5a21630d261b3791501f7c83740fb9f`.

Standards discovery: checked filesystem ancestry from `/` through Projects,
`<devwork>/stdbench`, and searched the stdbench subtree for
AGENTS.md, LOCAL-AGENT.md, CONTRIBUTING*, pyproject.toml, setup.cfg, tox.ini, Makefile,
.ruff*, package.json and .pre-commit-config.yaml. None applicable were present.
The new format-retest directory did not exist at intake. This is the assigned
non-git artifact slice; no branch/worktree/commit/PR operations apply.

Relevant skill read: assigned `principle-prove-it-works` SKILL.md at revision 1,
content `4c947d4b7b09661e7be5f838442bcc82bf04e66908995e5b180381039a6eb0d1`.
Source inspected directly; actual verification remains parent-owned because
this task explicitly prohibits executing Python/tests here.

Check commands recorded in HANDOFF.md: AST parse of both new Python files and
`PYTHONDONTWRITEBYTECODE=1 python3 handwritten_checks.py --grader <pinned grade.py>`.
Both pending parent execution under node 2 GiB guard; no check-pass artifact is
fabricated. Existing strict grader was read only and remains unchanged.

Artifacts: reporter.py, handwritten_checks.py, RESULTS.md (NEEDS-T5), HANDOFF.md,
this ledger. Actual JSON/CSV/readable report are emitted only from supplied
real inputs at the parent's eventual invocation. No synthetic production results.

Source-only clipping follow-up: reread frozen T5 format spec, A1 section 1,
native truncation proof source and assigned prove-it-works skill. A1 expressly
permits independently qualified policy-clipped presentations and requires them
to remain distinct from complete values. Replaced the initial stronger blanket
ban with explicit policy-clip receipts/counts, separate unresolved cuts, exact
source/status/frame/landed hashes and independent full-reference metadata.
r7 scope now explicitly records pinned owned-runner oracle coverage, zero owned
cells with all optional runners OFF, without claiming native-code verification.
Added handwritten metadata controls and updated HANDOFF/RESULTS interpretation.
API unchanged: analyze(rows, tasks, answers, audit, grader) -> dict.
No Python, tests, AST checks or analysis were run; parent checks remain pending.
