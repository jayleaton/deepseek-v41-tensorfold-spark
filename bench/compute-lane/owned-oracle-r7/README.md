# owned-oracle

Checks every value the engine may have written into a reply against an independent evaluation. It reads only receipt
text. It runs no engine code, no tests and no builds. Evaluation is CPython 3.12 in a sandbox.

**Rule: every experiment child runs owned-oracle on its run's receipts before reporting "owned correct".** Put the
`report.md` path and the owned-mismatch count (owner `engine` or `owned-form`) in the report. Any owned mismatch is
a wrong engine write until shown otherwise. Do not report "owned correct" with one open.

## Running it (<rank0-host>/<rank1-host> only, never <dev-host>)

The evaluation is compute, so it runs on a Spark:

```
scp -r tools/owned-oracle/{oracle.py,run.sh,selftest.sh,selftest} <rank0-user>@<rank0-ip>:work/owned-oracle/tool/
ssh <rank0-user>@<rank0-ip> 'bash ~/work/owned-oracle/tool/selftest.sh'                       # must print selftest PASS
ssh <rank0-user>@<rank0-ip> 'bash ~/work/owned-oracle/tool/run.sh ~/work/owned-oracle/out/<run> <receipt dir>...'
scp <rank0-user>@<rank0-ip>:work/owned-oracle/out/<run>/{report.md,mismatches.jsonl} <evidence dir>/
```

Pass any number of receipt directories or files; JSON files are found recursively. Runner counts are read from `tensorfold`, `response.tensorfold` or `extra.tensorfold` (imr-measure receipts). `run.sh` does the following:

- **Waits.** It waits out `~/work/queue/QUIET`, a held T4 window and an `imr-measure` at the head of ORDER before each step.
- **Extract and report.** These steps run in a `systemd-run --user --scope` with MemoryMax=2G, MemorySwapMax=0, nice 19 and ionice c3.
- **Evaluate.** This step runs in `docker run --network none --read-only` (2 GiB, 1 CPU, cpuset 0-4, pids 256, nice 19). The container is `docker pause`d while QUIET or a T4 window is held. Fence and REPL code runs there under `prlimit --as=1G --cpu=5 python3 -I` in an empty temp directory.
- **Image.** It uses `tensorfold-dsv41-imr:r-6cdaf1bed8ae` (python 3.12.3; only its python is used). Override with `ORACLE_IMAGE`.

Speed: 674 replies (19 roots, today's whole corpus) take 18 s; TE's 57-reply arms take 10 s.

Outputs in OUT:

| file | contents |
|---|---|
| `report.md` | tally by form/verdict/owner, then every mismatch, owned first |
| `mismatches.jsonl` | one record per mismatch: `paths` (all receipt copies of that text), `field` (reasoning/content), `offset` (character offset of the written value in that field), `visible`, `visible_value`, `parsed_span`, `written`, `code`, `cpython_stdout`, `reappears_later`, `in_final_content`, `questioned`, runner `counts` |
| `cells.jsonl`, `evals.jsonl`, `replies.jsonl` | every reconstructed cell, its verdict, each unique reply once (text plus its runner counts and records) |

## Engine records (exact mode)

An engine built from `dsv41/imr-receipt` writes one record per owned cell into the receipt, at
`tensorfold.runner.cells`. This is on by default; set `TF_DSV41_RUNNER_CELLS=0` to turn it off.

```json
{"cell":3,"form":"inline","route":"graph","status":"ok","lang":"python","span":"10^18","value":" 1000000000000000000",
 "field":"reasoning","at":15837,"token":4211,"ids":7,"landed":7,"drafted":0,"forced":7,"path":"held"}
```

| field | meaning |
|---|---|
| `span` | the exact bytes the scanner parsed and ran; over 512 bytes it is `span_sha256` + `span_len`, matched to the reply's own copy |
| `value` | the bytes of the result tokens that actually committed; `value_hex` if they end mid-character |
| `at`, `token` | byte offset in `field` and reply token index where the write starts |
| `ids`, `landed` | result tokens the engine gave, and how many committed in order (fewer means the reply's budget cut them) |
| `drafted`, `forced` | `drafted` landed in the trigger's own commit because the model's verify produced them (sync_graph only); `forced` were forced rows |
| `path` | `held` (forced after a hold) or `sync` (settled at the trigger) |

When a receipt has `cells` (even `[]`), the oracle judges each record:

| verdict | meaning |
|---|---|
| `ok` | the record is right |
| `wrong-value` | the engine's own span has another value. Inline and trace are checked by the exact evaluator, fence and REPL by CPython. |
| `wrong-span` | the value is right for the span the engine parsed, but the reader's expression around it is larger. This is the `mod 10^18` / `φ(2^18)` bug class, now named directly. |
| `not-in-text` | the landed bytes are not in the reply |
| `cut` | the reply ended on its length cap (finish `length`/`cap`), or on the server's own deadline (`deadline_s`, finish `deadline`), in a final receipt, inside its **last** record, and the landed bytes are a byte prefix of the whole write the record's own span gives here (exact evaluator for inline/trace values, CPython for fence/REPL output; a divmod trace line `a ÷ b = q remainder r` and its check rebuilt from the operands, T5's r5 OBJECT). A trace value always comes from the record's span alone: a display or visible echo that computes to anything else (divmod: other operands) is `wrong-span`, cut or complete, and a check that contradicts its line is never approved (T5's r6 OBJECT). Counted, never a verified value, never a failure. A whitespace-only prefix is a cut, not `ok`. Any landed byte off the prefix is `wrong-value`, and an inline span shorter than the reader's visible expression (`2*3` of `2*3*4 =`) is `wrong-span`, checked before the prefix (T5's review); a partial record that is not the capped reply's last, or whose whole write cannot be computed here, is judged as any other (an unreadable one is `unparsed` and fails). |
| `inconclusive` | a client close (measure.py `timeout`/`cut`) that kept only the stream's live records (`live: true`, no final receipt), and the last record's landed bytes are a correct prefix. Text and records are separate stream events, so the client's text may run past its newest record (T5's r3 OBJECT): never a cut, never verified. Any receipt with only live records makes the gate **INCONCLUSIVE** (rc 3) when nothing failed, even if every record is `ok` (a newer record may be missing). Wrong landed bytes are still `wrong-value`. |
| `wrong-check` | a trace record's text after its `]` is not the opt-in check (`runner_trace_check`, S2 R7) computed here from the line's own display and value: ` check: L ≡ v (mod 9), L ≡ v (mod 1000)` (each literal as its residue, mod 1000 keeping 3 digits for 3+ digit numbers; divmod `q*B + r ≡ A`), or a check where the line takes none (more than 4 literals, a power past 4, unary minus, hex, …). Fails the gate. No text after `]` is fine (the check is opt-in), and the value is judged as before. A capped last record cut inside its check is `cut` when its landed bytes are a prefix of the whole checked line. |
| `short` | every byte of the value landed and it is right, but fewer result tokens landed than the engine gave, with no length cap (listed as an engine mismatch, counted, not failed; this was `cut` before 2026-11-03) |

Every other value in that reply is the model's (owner `model (records: not owned)`). Receipts without `cells` fall back
to text reconstruction (below), so old runs still check.

## Hard gate (gate.sh)

`gate.sh OUT ROOT...` runs `run.sh`, then `oracle.py gate OUT`. Its exit code is 0 for a pass, 1 for a wrong engine write and 2 when the oracle did not run. It prints one `WRONG` line per write and a summary line, and writes `OUT/gate.json`.

- **Records:** a `wrong-value`, `wrong-span`, `not-in-text` or `unparsed` record fails the gate. An `unparsed` record is an owned write whose landed value the oracle cannot read, so it would otherwise go unchecked (until 2026-11-03 it passed silently: S2's trace4 run on the earlier tool had 239 such trace records). `cut` and `unknown` records are counted but do not fail it. With ORACLE_GROUP, the arm table's `correct` in records mode is the count of records judged `ok`; `cut`, `unknown` and `unparsed` have their own columns, and a missing record is never counted (until 2026-11-03 it was owned less wrong, so a cut counted as correct).
- **INLINE_MARK (` @engine`, T5 9455a17e7).** A record's value keeps its mark; the oracle strips the exact
  terminal ` @engine` only to judge the number, so a wrong marked write is `wrong-value` and fails. A mark is the
  engine's only where a record covers its exact bytes (`marks.jsonl` authority `engine-inline`, or `-partial` for a cut).
  A recorded mark not at its own offset is `not-in-text`; a copy elsewhere is never borrowed. Any other literal
  ` @engine`, with records empty or absent (even with nonzero owned counts), is `model-imitation`: listed and counted
  in gate.json `marks`, never engine output and never a failure. Run a marked arm with `TF_DSV41_RUNNER_CELLS=1`;
  without records every mark is classed as imitation and nothing is authenticated. Built from T5's candidate
  (1a3d18aa) with the Lead's rule for text-only marks; T5's rc-2 "inconclusive" state and its trace-label change are
  not taken, so text-mode trace receipts gate as before.
- **Trace records** are read in any wording: `[exact: S = V]`, `[exact, computed by the engine: S = V]`, or any other `[exact…]` line, split at its last ` = `.
- **Receipts without records:** any owned mismatch fails the gate, whether its owner is `engine` (trace) or `owned-form`.
- **Where it runs:** TE's `imr-build-turn.sh` runs it on the image smoke after `imr-turn rc=0`, still under both locks. A failure ends the turn with rc=20. The `WRONG` lines and the summary go to turn2.log, and the full output goes to `logs/oracle-<run>.log`.

## One report per experiment window (window-report.sh)

```
ssh <rank0-user>@<rank0-ip> 'bash ~/work/owned-oracle/tool/window-report.sh <receipts dir> "a=none,rg=@d_mirror,rs=@trust" [OUT]'
```

This runs on <rank0-host>/<rank1-host> only. It waits out QUIET, a held T4 window and an imr-measure at the head of ORDER, then runs in a 2 GiB scope at nice 19 / ionice c3. Its exit code is the gate's: 0 pass, 1 a wrong engine write, 2 the tool did not run. It writes `OUT/window-report.md` and `.json`, by default to `~/work/owned-oracle/out/window-<dir>`. The report has:

- **owned-oracle gate** (`gate.sh`): judges every record, counts UNKNOWN, and lists `WRONG` lines.
- **Per arm** (the receipt's `arm`, else `variant`, else its directory; `gates/` is skipped):
  - replies, reasoning and completion tokens (mean and total);
  - solve, from the window's `grades.jsonl` (T4's analyze.py), else the receipt's grade (S1), else T5's reference answers for the t6 smoke prompts;
  - cap hits (finish `length`);
  - T5's doubt detector: denial and mental-only. `detector.json` is a byte copy of t5-doubt-evidence/20261103/detector.json, md5 144e6a36, and the detection logic is T5's `detect()`.
- **Owned cells per arm:** record verdicts and recheck.py's re-verification cost (inline and trace cells; fence and REPL outputs are not followed).
- **Per guide:** the same totals grouped by the map's guide labels.

Validation (runs/window-w5-validate/): on T4 window 5, the report reproduces T5's published W5 numbers exactly for every arm: denial a 0, d 0, r 1, rg 14, rs 3; mental-only rg 7, rs 2; solve 43/43/43/43/44 of 45; reasoning and completion token totals and means; 0 cap hits. It also reproduces recheck.py's W5 figures (3252 / 2968 / 1665 tokens).

### Post-turn hook (window-hook.sh)

Experiment turns start the hook detached, after G is restored. It holds no GPU lock:

```
setsid nohup bash <rank0-work>/owned-oracle/tool/window-hook.sh "<receipts dir>" auto > /dev/null 2>&1 < /dev/null &
```

On <rank0-host> it waits out QUIET. It then copies the receipts and the tool to <rank1-host> (root, `<root-home>/work/t2-window/<run>`) and runs window-report.sh there with the image's python in the oracle container. <rank0-host>'s QUIET is mirrored to <rank1-host> every 5 s, through `QUIET_FILE`, so a running report pauses. The report comes back as `<receipts dir>/window-report/{window-report.md,window-report.json,gate.json,RC}`, with a log at `<receipts dir>/window-report.log`, and the <rank1-host> copy is removed. A second hook for the same run exits at once. The map `auto` labels each arm from its runner_* request fields (the guide text by its sha256 prefix); pass `arm=label,...` for names.

## What it checks (text reconstruction)

| form | span and written value | independent value |
|---|---|---|
| inline | text left of `=` and the number right after it | the longest valid expression ending at `=`, with an exact rational evaluator (`+ - * / % mod ^ ! ()`, hex, LaTeX, gcd/lcm/isqrt/comb/perm/factorial/φ…, `a^b mod m` via pow) |
| trace | `[exact: SPAN = VALUE]`, or the default since ce89c50a7 `[exact, computed by the engine: SPAN = VALUE]` (any `[exact…:` opener) | the same evaluator on SPAN |
| repl | `>>> code` and the lines after it | CPython `single` mode, replaying the session's earlier lines |
| fence | a closed ```` ```python ```` fence, then an optional result line, then the output (fenced or not) | CPython running the fence; ok if CPython's stdout lines are a prefix of the written lines (prose may follow) |

Verdicts:

- `ok`: the written value is right.
- `boundary`: the written value equals a strictly shorter computed span glued inside the visible expression, either right after `mod`/`%` or the argument of a call. This is the engine-cell signature: `3^400 mod 10^18 = 1000000000000000000` (the cell took `10^18`) and `φ(2^18)= 262144` (the cell took `(2^18)`).
- `value`: any other wrong value.
- `no-cell`: a bare number left of `=`.
- `unknown` / `unparsed`: not decidable from text.

Owner of a mismatch:

| owner | meaning |
|---|---|
| `engine` | trace form: only the engine writes `[exact:` or `[exact, computed by the engine:` |
| `owned-form` | the receipt's runner owned cells of this form, and nothing marks it as the model's: **treat as an engine write** |
| `unexplained (model or engine)` | an inline `value` in a reply that owned inline cells: read it |
| `model? (chained line)`, `model? (=> step)` | inline refuses a line with an earlier `=`; `=>` steps are the model's partial terms |
| `model (grouped digits)` | the engine writes plain digits, so a comma-grouped value is the model's |
| `model (form owned 0)` | that form owned nothing in this reply |

For each mismatch, "did the model use it" is reported three ways: the value reappears later in the reply, it is in the final content, and the next text questions it (`? Wait`, `Actually`, …).

## Limits

- **Receipts before `dsv41/imr-receipt` hold no spans.** For those, the oracle reconstructs each cell from the reply text and the per-form owned counts. It cannot tell which cell in a reply the engine owned, only that the form owned some. Engines with records (above) remove this limit.
- **Fence/REPL notebooks.** A record's REPL span is replayed after the reply's earlier `>>>` lines. A fence that needs names from an earlier cell is `unknown`, because the notebook is not replayed.
- **`unexplained` inline values need a reader.** In today's 34 they are all model text: prose-split spans (`first 3 digits + 12*3 = 39`), position ranges (`12-20 = 2 3 4 …`), and the model's own unfinished arithmetic (`823,543^2 = 678,?`).
- **Undecidable cells.** An `unknown` or `unparsed` cell is not checked. Fences that read input, files or the network fail in the sandbox.

## Validation, 2026-11-02 (validation-20261102/)

`today/`: every receipt root of 2026-11-01/02 on <rank0-host> `~/work/dsv41-bench/out/` (imr-*, imr-measure-*, trace-*, trace2-*, s1-guide*) plus `~/work/imr-s1-guide3/live`. That is 674 unique reply texts and 26,509 cells, in 18 s.

- **21 owned-form boundaries, matching STATUS-S3's tables text for text.**
  - **12 `mod` operand writes:** the 10^18 or 5^18 after `mod`, written as the value of `3^N mod 10^18`. The receipts are imr-guide 184605Z, graph-c4-settle (= graph-c4-A), 044547Z smoke-golden, 071801Z smoke-golden ×2, trace-on, trace2 guide ×3, trace2 exact, s1-guide3 ×2.
  - **9 `φ` call-operand writes:** `φ(2^18)` ×4, `φ(5^18)` ×4, `φ(10^9)` ×1, in graph-c4-settle, 044547Z, 071801Z and trace2 exact.
  - The model questioned or rewrote every one; none reached the final content.
- **0 engine trace mismatches.** All 115 decidable trace lines are ok, including `(0x0123456789ABCDEF + 1)*2^64`.
- **0 owned fence/repl mismatches.** The 13 fence/repl mismatches are the model's own text in replies where that form owned nothing (e.g. `sum([0.1]*10)` written as 0.9999999999999999, but CPython 3.12 gives 1.0).

`te-r0-0c1c83b1a668/`: TE's arms `imr-20261102T115935Z/{graph-c4-off,graph-c4-sync,handoff-rg}`, 57 replies, in 10 s.

- No owned mismatches, and no list-cell mismatches. The 4 `unexplained` inline values are model prose (`positions 12-20 = 2 3 4 …`, `12*3 = 39 digits`).
- One fence and three REPL cells were owned. All are ok (`a*b`, `314159 * 271828`, `2**100 - 3**50`).
- Every list-using block in these arms is model text (form owned 0). The list shapes that T5 withheld 3af553e61 for never ran live here.

`selftest/`: synthetic receipts with the known defects, plus `bad-record-*` engine records (wrong span, wrong value, cut, bytes not in the text) beside `good-record-*` twins. These are T5's 3af553e61 FAIL shapes (`min([2, 3])` → 4, `sorted([2, 3])` → `[4, 3]`, filtered `len` → 0), the mod-operand and φ writes, and a wrong trace line, each beside a correct twin. `selftest.sh` must flag every `bad-*` and no `good-*`. On 2026-11-02 it gave PASS, 8/8, and PASS 16/16 with the record cases. Since 2026-11-03 it also requires every `good-record-*` record to be judged `ok`, and checks gate rc 0 on a new-wording trace record and rc 1 on `bad-record-unparsed`. PASS 26/26; with the earlier oracle.py (6fb8d979) the same selftest FAILs on those 3 checks.

## W9 report (`w9-report.sh`, `w9_report.py`)

`w9-report.sh RUN BLOCKS OUT [r2|r5] [ONLY]` on <rank0-host>/<rank1-host> only. It gives one verdict per 45-task block and one pooled verdict over T4's `w9-block-NN-manifest.json` (from w9_kit.py). The receipts are `RUN/run-NN/<label>/<id>.json`.

- **Oracle pin:** the pinned oracle's gate.sh judges each block's 270 receipts. That oracle is r2 by sha256 (`ORACLES` in w9_report.py). r5 is refused (rc 2) until T5 AGREEs; moving to r5 means setting its `agreed`. Each block's `block-NN/oracle/gate.json` is the file `w8_gate.py --oracle` reads.
- **Per-arm fields** (rule names from T5-W8-DECISION.md, carried by T5-W9-SPEC.md):
  - `engine_bytes`: rule 1 and the BLOCK list.
  - `cut_gate`: rule 5. Cut records must be at most 1% of owned records, never rounded, and must end at a native end: length, cap or deadline.
  - Owned writes without a record: per-form runner counters above that form's record count.
  - `provenance`: the final receipt checked against measure.py's `live_cells`. A live record must be its final record's prefix. A live-only reply (a client close) is inconclusive.
  - `imitations`: model-written `[exact` labels, per reply and per label (values are not checked here).
- **Verdicts:**
  - FAIL = BLOCK / investigate: a wrong engine write, an unreconciled live record, or a cut with no native end.
  - INCONCLUSIVE: missing replies or blocks, live-only replies, owned writes without a record, unknown/inconclusive records, or an oracle that did not run.
  - The cut share is reported beside the verdict, as a "keep opt-in" condition.
- **rc:** 0 PASS, 1 FAIL, 3 INCONCLUSIVE, 2 error.
- **Fixture:** `selftest/w9/make_w9_fixture.py` builds 10 blocks × 270 replies from real receipts, in three variants (clean, cutrec, faults); expected results are in its docstring. A 2,700-reply report takes about 1 minute on <rank0-host>.
