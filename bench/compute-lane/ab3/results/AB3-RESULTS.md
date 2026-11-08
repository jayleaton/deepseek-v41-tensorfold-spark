# AB3 results: **DOES NOT MEET** (T5-AB3-SPEC.md, sha256 81b34c1a…)

The analysis ran once, at 2026-11-04T12:33:33Z, after the last turn. It used the sealed `ab3_analysis.py` (sha256 26d7bec8…, equal to the
freeze). Output: `ab3-final.json` (sha256 0a3230f8…).

## Chronology

| Step | When (UTC) | What |
|---|---|---|
| Freeze | 09:18:35 | `AB3-FREEZE.json` (sha256 93417d27…), committed as `86dda37c` with the 60 unit IDs |
| First request | ~09:25 | plain boot ready after 319 s |
| Turns | 09:19:44 to 12:21:12 | 6 turns, held 25, 18, 20.5, 30, 20 and 30.5 min, all rc 0 |
| Tool re-runs | 12:33 | 444/444 |
| Analysis | 12:33:33 | |

- The fence was on and off once per turn. r7 `tool/oracle.py` was 35f2bdf0… at queue time and after every turn.
- Image `r-3bf584af1c54` was label-checked on both nodes in every turn.
- No outcome was computed before the analysis. Per turn, only safety was posted (STATUS-TE).

## Gates

| Gate | Result | Evidence |
|---|---|---|
| Primary token (full vs lane, 180 pairs) | **FAIL** | −13.17%, CI95 [−26.08, +0.98]. The point estimate passes (≤ −10%), but the upper bound is not ≤ −5%. Per seed: −10.83 / −14.99 / −13.75 (all ≤ 0). |
| Solves (operational non-inferiority) | PASS | L = 2 (div_big-030 s103, div_big-054 s103), G = 4, no repeated-loss task. Paired diff CI95 [−0.022, +0.050] (task cluster). |
| Censoring | PASS | Censored: full 0, lane 0 (plain 7). Complete-both n = 180: −13.17% [−26.08, +0.98]. No-cut −13.17%. Censored share of savings 0 / 0. |
| Safety | **FAIL** | r7 t02 FAIL: 1 not-in-text record (below). Every other turn passes, and wrong-value is 0 everywhere. Cuts 0 (full 0/40, only 0/48 owned occurrences). Tool results: 444/444 ok, re-run independently in the engine-faithful jail (27/27 arm-runs PASS). |
| Provenance / integrity | PASS | 720/720 receipts, all id hashes match, 0 integrity problems. Lane = only = plain + 306 prompt tokens in every pair (the namespace was never removed). The guide adds exactly 112. |

**The safety failure** is tool-full `w9-bignum-017`, seed 101 (`t02-window-report.json`, receipt in `../ab3-safety/`):
- A trace cell landed right after `</think>`, in content: `[exact, computed by the engine: 2^66 = 73786976294838206464]\n`, all 23 ids.
- The model then opened a tool call. The server's content projection trims the whitespace before the tool-call block, so the client's `content` lacks that `\n`.
- The value is right and the native ids carry the bytes, but the record's span is not in the visible text. The spec says unplaced provenance is not a pass and allows no outcome-driven checker change, so the gate fails on this run.

## All four arms (180 task/seed units each; T = all returned native ids)

| Arm | T | Model-sampled | Inserted results | Solved | Censored / deadlines | Wall sum s (median) | Tool calls (executors) | Tool exec s |
|---|---|---|---|---|---|---|---|---|
| plain | 301,009 | 301,009 | 0 | 170 | 7 / 7 | 8,108 (19.2) | 0 | 0 |
| lane | 65,499 | 62,094 | 3,405 | 172 | 0 / 0 | 2,274 (8.6) | 144 (sandbox) | 23.2 |
| tool-full | 56,872 | 53,060 | 3,812 | 174 | 0 / 0 | 2,048 (7.0) | 159 (S3 runner 89, sandbox fallback 70) | 12.2 |
| tool-only | 65,558 | 62,224 | 3,334 | 172 | 0 / 0 | 2,300 (8.4) | 141 (sandbox) | 22.7 |

Exploratory contrasts, on the same draws (none can rescue the primary):

| Contrast | Change | CI95 |
|---|---|---|
| full vs plain | −81.11% | [−84.66, −76.65] |
| lane vs plain | −78.24% | [−83.33, −71.75] |
| only vs lane | +0.09% | [−1.72, +2.19] |
| full vs only | −13.25% | [−26.04, +0.80] |

Tool-full vs lane, per family (T_full / T_lane):

| Family | T_full / T_lane | Change |
|---|---|---|
| add30 | 3,532 / 3,819 | −7.5% |
| algo | 8,129 / 7,196 | **+13.0%** |
| bignum | 1,734 / 2,771 | −37.4% |
| div_big | 2,819 / 3,382 | −16.6% |
| mbpp | 35,224 / 40,699 | −13.5% |
| mul30 | 1,173 / 1,750 | −33.0% |
| mul_big | 4,261 / 5,882 | −27.6% |

Robustness:
- The median task paired change is −6.8%.
- Leave-one-task-out ranges from −15.78% to −6.64%. The most influential task is mbpp-580: dropping it gives −6.64%.

## What this says (built configuration only)

- **The pilot's 25% gain over the lane did not replicate at the precision the spec demands.** The point estimate is −13%, but the CI reaches +1%, driven by task variance in mbpp and algo, where code is. algo goes the other way (+13%).
- **Tool-only is the same as the lane** (+0.09% [−1.7, +2.2]): adding the in-model runner, with the tool declared and no guide or S3, changes nothing.
  - Everything tool-full gains comes from the joint guide + S3 addition, which can't be split here.
- All three tool arms cut tokens by about 80% against plain and remove plain's 7 deadlines, with solves at or above plain.
- **The prefix cache was on.** Cached prompt tokens: lane and only 47,632 each, full 61,856, plain 5,440. The shared order makes cache use equal per arm-pair. Cached tokens are prompt-side only, so T is unaffected. This is disclosed, not a gate.
- **No claim is made** about latency, compute, answer-commit, the namespace alone, the guide alone, or S3 alone.

## Files

- `ab3-final.json` and `ab3-final.md`
- `toolcheck-verify.json` and `toolcheck-verify.md` (per-call verdicts and executors)
- `ledger.json` and `ledger.log`
- `t0N-window-report.json` (r7 per turn)
- `ab3_turnsafety.py`
- The receipts are under `<rank0-host>:<rank0-work>/dsv41-bench/out/imr-measure-20261104T{091946,095232,101747,104533,112309,115045}Z`.
