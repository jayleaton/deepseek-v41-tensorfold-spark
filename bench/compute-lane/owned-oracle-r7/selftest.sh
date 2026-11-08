#!/usr/bin/env bash
# owned-oracle self-test on rank0-host/rank1-host (never dev-host): every selftest/receipts/bad-* must give an owned mismatch
# (owner engine or owned-form), no good-* may. The bad shapes are the known engine defects: T5's 3af553e61 list FAIL
# (min([2,3]) -> 4, sorted -> [4, 3], filtered len), lane's mod-operand write, the φ call-operand write, a wrong trace.
# bad-measure-*: the runner counts at extra.tensorfold (imr-measure receipts).
# bad-record-*: engine records (runtime.runner.cells) with a wrong span, value, cut or bytes not in the text, or a
# wrong divmod trace line (`a ÷ b = q remainder r`), a wrong trace line in the new wording, or (bad-record-unparsed) a
# value the oracle cannot read, which must fail the gate. good-record-trace-*: both trace wordings and a third.
# INLINE_MARK (` @engine`): good-record-mark is a recorded mark (engine-inline, ok); bad-record-mark a recorded mark on a
# wrong value; bad-record-mark-shifted a record whose mark is not at its offset (a copy elsewhere is never borrowed);
# good-mark-imitation* text-only marks (records empty, or none with owned counts): model imitation, never engine output.
# Length cap cuts (selftest/make_cut_receipts.py writes them): cut-record-* end a capped reply (finish length/cap) inside
# its last record, landed bytes a prefix of the whole write (S1 032339Z's one landed space, digits, a partial mark, a
# trace value, a trace header, fence output): that record must be `cut`, every other `ok`, gate rc 0. bad-record-capcut-*
# are the negative controls, each rc 1: a wrong landed digit or sign, a partial that is not the reply's last record, a
# partial with no cap (finish stop), a wrong trace or fence digit. cut-record-deadline: the server's deadline_s finish.
# live-record-*: a client close with only the stream's live records (`live: true`): never a cut, gate rc 3 INCONCLUSIVE.
# selftest/t5-r6/*.json: T5's r6 boundary; bad-record-span-*: a display or echo that is not the span (rc 1).
# selftest/t5-r5/*.json: T5's r5 matrix, each gated alone against its rc in selftest/t5-r5/EXPECTED.
# selftest/t5-r3/*.json: T5's r3 review fixtures (T5-ORACLE-R3-REVIEW.md), each gated alone against its rc below.
#   selftest.sh [OUT]       default OUT ./oracle-selftest
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/cpu-scope.sh"
oracle_require_scope || exit 2
out=${1:-$PWD/oracle-selftest}
# Missing synthetic cases must fail before starting any evaluation, including the EXPECTED matrices.
[[ -s "$here/selftest/fixtures.sha256" ]] || { echo 'selftest: missing fixture hash inventory' >&2; exit 1; }
(cd "$here/selftest" && sha256sum --quiet -c fixtures.sha256) || { echo 'selftest: fixture hashes failed' >&2; exit 1; }
for matrix in t5-r5 t5-r6; do
  [[ -s "$here/selftest/$matrix/EXPECTED" ]] || { echo "selftest: missing $matrix/EXPECTED" >&2; exit 1; }
  for pair in $(< "$here/selftest/$matrix/EXPECTED"); do
    n=${pair%:*}; want=${pair#*:}
    [[ -f "$here/selftest/$matrix/$n.json" && "$want" =~ ^[0-3]$ ]] || { echo "selftest: invalid $matrix fixture $n" >&2; exit 1; }
  done
done
compgen -G "$here/selftest/receipts/*.json" >/dev/null || { echo 'selftest: missing receipt fixtures' >&2; exit 1; }
bash "$here/run.sh" "$out" "$here/selftest/receipts" >/dev/null || { echo "selftest: run failed"; exit 1; }
fail=0
for f in "$here"/selftest/receipts/*.json; do
  n=$(basename "$f" .json)
  hits=$(grep -F "\"$f\"" "$out/mismatches.jsonl" | grep -cE '"owner": "(engine|owned-form)')
  case $n in
    bad-*) [ "$hits" -gt 0 ] && r=ok || { r=MISSED; fail=1; } ;;
    *) [ "$hits" = 0 ] && r=ok || { r=FALSE-FLAG; fail=1; } ;;
  esac
  echo "$n: $hits owned mismatches: $r"
done
# every good-record-* record must be judged ok (an unparsed one used to pass unseen), and the gate's rc must follow:
# 0 on the new-wording trace record, 1 on an owned record whose value cannot be read
for f in "$here"/selftest/receipts/good-record-*.json; do
  v=$(python3 -I -c 'import json,sys; print(" ".join(sorted({c["verdict"] for c in map(json.loads, open(sys.argv[1]))
      if c.get("source") == "record" and sys.argv[2] in c["paths"]})))' "$out/evals.jsonl" "$f")
  [ "$v" = ok ] && r=ok || { r="NOT-JUDGED ($v)"; fail=1; }
  echo "$(basename "$f" .json): record verdicts $v: $r"
done
# a capped reply's cut last record is `cut` (never ok, a whitespace-only prefix included), its other records ok
for f in "$here"/selftest/receipts/cut-record-*.json; do
  v=$(python3 -I -c 'import json,sys; r=[c for c in map(json.loads, open(sys.argv[1])) if c.get("source") == "record" and sys.argv[2] in c["paths"]]
r.sort(key=lambda c: c["at"]); print(" ".join(c["verdict"] for c in r))' "$out/evals.jsonl" "$f")
  case "$v" in cut|"ok cut"|"ok ok cut") r=ok ;; *) r="WRONG-CUT ($v)"; fail=1 ;; esac
  echo "$(basename "$f" .json): record verdicts $v: $r"
done
# a live set's correct partial last record is `inconclusive`, never `cut` (its text may run past the record)
for f in "$here"/selftest/receipts/live-record-*.json; do
  v=$(python3 -I -c 'import json,sys; r=[c for c in map(json.loads, open(sys.argv[1])) if c.get("source") == "record" and sys.argv[2] in c["paths"]]
r.sort(key=lambda c: c["at"]); print(" ".join(c["verdict"] for c in r))' "$out/evals.jsonl" "$f")
  case "$v" in inconclusive|"ok inconclusive"|"ok ok inconclusive") r=ok ;; *) r="WRONG-LIVE ($v)"; fail=1 ;; esac
  echo "$(basename "$f" .json): record verdicts $v: $r"
done
gates="good-record-trace-new:0 bad-record-unparsed:1 good-record-mark:0 bad-record-mark:1 good-mark-imitation-norecords:0"
for f in "$here"/selftest/receipts/cut-record-*.json; do gates="$gates $(basename "$f" .json):0"; done
for f in "$here"/selftest/receipts/bad-record-capcut-*.json; do gates="$gates $(basename "$f" .json):1"; done
for f in "$here"/selftest/receipts/live-record-*.json; do gates="$gates $(basename "$f" .json):3"; done
# T5's r6 OBJECT: a display or visible echo that is not the span's own computation fails, cut or complete
for f in "$here"/selftest/receipts/bad-record-span-*.json; do gates="$gates $(basename "$f" .json):1"; done
for f in "$here"/selftest/receipts/good-record-span-*.json; do gates="$gates $(basename "$f" .json):0"; done
for c in $gates; do
  n=${c%:*}; want=${c#*:}
  mkdir -p "$out-gate/$n/in" && cp "$here/selftest/receipts/$n.json" "$out-gate/$n/in/"
  bash "$here/gate.sh" "$out-gate/$n/out" "$out-gate/$n/in" > /dev/null 2>&1
  rc=$?
  [ "$rc" = "$want" ] && r=ok || { r=WRONG-RC; fail=1; }
  echo "gate on $n: rc $rc (want $want): $r"
done
# T5's r3 fixtures (all live sets): no correct prefix passes, stale ones included; a wrong landed byte or span still fails
for c in prefix-length:3 prefix-cap:3 prefix-timeout:3 prefix-cut:3 stop-prefix:3 whitespace-only:3 \
         stale-live-correct-tail:3 stale-live-wrong-digit:3 stale-live-wrong-sign:3 missing-new-record:3 fresh-live-wrong-digit:1 \
         shorter-span-length:1 shorter-span-cap:1 shorter-span-timeout:1 shorter-span-cut:1 \
         wrong-recorded-length:1 wrong-recorded-cap:1 wrong-recorded-timeout:1 wrong-recorded-cut:1; do
  n=${c%:*}; want=${c#*:}
  mkdir -p "$out-t5/$n/in" && cp "$here/selftest/t5-r3/$n.json" "$out-t5/$n/in/"
  bash "$here/gate.sh" "$out-t5/$n/out" "$out-t5/$n/in" > /dev/null 2>&1
  rc=$?
  [ "$rc" = "$want" ] && r=ok || { r=WRONG-RC; fail=1; }
  echo "T5 r3 $n: gate rc $rc (want $want): $r"
done
# T5's r5 matrix (T5-ORACLE-R5-REVIEW.md, selftest/t5-r5/EXPECTED): divmod cuts, multiply cuts, caps, live sets
for c in $(cat "$here/selftest/t5-r5/EXPECTED"); do
  n=${c%:*}; want=${c#*:}
  mkdir -p "$out-t5r5/$n/in" && cp "$here/selftest/t5-r5/$n.json" "$out-t5r5/$n/in/"
  bash "$here/gate.sh" "$out-t5r5/$n/out" "$out-t5r5/$n/in" > /dev/null 2>&1
  rc=$?
  [ "$rc" = "$want" ] && r=ok || { r=WRONG-RC; fail=1; }
  echo "T5 r5 $n: gate rc $rc (want $want): $r"
done
# T5's r6 boundary (T5-ORACLE-R6-REVIEW.md, selftest/t5-r6/EXPECTED): span divmod(1000,7) under another visible dividend
for c in $(cat "$here/selftest/t5-r6/EXPECTED"); do
  n=${c%:*}; want=${c#*:}
  mkdir -p "$out-t5r6/$n/in" && cp "$here/selftest/t5-r6/$n.json" "$out-t5r6/$n/in/"
  bash "$here/gate.sh" "$out-t5r6/$n/out" "$out-t5r6/$n/in" > /dev/null 2>&1
  rc=$?
  [ "$rc" = "$want" ] && r=ok || { r=WRONG-RC; fail=1; }
  echo "T5 r6 $n: gate rc $rc (want $want): $r"
done
# the arm table counts only `ok` records correct (a cut never), and shows the cut beside them
for c in cut-record-s1:2:1 cut-record-digits:2:1 cut-record-trace-head:0:1; do
  n=${c%%:*}; w=${c#*:}
  mkdir -p "$out-arms/$n/in" && cp "$here/selftest/receipts/$n.json" "$out-arms/$n/in/"
  ORACLE_GROUP='^(cut)-' bash "$here/gate.sh" "$out-arms/$n/out" "$out-arms/$n/in" > /dev/null 2>&1
  got=$(awk -F' *[|] *' '$2 == "cut" && $NF == "" {print $5 ":" $12}' "$out-arms/$n/out/report.md")
  [ "$got" = "$w" ] && r=ok || { r=WRONG-ARMS; fail=1; }
  echo "arm table on $n: correct:cut $got (want $w): $r"
done
# every literal ` @engine` gets an authority from the records alone
for c in good-record-mark:engine-inline good-mark-imitation:model-imitation good-mark-imitation-wrong:model-imitation \
         good-mark-imitation-norecords:model-imitation; do
  n=${c%:*}; want=${c#*:}
  got=$(python3 -I -c 'import json,sys; print(" ".join(sorted({m["authority"] for m in map(json.loads, open(sys.argv[1]))
      if sys.argv[2] in m["paths"]})))' "$out/marks.jsonl" "$here/selftest/receipts/$n.json")
  [ "$got" = "$want" ] && r=ok || { r=WRONG-AUTHORITY; fail=1; }
  echo "$n: @engine authority $got (want $want): $r"
done
[ $fail = 0 ] && echo "selftest PASS" || echo "selftest FAIL"
exit $fail
