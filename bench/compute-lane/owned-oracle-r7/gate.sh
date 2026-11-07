#!/usr/bin/env bash
# owned-oracle as a hard gate (rank0-host/rank1-host only): run.sh on the receipts, then fail on any wrong engine write.
#   gate.sh OUT ROOT...    rc 0 pass, 1 a wrong write (WRONG lines + OUT/gate.json), 2 the oracle did not run,
#                          3 INCONCLUSIVE: no wrong write, but a receipt has only live records (a client close)
# Records (TF_DSV41_RUNNER_CELLS) are judged exactly: wrong-value, wrong-span, not-in-text, wrong-check (a trace line's
# opt-in check text is not the one computed here) or unparsed (an owned value the oracle cannot read, so unchecked) fail; cut/short/unknown are counted. cut: the reply ended on its length cap
# (a final receipt's finish length/cap, or the server's deadline_s finish `deadline`) inside its last record, whose landed bytes are a byte prefix of the whole write its span gives
# (computed here) and whose span is the whole visible expression; a cut is never a verified value, any landed byte
# off that prefix is wrong-value, and a span shorter than the reader's expression is wrong-span as when uncapped.
# INLINE_MARK ` @engine`: a mark is the engine's only at a record's exact span (a wrong marked write fails); a
# text-only mark is model imitation (counted in gate.json marks, never a failure).
# Receipts without records fall back to text mode, where an owned mismatch (owner engine or owned-form) fails.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/cpu-scope.sh"
oracle_require_scope || exit 2
out=$(realpath -m "$1"); shift
mkdir -p "$out"
echo "owned-oracle $(md5sum "$here/oracle.py" | cut -c1-8) on $*"
bash "$here/run.sh" "$out" "$@" > "$out.run.log" 2>&1 || { tail -3 "$out.run.log"; echo "owned-oracle gate ERROR: run failed"; exit 2; }
python3 -I "$here/oracle.py" gate "$out"
rc=$?
[ $rc -le 1 ] || [ $rc = 3 ] || { echo "owned-oracle gate ERROR: gate rc=$rc"; exit 2; }
exit $rc
