#!/usr/bin/env bash
# The single-pass window's served checks (quant repo results/zig-window-plan.md, step 3), one server on URL, in order:
#   1. replies: diverge.py ask --bench (prompts.txt, thinking on, + the bench's 8 prompts) -> replies.jsonl; for a label
#      other than prod: diverge.py compare against out/bench-prod/replies.jsonl (ids: return_token_ids), must be 16/16
#   2. speed: code and prose, T0 and T0.7, 1 and 4 streams, 2 reps; one bench_http.py run a cell, phases.json copied
#      after each (cumulative: phases/<n>-<cell>.json)
#   3. long prompts: 32K and 131K, one stream each (prompt tok/s, TTFT, decode, reply sha, tokens a round)
#   4. one 4-stream long-context check: a 131K prompt while 4 code streams run (no error, both finish)
#   5. a label other than prod: --compare tables against bench-prod and the verdict in verdict.txt: PASS only when every
#      expected cell (8 speed cells, each LONG length, the mix's two halves) was measured on both sides with no failed
#      stream, Zig >= Python on each (decode tok/s; prompt tok/s for long prompts), every T0 reply equal, replies 16/16
# A failed cell never stops the bench: it is logged, listed in failed.txt and the next cell runs. The server is checked
# after every part; a server that stopped answering is recorded in server-died (the cell after which it went).
# Run on rank 0 host from the kit dir:
#   ./zig-bench.sh prod      before the window, Python prod up (the reference replies and cells)
#   ./zig-bench.sh zig       in the window, the Zig server up with TF_DSV41_PHASES=1 TF_DSV41_PHASES_OUT=/state/phases.json
# Env: URL (http://localhost:8000), PHASES (the host path of the server's /state/phases.json; default $KIT/phases.json),
#   6. (PARTS with mixwin, not in the default set) one long stream decoding in the same windows as 3 code streams:
#      bench_http.py --mixwin MIX_LONG -> mixwin.json; the verdict compares its aggregate and the long stream's live tok/s
# WARM (1: an untimed warm-up a speed cell), REPS (2), LONG (32768,131072), MIX_LONG (131072), PARTS (replies speed long mix [mixwin]: a subset, e.g. the window's
# crash repro `PARTS=speed CELLS=code-t0-s4 REPS=1`), CELLS (speed cells by name; default all 8).
# Exit: 0 verdict PASS (or the prod label), 1 verdict FAIL, 2 no server at the start, 3 the server died during the bench.
set -u
LABEL=${1:?label: prod | zig}
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
URL=${URL:-http://localhost:8000}; PH=${PHASES:-$KIT/phases.json}
PARTS=${PARTS:-replies speed long mix}; LONG=${LONG:-32768,131072}; MIX_LONG=${MIX_LONG:-131072}
OUT=$KIT/out/bench-$LABEL; PROD=$KIT/out/bench-prod
mkdir -p "$OUT/cells" "$OUT/phases"; rm -f "$OUT/failed.txt" "$OUT/server-died"
B="python3 $KIT/bench_http.py --url $URL --label $LABEL"
D="python3 $KIT/tools/zig/dsv41_diverge/diverge.py"
log() { echo "[zig-bench $(date +%T)] $*" | tee -a "$OUT/bench.log"; }
n=0
phases() { n=$((n + 1)); [[ -f "$PH" ]] && cp "$PH" "$OUT/phases/$(printf %02d $n)-$1.json"; }
alive() { # a 1-token reply, not just /v1/models: rank 0's HTTP outlives a dead rank 1 (every request then fails)
    curl -sf -m 60 "$URL/v1/chat/completions" -H 'content-type: application/json' \
        -d '{"model":"dsv41","messages":[{"role":"user","content":"Say ok."}],"max_tokens":1,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
        | grep -q '"choices"'
}
failed() { echo "$1" >> "$OUT/failed.txt"; log "FAILED: $1"; }
after() { # $1 = what just ran: a dead server is recorded once (the first part it did not survive)
    alive && return 0
    [[ -f "$OUT/server-died" ]] || { echo "$1" > "$OUT/server-died"; log "SERVER DOWN after $1"; }
    return 1
}
part() { [[ " $PARTS " == *" $1 "* ]]; }
alive || { log "no server on $URL"; exit 2; }
t0=$(date +%s)

if part replies; then
    log "1. replies (diverge.py ask --bench)"
    $D ask --url "$URL" --prompts "$KIT/prompts.txt" --bench --out "$OUT/replies.jsonl" >> "$OUT/bench.log" 2>&1 || failed replies
    after replies; phases replies
    if [[ "$LABEL" != prod ]]; then
        if [[ -f "$PROD/replies.jsonl" ]]; then
            $D compare "$PROD/replies.jsonl" "$OUT/replies.jsonl" > "$OUT/compare.txt" 2>&1
            log "replies vs prod: $(tail -1 "$OUT/compare.txt")"
        else log "no $PROD/replies.jsonl: run ./zig-bench.sh prod before the window"; fi
    fi
fi

if part speed; then
    log "2. speed cells"
    for work in code prose; do for temp in 0 0.7; do for s in 1 4; do
        c="$work-t$temp-s$s"
        [[ -n "${CELLS:-}" && " $CELLS " != *" $c "* ]] && continue
        rm -f "$OUT/cells/$c.json"
        # WARM (1): one untimed request set at the cell's own temperature and streams first, so rep 0 is not the
        # engine's cold first pass at that shape
        [[ "${WARM:-1}" == 1 ]] && $B --work "$work" --temps "$temp" --streams "$s" --reps 1 --max-tokens 64 > /dev/null 2>> "$OUT/bench.log"
        $B --work "$work" --temps "$temp" --streams "$s" --reps "${REPS:-2}" --out "$OUT/cells/$c.json" > /dev/null 2>> "$OUT/bench.log" \
            || failed "$c"
        after "$c"; phases "$c"
        log "  $c: $(python3 -c "import json,sys;print([(c['mean_tok_s'] if c['streams']==1 else c['aggregate_tok_s'], c.get('failed_streams', 0)) for c in json.load(open(sys.argv[1]))['cells']])" "$OUT/cells/$c.json" 2>/dev/null)"
    done; done; done
    python3 - "$OUT/cells" "$OUT/cells.json" "$LABEL" <<'PY'
import glob, json, sys
cells = [c for f in sorted(glob.glob(sys.argv[1] + "/*.json")) for c in json.load(open(f))["cells"]]
json.dump({"label": sys.argv[3], "cells": cells}, open(sys.argv[2], "w"), indent=1)
PY
fi

if part long; then
    log "3. long prompts ($LONG)"
    rm -f "$OUT/long.json"
    $B --long "$LONG" --out "$OUT/long.json" > /dev/null 2>> "$OUT/bench.log" || failed long
    after long; phases long
fi

if part mix; then
    log "4. 4-stream long-context check ($MIX_LONG + 4 code streams)"
    rm -f "$OUT/mix-long.json" "$OUT/mix-short.json"
    $B --long "$MIX_LONG" --out "$OUT/mix-long.json" > /dev/null 2>> "$OUT/bench.log" &
    pl=$!; sleep 20
    $B --work code --temps 0 --streams 4 --reps 1 --out "$OUT/mix-short.json" > /dev/null 2>> "$OUT/bench.log"; rs=$?
    wait "$pl"; rl=$?
    log "  long rc $rl, 4 streams rc $rs"
    [[ $rl == 0 ]] || failed mix-long; [[ $rs == 0 ]] || failed mix-short
    after mix; phases mix
fi

if part mixwin; then
    log "4b. mixed windows ($MIX_LONG-token stream decoding beside 3 code streams)"
    rm -f "$OUT/mixwin.json"
    $B --mixwin "$MIX_LONG" --out "$OUT/mixwin.json" > /dev/null 2>> "$OUT/bench.log" || failed mixwin
    log "  mixwin: $(python3 -c "import json,sys;c=json.load(open(sys.argv[1]))['cells'][0];print('aggregate', c['aggregate_tok_s'], 'long live', c['long_live_tok_s'], 'span', c['live_span_s'], 's, per stream', c['per_stream'])" "$OUT/mixwin.json" 2>/dev/null)"
    after mixwin; phases mixwin
fi

rc=0
if [[ "$LABEL" != prod ]]; then
    log "5. against prod"
    { python3 "$KIT/bench_http.py" --compare "$PROD/cells.json" "$OUT/cells.json"; echo
      python3 "$KIT/bench_http.py" --compare "$PROD/long.json" "$OUT/long.json"; } > "$OUT/compare.md" 2>&1
    python3 - "$PROD" "$OUT" "$LONG" "$MIX_LONG" "$PARTS" > "$OUT/verdict.txt" <<'PY'
import json, os, sys
p, z, longs, mix_long, parts = sys.argv[1:]
parts = parts.split()

def load(d, f):
    try:
        return json.load(open(f"{d}/{f}"))["cells"]
    except (OSError, ValueError, KeyError):
        return None

def rate(c):
    return c["mean_tok_s"] if c["streams"] == 1 else (c["aggregate_tok_s"] or c["mean_tok_s"])

def best(cells, val):
    """(work, temp, streams) -> the best rep's value, over reps without a failed stream; a key whose every rep failed: 'failed'."""
    out = {}
    for c in cells or []:
        k = (c["work"], c["temp"], c["streams"])
        if c.get("failed_streams"):
            out.setdefault(k, "failed")
            continue
        v = val(c)
        if v is None:
            out.setdefault(k, "failed")
            continue
        prev = out.get(k)
        out[k] = v if not isinstance(prev, float) else max(prev, v)
    return out

# every cell the bench must measure: (file, key, value, label)
expect = []
if "speed" in parts:
    expect += [("cells.json", (w, t, s), rate, "decode") for w in ("code", "prose") for t in (0.0, 0.7) for s in (1, 4)]
if "long" in parts:
    for n in longs.split(","):
        k = (f"long-{n}", 0.0, 1)
        expect += [("long.json", k, lambda c: c["mean_tok_s"], "decode"), ("long.json", k, lambda c: c.get("prompt_tok_s"), "prompt")]
if "mix" in parts:
    expect += [("mix-short.json", ("code", 0.0, 4), rate, "decode"),
               ("mix-long.json", (f"long-{mix_long}", 0.0, 1), lambda c: c["mean_tok_s"], "decode"),
               ("mix-long.json", (f"long-{mix_long}", 0.0, 1), lambda c: c.get("prompt_tok_s"), "prompt")]
if "mixwin" in parts:
    k = (f"mixwin-{mix_long}", 0.0, 4)
    expect += [("mixwin.json", k, lambda c: c.get("aggregate_tok_s"), "decode (all live)"),
               ("mixwin.json", k, lambda c: c.get("long_live_tok_s"), "long stream decode (live)")]
rows, worse, missing = [], [], []
for f, k, val, what in expect:
    a, b = best(load(p, f), val).get(k), best(load(z, f), val).get(k)
    name = f"{f[:-5]} {k[0]} T{k[1]} x{k[2]} {what}"
    if not isinstance(a, float) or not isinstance(b, float):
        missing.append(name)
        rows.append(f"{name}: prod {a if a is not None else 'MISSING'} zig {b if b is not None else 'MISSING'}  FAIL (not measured)")
        continue
    ok = b >= a
    rows.append(f"{name}: prod {a:.1f} zig {b:.1f}  {'ok' if ok else 'SLOWER'}")
    if not ok:
        worse.append(name)
diff = []
if "speed" in parts:
    def shas(cells):
        return {(c["work"], c["temp"], c["streams"], c.get("rep")): c.get("sha") for c in cells or [] if c["temp"] == 0}
    sa, sb = shas(load(p, "cells.json")), shas(load(z, "cells.json"))
    diff = [k for k in sa if k in sb and sa[k] != sb[k]]
if "mixwin" in parts:
    ma, mb = load(p, "mixwin.json") or [], load(z, "mixwin.json") or []
    if ma and mb and ma[0].get("sha") != mb[0].get("sha"):
        diff.append(("mixwin", mix_long))
cmp = "not run"
if "replies" in parts:
    cmp = open(f"{z}/compare.txt").read().strip().splitlines()[-1] if os.path.exists(f"{z}/compare.txt") else "no compare"
died = open(f"{z}/server-died").read().strip() if os.path.exists(f"{z}/server-died") else ""
failed = open(f"{z}/failed.txt").read().split() if os.path.exists(f"{z}/failed.txt") else []
print("\n".join(rows))
print(f"replies: {cmp}")
print(f"T0 cells with a differing reply: {len(diff)} {diff[:8]}")
print(f"cells not measured: {len(missing)} {missing}")
print(f"failed parts / cells: {len(failed)} {failed}")
print(f"server died after: {died or '-'}")
print(f"cells where Zig is slower: {len(worse)} {worse}")
ok = not worse and not diff and not missing and not failed and not died and (cmp.startswith("16/16") or "replies" not in parts)
full = all(x in parts for x in ("replies", "speed", "long", "mix"))
print("VERDICT: PASS - Zig >= Python on every cell, replies equal, no failure" if ok and full else
      "VERDICT: PASS (partial run: " + " ".join(parts) + ")" if ok else
      "VERDICT: FAIL - " + "; ".join(x for x in (
          f"{len(missing)} cell(s) not measured" if missing else "", f"{len(worse)} slower" if worse else "",
          f"{len(diff)} T0 replies differ" if diff else "", f"server died after {died}" if died else "",
          f"replies {cmp}" if "replies" in parts and not cmp.startswith("16/16") else "") if x))
sys.exit(0 if ok else 1)
PY
    rc=$?
    log "$(tail -1 "$OUT/verdict.txt")"
fi
log "done in $(( ($(date +%s) - t0) / 60 )) min: $OUT"
[[ -f "$OUT/server-died" ]] && exit 3
exit $rc
