#!/usr/bin/env bash
# rank0-host CPU dry run of one AB3 turn's configs (a scratch kit copy): every boot through window_boot.py against T4's
# fake server, as the window runs them (place off -> fake mode off, else inline), then analyze. The fake knows no env
# knob: this proves the kit (names, arms, guides, units, report), not the engine. Usage: ab3-dry.sh KITCOPY TURN (e.g. dry t01)
set -uo pipefail
D=<rank0-work>/ab3-20261104/$1
cd "$D" || exit 3
configs=ab3-$2-configs.json
run=$D/dry-$(date -u +%H%M%S)
mkdir -p "$run"
cp ab3-$2-units.jsonl "$run/tasks.jsonl"
python3 -I make_window_driver.py <rank0-work>/imr-te/imr-turn.py "$run/imr-measure.py" && grep -c "$D/window_boot.py" "$run/imr-measure.py" || echo DRIVER_FAILED
python3 -B <rank0-work>/imr-te/make-turn-driver.py <rank0-work>/r5/cand-e/efast2.py "$run/imr-turn.check.py" "$D/$configs" > /dev/null && echo DRIVER_KEYS_OK || echo DRIVER_KEYS_FAILED
n=$(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "$configs")
for ((i = 0; i < n; i++)); do
  cfg=$(python3 -c 'import json, sys; print(json.dumps(json.load(open(sys.argv[1]))[int(sys.argv[2])]))' "$configs" "$i")
  read -r name place < <(python3 -c 'import json, sys; c = json.loads(sys.argv[1]); print(c["name"], c["measure"]["place"])' "$cfg")
  mode=inline; [[ $place == off ]] && mode=off
  mkdir -p "$run/$name"
  python3 -I fake_server.py --port $((18291 + i)) --census census --tasks "$run/tasks.jsonl" --mode "$mode" \
    --rank-log "$run/$name/rank0.log" > "$run/$name/fake.log" 2>&1 &
  p=$!
  sleep 2
  T4_BASE=http://127.0.0.1:$((18291 + i)) timeout 1200 python3 -B window_boot.py "$run/$name" "$cfg" > "$run/$name/window-boot.log" 2>&1
  echo "$name=$?"
  kill "$p"
done
grep -h "Error\|Traceback" "$run"/*/fake.log "$run"/*/window-boot.log | head -5
for f in "$run"/*/*/arm.json; do python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1].split("/")[-2], len(d.get("finished_order", d.get("order", []))), {k: (v[:12] if isinstance(v, str) else v) for k, v in d.get("request_opts", {}).items()})' "$f"; done
ls "$run"/*/
python3 -I analyze.py --run "$run" --tasks "$run/tasks.jsonl" > "$run/analyze.log" 2>&1; echo "analyze=$?"
head -30 "$run/report.md"
echo "RUN $run"
