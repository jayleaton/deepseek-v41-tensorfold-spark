#!/bin/bash
set -uo pipefail
out=$T5_OUT;mkdir -p "$out/audit"
ps -o sid= -p $$ | tr -d ' ' > "$out/sid"
python3 -B - "$out/audit" <<'PY'
import pathlib,hashlib,json,sys,datetime,math,decimal
O=pathlib.Path(sys.argv[1]);P=pathlib.Path.home()/'work/imr-t5-s1/transport/T5-COMPUTE-DL-SPEC.md';b=P.read_bytes();s=b.decode()
for term in ['480 scheduled replies','No plain arm','20261105','1000 collisions','30 distinct task IDs','4h25m','INSUFFICIENT EXECUTION EXPOSURE','INCOMPLETE','unknown full processed-input totals']:
 assert term in s,term
assert 6*20*2*2==480 and 6*10*2*2==240 and math.comb(18,9)==48620
with decimal.localcontext() as c:
 c.prec=256;c.traps[decimal.Inexact]=True;c.traps[decimal.Rounded]=True
 a=decimal.Decimal(10**79+987654321);d=decimal.Decimal(10**28+123)
 q=a//d;r=a%d;assert int(q)==int(a)//int(d) and int(r)==int(a)%int(d);assert a==q*d+r
record={'spec_sha256':hashlib.sha256(b).hexdigest(),'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bytes':len(b),'families':6,'tasks':120,'seeds':[101,102],'arms':['D','L'],'scheduled_replies':480,'scheduled_turns':2,'gpu_budget_hours':5,'grid_max_paths':48620,'decimal_reference_control':'PASS','task_manifest_generated':False,'disjointness_gate':'PENDING_CPU_PREPARATION','new_outcomes_read':False}
(O/'freeze.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
PY
rc=$?;echo "$rc" > "$out/job-exit";exit "$rc"
