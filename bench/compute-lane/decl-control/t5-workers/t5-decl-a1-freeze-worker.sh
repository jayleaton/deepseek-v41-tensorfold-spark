#!/bin/bash
set -uo pipefail
out=$T5_OUT;mkdir -p "$out/audit"
ps -o sid= -p $$ | tr -d ' ' > "$out/sid"
python3 -B - "$out/audit" <<'PY'
import pathlib,sys,json,hashlib,datetime
O=pathlib.Path(sys.argv[1]);T=pathlib.Path.home()/'work/imr-t5-s1/transport';B=pathlib.Path.home()/'work/stdbench/decl-control';sha=lambda b:hashlib.sha256(b).hexdigest();J=lambda p:json.loads(p.read_text());p=T/'T5-DECL-CONTROL-SPEC-A1.md';text=p.read_text();parent=T/'T5-DECL-CONTROL-SPEC.md';assert sha(parent.read_bytes())=='adb05167fff21469ee7251555cb243f1f109c4d8e3de237fdcaa59ada8eb55e6'
for word in ['552','same-length','wire body','dispatch','all120','MULTI-ROUND/INPUT-ACCOUNTING-UNKNOWN','strict accuracy']:assert word in text,word
proof=J(B/'cpu-proof.json');a=B/'prompt-arrays.jsonl';assert sha(a.read_bytes())==proof['prompt_arrays_sha256']=='7887963d2c6b3f4de450792ba5b89655b3ab80f64e9cc1f4a5300d12ac7187db';rows=[json.loads(l) for l in a.read_text().splitlines()];assert len(rows)==186
for r in rows:assert r['declared_off_ids']==r['declared_lane_ids']==r['original_builtin_lane_ids'] and r['held_off'] is False and r['held_lane'] is True
by={}
for r in J(B/'DRAFT-requests-NOT-QUALIFIED.json')['requests']:by.setdefault(tuple(r['key']),{})[r['arm']]=r
assert len(by)==184
for k,v in by.items():
 assert set(v)=={'plain','declared_off','lane'};d=v['declared_off'];l=v['lane'];assert d['wire']==l['wire'];assert sha(d['wire'].encode())==d['wire_sha256']==l['wire_sha256'];assert d['execution_policy']=={'TF_DSV41_COMPUTE_LANE':'0'} and l['execution_policy']=={'TF_DSV41_COMPUTE_LANE':'1'}
assert proof['dispatch_negative_control_qualified'] is False
result={'amendment_sha256':sha(p.read_bytes()),'parent_sha256':sha(parent.read_bytes()),'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cpu_equal_arrays':186,'draft_equal_wire_pairs':184,'arrays_sha256':sha(a.read_bytes()),'dispatch_gate':'PENDING_NOT_WAIVED','collection_qualified':False,'new_dataset_outcomes_read':False}
(O/'freeze.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
PY
rc=$?;echo "$rc" > "$out/job-exit";exit "$rc"
