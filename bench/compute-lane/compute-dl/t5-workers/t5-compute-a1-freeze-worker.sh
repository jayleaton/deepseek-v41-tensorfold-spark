#!/bin/bash
set -uo pipefail
out=$T5_OUT;mkdir -p "$out/audit"
ps -o sid= -p $$ | tr -d ' ' > "$out/sid"
python3 -B - "$out/audit" <<'PY'
import pathlib,json,hashlib,sys,datetime,base64,collections
H=pathlib.Path.home();B=H/'work/stdbench/compute-dl';Q=B/'diagnosis-v1';T=H/'work/imr-t5-s1/transport';O=pathlib.Path(sys.argv[1]);J=lambda p:json.loads(p.read_text());L=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()];sha=lambda b:hashlib.sha256(b).hexdigest()
parent=B/'T5-COMPUTE-DL-SPEC.md';assert sha(parent.read_bytes())=='c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141'
p=T/'T5-COMPUTE-DL-SPEC-A1.md';text=p.read_text()
for term in ['720 scheduled replies','18 L +120 D','add240 P','outcome-informed','L versus P','post hoc for retained outcomes','14586.481','3413.519','uniformly','512','D answers']:assert term in text,term
case=J(Q/'retained-case.json');proof=J(Q/'truncation-proof.json');refs={r['call_id']:r for r in case['safety']['references']};calls=[c for c in case['calls'] if c['inserted'] and not refs[c['id']]['matches']];assert len(calls)==1;c=calls[0];ref=refs[c['id']]['result'];body=base64.b64decode(ref['normalized_body_base64'],validate=True);native=c['output'].encode();assert c['status']==ref['status']=='ok'
assert sha(body)==proof['full_reference_sha256'];assert sha(native)==proof['native_output_sha256'];assert sha(pathlib.Path(proof['production_source_path']).read_bytes())==proof['production_source_sha256'];assert sha((Q/'truncation-render.jsonl').read_bytes())==proof['native_arrays_sha256']
cfg=J(B/'stdbench-compute-dl-20261105-v1/campaign-v1.json');tok=pathlib.Path(cfg['tokenizer']);assert sha(tok.read_bytes())==cfg['tokenizer_sha256']==proof['tokenizer_sha256'];added={r['content']:r['id'] for r in J(tok)['added_tokens']};ids=L(Q/'truncation-render.jsonl')[0]['plain_ids'];assert ids.count(added['<｜User｜>'])==ids.count(added['<｜Assistant｜>'])==1;ids=ids[ids.index(added['<｜User｜>'])+1:ids.index(added['<｜Assistant｜>'])]
sys.path.insert(0,str(H/'work/stdbench/phase2'));from client import NativeDecoder
d=NativeDecoder(tok,cfg['tokenizer_sha256']);assert d.decode(ids)==body and len(ids)==554;n=512
while n:
 prefix=d.decode(ids[:n])
 try:prefix.decode('utf-8');break
 except UnicodeDecodeError:n-=1
expected=prefix+b'\n[output truncated at 512 tokens]' if n else b'[output truncated at 512 tokens]';assert expected==native and n==512
for i in range(3):
 rr=J(Q/('replay-dd01e167f94ef3e1-'+str(i)+'.json'));assert rr['normalized_body'].encode()==body and rr['status']=='ok'
# Check the retained grid and exact remaining-key/wire mask without regrading.
C=B/'stdbench-compute-dl-20261105-v1';F=C/'frozen-v1';expected_keys={};observed={};hashes={}
for boot in ['t01-D','t01-L','t02-L','t02-D']:
 bb=J(F/'boots'/(boot+'.json'));arm=boot[-1]
 for r in bb['requests']:expected_keys[r['id'],r['seed'],arm]=r
for file in (B/'runs').glob('stdbench-compute-dl-20261105-v1-*/*/responses.jsonl'):
 hashes[str(file)]=sha(file.read_bytes())
 for r in L(file):
  a={'declared_off':'D','lane':'L'}.get(r['arm'],r['arm']);key=(r['id'],r['seed'],a);assert key in expected_keys and key not in observed;assert r['actual_request_wire']==expected_keys[key]['wire'];observed[key]=r
assert len(expected_keys)==480 and len(observed)==342 and collections.Counter(k[2] for k in observed)=={'D':120,'L':222}
remaining=J(B/'remaining-collection.json');R={(r['id'],r['seed'],r['arm']):r for r in remaining['remaining_original_requests']};assert len(R)==138 and set(R)==set(expected_keys)-set(observed);assert collections.Counter(k[2] for k in R)=={'D':120,'L':18}
for k,r in R.items():assert r['wire_sha256']==sha(expected_keys[k]['wire'].encode())
assert ('compute-dl-v1-aggregation-015',101,'L') in observed and ('compute-dl-v1-aggregation-015',101,'L') not in R
budget=J(pathlib.Path(cfg['budget_ledger']));used=budget['held_seconds'];assert abs(used-3413.519)<.001 and budget['active'] is None
record={'amendment_sha256':sha(p.read_bytes()),'parent_sha256':sha(parent.read_bytes()),'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_observed':342,'retained_D':120,'retained_L':222,'remaining_D':120,'remaining_L':18,'new_P':240,'total_scheduled':720,'full_reference_tokens':len(ids),'retained_prefix_tokens':n,'known_clipped_result_exact_byte_match':True,'fresh_replay_bodies_equal':3,'remaining_manifest_sha256':sha((B/'remaining-collection.json').read_bytes()),'held_seconds_used':used,'held_seconds_remaining':18000-used,'new_uniform_verifier_gate':'PENDING_IMPLEMENTATION_AND_ALL_342_REVALIDATION','new_GPU_requests':0}
(O/'freeze.json').write_text(json.dumps(record,indent=2));(O/'original-source-hashes.json').write_text(json.dumps(hashes,indent=2));print(json.dumps(record,indent=2))
PY
rc=$?;echo "$rc" > "$out/job-exit";exit "$rc"
