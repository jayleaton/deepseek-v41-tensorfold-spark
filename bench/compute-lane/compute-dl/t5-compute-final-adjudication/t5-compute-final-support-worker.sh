#!/bin/bash
set -uo pipefail
out=$T5_OUT;mkdir -p "$out/audit";ps -o sid= -p $$ | tr -d ' ' > "$out/sid"
python3 -B - "$out/audit" <<'PY'
import json,pathlib,sys,hashlib,collections
B=pathlib.Path.home()/'work/stdbench/compute-dl';O=pathlib.Path(sys.argv[1]);J=lambda p:json.loads(p.read_text());H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();A=J(B/'a2-report/final-audit-v1/audit.json');checks=0
for p,h in A['provenance']['input_file_sha256'].items():assert H(pathlib.Path(p))==h,p;checks+=1
paths=[pathlib.Path(p) for p in A['provenance']['input_file_sha256'] if p.endswith('responses.jsonl')];P=[]
for p in paths:
 for r in map(json.loads,p.open()):
  if r.get('suite')=='compute_dl' and not r.get('diagnostic') and r['arm']=='plain':P.append((r['id'],r['seed']))
expected=[]
for b in ('t01-D','t02-D'):
 expected.extend((r['id'],r['seed']) for r in J(B/('stdbench-compute-dl-20261105-v1/frozen-v1/boots/'+b+'.json'))['requests'])
assert P==expected and len(P)==240 and len(set(P))==240
audited=list(map(json.loads,(B/'a2-report/final-audit-v1/audited-rows.jsonl').open()));present=[ref['presentation'] for r in audited for ref in r['safety']['references']];assert len(present)==243 and all(p['qualified'] for p in present);print('PRESENTATIONS',len(present))
res=J(pathlib.Path.home()/'work/imr-t5-s1/receipts/compute-final-audit4/audit/residuals.json');print('NONWRAPPER',json.dumps([r for r in res if not r['wrapper_payload_equal_reference']],indent=2))
answers=J(B/'dataset-v1/answers-a.json');assert answers['compute-dl-v1-modular-005']==997375194580461993;assert answers['compute-dl-v1-aggregation-010']=={'A':8364,'B':58273,'C':40126,'D':46271,'E':-23653,'F':71559}
rs=list(map(json.loads,(B/'a2-results-v1/graded.jsonl').open()));by={}
for r in rs:
 a={'lane':'L','plain':'P','declared_off':'D'}.get(r['arm'],r['arm']);by[r['id'],r['seed'],a]=r
weights=[]
for i in sorted({r['id'] for r in rs}):weights.append(sum((by[i,s,'L']['grade']=='correct')-(by[i,s,'P']['grade']=='correct') for s in (101,102)))
dist={0:1}
for w in weights:
 if not w:continue
 nxt=collections.Counter()
 for x,n in dist.items():nxt[x+w]+=n;nxt[x-w]+=n
 dist=dict(nxt)
p=sum(n for x,n in dist.items() if abs(x)>=abs(sum(weights)))/sum(dist.values())
clipped=[x for x in present if x.get('presentation_verified_clipped') or x.get('classification')=='presentation-verified clipped' or x.get('full_reference_tokens',0)>512]
print('CLIPPED',json.dumps(clipped,indent=2))
(O/'support.json').write_text(json.dumps({'provenance_files_hash_checked':checks,'P_exact_original_order':True,'P_prefix_suffix':[116,124],'exact_task_signflip_p':p,'presentations':len(present),'presentation_keys':list(present[0]),'clipped':clipped,'nonwrapper_residuals':[r for r in res if not r['wrapper_payload_equal_reference']]},indent=2))
PY
rc=$?;echo "$rc" > "$out/job-exit";exit "$rc"
