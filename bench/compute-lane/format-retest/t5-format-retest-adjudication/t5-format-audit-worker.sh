#!/bin/bash
set -uo pipefail
out=$T5_OUT;mkdir -p "$out/audit";ps -o sid= -p $$ | tr -d ' ' > "$out/sid"
python3 -B - "$out/audit" <<'PY'
import pathlib,json,sys,hashlib,importlib.util,random,collections
B=pathlib.Path.home()/'work/stdbench/format-retest';O=pathlib.Path(sys.argv[1]);J=lambda p:json.loads(p.read_text());H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
A=J(B/'audit-v2/raw-audit.json');assert A['qualified'] and not A['gaps'];checked=0
for p,h in A['provenance']['input_file_sha256'].items():assert H(pathlib.Path(p))==h,p;checked+=1
for d in A['outputs'].values():assert H(pathlib.Path(d['path']))==d['sha256']
G=B.parent/'compute-dl/grade.py';assert H(G)=='1a2249a1b5e0b06f03a4300995d0c320e5a21630d261b3791501f7c83740fb9f';s=importlib.util.spec_from_file_location('pinned',G);g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
tasks={t['id']:t for t in map(json.loads,(B/'dataset-v1/tasks.jsonl').open())};ans=J(B/'dataset-v1/reference-a.json');assert ans==J(B/'dataset-v1/reference-b.json');audited={(r['id'],r['seed'],r['study_arm']):r for r in map(json.loads,(B/'audit-v2/rows.jsonl').open())}
rows={};raw_orders={}
for p in A['provenance']['input_file_sha256']:
 if not p.endswith('responses.jsonl'):continue
 order=[]
 for r in map(json.loads,pathlib.Path(p).open()):
  if r.get('diagnostic') or r.get('suite')!='compute_dl':continue
  k=(r['id'],r['seed'],r['study_arm']);assert k not in rows;order.append(k);ar=audited[k]
  for field in ('content','finish_reason','status','elapsed_seconds','actual_request_wire','raw_response'):assert r[field]==ar[field],field
  gr=g.grade(r,tasks[r['id']],ans[r['id']]);ids=json.loads(r['raw_response'])['tensorfold']['token_ids'];assert len(ids)==r['native_accounting']['returned_tokens'];assert r['native_accounting']['hash_matches'] and r['native_accounting']['ids_equal_completion_plus_inserted']
  wire=r['actual_request_wire'];assert hashlib.sha256(wire.encode()).hexdigest()==r['actual_request_wire_sha256'];body=json.loads(wire)
  row={'id':k[0],'seed':k[1],'arm':k[2],'family':tasks[k[0]]['family'],'correct':gr['grade']=='correct','format_failure':gr['grader']['reason'] not in ('exact_match','wrong_answer'),'wrong_value':gr['grader']['reason']=='wrong_answer','returned':len(ids),'wall':r['elapsed_seconds'],'cap':r['finish_reason']=='length','reason':gr['grader']['reason'],'body':body,'calls':r['call_accounting']}
  rows[k]=row
 raw_orders[p]=order
assert len(rows)==360 and set(rows)==set(audited)
newscalar='Final answer: only the integer, written as a bare JSON number (for example -42): no object, no key, no quotes, no code fence.'
news={'division':'Final answer: only the JSON array [q,r] itself, with nothing around it and no code fence.','aggregation':'Final answer: only the JSON object with keys A through F and integer values, on its own line, with no code fence.'}
for i,t in tasks.items():
 for seed in (101,102):
  for prefix in ('L','P'):
   if (i,seed,prefix+'-old') not in rows:continue
   old=rows[i,seed,prefix+'-old']['body'];new=rows[i,seed,prefix+'-new']['body'];expected=t['prompt'].split('Final answer format:')[0]+news.get(t['family'],newscalar)
   assert old['messages']==[{'role':'user','content':t['prompt']}];assert new['messages']==[{'role':'user','content':expected}]
   a=dict(old);b=dict(new);a.pop('messages');b.pop('messages');assert a==b
  if (i,seed,'P-new') in rows:
   for suffix in ('old','new'):
    a=dict(rows[i,seed,'L-'+suffix]['body']);a.pop('tools',None);a.pop('tool_choice',None);assert a==rows[i,seed,'P-'+suffix]['body']
# Independently reconstruct frozen interleaving.
for path,order in raw_orders.items():
 ids=sorted({k[0] for k in order});pairs=[(i,s) for i in ids for s in (101,102)];random.Random(20261107).shuffle(pairs);half=len(pairs)//2;directions=[False]*half+[True]*half;random.Random(20261108).shuffle(directions);prefix=order[0][2][0];expected=[]
 for (i,s),reverse in zip(pairs,directions):expected.extend((i,s,prefix+'-'+suf) for suf in (('new','old') if reverse else ('old','new')))
 assert order==expected,('order',path)
def stats(rs):return {'n':len(rs),'solved':sum(r['correct'] for r in rs),'format_failures':sum(r['format_failure'] for r in rs),'wrong_values':sum(r['wrong_value'] for r in rs),'caps':sum(r['cap'] for r in rs),'returned_sum':sum(r['returned'] for r in rs),'returned_mean':sum(r['returned'] for r in rs)/len(rs),'wall_sum':sum(r['wall'] for r in rs),'wall_mean':sum(r['wall'] for r in rs)/len(rs)}
byarm={a:stats([r for r in rows.values() if r['arm']==a]) for a in ('L-old','L-new','P-old','P-new')};fams=sorted({t['family'] for t in tasks.values()});byfam={f:{a:stats([r for r in rows.values() if r['arm']==a and r['family']==f]) for a in byarm if any(r['arm']==a and r['family']==f for r in rows.values())} for f in fams};pf=['aggregation','dp','graph'];matched={a:stats([r for r in rows.values() if r['arm']==a and r['family'] in pf]) for a in ('L-new','P-new')}
loss={};gains={}
for prefix in ('L','P'):
 keys=[(i,s) for i in tasks for s in (101,102) if (i,s,prefix+'-old') in rows];loss[prefix]=sum(rows[i,s,prefix+'-old']['correct'] and not rows[i,s,prefix+'-new']['correct'] for i,s in keys);gains[prefix]=sum(not rows[i,s,prefix+'-old']['correct'] and rows[i,s,prefix+'-new']['correct'] for i,s in keys)
rng=random.Random(20261107);vals=[[],[],[]];groups=[[i for i in sorted(tasks) if tasks[i]['family']==f] for f in fams]
for _ in range(20000):
 sampled=[group[rng.randrange(10)] for group in groups for z in range(10)];vals[0].append(sum(rows[i,s,'L-new']['format_failure']-rows[i,s,'L-old']['format_failure'] for i in sampled for s in (101,102))/120);sub=[i for i in sampled if tasks[i]['family'] in pf]
 for pos,field in ((1,'returned'),(2,'wall')):vals[pos].append(sum(rows[i,s,'L-new'][field] for i in sub for s in (101,102))/sum(rows[i,s,'P-new'][field] for i in sub for s in (101,102)))
ci={k:[sorted(v)[500],sorted(v)[19499]] for k,v in zip(('format_difference','matched_returned_ratio','matched_wall_ratio'),vals)}
present=list(map(json.loads,(B/'audit-v2/presentations.jsonl').open()));assert len(present)==245 and all(not p['clipped'] and p['landed_body_sha256']==p['full_reference_sha256'] and p['provenance']['presentation']['qualified'] for p in present)
summary={'rows':len(rows),'source_hashes_checked':checked,'raw_audited_fields_equal':True,'old_new_only_format_change':True,'raw_order_matches_frozen_randomization':True,'byarm':byarm,'byfamily':byfam,'matched_new':matched,'paired_losses':loss,'paired_gains':gains,'ci95':ci,'matched_returned_ratio':matched['L-new']['returned_sum']/matched['P-new']['returned_sum'],'matched_wall_ratio':matched['L-new']['wall_sum']/matched['P-new']['wall_sum'],'presentations':len(present),'dataset_presentations':sum(not p['diagnostic'] for p in present),'clipped':0,'budget':A['budget']}
(O/'audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ('byfamily','budget')},indent=2))
PY
rc=$?;echo "$rc" > "$out/job-exit";exit "$rc"
