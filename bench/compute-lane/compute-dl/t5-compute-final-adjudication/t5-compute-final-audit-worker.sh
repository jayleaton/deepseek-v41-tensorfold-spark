#!/bin/bash
set -uo pipefail
out=$T5_OUT; mkdir -p "$out/audit"
ps -o sid= -p $$ | tr -d ' ' > "$out/sid"
python3 -B - "$out/audit" <<'PY'
import pathlib,json,sys,hashlib,importlib.util,collections,random,re
from fractions import Fraction
B=pathlib.Path.home()/'work/stdbench/compute-dl';O=pathlib.Path(sys.argv[1]);J=lambda p:json.loads(p.read_text()); H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def mod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
sys.path.insert(0,str(B.parent/'phase2'));g=mod('strict',B/'grade.py');sel=mod('selector',B.parent/'phase2/grade_all.py')
tasks={t['id']:t for t in map(json.loads,(B/'dataset-v1/tasks-public.jsonl').open())};answers=J(B/'dataset-v1/answers-a.json');assert answers==J(B/'dataset-v1/answers-b.json')
audit=J(B/'a2-report/final-audit-v1/audit.json');files=audit['provenance']['input_file_sha256']; paths=[pathlib.Path(p) for p in files if p.endswith('responses.jsonl')];rows={};sources={};bad=[]
published={(r['id'],r['seed'],{'declared_off':'D','lane':'L','plain':'P'}.get(r['arm'],r['arm'])):r for r in map(json.loads,(B/'a2-results-v1/graded.jsonl').open())}
for p in paths:
 assert H(p)==files[str(p)],str(p)
 for r in map(json.loads,p.open()):
  if r.get('diagnostic') or r.get('suite')!='compute_dl':continue
  arm={'declared_off':'D','lane':'L','plain':'P'}.get(r['arm'],r['arm']);key=(r['id'],r['seed'],arm);assert key not in rows
  wire=r['actual_request_wire'];assert hashlib.sha256(wire.encode()).hexdigest()==r['actual_request_wire_sha256'];assert len(wire.encode())==r['actual_request_wire_bytes']
  gr=g.grade(r,tasks[r['id']],answers[r['id']]);pr=published[key];assert gr['grade']==pr['grade'] and gr['grader']==pr['grader']
  raw=json.loads(r['raw_response']);ids=raw['tensorfold']['token_ids'];assert len(ids)==r['native_accounting']['returned_tokens'];assert r['native_accounting']['hash_matches'] and r['native_accounting']['ids_equal_completion_plus_inserted']
  rows[key]={'grade':gr['grade'],'grader':gr['grader'],'content':r.get('content',''),'family':tasks[r['id']]['family'],'returned':len(ids),'wall':r['elapsed_seconds'],'finish':r['finish_reason'],'status':r['status'],'wire':wire,'calls':r['call_accounting'],'input':r['input_accounting'],'source':str(p)}
  sources[str(p)]=sources.get(str(p),0)+1
assert len(rows)==720
for i in tasks:
 for s in (101,102):
  assert rows[i,s,'D']['wire']==rows[i,s,'L']['wire']
  d=json.loads(rows[i,s,'D']['wire']);p=json.loads(rows[i,s,'P']['wire']);d.pop('tools',None);d.pop('tool_choice',None);assert d==p,(i,s,'P wire')
num=r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:\s*/\s*[+-]?\d+)?';pattern=r"\s*[A-Za-z\s'’]*\$?\s*("+num+r")[A-Za-z\s\-'’]*[.!]?\s*";blocked=r'\b(or|about|approximately|million|billion|thousand|hundred|percent)\b'
res=[];rescues=collections.Counter();wrappers=collections.Counter()
for (i,s,a),r in rows.items():
 if r['grade']=='correct':continue
 span=sel.gsm_final(r['content']);m=re.fullmatch(pattern,span or '');ok=False
 if r['family'] in ('integer','modular','dp','graph') and r['status']=='ok' and r['finish'] not in ('tool_calls','function_call','error','failed','cancelled') and m and not re.search(blocked,span,re.I):
  try:ok=Fraction(m[1].replace(',','').replace(' ',''))==answers[i]
  except (ValueError,ZeroDivisionError):pass
 if ok:rescues[a]+=1
 if a=='L':
  v=r['grader']['parsed_value'];match=False
  if isinstance(v,dict) and len(v)==1:match=next(iter(v.values()))==answers[i]
  wrappers[str(match)]+=1;res.append({'id':i,'seed':s,'family':r['family'],'reason':r['grader']['reason'],'content':r['content'],'wrapper_payload_equal_reference':match,'sensitivity_rescue':ok})
def agg(rs):return {'n':len(rs),'solved':sum(r['grade']=='correct' for r in rs),'returned':sum(r['returned'] for r in rs),'wall':sum(r['wall'] for r in rs),'caps':sum(r['finish']=='length' for r in rs)}
families=collections.defaultdict(dict)
for a in 'DLP':
 for f in ('overall',*g.FAMILIES):families[f][a]=agg([r for k,r in rows.items() if k[2]==a and (f=='overall' or r['family']==f)])
pairs=[(rows[i,s,'L'],rows[i,s,'P']) for i in sorted(tasks) for s in (101,102)];wins=sum(l['grade']=='correct' and p['grade']!='correct' for l,p in pairs);losses=sum(l['grade']!='correct' and p['grade']=='correct' for l,p in pairs)
complete=[(l,p) for l,p in pairs if l['finish']!='length' and p['finish']!='length'];capped=[(l,p) for l,p in pairs if l['finish']=='length' or p['finish']=='length']
ratios={}
for field in ('returned','wall'):
 ratios[field]={'all':sum(l[field] for l,p in pairs)/sum(p[field] for l,p in pairs),'complete_both':sum(l[field] for l,p in complete)/sum(p[field] for l,p in complete),'capped_signed_saving_share':sum(p[field]-l[field] for l,p in capped)/sum(p[field]-l[field] for l,p in pairs)}
rng=random.Random(20261105);draws=[[],[],[]];strata=[[i for i in sorted(tasks) if tasks[i]['family']==f] for f in g.FAMILIES]
for z in range(20000):
 ks=[i for group in strata for i in [group[rng.randrange(len(group))] for _ in range(len(group))]];lp=[(rows[i,s,'L'],rows[i,s,'P']) for i in ks for s in (101,102)];draws[0].append(sum((l['grade']=='correct')-(p['grade']=='correct') for l,p in lp)/240)
 for n,field in enumerate(('returned','wall'),1):draws[n].append(sum(l[field] for l,p in lp)/sum(p[field] for l,p in lp))
ci={name:[sorted(v)[500],sorted(v)[19499]] for name,v in zip(('solve_difference','returned_ratio','wall_ratio'),draws)}
D=collections.defaultdict(lambda:collections.Counter())
for (i,s,a),r in rows.items():
 if a=='D':
  c=r['calls'];d=D[r['family']+':'+str(s)];d['replies']+=1;d['attempted']+=c['attempted'];d['blocked']+=c['blocked_dispatches'];d['executed']+=c['successful_executions'];d['policy_terminated']+=r['status']=='policy_terminated';d['ordinary_final']+=r['status']=='ok';d['solved']+=r['grade']=='correct'
summary={'raw_source_counts':sources,'rows':len(rows),'strict_grades_equal':True,'wire_D_L_equal':240,'P_wire_equal_after_declaration_removal':240,'families':dict(families),'paired_wins':wins,'paired_losses':losses,'ci95_independent':ci,'sensitivity_rescues':dict(rescues),'L_residual_wrapper_payload_equal':dict(wrappers),'D_by_family_seed':dict(D),'L_calls':sum(r['calls']['successful_executions'] for k,r in rows.items() if k[2]=='L'),'L_distinct_executing':len({k[0] for k,r in rows.items() if k[2]=='L' and r['calls']['successful_executions']}),'ratios':ratios,'complete_both':{'pairs':len(complete),'L_solved':sum(l['grade']=='correct' for l,p in complete),'P_solved':sum(p['grade']=='correct' for l,p in complete)},'budget':audit['budget'],'P_batches':audit['P_batches'],'hashes':{'strict':H(B/'grade.py'),'selector':H(B.parent/'phase2/grade_all.py')}}
(O/'summary.json').write_text(json.dumps(summary,indent=2));(O/'residuals.json').write_text(json.dumps(res,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ('budget','P_batches','D_by_family_seed')},indent=2))
PY
rc=$?;echo "$rc" > "$out/job-exit";exit "$rc"
