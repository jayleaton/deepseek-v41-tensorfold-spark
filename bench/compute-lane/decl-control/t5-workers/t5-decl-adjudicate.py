
import sys as _archive_sys
import pathlib,json,sys,hashlib,random,collections,re,datetime
from fractions import Fraction

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)
H=pathlib.Path.home();B=H/'work/stdbench/decl-control';F=B/'frozen-a1-v1';O=pathlib.Path(sys.argv[1]);O.mkdir(parents=True,exist_ok=True);J=lambda p:json.loads(p.read_text());L=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()];sha=lambda b:hashlib.sha256(b).hexdigest()
summary={};inputs={};read=lambda p:p.read_bytes()
for f,h in [('T5-DECL-CONTROL-SPEC.md','adb05167fff21469ee7251555cb243f1f109c4d8e3de237fdcaa59ada8eb55e6'),('T5-DECL-CONTROL-SPEC-A1.md','e7308621c645aef4402924a1b47c1a52e05baabe00239f915011f8431bbd0e89')]:assert sha(read(B/f))==h
cpu=J(B/'cpu-final-a1-v2/verification.json');seal=J(F/'manifest.json');assert sha(read(F/'manifest.json'))==cpu['freeze_sha256']=='785a8f5b9625427d50fc1e18b26bbdd40198ddd55ca18d3a635741edaab94d21'
for old,h in cpu['code_sha256'].items():
 p=H/'work/stdbench'/old.split('/stdbench/')[1];assert sha(read(p))==h,old
summary['presealed_code_files']=len(cpu['code_sha256']);prompts={tuple(r['key']):r for r in L(B/'prompt-arrays.jsonl')};assert len(prompts)==186
for r in prompts.values():assert r['declared_off_ids']==r['declared_lane_ids']==r['original_builtin_lane_ids']
D={};diagnostics=[];specs={};reqpairs={};snapshots=0;first=[]
for entry in seal['boots']:
 bp=F/entry['file'];assert sha(read(bp))==entry['sha256'];boot=J(bp);name=boot['boot'];directory=B/'runs'/('stdbench-decl-a1-20261105-v1-'+name.split('-')[0])/name
 for filename,templates in [('responses.jsonl',boot['requests']),('diagnostics.jsonl',boot['diagnostics'])]:
  p=directory/filename;inputs[str(p)]=sha(read(p));rows=L(p);assert len(rows)==len(templates)
  for r,t in zip(rows,templates):
   k=(r['suite'],r['id'],r['seed']);arm=r['arm'];assert list(k)==t['key'] and arm==boot['arm'];wire=r['actual_request_wire'];assert wire==t['wire'] and sha(wire.encode())==t['wire_sha256']==r['actual_request_wire_sha256'];assert len(wire.encode())==r['actual_request_wire_bytes'];request=json.loads(wire);assert r['request']==request
   pr=prompts[k];n=len(pr[{'plain':'plain_ids','declared_off':'declared_off_ids','lane':'declared_lane_ids'}[arm]]);raw=json.loads(r['raw_response']);tf=raw['tensorfold'];assert tf['prompt']==raw['usage']['prompt_tokens']==r['usage']['prompt_tokens']==n
   for frame in r['sse_frames']:
    if 'usage' in frame and frame['usage'] is not None:assert frame['usage']['prompt_tokens']==n
   ids=tf['token_ids'];assert sha(','.join(map(str,ids)).encode())==tf['token_sha256'];assert len(ids)==r['native_accounting']['returned_tokens']==raw['usage']['completion_tokens']+tf.get('compute',{}).get('inserted_tokens',0)
   for field,target in [('content','content'),('reasoning_content','reasoning')]:
    s=''.join(c.get('delta',{}).get(field,'') or '' for frame in r['sse_frames'] for c in frame.get('choices',[]));assert s==r[target],(k,arm,field)
   assert request['seed']==k[2] and request['max_tokens']==8192 and request['deadline_s']==270 and request['temperature']==1 and request['top_p']==.95 and request['top_k']==20 and request['reasoning_effort']=='high'
   if filename=='responses.jsonl':assert r['status']=='ok', (k,arm,r['status'])
   reqpairs.setdefault((name.split('-')[0] if filename=='diagnostics.jsonl' else '',k),{})[arm]=wire
   if filename=='diagnostics.jsonl':diagnostics.append(r)
   else:
    assert (k,arm) not in D;D[k,arm]=r;specs[k]=t
    assert not r['live_compute_reports'] and not r['external_tool_calls'] and not tf.get('compute',{}).get('calls',[]);assert not r['attempts']['calls'] and not r['attempts']['malformed'];assert r['native_accounting']['inserted_tokens']==0
 for rank in [0,1]:
  rr=[J(p) for p in (B/'evidence-a1-v1'/('rank'+str(rank))/name).glob('*.json')];rr=[r for r in rr if r.get('command')=='snapshot' and r.get('rank')==rank and r.get('dry_run') is False];assert len(rr)==1
  rr=rr[0];m=B/'stdbench-decl-a1-20261105-v1/boots'/name/'manifest.json';mm=J(m);assert rr['manifest_sha256']==sha(read(m));assert rr['present'] and rr['image_id']==mm['arms']['D']['ranks'][rank]['image_id'];env=rr['native_process']['env'];ex=mm['phase2']['ranks'][rank]['effective_env']
  for key,val in ex.items():
   if key.startswith(('TF_','GLM53_TF','NCCL')):assert env.get(key)==val,(name,rank,key)
  assert env['TF_DSV41_COMPUTE_LANE']==str(int(boot['arm']=='lane'));snapshots+=1
assert len(D)==552 and len(diagnostics)==18 and len(reqpairs)==190 and snapshots==18
for v in reqpairs.values():assert v['declared_off']==v['lane']
G={}
for r in L(B/'finalized-a1-v1/all-final.jsonl'):
 k=(r['suite'],r['id'],r['seed']);a=r['arm'];q=D[k,a]
 for f in ['raw_response','content','reasoning','request','actual_request_wire','elapsed_seconds','finish_reason','native_accounting','status']:assert r[f]==q[f],(k,a,f)
 assert (k,a) not in G;G[k,a]=r
assert len(G)==552
# The output hashes, not just lengths or aggregate grades, must match per paired reply.
ident=collections.Counter();old={}
for r in L(H/'work/stdbench/phase2/post-v1/nonmath-graded.jsonl')+L(H/'work/stdbench/phase2/post-v1/math-reviewed.jsonl'):old[(r['suite'],r['id'],r['seed']),r['arm']]=r
for k in specs:
 d,l,p=(G[k,a] for a in ['declared_off','lane','plain'])
 for f in ['content','reasoning','grade','finish_reason']:assert d[f]==l[f],(k,f)
 assert json.loads(d['raw_response'])['tensorfold']['token_ids']==json.loads(l['raw_response'])['tensorfold']['token_ids'];ident[k[0]]+=1
 for a in ['plain','lane']:
  r=G[k,a];q=old[k,a]
  for f in ['content','reasoning','grade','finish_reason']:assert r[f]==q[f],(k,a,'prior',f)
  assert json.loads(r['raw_response'])['tensorfold']['token_ids']==json.loads(q['raw_response'])['tensorfold']['token_ids']
summary.update({'replies':552,'matched_declared_wire_pairs':190,'cpu_arrays':186,'actual_rank_snapshots':18,'D_L_native_and_text_identity':dict(ident),'prior_plain_lane_native_and_text_identity':368,'diagnostics':{a:{'replies':sum(r['arm']==a for r in diagnostics),'executed_calls':sum(len(r['live_compute_reports']) for r in diagnostics if r['arm']==a)} for a in ['plain','declared_off','lane']}})
for r in diagnostics:
 for c in r['live_compute_reports']:assert r['arm']=='lane' and c['status']=='ok' and c['inserted'] and c['output'].strip()==('323' if r['id']=='python-17x19' else '55')
# Primary and registered extraction sensitivity independently from the frozen final span.
num=r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:\s*/\s*[+-]?\d+)?';pattern=r"\s*[A-Za-z\s'’]*\$?\s*("+num+r")[A-Za-z\s\-'’]*[.!]?\s*";blocked=r'\b(or|about|approximately|million|billion|thousand|hundred|percent)\b'
for (k,a),r in G.items():
 if k[0]!='gsm8k':continue
 text=r['grading']['extracted_final'];m=re.fullmatch(pattern,text or '');v=None
 if m and not re.search(blocked,text,re.I):
  try:v=Fraction(m[1].replace(',','').replace(' ',''))
  except (ValueError,ZeroDivisionError):pass
 ok=r['grade']=='correct' or(v is not None and v==Fraction(r['grading']['reference_fraction']));assert ok==(r['gsm_sensitivity']['grade']=='correct')
def exact(ds):
 states={0:1}
 for d in ds:
  if d:
   nxt=collections.defaultdict(int)
   for x,n in states.items():nxt[x-abs(d)]+=n;nxt[x+abs(d)]+=n
   states=nxt
 return sum(n for x,n in states.items() if abs(x)>=abs(sum(ds)))/sum(states.values())
summary['suites']={}
for s in ['gsm8k','math500','humaneval']:
 ids=sorted({k[1] for k in specs if k[0]==s});out={'arms':{},'contrasts':{}}
 for a in ['plain','declared_off','lane']:
  rr=[G[(s,i,z),a] for i in ids for z in [101,102]];out['arms'][a]={'solves':sum(r['grade']=='correct' for r in rr),'returned':sum(r['native_accounting']['returned_tokens'] for r in rr),'prompt':sum(json.loads(r['raw_response'])['usage']['prompt_tokens'] for r in rr),'wall':sum(r['elapsed_seconds'] for r in rr),'caps':sum(r['finish_reason']=='length' for r in rr)}
  if s=='gsm8k':out['arms'][a]['sensitivity_solves']=sum(r['gsm_sensitivity']['grade']=='correct' for r in rr)
 for name,(a,b) in {'D-P':('plain','declared_off'),'L-D':('declared_off','lane'),'L-P':('plain','lane')}.items():
  pairs=[(G[(s,i,z),a],G[(s,i,z),b]) for i in ids for z in [101,102]];correct=lambda r:int(r['grade']=='correct');ds=[sum(correct(G[(s,i,z),b])-correct(G[(s,i,z),a]) for z in [101,102]) for i in ids];co={'p':exact(ds),'losses':[(p['id'],p['seed']) for p,l in pairs if correct(p)>correct(l)],'gains':[(p['id'],p['seed']) for p,l in pairs if correct(l)>correct(p)],'metrics':{}}
  for metric,fn in [('returned',lambda r:r['native_accounting']['returned_tokens']),('total',lambda r:r['native_accounting']['returned_tokens']+json.loads(r['raw_response'])['usage']['prompt_tokens']),('wall',lambda r:r['elapsed_seconds'])]:
   vals=[(sum(fn(G[(s,i,z),a]) for z in [101,102]),sum(fn(G[(s,i,z),b]) for z in [101,102])) for i in ids];P=sum(x[0] for x in vals);Q=sum(x[1] for x in vals);rng=random.Random(20261104);rat=[]
   for _ in range(20000):
    vv=[vals[rng.randrange(len(ids))] for __ in ids];rat.append(sum(x[1] for x in vv)/sum(x[0] for x in vv))
   rat.sort();complete=[(p,l) for p,l in pairs if p['finish_reason']!='length' and l['finish_reason']!='length'];censored=[(p,l) for p,l in pairs if p['finish_reason']=='length' or l['finish_reason']=='length']
   co['metrics'][metric]={'ratio':Q/P,'ci':[rat[500],rat[19499]],'complete_pairs':len(complete),'complete_ratio':sum(fn(l) for p,l in complete)/sum(fn(p) for p,l in complete),'net_censor_share':sum(fn(p)-fn(l) for p,l in censored)/(P-Q) if P!=Q else None}
  out['contrasts'][name]=co
 summary['suites'][s]=out
# Archive equality: independently retained old report plus node originals and newly named archive.
files=[H/'work/stdbench/phase2/post-v1/RESULTS.md',H/'work/stdbench/RESULTS-PHASE2-TWO-ARM.md',H/'work/imr-t5-s1/transport/t5-prior-report-retained.md',H/'work/imr-t5-s1/transport/t5-prior-report-archive.md'];assert all(read(p)==read(files[0]) for p in files);assert b'Rendered 2026-11-04T23:19:41.732233+00:00' in read(files[0]);summary['old_report_sha256']=sha(read(files[0]));summary['old_report_byte_identical_copies']=len(files)
(O/'summary.json').write_text(json.dumps(summary,indent=2));(O/'input-hashes.json').write_text(json.dumps(inputs,indent=2));print(json.dumps(summary,indent=2))
