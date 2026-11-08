
import sys as _archive_sys
import pathlib,json,sys,hashlib,collections,math

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)
H=pathlib.Path.home();B=H/'work/stdbench/decl-control';T=B/'dispatch-high';O=pathlib.Path(sys.argv[1]);J=lambda p:json.loads(p.read_text());L=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()];sha=lambda b:hashlib.sha256(b).hexdigest()
proof=J(T/'dispatch-proof.json');cpu=J(B/'cpu-final-a1-v2/verification.json');assert sha((T/'dispatch-proof.json').read_bytes())==cpu['dispatch_proof_sha256']
for rel,h in proof['sha256'].items():assert sha((T/rel).read_bytes())==h,rel
baseline=J(T/'unmodified-source-sha256.json');checked=0
for rel,h in baseline.items():
 assert sha((T/rel).read_bytes())==h,('copy',rel)
 assert sha((H/'work/imr-te/build-3a429d6f122d'/rel).read_bytes())==h,('pinned',rel)
 checked+=1
R={r['case']:r for r in L(T/'fixture-receipts.jsonl')};assert len(R)==8
for form in ['whole','stream']:
 d,l=R['D-'+form],R['L-'+form];assert d['dispatch_attempts']==d['dispatch_forwarded']==d['resume_count']==0;assert l['dispatch_attempts']==l['dispatch_forwarded']==1 and l['resume_count']==2;assert not d['held'] and l['held'];assert d['submits']==l['submits']==1;assert d['submitted_prompt_ids']==l['submitted_prompt_ids'] and d['request_wire']==l['request_wire'];assert R['noncall-'+form]['dispatch_attempts']==0 and R['wrongly-armed-D-'+form]['dispatch_attempts']==1
 if form=='whole':
  assert d['response']['choices'][0]['finish_reason']=='tool_calls';c=d['response']['choices'][0]['message']['tool_calls'][0]['function'];assert c['name']=='python' and json.loads(c['arguments'])=={'code':'print(17*19)'};cc=l['response']['tensorfold']['compute']['calls'];assert len(cc)==1 and cc[0]['status']=='ok' and cc[0]['output']=='323' and cc[0]['inserted']
 else:
  ds=''.join(str(x) for x in d['sse_payloads']);assert 'python' in ds and 'tool_calls' in ds;cc=next(x['tensorfold']['compute']['calls'] for x in reversed(l['sse_payloads']) if 'compute' in x.get('tensorfold',{}));assert len(cc)==1 and cc[0]['output']=='323' and cc[0]['inserted']
assert proof['guard_mutant_compiled'] and proof['guard_mutant_behaviorally_caught'];log=(T/'guard-mutant/check.log').read_text();assert '1 crash' in log and 'attempt to use null value' in log and 'chat.zig' in log and 'openai.zig' in log
statuses=collections.Counter();attempts=collections.Counter();diagnostic_calls=[];mathreview=J(B/'finalized-a1-v1/math-review-audit.json');assert len(mathreview['labels'])==120
for r in mathreview['labels']:assert r['decision']==r['effective_grade']=='correct' and len(r['labels'])==2 and all(x['label']=='correct' for x in r['labels'])
for p in (B/'runs').glob('*/*/diagnostics.jsonl'):
 for r in L(p):
  statuses[r['arm']+':'+r['status']]+=1;attempts[r['arm']]+=len(r['attempts']['calls'])
  for c in r['live_compute_reports']:diagnostic_calls.append({'boot':r['boot'],'id':r['id'],'source':c['code'],'output':c['output']})
result={'dispatch_fixture_count':8,'unchanged_source_hashes_vs_pinned_snapshot':checked,'compiled_guard_mutant':'behavioral invalid-transition crash, not compile failure','diagnostic_statuses':dict(statuses),'diagnostic_attempts':dict(attempts),'diagnostic_calls':diagnostic_calls,'math_review_pairs':120,'planning':{'tasks':120,'seeds':2,'replies':480,'solve_effect_80pct_proxy_q_point2':2.8*math.sqrt(.2/120),'token_reduction_80pct_proxy_logsd_point5':1-math.exp(-2.8*.5/math.sqrt(120)),'wall_hours_10s':480*10/3600,'wall_hours_20s':480*20/3600,'deadline_hours':480*270/3600}}
(O/'support.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
