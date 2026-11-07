import json,pathlib,sys,hashlib,collections,random,subprocess,datetime,math
H=pathlib.Path.home();O=pathlib.Path(sys.argv[1]);O.mkdir(parents=True,exist_ok=True);ROOT=H/'work/ab3-20261104';KIT=ROOT/'kit2'
J=lambda p:json.loads(p.read_text());L=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()];sha=lambda b:hashlib.sha256(b).hexdigest()
f=J(ROOT/'AB3-FREEZE.json');layout=J(KIT/'ab3-layout.json');assert sha((H/'work/imr-t5-w9/T5-AB3-SPEC.md').read_bytes())==f['spec_sha256']=='81b34c1a79a754efca01ebee85f47c74490a029f23e4e0533799205b60dd524e'
file_fail=[]
for base in [KIT]+[pathlib.Path(x) for x in f['turn_dirs']]:
 for rel,s in f['kit_files_sha256'].items():
  p=base/rel
  if not p.exists() or sha(p.read_bytes())!=s:file_fail.append(str(p))
assert not file_fail,file_fail
allrows=L(H/'work/imr-t5-w9/frozen/tasks-private.jsonl');allunits={u['id']:u for u in allrows};units=L(KIT/'ab3-units.jsonl');families={u['id']:u['set'] for u in units};assert sha((KIT/'ab3-units.jsonl').read_bytes())==f['units_sha256']==layout['units_sha256']
picked={g:sorted([i for i,u in allunits.items() if u['set']==g and i not in layout['excluded_pilot_units']],key=lambda i:(sha(('AB3-v1:'+i).encode()),i))[:n] for g,n in layout['k'].items()};assert set(sum(picked.values(),[]))==set(families) and len(families)==60
assert len(set(layout['excluded_pilot_units']))==11
for u in units:assert u==allunits[u['id']]
runs=[H/'work/dsv41-bench/out'/('imr-measure-20261104T'+t+'Z') for t in ['091946','095232','101747','104533','112309','115045']]
# Independent byte-token decoder; no analysis or oracle helpers imported.
tok=J(H/'ai/models/dsv41-flash17-k6-mtp/tokenizer.json');bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
for b in range(256):
 if b not in bs:bs.append(b);cs.append(256+n);n+=1
inverse={chr(c):b for b,c in zip(bs,cs)};added={z['id']:z['content'].encode() for z in tok['added_tokens']};idword={i:w for w,i in tok['model']['vocab'].items()};cache={}
def piece(i):
 if i not in cache:cache[i]=added[i] if i in added else bytes(inverse[c] for c in idword[i])
 return cache[i]
rr=L(ROOT/'rerun-ab3/results.jsonl');refs={z['key']:z for z in rr if 'key' in z};assert len(refs)==444 and len(rr)==445
letters={'plain':'a','lane':'d','full':'rn','only':'ro'};boots={b['name']:(bl,b) for bl in layout['blocks'] for b in bl['boots']};res={a:{} for a in letters};calls=[];rawhash={};metadata={};safety=[];first=[];toolkeys=set();configs=[]
for run in runs:
 grades=L(run/'grades.jsonl');gr={(g['arm'],g['id']):g for g in grades};assert len(gr)==len(grades)==120
 status=J(run/'status.json');assert status['engine_sha']==f['engine_sha']
 for att in status['attempts']:
  name=att['name'];bl,b=boots[name];a=b['arm'];assert att['pass'] and att['measure_rc']==0;assert att['extra']==layout['arms'][a]['extra'];meta=J(run/name/'arm-metadata.json');assert meta['engine_sha']==f['engine_sha'];assert [z['image_id'] for z in meta['ranks']]==list(f['image_ids'].values());metadata[name]=meta
  reqs=J(KIT/b['templates'])['requests'];assert sha((KIT/b['templates']).read_bytes())==b['templates_sha256'];order=sorted((i,s) for i in bl['units'] for s in layout['seeds']);random.Random(20261104).shuffle(order);assert [(z['task_id'],z['seed']) for z in reqs]==order
  temps={(r['task_id'],r['seed']):r['body_template'] for r in reqs};bootraw=[]
  for s in layout['seeds']:
   for i in bl['units']:
    p=run/name/(letters[a]+'-s'+str(s))/(i+'.json');x=J(p);rq=x['request'];temp=temps[i,s];expected=dict(temp);expected['model']=rq['model'];expected.update(return_token_ids=True,deadline_s=300.0);assert expected==rq,(str(p),expected,rq);assert rq['messages'][0]['content']==allunits[i]['prompt']
    tf=x['extra']['tensorfold'];ids=tf['token_ids'];assert sha(','.join(map(str,ids)).encode())==tf['token_sha256'];assert x['ids_check']=='match';comp=tf.get('compute',{});assert len(ids)==x['usage']['completion_tokens']+comp.get('inserted_tokens',0)
    raw=b''.join(piece(z) for z in ids);key=(i,s);assert key not in res[a];g=gr[name+'/'+letters[a]+'-s'+str(s),i];cells=tf.get('runner',{}).get('cells',[]);cut=any(c.get('status')=='cut' or c.get('landed',0)<c.get('ids',0) for c in cells)
    r={'id':i,'seed':s,'family':families[i],'T':len(ids),'completion':x['usage']['completion_tokens'],'inserted':comp.get('inserted_tokens',0),'solved':bool(g.get('ok_signed',g.get('ok'))),'finish':x['finish'],'wall':x['elapsed_s'],'prompt':x['usage']['prompt_tokens'],'cached':x['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0),'censored':x['finish']!='stop' or len(ids)>=32768,'cut':cut,'owned':len(cells),'path':str(p),'sent':x['sent_unix']};res[a][key]=r;bootraw.append(r);rawhash[str(p)]=sha(p.read_bytes());seek=0
    for ci,c in enumerate(comp.get('calls',[])):
     k=name+'/'+letters[a]+'-s'+str(s)+'/'+i+'#'+str(ci);ref=refs[k];assert k not in toolkeys;toolkeys.add(k);assert sha(c['code'].encode())==ref['code_sha256'];assert c['status']==ref['status']=='ok';assert ref['exit_code']==0
     body=ref['stdout'];err=ref['stderr']
     if err:body+= ('[stderr]\n' if not body else '\n[stderr]\n')+err
     if body.endswith('\n'):body=body[:-1]
     body=body or '[no output]';assert body==c['output'],(k,body,c['output']);encoded=('<tool_result>'+body.replace('<｜deepseek_image｜>','<​｜deepseek_image｜>')+'</tool_result>').encode();loc=raw.find(encoded,seek);assert loc>=0,(k,'missing delivered tool bytes');seek=loc+len(encoded);assert c['code'].encode() in raw[:loc],(k,'callsource not found')
     calls.append({'key':k,'executor':c.get('executor','sandbox'),'ref_match':True,'native_result_at':loc,'code_sha256':ref['code_sha256']})
    if a=='full' and i=='w9-bignum-017' and s==101:
     want='[exact, computed by the engine: 2^66 = 73786976294838206464]\n';idx=raw.find(want.encode());assert idx>=0 and 2**66==73786976294838206464;assert want not in x['content'];assert want[:-1] in x['content'];(O/'safety-native-proof.json').write_text(json.dumps({'native_byte_at':idx,'native_contains_full_record':True,'full_record_in_content':False,'without_LF_in_content':True,'content':x['content'],'record_cells':cells,'token_hash':tf['token_sha256']},indent=2)+'\n')
  first.append({'boot':name,'arm':a,'first':min(bootraw,key=lambda z:z['sent'])})
 safety.append(J(run/'window-report/window-report.json')['gate'])
assert len(rawhash)==720 and len(toolkeys)==444 and toolkeys==set(refs)
keys={(i,s) for i in families for s in [101,102,103]};assert all(set(z)==keys for z in res.values());minsent=min(v['sent'] for z in res.values() for v in z.values());assert minsent>datetime.datetime.fromisoformat(f['frozen_at_utc'].replace('Z','+00:00')).timestamp()
pairs=[{'id':i,'seed':s,'family':families[i],**{a:res[a][i,s] for a in res}} for i,s in sorted(keys)]
summary={'freeze_sha256':sha((ROOT/'AB3-FREEZE.json').read_bytes()),'frozen_at':f['frozen_at_utc'],'first_request_utc':datetime.datetime.fromtimestamp(minsent,datetime.timezone.utc).isoformat(),'frozen_files_per_copy':len(f['kit_files_sha256']),'frozen_copies':1+len(f['turn_dirs']),'family_counts':dict(collections.Counter(families.values())),'receipts':720,'tool_matches':444,'safety_gate_sum':{k:sum(z.get(k,0) for z in safety) for k in ['records','ok','wrong-value','wrong-span','not-in-text','cut','unknown','unparsed']},'first_cache':first,'arms':{}}
for a,z in res.items():summary['arms'][a]={**{k:sum(r[k] for r in z.values()) for k in ['T','completion','inserted','solved','censored','cut','owned','wall','prompt','cached']},'finishes':dict(collections.Counter(r['finish'] for r in z.values()))}
for a in ['lane','full','only']:
 for key in keys:assert res[a][key]['prompt']-res['plain'][key]['prompt']==(418 if a=='full' else 306)
for name,obj in [('summary',summary),('pairs',pairs),('tool-bindings',calls),('raw-hashes',rawhash),('metadata',metadata)]: (O/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['first_cache']},indent=2));(O/'script.py').write_text(pathlib.Path(__file__).read_text())
