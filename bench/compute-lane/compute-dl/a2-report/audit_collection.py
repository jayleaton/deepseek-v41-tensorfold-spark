#!/usr/bin/env python3
"""Offline A2 continuation audit; immutable A1 audit plus the exact P suffix."""
import argparse
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'a1-prep'))
from reference_policy import NativeCap,reference_body
from audit_lifecycle import classify,allow_partial_boot
SPEC='8eaa9575592f519b504d08022d400602e4b552874c8424b02c59c34f4af174e2'
A1_SPEC='0dca1fa52838388ea1e7afbc3a4c5e9747775f64e871640df7a66496369e5453'
A1_AUDIT=ROOT/'a1-continuation/final-audit-v3'
load=lambda p:json.loads(Path(p).read_text())
def iterrows(path):
    with Path(path).open() as source:
        for line in source:
            if line.strip():yield json.loads(line)
def rows(path):return list(iterrows(path))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def require(ok,why):
    if not ok:raise ValueError(why)
def audit(campaign,out):
    cfg=load(campaign);pins={str(Path(campaign)):sha(campaign)};gaps=[]
    sys.path.insert(0,str(ROOT/'a2-continuation'))
    import protocol as authority
    from budget import validate_prior,campaign_kwargs
    require(authority.require_collection_seal(campaign)==cfg,'offline final campaign seal mismatch')
    def read(path):
        p=Path(path);pins[str(p)]=sha(p)
    def pinned(path,expected):
        require(sha(path)==expected,'source/evidence drift: '+str(path));read(path)
    require(cfg['spec_sha256']==SPEC,'A2 spec mismatch')
    pinned(cfg['spec'],SPEC)
    for path,expected in cfg['code_sha256'].items():pinned(path,expected)
    proof=load(cfg['cpu_receipt']);pinned(cfg['cpu_receipt'],cfg['cpu_receipt_sha256'])
    require(proof['passed'] is True and proof['spec_sha256']==SPEC,'CPU proof mismatch')
    require(proof['code_sha256']==cfg['code_sha256'],'CPU source seal mismatch')
    for path,expected in proof['preparation_file_sha256'].items():pinned(path,expected)
    prior_audit=load(A1_AUDIT/'audit.json');read(A1_AUDIT/'audit.json')
    for path,expected in cfg['a1_audit_files_sha256'].items():pinned(path,expected)
    require(cfg['a1_audit_sha256']==sha(A1_AUDIT/'audit.json'),'A1 audit authority hash mismatch')
    require(prior_audit['qualified'] is True and not prior_audit['gaps']
            and prior_audit['spec_sha256']==A1_SPEC,'retained A1 raw audit unqualified')
    for path,expected in prior_audit['provenance']['input_file_sha256'].items():pinned(path,expected)
    for filename in ('audited-rows.jsonl','presentations.jsonl','diagnostics.jsonl'):read(A1_AUDIT/filename)
    retained=rows(A1_AUDIT/'audited-rows.jsonl')
    require(len(retained)==prior_audit['observed'] and 480<=len(retained)<=720,'retained count mismatch')
    require(sum(r['arm']=='lane' for r in retained)==240
            and sum(r['arm']=='declared_off' for r in retained)==240,'D/L completion coverage mismatch')
    retained_presentations={(tuple(p['identity']),p['call_id']):p for p in rows(A1_AUDIT/'presentations.jsonl')}
    for r in retained:
        if r['arm']=='plain':r['retention_batch']='P-before-break'
        key=tuple(r[k] for k in ('suite','id','seed','arm','boot'))
        for ref in r['safety']['references']:
            presentation=retained_presentations.get((key,ref['call_id']),ref.get('presentation'))
            require(presentation is not None and presentation['qualified'] is True,'missing retained presentation receipt')
            ref['presentation']=presentation
    native=cfg['cap_policy'];cap=NativeCap(*(native[k] for k in ('binary','tokenizer','source','receipt')))
    module=importlib.util.spec_from_file_location('_sealed_legacy_audit',ROOT/'collect_report.py')
    old=importlib.util.module_from_spec(module);module.loader.exec_module(old)
    decoder=old.client3.NativeDecoder(cfg['tokenizer'],cfg['tokenizer_sha256'])
    worker=sha(ROOT.parent/'phase2/jail_worker.py')
    prompts={tuple(p['key']):p for p in rows(cfg['cpu_prompt_arrays'])}
    pinned(cfg['cpu_prompt_arrays'],cfg['cpu_prompt_arrays_sha256'])
    freeze=Path(cfg['freeze']);pinned(freeze/'manifest.json',cfg['freeze_sha256']);manifest=load(freeze/'manifest.json')
    a1_campaign=ROOT/'stdbench-compute-a1-20261106-v3/campaign-v1.json'
    a1cfg=load(a1_campaign);read(a1_campaign)
    require(a1cfg['spec_sha256']==A1_SPEC,'A1 campaign authority mismatch')
    a1freeze=Path(a1cfg['freeze']);pinned(a1freeze/'manifest.json',a1cfg['freeze_sha256'])
    pentries=[e for e in load(a1freeze/'manifest.json')['boots'] if e['arm']=='P']
    require(len(pentries)==1,'original P manifest ambiguity')
    originalp=a1freeze/pentries[0]['file'];pinned(originalp,pentries[0]['sha256'])
    planned=load(originalp)['requests'];priorp=[r for r in retained if r['arm']=='plain']
    require(len(planned)==240 and [(r['suite'],r['id'],r['seed']) for r in priorp]
            ==[tuple(r['key']) for r in planned[:len(priorp)]],'retained P is not the original prefix')
    require(len(manifest['boots'])==1 and manifest['boots'][0]['arm']=='P','A2 may boot only one P suffix')
    freshentry=manifest['boots'][0];freshboot=freeze/freshentry['file'];pinned(freshboot,freshentry['sha256'])
    freshrequests=load(freshboot)['requests']
    require([(r['key'],r['wire'],r['wire_sha256']) for r in freshrequests]
            ==[(r['key'],r['wire'],r['wire_sha256']) for r in planned[len(priorp):]],'A2 P suffix wire/order mismatch')
    states={};step_paths={}
    for turn in manifest['turns']:
        sp=ROOT/'a2-continuation/runs'/(cfg['prefix']+'-'+turn['name'])/'steps.jsonl'
        if sp.exists():read(sp);states[turn['name']]=classify(rows(sp));step_paths[turn['name']]=sp
    attempt_dir=ROOT/'a2-continuation/attempted'/cfg['prefix']
    markers={}
    if attempt_dir.exists():
        for p in attempt_dir.glob('*.json'):read(p);markers[p.name]=load(p)
    records=[];bootmeta=[];newpresentations=[]
    for entry in manifest['boots']:
        pinned(freeze/entry['file'],entry['sha256']);boot=load(freeze/entry['file'])
        turn=next(t['name'] for t in manifest['turns'] if entry['boot'] in t['boots'])
        directory=ROOT/'a2-continuation/runs'/(cfg['prefix']+'-'+turn)/entry['boot']
        if not directory.exists():
            bootmeta.append({'boot':entry['boot'],'observed':0,'started':False});continue
        require(turn in states,'boot directory lacks settled lifecycle evidence')
        paths={name:directory/name for name in ('responses.jsonl','diagnostics.jsonl','safety.jsonl','raw-observations.jsonl')}
        # A boot can be cut off before its first dataset request, but every attempt
        # must have a durable response and a reference sidecar before qualification.
        actuals={n:rows(p) if p.exists() else [] for n,p in paths.items() if n!='raw-observations.jsonl'}
        for path in paths.values():
            if path.exists():read(path)
        attempted=sum(m['boot']==entry['boot'] for m in markers.values())
        partial=allow_partial_boot(states[turn],entry['boot'],len(actuals['responses.jsonl']),len(actuals['diagnostics.jsonl']),attempted)
        if entry['boot'] not in states[turn]['ready']:
            require(not any(actuals.values()) and (not paths['raw-observations.jsonl'].exists()
                    or paths['raw-observations.jsonl'].stat().st_size==0),'unready boot contains durable model activity')
            log=directory/'rank0.log'
            if log.exists():read(log);require('POST ' not in log.read_text(),'unready boot has native POST evidence')
            bootmeta.append({'boot':entry['boot'],'started':True,'ready':False,'partial':True,'observed':0})
            continue
        safeties=old.indexed(actuals['safety.jsonl']);seen=set();bytransport={};perboot=[]
        for filename,templates,diag in [('responses.jsonl',boot['requests'],False),('diagnostics.jsonl',boot['diagnostics'],True)]:
            expected=[dict(t,boot=entry['boot']) for t in templates]
            observed=actuals[filename]
            require([old.key(r) for r in observed]==[old.key(t) for t in expected[:len(observed)]],'new request prefix/order mismatch')
            for raw,t in zip(observed,expected):
                key=old.key(raw);require(key not in seen,'duplicate new receipt');seen.add(key)
                require(key in safeties,'missing new safety sidecar')
                safety=safeties[key];calls=((raw.get('tensorfold') or {}).get('compute') or {}).get('calls') or []
                if raw['arm'] in ('plain','declared_off'):
                    require(not calls and raw.get('native_binding',{}).get('native_frame_count',0)==0,'effective-OFF execution violation')
                refs={r['call_id']:r for r in safety.get('references',[])}
                require(len(refs)==len(safety.get('references',[])),'duplicate independent replay ID')
                for call in calls:
                    if call.get('inserted') is True:
                        require(call['id'] in refs,'missing new independent replay')
                        ref=refs[call['id']];body=reference_body(ref['result'])
                        perboot.append((call,ref,body,key))
                records.append((raw,safety,t,diag,prompts[tuple(t['key'])]))
                bytransport[(tuple(t['key']),diag)]=raw
        require(set(safeties)==seen,'new safety coverage differs')
        tk=[]
        transports=iterrows(paths['raw-observations.jsonl']) if paths['raw-observations.jsonl'].exists() else iter(())
        for transport in transports:
            tk.append((tuple(transport['template_key']),transport['diagnostic']))
            raw=bytransport[(tuple(transport['template_key']),transport['diagnostic'])]
            require(all(transport['response'].get(f)==raw.get(f) for f in ('sse_frames','sse_done','actual_request_wire','elapsed_seconds','transport_status','content','reasoning','usage','tensorfold')),'new raw transport/frame mismatch')
        require(len(tk)==len(set(tk))==len(seen) and set(tk)==set(bytransport),'new attempted/observed key mask mismatch')
        log=directory/'rank0.log';require(log.exists(),'missing actual native access log');read(log)
        http=old.audit_records.http_audit(log.read_text(),len(seen));require(http['qualified'],'new native POST coverage mismatch')
        snapshots=[]
        for rank in (0,1):
            mp=Path(str(cfg['rank_manifests'][entry['boot']][str(rank)]).replace('<rank1-home>/','<rank0-home>/',1))
            pinned(mp,cfg['rank_manifest_sha256'][entry['boot']][str(rank)])
            rm=load(mp);snap=directory/f'snapshot-r{rank}.log';require(snap.exists(),'missing new native snapshot');read(snap)
            pointers=[json.loads(l) for l in snap.read_text().splitlines() if l.startswith('{')]
            pointers=[p for p in pointers if p.get('command')=='snapshot'];require(len(pointers)==1,'ambiguous native snapshot')
            pointer=pointers[0];pp=Path(pointer['proof'])
            if not pp.exists(): pp=Path(str(pp).replace('<rank1-home>/','<rank0-home>/',1))
            require(pp.exists(),'missing pointed native proof: '+str(pp));read(pp);p=load(pp)
            nativeproc=p.get('native_process') or {};env=nativeproc.get('env') or {}
            expectedenv=rm['phase2']['ranks'][rank]['effective_env']
            require(p.get('dry_run') is False and p.get('present') is True and p.get('boot')==entry['boot'] and p.get('rank')==rank,
                    'actual native snapshot identity mismatch')
            require(p.get('manifest_sha256')==sha(mp) and p.get('image_id')==cfg['image_ids'][rank] and nativeproc.get('pid_namespace',0)>1,'native source/image/process mismatch')
            selected=lambda e:{k:v for k,v in e.items() if k.startswith(('TF_','GLM53_TF','NCCL'))}
            require(selected(env)==selected(expectedenv),'actual native effective environment drift')
            require(env.get('TF_DSV41_COMPUTE_LANE')==str(int(entry['arm']=='L')),'native lane state drift')
            snapshots.append({'rank':rank,'proof':str(pp),'sha256':sha(pp)})
        bootmeta.append({'boot':entry['boot'],'started':True,'ready':True,'partial':partial,'observed':len(actuals['responses.jsonl']),'http':http,'snapshots':snapshots})
        newpresentations.extend(perboot)
        # Records retain the single response copy needed for native re-audit;
        # duplicate raw transport objects have already been compared and freed.
        actuals.clear();bytransport.clear();safeties.clear()
    caprows=cap.cap([p[2] for p in newpresentations]);cache={};presentations=[]
    for (call,ref,body,k),r in zip(newpresentations,caprows,strict=True):
        matched=call['status']==ref['result']['status'] and call['output']==r['output']
        require(ref.get('matches') is matched,'new reference comparison attestation mismatch')
        cache[old.client3.canonical([call,ref['result']])]=matched
        clipped=r['token_count']>512
        presentations.append({'identity':list(k),'call_id':call['id'],'qualified':matched,
            'classification':'presentation-verified clipped' if clipped and matched else 'presentation-verified unclipped' if matched else 'unqualified',
            'full_reference_tokens':r['token_count'],'retained_prefix_tokens':r['retained_prefix_tokens'],
            'full_reference_bytes':len(body),'landed_body_bytes':len(call['output'].encode()),
            'full_reference_sha256':old.digest(body),'landed_body_sha256':old.digest(call['output'].encode()),
            'marker_bytes':list(b'\n[output truncated at 512 tokens]') if clipped else []})
    old.reference_matches=lambda call,result:cache.get(old.client3.canonical([call,result]),False)
    output=list(retained);audits=[];diagnostics=[]
    for index,(raw,safety,t,diag,prompt) in enumerate(records):
        row,a=old.audit_row(raw,safety,t,prompt,decoder,worker,diag)
        if not a['qualified']:gaps.extend(a['gaps'])
        audits.append(a)
        if diag:diagnostics.append(row)
        else:
            require(row['arm']=='plain','A2 may collect only the unchanged P suffix')
            row['retention_batch']='P-after-break';output.append(row)
        # Later marker checks require identity/wire metadata, not another copy
        # of the full SSE stream after its independently audited row is saved.
        records[index]=({'arm':raw['arm'],'boot':raw['boot']},None,t,diag,None)
    for b in bootmeta:
        if not b['started'] or not b.get('ready'):continue
        ds=[r for r in diagnostics if r['boot']==b['boot']]
        require(len(ds)==2 or b['partial'],'actual boot diagnostic coverage failed')
        diagnostic_audits=[a for a in audits if a['diagnostic'] and a['identity'][-1]==b['boot']]
        successes=[a['verified_diagnostic_calls'] for a in diagnostic_audits]
        require(all(n>0 for n in successes) if b['boot'].endswith('-L') else all(n==0 for n in successes),'each fixed boot diagnostic must qualify')
    keys=[(r['id'],r['seed'],r['arm']) for r in output];require(len(keys)==len(set(keys)),'repeated task/seed/arm')
    allp=[r for r in output if r['arm']=='plain']
    require([(r['suite'],r['id'],r['seed']) for r in allp]
            ==[tuple(r['key']) for r in planned[:len(allp)]],'combined P is not the immutable original prefix')
    expected_markers={}
    for raw,safety,t,diag,prompt in records:
        if not diag:
            name=old.digest(old.client3.canonical([raw['arm'],*t['key']]).encode())+'.json'
            expected_markers[name]=(raw,t)
    require(set(markers)==set(expected_markers),'attempted dataset marker/receipt coverage mismatch')
    for name,(raw,t) in expected_markers.items():
        m=markers[name]
        require(m['boot']==raw['boot'] and m['key']==t['key'] and m['arm']==raw['arm'] and m['wire_sha256']==t['wire_sha256'],'attempt marker source/wire mismatch')
    turns=[]
    for turn in manifest['turns']:
        sp=ROOT/'a2-continuation/runs'/(cfg['prefix']+'-'+turn['name'])/'steps.jsonl'
        started=any(b['started'] and b['boot'] in turn['boots'] for b in bootmeta)
        if not sp.exists():
            require(not started,'started turn has no settled step evidence')
            continue
        state=states[turn['name']];verdict=state['verdict']
        if verdict not in ('QUALIFIED','INCOMPLETE'):gaps.append('operator turn '+verdict)
        item={'turn':turn['name'],'verdict':verdict,'held':state['held'],'budget_event':state['budget_event']}
        restoration=state['restore']
        if restoration:
            rp=sp.parent/'G-restore.log';read(rp)
            require(restoration.get('actual_step_path')==str(rp) and restoration.get('actual_step_sha256')==sha(rp),'G restoration log hash/path mismatch')
            text=rp.read_text()
            require(all(name in text for name in ('SERVING ','/v1/models 200','POPPED; ORDER:')),'G restoration log steps missing')
            require('58bece1d3184269059fd3f7f755c52f4f295113d' in text,'restored G source mismatch')
            item.update(restore_log=str(rp),restore_log_sha256=sha(rp))
        turns.append(item)
    budget=load(cfg['budget_ledger']);read(cfg['budget_ledger'])
    original=validate_prior(**campaign_kwargs(cfg))
    require(budget['schema']==4 and budget['spec_sha256']==SPEC and budget['parent_spec_sha256']==cfg['base_spec_sha256']
        and budget['a1_spec_sha256']==A1_SPEC
        and budget['original_ledger_sha256']==cfg['original_budget_sha256']
        and budget['prior_ledger_sha256']==cfg['prior_budget_sha256']
        and budget['a1_audit_sha256']==cfg['a1_audit_sha256']
        and budget['carry_seconds']==original['held_seconds'] and budget['run_prefix']==cfg['prefix']
        and budget['active'] is None and budget['limit_seconds']==36000 and budget['admission_seconds']==33900,
        'held-time carry/release schema mismatch')
    n=len(original['turns']);require(budget['turns'][:n]==original['turns'],'original held intervals changed')
    intervals=budget['turns'][n:]
    held_turns=[t for t in turns if t['held']]
    require([r['turn'] for r in intervals]==[r['turn'] for r in held_turns],'held intervals/settled turns differ')
    cumulative=original['held_seconds']
    for interval,turn in zip(intervals,held_turns,strict=True):
        require(interval.get('run_prefix')==cfg['prefix'],'new held interval namespace mismatch')
        require(interval['held_seconds']>=0,'negative held interval');cumulative+=interval['held_seconds']
        require(abs(interval['cumulative_seconds']-cumulative)<1e-6,'held interval cumulative mismatch')
        require(abs(turn['budget_event']['held_seconds']-cumulative)<1e-6,'step/ledger held charge mismatch')
    require(abs(budget['held_seconds']-cumulative)<1e-6,'held time total mismatch')
    require(abs(budget.get('overrun_seconds',0)-max(0,cumulative-36000))<1e-6,'safety overrun not reported separately')
    for p,d in pins.items():require(sha(p)==d,'input changed during audit')
    out=Path(out);out.mkdir(exist_ok=False)
    for name,values in [('audited-rows.jsonl',output),('diagnostics.jsonl',diagnostics),('presentations.jsonl',presentations)]:
        with (out/name).open('x') as f:
            for row in values:f.write(old.client3.canonical(row)+'\n')
    integrity={'qualified':not gaps,'spec_sha256':SPEC,'a1_spec_sha256':A1_SPEC,'parent_spec_sha256':cfg['base_spec_sha256'],'gaps':gaps,
               'provenance':{'input_file_sha256':pins},'observed':len(output),'scheduled':720,'missing':720-len(output),
               'boots':bootmeta,'turns':turns,'budget':budget,'row_audits':audits,
               'retained_A1_audit_sha256':sha(A1_AUDIT/'audit.json'),'retained_rows':len(retained),
               'P_batches':{'before_break':sum(r['arm']=='plain' for r in retained),
                            'after_break':sum(r.get('retention_batch')=='P-after-break' for r in output)}}
    (out/'audit.json').write_text(json.dumps(integrity,indent=2))
    return integrity
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    r=audit(a.campaign,a.out);print(json.dumps({k:r[k] for k in ('qualified','observed','missing')}))
