#!/usr/bin/env python3
"""Postcollection v2: read families from the sealed public manifest, not templates.

The original sealed auditor and all collection code remain byte-identical.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'compute-dl'
SPEC='b47ede697eed3a38d244ad5ff7447690732bd12945ff4592554d45bc2d8527ec'
GRADER='1a2249a1b5e0b06f03a4300995d0c320e5a21630d261b3791501f7c83740fb9f'
load=lambda p:json.loads(Path(p).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def key(row):return (row.get('study_arm'),row['suite'],row['id'],row['seed'],row['boot'])
def require(ok,why):
    if not ok:raise ValueError(why)
def local(path):return Path(str(path).replace('<rank1-home>/','<rank0-home>/',1))
def public_tasks():
    path=HERE/'dataset-v1/tasks.jsonl'
    require(sha(path)=='a697b7e556d793f3f712bbc043b496b1f1ed7a343289c2f32358e4d6b00238c6','frozen public tasks differ')
    values=rows(path);byid={t['id']:t for t in values}
    require(len(byid)==len(values)==60,'public task identity coverage')
    return byid
def audit(campaign,out):
    sys.path.insert(0,str(HERE/'runtime'))
    import protocol
    import budget as budgets
    cfg=protocol.require_collection_seal(campaign)
    require(cfg['spec_sha256']==SPEC,'spec identity')
    sys.path.insert(0,str(OLD/'a1-prep'))
    from reference_policy import NativeCap,reference_body
    from audit_lifecycle import classify,allow_partial_boot
    module=importlib.util.spec_from_file_location('_format_frozen_native_audit',OLD/'collect_report.py')
    old=importlib.util.module_from_spec(module);module.loader.exec_module(old)
    decoder=old.client3.NativeDecoder(cfg['tokenizer'],cfg['tokenizer_sha256'])
    worker_path=HERE.parent/'phase2/jail_worker.py'
    worker=cfg['code_sha256'][str(worker_path)]
    require(sha(worker_path)==worker,'qualified replay worker drift')
    cap=NativeCap(*(cfg['cap_policy'][k] for k in ('binary','tokenizer','source','receipt')))
    byid=public_tasks()
    pins={str(Path(__file__).resolve()):sha(__file__)};gaps=[];records=[];bootmeta=[];allpresentations=[]
    def read(path,expected=None):
        path=Path(path);digest=sha(path)
        require(expected is None or digest==expected,'pinned file drift: '+str(path))
        require(str(path) not in pins or pins[str(path)]==digest,'file changed during audit')
        pins[str(path)]=digest
    read(campaign);read(cfg['spec'],SPEC)
    read(HERE/'dataset-v1/tasks.jsonl')
    for p,h in cfg['code_sha256'].items():read(p,h)
    for q in cfg['qualification'].values():read(q['path'],q['sha256'])
    read(cfg['cpu_receipt'],cfg['cpu_receipt_sha256'])
    prompts={tuple(p['key']):p for p in rows(cfg['cpu_prompt_arrays'])}
    read(cfg['cpu_prompt_arrays'],cfg['cpu_prompt_arrays_sha256'])
    freeze=Path(cfg['freeze']);read(freeze/'manifest.json',cfg['freeze_sha256']);manifest=load(freeze/'manifest.json')
    states={};steps={}
    for turn in manifest['turns']:
        path=HERE/'runtime/runs'/(cfg['prefix']+'-'+turn['name'])/'steps.jsonl'
        if path.exists():read(path);states[turn['name']]=classify(rows(path));steps[turn['name']]=path
    marker_dir=HERE/'runtime/attempted'/cfg['prefix']
    markers={}
    if marker_dir.exists():
        for path in marker_dir.glob('*.json'):read(path);markers[path.name]=load(path)
    expected_markers={};replay_inputs=[]
    for entry in manifest['boots']:
        path=freeze/entry['file'];read(path,entry['sha256']);boot=load(path)
        turn=next(t['name'] for t in manifest['turns'] if entry['boot'] in t['boots'])
        directory=HERE/'runtime/runs'/(cfg['prefix']+'-'+turn)/entry['boot']
        if not directory.exists():bootmeta.append({'boot':entry['boot'],'started':False,'observed':0});continue
        require(turn in states,'started boot lacks settled lifecycle')
        paths={n:directory/n for n in ('responses.jsonl','diagnostics.jsonl','safety.jsonl','raw-observations.jsonl')}
        for path in paths.values():
            if path.exists():read(path)
        actual={n:rows(p) if p.exists() else [] for n,p in paths.items() if n!='raw-observations.jsonl'}
        attempted=sum(m['boot']==entry['boot'] for m in markers.values())
        partial=allow_partial_boot(states[turn],entry['boot'],len(actual['responses.jsonl']),len(actual['diagnostics.jsonl']),attempted)
        require(entry['boot'] in states[turn]['ready'] or not any(actual.values()),'unready boot has responses')
        if entry['boot'] not in states[turn]['ready']:
            require(not paths['raw-observations.jsonl'].exists() or paths['raw-observations.jsonl'].stat().st_size==0,'unready boot has raw requests')
            bootmeta.append({'boot':entry['boot'],'started':True,'ready':False,'observed':0});continue
        safeties={key(r):r for r in actual['safety.jsonl']}
        require(len(safeties)==len(actual['safety.jsonl']),'duplicate safety key')
        seen=set();transport_rows={}
        for filename,templates,diagnostic in [('responses.jsonl',boot['requests'],False),('diagnostics.jsonl',boot['diagnostics'],True)]:
            expected=[dict(t,boot=entry['boot']) for t in templates]
            observed=actual[filename]
            require([key(r) for r in observed]==[key(t) for t in expected[:len(observed)]],'observed prefix/order/wording mismatch')
            for raw,template in zip(observed,expected,strict=False):
                k=key(raw);require(k not in seen,'duplicate observation');seen.add(k)
                require(k in safeties,'missing independent replay sidecar');safety=safeties[k]
                require(raw.get('study_arm')==template.get('study_arm'),'logical arm drift')
                if not diagnostic:
                    name=old.digest(old.client3.canonical([raw['study_arm'],*template['key']]).encode())+'.json'
                    expected_markers[name]=(raw,template)
                calls=((raw.get('tensorfold') or {}).get('compute') or {}).get('calls') or []
                if entry['arm']=='P':require(not calls and raw.get('native_binding',{}).get('native_frame_count',0)==0,'plain executed a native call')
                refs={r['call_id']:r for r in safety.get('references',[])}
                require(len(refs)==len(safety.get('references',[])),'duplicate reference call')
                for call in calls:
                    if call.get('inserted') is True:
                        require(call['id'] in refs,'delivered result has no isolated reference')
                        ref=refs[call['id']];replay_inputs.append((call,ref,reference_body(ref['result']),k,diagnostic))
                prompt=prompts[tuple(template.get('prompt_key',template['key']))]
                require(prompt['key']==template.get('prompt_key',template['key']),'logical prompt identity differs')
                prompt=dict(prompt,key=template['key'])
                records.append((raw,safety,template,prompt,diagnostic))
                transport_rows[(raw.get('study_arm'),tuple(template['key']),diagnostic)]=raw
        require(set(safeties)==seen,'extra safety sidecar')
        transport_seen=set()
        if paths['raw-observations.jsonl'].exists():
            with paths['raw-observations.jsonl'].open() as source:
                for line in source:
                    transport=json.loads(line);k=(transport.get('study_arm'),tuple(transport['template_key']),transport['diagnostic'])
                    require(k in transport_rows and k not in transport_seen,'raw transport key mismatch')
                    transport_seen.add(k);raw=transport_rows[k]
                    require(all(transport['response'].get(f)==raw.get(f) for f in ('sse_frames','sse_done','actual_request_wire','elapsed_seconds','transport_status','content','reasoning','usage','tensorfold')),'raw transport bytes/frame mismatch')
        require(transport_seen==set(transport_rows),'raw/processed coverage differs')
        access=directory/'rank0.log';read(access)
        http=old.audit_records.http_audit(access.read_text(),len(seen));require(http['qualified'],'native HTTP coverage differs')
        snapshots=[]
        for rank in (0,1):
            mp=local(cfg['rank_manifests'][entry['boot']][str(rank)]);read(mp,cfg['rank_manifest_sha256'][entry['boot']][str(rank)]);rm=load(mp)
            snap=directory/f'snapshot-r{rank}.log';read(snap)
            pointers=[json.loads(l) for l in snap.read_text().splitlines() if l.startswith('{')]
            pointers=[p for p in pointers if p.get('command')=='snapshot'];require(len(pointers)==1,'ambiguous actual snapshot')
            pp=local(pointers[0]['proof']);read(pp);proof=load(pp)
            process=proof.get('native_process') or {};env=process.get('env') or {}
            require(proof.get('dry_run') is False and proof.get('present') is True and proof.get('boot')==entry['boot'] and proof.get('rank')==rank,'native snapshot identity')
            require(proof.get('manifest_sha256')==sha(mp) and proof.get('image_id')==cfg['image_ids'][rank] and process.get('pid_namespace',0)>1,'native image/source/process')
            select=lambda e:{k:v for k,v in e.items() if k.startswith(('TF_','GLM53_TF','NCCL'))}
            require(select(env)==select(rm['phase2']['ranks'][rank]['effective_env']),'actual effective env drift')
            require(env.get('TF_DSV41_COMPUTE_LANE')==str(int(entry['arm']=='L')),'actual lane state differs')
            snapshots.append({'rank':rank,'path':str(pp),'sha256':sha(pp)})
        bootmeta.append({'boot':entry['boot'],'started':True,'ready':True,'observed':len(actual['responses.jsonl']),'partial':partial,'http':http,'snapshots':snapshots})
    require(set(markers)==set(expected_markers),'attempt marker/observation coverage differs')
    for name,(raw,t) in expected_markers.items():
        m=markers[name]
        require(m['boot']==raw['boot'] and m['key']==t['key'] and m['study_arm']==raw['study_arm'] and m['wire_sha256']==t['wire_sha256'],'attempt identity/wire drift')
    caprows=cap.cap([r[2] for r in replay_inputs]);cache={}
    for (call,ref,body,k,diagnostic),r in zip(replay_inputs,caprows,strict=True):
        matched=call['status']==ref['result']['status'] and call['output']==r['output']
        require(ref.get('matches') is matched,'replay byte/status attestation drift')
        cache[old.client3.canonical([call,ref['result']])]=matched
        allpresentations.append({'identity':list(k),'call_id':call['id'],'diagnostic':diagnostic,'qualified':matched,
            'clipped':r['token_count']>512,'full_reference_tokens':r['token_count'],'retained_prefix_tokens':r['retained_prefix_tokens'],
            'full_reference_bytes':len(body),'landed_body_bytes':len(call['output'].encode()),
            'full_reference_sha256':old.digest(body),'landed_body_sha256':old.digest(call['output'].encode()),
            'source_sha256':old.digest(call['code'].encode()),'provenance':ref})
    old.reference_matches=lambda c,r:cache.get(old.client3.canonical([c,r]),False)
    output=[];diagnostics=[];row_audits=[]
    for raw,safety,t,prompt,diagnostic in records:
        row,proof=old.audit_row(raw,safety,t,prompt,decoder,worker,diagnostic)
        if not proof['qualified']:gaps.extend(proof['gaps'])
        row['family']=None if diagnostic else byid[t['id']]['family']
        proof['study_arm']=raw.get('study_arm');row_audits.append(proof)
        (diagnostics if diagnostic else output).append(row)
    for b in bootmeta:
        if not b.get('ready'):continue
        ds=[p for p in row_audits if p['diagnostic'] and p['identity'][-1]==b['boot']]
        require(len(ds)==2 or b['partial'],'diagnostic coverage differs')
        require(all(p['verified_diagnostic_calls']>0 for p in ds) if b['boot']=='format-L' else all(p['verified_diagnostic_calls']==0 for p in ds),'fixed activation diagnostics failed')
    require(len({key(r) for r in output})==len(output),'duplicate dataset identity')
    turns=[]
    for turn,state in states.items():
        if state['verdict'] not in ('QUALIFIED','INCOMPLETE'):gaps.append('turn '+state['verdict'])
        restore=state['restore'];require(restore is not None,'normal G restoration missing')
        path=steps[turn].parent/'G-restore.log';read(path)
        require(restore['actual_step_path']==str(path) and restore['actual_step_sha256']==sha(path),'restore log binding differs')
        require(all(s in path.read_text() for s in ('SERVING ','/v1/models 200','POPPED; ORDER:','58bece1d3184269059fd3f7f755c52f4f295113d')),'G restoration proof missing')
        export_state={k:sorted(v) if isinstance(v,set) else v for k,v in state.items()}
        turns.append({'name':turn,'state':export_state,'restore_log':str(path),'restore_log_sha256':sha(path)})
    ledger=budgets.validate_history(load(cfg['budget_ledger']));read(cfg['budget_ledger'])
    require(ledger['active'] is None,'unsettled held interval')
    require(len(ledger['turns'])==sum(s['held'] for s in states.values()),'held interval count differs')
    for turn in turns:
        require(abs(turn['state']['budget_event']['held_seconds']-ledger['held_seconds'])<1e-6,'step/ledger time differs')
    for p,h in pins.items():require(sha(p)==h,'input changed during audit')
    out=Path(out);out.mkdir(exist_ok=False)
    for name,values in [('rows.jsonl',output),('diagnostics.jsonl',diagnostics),('presentations.jsonl',allpresentations)]:
        with (out/name).open('x') as stream:
            for row in values:stream.write(old.client3.canonical(row)+'\n')
    artifacts={name:{'path':str(out/file),'sha256':sha(out/file)} for name,file in [('rows','rows.jsonl'),('diagnostics','diagnostics.jsonl'),('presentations','presentations.jsonl')]}
    result={'qualified':not gaps,'spec_sha256':SPEC,'grader_sha256':GRADER,'gaps':gaps,'observed':len(output),'scheduled':360,'missing':360-len(output),'outputs':artifacts,
            'provenance':{'input_file_sha256':pins},'boots':bootmeta,'turns':turns,'budget':ledger,'row_audits':row_audits,
            'native_presentations':{'observed':len(allpresentations),'clipped':sum(p['clipped'] for p in allpresentations),'qualified':all(p['qualified'] for p in allpresentations)}}
    (out/'raw-audit.json').write_text(json.dumps(result,indent=2));return result
def audit_or_block(campaign,out):
    """Preserve failed/raw/ambiguous attempts without accepting their attestations.

    A gate failure cannot prevent an explicit BLOCKED artifact. Raw-only responses
    remain observations; markers without a response remain ambiguous scheduled
    missingness. Neither is permission to retry a key.
    """
    try:return audit(campaign,out)
    except Exception as error:
        cfg=load(campaign);freeze=Path(cfg['freeze']);manifest=load(freeze/'manifest.json');byid=public_tasks()
        observations=[];attempts=[];pins={str(campaign):sha(campaign),str(Path(__file__).resolve()):sha(__file__),str(HERE/'dataset-v1/tasks.jsonl'):sha(HERE/'dataset-v1/tasks.jsonl')};scheduled=[]
        for entry in manifest['boots']:
            boot=load(freeze/entry['file']);turn=next(t['name'] for t in manifest['turns'] if entry['boot'] in t['boots'])
            directory=HERE/'runtime/runs'/(cfg['prefix']+'-'+turn)/entry['boot']
            templates={(r['study_arm'],tuple(r['key'])):r for r in boot['requests']}
            scheduled.extend((r['study_arm'],r['id'],r['seed']) for r in boot['requests'])
            processed=directory/'responses.jsonl';rawfile=directory/'raw-observations.jsonl'
            found={}
            if processed.exists():
                pins[str(processed)]=sha(processed)
                for row in rows(processed):
                    k=(row.get('study_arm'),(row['suite'],row['id'],row['seed']))
                    if k in templates:row=dict(row,family=byid[templates[k]['id']]['family'])
                    found[k]=row
            if rawfile.exists():
                pins[str(rawfile)]=sha(rawfile)
                for record in rows(rawfile):
                    if record.get('diagnostic'):continue
                    k=(record.get('study_arm'),tuple(record['template_key']))
                    if k not in templates:continue
                    if k not in found:
                        t=templates[k];row=dict(record['response'])
                        row.update(suite=t['suite'],id=t['id'],seed=t['seed'],study_arm=t['study_arm'],arm=t['arm'],boot=entry['boot'],family=byid[t['id']]['family'],
                                   answer_eligible=False,unqualified_raw_observation=True,audit_gap=str(error))
                        # A raw receipt does not acquire trusted usage or safety.
                        row.pop('native_accounting',None);row.pop('input_accounting',None)
                        row.pop('call_accounting',None);row.pop('initial_prompt_reconciliation',None)
                        found[k]=row
            for k,t in templates.items():
                if k in found:observations.append(found[k])
        marker_dir=HERE/'runtime/attempted'/cfg['prefix']
        if marker_dir.exists():
            for path in marker_dir.glob('*.json'):pins[str(path)]=sha(path);attempts.append(load(path))
        observed={(r['study_arm'],r['id'],r['seed']) for r in observations}
        missing=[{'study_arm':arm,'id':ident,'seed':seed} for arm,ident,seed in scheduled if (arm,ident,seed) not in observed]
        ambiguous=[m for m in attempts if (m['study_arm'],m['key'][1],m['key'][2]) not in observed]
        result={'qualified':False,'status':'BLOCKED','spec_sha256':SPEC,'gaps':[type(error).__name__+': '+str(error)],
                'observed':len(observations),'scheduled':360,'missing':len(missing),'missing_keys':missing,
                'ambiguous_attempts':ambiguous,'provenance':{'input_file_sha256':pins},'no_retry_authorization':True}
        out=Path(out);out.mkdir(exist_ok=True)
        # If the strict path wrote a partial artifact before failing, keep it;
        # fallback names never replace a previously emitted audit or rows.
        auditname='blocked-audit.json' if (out/'raw-audit.json').exists() else 'raw-audit.json'
        rowname='blocked-rows.jsonl' if (out/'rows.jsonl').exists() else 'rows.jsonl'
        with (out/rowname).open('x') as f:
            for row in observations:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        result['outputs']={'rows':{'path':str(out/rowname),'sha256':sha(out/rowname)}}
        with (out/auditname).open('x') as f:json.dump(result,f,indent=2)
        return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=audit_or_block(a.campaign,a.out);print(json.dumps({k:r[k] for k in ('qualified','observed','missing')}))
