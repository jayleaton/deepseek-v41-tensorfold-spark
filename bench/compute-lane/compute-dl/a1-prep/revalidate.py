#!/usr/bin/env python3
"""Uniform offline A1 re-gate, consuming original raw bytes and independent replays."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from reference_policy import NativeCap,reference_body,sha
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
A1='0dca1fa52838388ea1e7afbc3a4c5e9747775f64e871640df7a66496369e5453'
BASE='c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141'
OLD='stdbench-compute-dl-20261105-v1'
load=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines() if x]
digest=lambda b:hashlib.sha256(b).hexdigest()
def require(ok,why):
    if not ok: raise ValueError(why)
def main():
    cfg=load(ROOT/OLD/'campaign-v1.json')
    require(cfg['spec_sha256']==BASE,'old spec identity')
    require(sha(ROOT/'T5-COMPUTE-DL-SPEC-A1.md')==A1,'A1 bytes drift')
    policy=load(HERE/'a1-policy-check-receipt.json')
    require(policy.get('passed') is True and policy.get('spec_sha256')==A1,'A1 clipping controls missing')
    require(policy['native_build_receipt_sha256']==sha(HERE/'native-build-receipt.json'),'native policy build changed')
    for path,expected in policy['code_sha256'].items():require(sha(path)==expected,'policy control code drift')
    require(sha(ROOT/'remaining-collection.json')=='6fdf99b3dbfd5d77e9ae3c04c3a23c85c18e78a0762933561aa34c1fbd57da43','remaining inventory drift')
    require(sha(ROOT/'report-v1/audit.json')=='b29e6dc6b7b791bb771c5249571e9d9d9d83b6dbdfe9a0253674a09b758b1feb','original audit drift')
    oldaudit=load(ROOT/'report-v1/audit.json')
    pins=dict(cfg['code_sha256'])
    pins.update(oldaudit['provenance']['input_file_sha256'])
    for path,expected in pins.items(): require(sha(path)==expected,'original source/evidence drift: '+path)
    mod=importlib.util.spec_from_file_location('original_collect_report',ROOT/'collect_report.py')
    old=importlib.util.module_from_spec(mod);mod.loader.exec_module(old)
    cap=NativeCap(HERE/'zig-out/bin/native-cap',cfg['tokenizer'],ROOT.parent/'decl-control/dispatch-high',HERE/'native-build-receipt.json')
    decoder=old.client3.NativeDecoder(cfg['tokenizer'],cfg['tokenizer_sha256'])
    worker=sha(ROOT.parent/'phase2/jail_worker.py')
    prompts={tuple(r['key']):r for r in rows(cfg['cpu_prompt_arrays'])}
    freeze=Path(cfg['freeze']); manifest=load(freeze/'manifest.json')
    originals=[]; observed=set(); grid=set(); sourcehashes={}; pairs=[]
    for boot in manifest['boots']:
        template=load(freeze/boot['file'])
        arm={'D':'declared_off','L':'lane'}[boot['arm']]
        for t in template['requests']: grid.add((t['id'],t['seed'],boot['arm']))
        folder=ROOT/'runs'/(OLD+'-'+boot['boot'].split('-')[0])/boot['boot']
        if not (folder/'responses.jsonl').exists():
            require(not (folder/'raw-observations.jsonl').exists(),'unrecorded original attempt')
            continue
        for name in ('responses.jsonl','diagnostics.jsonl','safety.jsonl','raw-observations.jsonl','rank0.log','snapshot-r0.log','snapshot-r1.log'):
            path=folder/name
            require(path.exists(),'missing original native/evidence file '+str(path))
            sourcehashes[str(path)]=sha(path)
        safeties=old.indexed(rows(folder/'safety.jsonl')); transports=rows(folder/'raw-observations.jsonl')
        allseen=set(); native_by_key={}
        for filename,templates,diag in [('responses.jsonl',template['requests'],False),('diagnostics.jsonl',template['diagnostics'],True)]:
            actual=rows(folder/filename)
            expected=[dict(t,arm=arm,boot=boot['boot']) for t in templates]
            require([old.key(r) for r in actual]==[old.key(r) for r in expected[:len(actual)]],'original prefix order mismatch')
            for raw,t in zip(actual,expected):
                k=old.key(raw);require(k not in allseen,'duplicate original receipt');allseen.add(k)
                require(k in safeties,'missing original safety sidecar')
                if not diag:
                    logical=(raw['id'],raw['seed'],boot['arm'])
                    require(logical not in observed,'duplicate observed task/seed/arm');observed.add(logical)
                safety=copy.deepcopy(safeties[k]);calls=((raw.get('tensorfold') or {}).get('compute') or {}).get('calls') or []
                byid={r['call_id']:r for r in safety.get('references',[])}
                require(len(byid)==len(safety.get('references',[])),'duplicate replay ID')
                for call in calls:
                    if call.get('inserted') is True:
                        require(call['id'] in byid,'missing independent replay')
                        ref=byid[call['id']]; body=reference_body(ref['result'])
                        pairs.append((call,ref,body,k))
                originals.append((raw,safety,t,diag))
                native_by_key[(tuple(t['key']),diag)]=raw
        require(set(safeties)==allseen,'original safety coverage mismatch')
        transportkeys=[(tuple(t['template_key']),t['diagnostic']) for t in transports]
        require(len(transportkeys)==len(set(transportkeys))==len(allseen),'durable attempted key coverage mismatch')
        require(set(transportkeys)==set(native_by_key),'durable attempt/response identities mismatch')
        for t in transports:
            raw=native_by_key[(tuple(t['template_key']),t['diagnostic'])]
            require(all(t['response'].get(f)==raw.get(f) for f in ('sse_frames','sse_done','actual_request_wire','elapsed_seconds','transport_status','content','reasoning','usage','tensorfold')),'raw transport mismatch')
        http=old.audit_records.http_audit((folder/'rank0.log').read_text(),len(allseen))
        require(http['qualified'],'native HTTP POST coverage mismatch')
    require(len(observed)==342 and len(grid)==480,'original observed/grid coverage differs')
    remain=load(ROOT/'remaining-collection.json')['remaining_original_requests']
    missing={(r['id'],r['seed'],r['arm']) for r in remain}
    require(len(missing)==138 and grid-observed==missing and not observed&missing,'observed/unobserved mask differs')
    require(('compute-dl-v1-aggregation-015',101,'L') in observed,'stopped receipt must remain observed')
    expected=cap.cap([p[2] for p in pairs])
    cache={}; presentations=[]
    for (call,ref,body,k),row in zip(pairs,expected,strict=True):
        result=ref['result']; landed=call['output'].encode()
        ok=call.get('status')==result.get('status') and landed==row['output'].encode()
        ref['matches']=ok
        key=digest(old.client3.canonical([call,result]).encode());cache[key]=ok
        clipped=row['token_count']>512
        presentations.append({'identity':list(k),'call_id':call['id'],'qualified':ok,
            'classification':'presentation-verified clipped' if clipped and ok else 'presentation-verified unclipped' if ok else 'unqualified',
            'full_reference_tokens':row['token_count'],'retained_prefix_tokens':row['retained_prefix_tokens'],
            'full_reference_bytes':len(body),'landed_body_bytes':len(landed),'full_reference_sha256':digest(body),
            'landed_body_sha256':digest(landed),'marker_bytes':list(b'\n[output truncated at 512 tokens]') if clipped else [],
            'authenticates':'landed presentation, not invisible tail or complete uncropped answer'})
    def compare(call,result): return cache.get(digest(old.client3.canonical([call,result]).encode()),False)
    old.reference_matches=compare
    qualified=[]; audits=[]; activation=set(); controls=[]
    for raw,safety,t,diag in originals:
        calls=((raw.get('tensorfold') or {}).get('compute') or {}).get('calls') or []
        gaps=old.reference_audit(calls,safety,worker)
        safety['qualified']=safety.get('binding',{}).get('qualified') is True and not gaps
        row,audit=old.audit_row(raw,safety,t,prompts[tuple(t['key'])],decoder,worker,diag)
        require(audit['qualified'],'retained receipt failed new uniform audit: '+str(audit))
        audits.append(audit)
        if not diag:
            row['retention_batch']='before-amendment'; qualified.append(row)
            if row['arm']=='lane' and row['native_binding'].get('delivered_successes',0)>0: activation.add(row['id'])
        # Prove actual native source/frame and status are necessary, not just landed prefix.
        if not controls and calls:
            for mode in ('source_swap','missing_frame','status_swap'):
                mutation=copy.deepcopy(raw)
                if mode=='missing_frame': mutation['sse_frames']=[]
                else:
                    c=mutation['tensorfold']['compute']['calls'][0]
                    c['code' if mode=='source_swap' else 'status']='print(0)' if mode=='source_swap' else 'timeout'
                _,bad=old.audit_row(mutation,safety,t,prompts[tuple(t['key'])],decoder,worker,diag)
                require(not bad['qualified'],'binding mutation accepted: '+mode)
                controls.append(mode)
    require(len(qualified)==342 and len(activation)>=30,'retained coverage/activation failed')
    counts={a:sum(r['arm']==alias for r in qualified) for a,alias in [('D','declared_off'),('L','lane')]}
    require(counts=={'D':120,'L':222},'retained arm counts differ')
    for path,expectedhash in pins.items():require(sha(path)==expectedhash,'input changed during audit')
    out=HERE/'revalidation-v2';out.mkdir(exist_ok=False)
    for name,values in [('audited-rows.jsonl',qualified),('presentations.jsonl',presentations)]:
        with (out/name).open('x') as f:
            for value in values:f.write(old.client3.canonical(value)+'\n')
    mask={'original_grid':sorted(map(list,grid)),'observed':sorted(map(list,observed)),
          'unobserved':sorted(map(list,missing)),'raw_source_sha256':sourcehashes}
    (out/'mask.json').write_text(json.dumps(mask,indent=2))
    (out/'audit.json').write_text(json.dumps({'qualified':True,'rows':audits,'binding_mutations_rejected':controls},indent=2))
    result={'passed':True,'spec_sha256':A1,'parent_spec_sha256':BASE,'observed':342,'remaining':138,
            'counts':counts,'activation_distinct_L':len(activation),'raw_source_sha256':sourcehashes,
            'preserved_original_seal':True,'original_sealed_files':len(cfg['code_sha256']),
            'all_input_sha256':pins,'new_code_sha256':{str(p):sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.zig','.sh')},
            'clipped_presentations':sum(p['classification']=='presentation-verified clipped' for p in presentations),
            'independent_replays_reused':len(pairs),'GPU_requests':0,
            'files_sha256':{p.name:sha(p) for p in out.iterdir()},'native_build_receipt_sha256':sha(HERE/'native-build-receipt.json')}
    result['policy_controls_sha256']=sha(HERE/'a1-policy-check-receipt.json')
    (out/'qualification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('passed','observed','remaining','activation_distinct_L','clipped_presentations','GPU_requests')}))
if __name__=='__main__':main()
