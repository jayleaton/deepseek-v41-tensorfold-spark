#!/usr/bin/env python3
"""Bind independent raw/replay/r7 evidence to the unchanged strict format analysis."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'compute-dl'
SPEC='b47ede697eed3a38d244ad5ff7447690732bd12945ff4592554d45bc2d8527ec'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashbytes=lambda b:hashlib.sha256(b).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def require(ok,why):
    if not ok:raise ValueError(why)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def validate_audit_outputs(raw,auditdir):
    outputs=raw.get('outputs') or {}
    require('rows' in outputs,'audited output manifest missing')
    if raw['qualified']:require(set(outputs)=={'rows','diagnostics','presentations'},'qualified audit output coverage')
    paths={}
    for name,entry in outputs.items():
        path=Path(entry['path'])
        require(path.resolve().is_relative_to(auditdir.resolve()),'audit output path escapes directory')
        require(sha(path)==entry['sha256'],'audited output drift: '+name)
        paths[name]=path
    return paths
def native_receipt(row,p,decoder,provenance):
    ident,callid=p['identity'],p['call_id']
    call=next(c for c in row['tensorfold']['compute']['calls'] if c['id']==callid)
    binding=next(c for c in row['native_binding']['calls'] if c['id']==callid)
    require(p['qualified'] is True and row['native_binding']['qualified'] is True and binding.get('native_bound') is True,'unqualified presentation/frame')
    require(binding['source_sha256']==p['source_sha256']==hashbytes(call['code'].encode()),'source binding drift')
    require(binding['output_sha256']==p['landed_body_sha256']==hashbytes(call['output'].encode()),'native output binding drift')
    frame=row['native_binding']['frames'][binding['native_frame']]
    ids=row['tensorfold']['token_ids'];framebytes=decoder.decode(ids[frame['eos_token']:frame['end_token']])
    full=decoder.decode(ids)
    require(full[binding['output_at']:binding['output_end']]==call['output'].encode(),'landed output byte offsets differ')
    reference=p['provenance']['result'];clipped=p['clipped']
    marker=('\n' if p['retained_prefix_tokens'] else '')+'[output truncated at 512 tokens]' if clipped else ''
    landed=call['output'].encode();prefix=len(landed)-len(marker.encode())
    require(not marker or landed.endswith(marker.encode()),'known policy marker differs')
    require(reference['status']==call['status'],'native/reference status mismatch')
    return dict(id=row['id'],seed=row['seed'],study_arm=row['study_arm'],call_id=callid,
        qualified=True,provenance=provenance,presentation_verified=True,full_reference_verified=True,
        policy_transform_verified=True,frame_binding_verified=True,source_binding_verified=True,
        native_prefix_exact=True,matches_bytes=True,in_text=True,native_frame_sha256=hashbytes(framebytes),
        native_status=call['status'],reference_status=reference['status'],source_sha256=p['source_sha256'],
        landed_output_sha256=p['landed_body_sha256'],expected_presentation_sha256=p['landed_body_sha256'],
        full_reference_sha256=p['full_reference_sha256'],full_reference_bytes=p['full_reference_bytes'],
        landed_output_bytes=p['landed_body_bytes'],full_reference_body_tokens=p['full_reference_tokens'],
        retained_prefix_bytes=prefix,retained_prefix_tokens=p['retained_prefix_tokens'],marker_bytes=marker,
        clipped=clipped,complete_value=not clipped)
def publish(campaign,auditdir,r7path,out):
    require(not out.exists(),'publication output must be fresh')
    cfg=load(campaign);rawpath=auditdir/('blocked-audit.json' if (auditdir/'blocked-audit.json').exists() else 'raw-audit.json')
    raw=load(rawpath);r7=load(r7path)
    require(raw['spec_sha256']==cfg['spec_sha256']==SPEC,'publication authority mismatch')
    for path,h in raw['provenance']['input_file_sha256'].items():require(sha(path)==h,'raw audit input drift')
    reporter=module(HERE/'analysis/reporter.py','_format_reporter')
    outputs=validate_audit_outputs(raw,auditdir);rowspath=outputs['rows']
    rows=[json.loads(l) for l in rowspath.read_text().splitlines()]
    tasks=[json.loads(l) for l in (HERE/'dataset-v1/tasks.jsonl').read_text().splitlines()]
    answers=HERE/'dataset-v1/reference-a.json'
    require(r7['input_rows_sha256']==sha(rowspath),'r7 row binding differs')
    require(r7['oracle_sha256']=='35f2bdf0243df7917b2b61098a1ff4a91bf1044854634912ab4800495db853fc','r7 pin')
    for p,h in cfg['code_sha256'].items():require(sha(p)==h,'sealed collection source drift')
    sys.path.insert(0,str(HERE.parent/'phase2'))
    from client import NativeDecoder
    decoder=NativeDecoder(cfg['tokenizer'],cfg['tokenizer_sha256'])
    indexed={(r['study_arm'],r['suite'],r['id'],r['seed'],r['boot']):r for r in rows}
    ps=[json.loads(l) for l in outputs['presentations'].read_text().splitlines()] if 'presentations' in outputs else []
    validate_audit_outputs(raw,auditdir)
    receipts=[]
    for p in ps:
        if p.get('diagnostic'):continue
        row=indexed[tuple(p['identity'])]
        receipts.append(native_receipt(row,p,decoder,{'raw_audit_path':str(rawpath),'raw_audit_sha256':sha(rawpath),
                         'presentations_path':str(auditdir/'presentations.jsonl'),'presentations_sha256':sha(auditdir/'presentations.jsonl')}))
    provenance={'raw_audit_path':str(rawpath),'raw_audit_sha256':sha(rawpath),'campaign_sha256':sha(campaign)}
    gates={name:dict(qualified=raw['qualified'],provenance=provenance) for name in ('sealed_inputs','disjointness','prompt_neutrality','configuration','dispatch_boundary','source_image_rank_environment','post_run_bindings')}
    r7meta=dict(r7,scope='owned_runner_cells',all_optional_runners_off=raw['qualified'],oracle_pinned=True,
                owned_cell_count=r7['runner_owned_count'],checked_owned_cell_count=r7['verdict']['records'])
    reportaudit={'kind':'qualified-real-collection','spec_sha256':SPEC,'grader_sha256':sha(OLD/'grade.py'),
        'input_sha256':{'rows':sha(rowspath),'tasks':sha(HERE/'dataset-v1/tasks.jsonl'),'answers':sha(answers)},
        'scheduled_keys':[reporter.key_json(k) for k in reporter.expected_grid(tasks)],
        'integrity':dict(qualified=raw['qualified'],provenance=provenance,**gates),
        'r7':r7meta,'native_replay':dict(qualified=raw['qualified'],provenance=provenance,independently_isolated_cpython=True,
            presentation_token_limit=512,clipping_policy_qualified=True,policy_source_sha256=sha(OLD/'a1-prep/reference_policy.py'),
            tokenizer_sha256=cfg['tokenizer_sha256'],wrong_values=0 if raw['qualified'] else None,
            wrong_bytes=0 if raw['qualified'] else None,not_in_text=0 if raw['qualified'] else None,
            unresolved_cuts=0 if raw['qualified'] else None,unknowns=0 if raw['qualified'] else None,
            policy_clipped_results=sum(p['clipped'] for p in receipts),receipts=receipts),
        'operations':{'budget':raw.get('budget'),'boots':raw.get('boots'),'turns':raw.get('turns'),
            'prompt_limitation':'CPU arrays, exact wires and served initial counts; actual served arrays unobserved.',
            'cache_state':'Fresh L and P cold boots; old/new interleaved within each. No KV-state equality claim.'}}
    target=auditdir/'report-input-audit.json'
    with target.open('x') as f:json.dump(reportaudit,f,indent=2)
    graded=reporter.load_grader(OLD/'grade.py')
    result=reporter.analyze(rows,tasks,load(answers),reportaudit,graded)
    result['input_sha256']=reportaudit['input_sha256']
    reporter.write_report(result,out)
    with (out/'NEEDS-T5.md').open('x') as f:f.write('# NEEDS-T5\n\nFormat re-test specification SHA256 `'+SPEC+'`.\n\nSee [results](RESULTS.md) and [complete report](report.json). No default promotion.\n')
    print(json.dumps({'report':str(out),'spec_sha256':SPEC,'raw_audit_qualified':raw['qualified'],'observed':raw['observed']}))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('campaign','audit','r7','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();publish(a.campaign,a.audit,a.r7,a.out)
