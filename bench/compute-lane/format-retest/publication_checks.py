#!/usr/bin/env python3
"""Actual historical native receipt controls; never new-study collection evidence."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import publish
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'compute-dl'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    for p in [*HERE.glob('*.py'),*(HERE/'runtime').glob('*.py'),*(HERE/'analysis').glob('*.py')]:ast.parse(p.read_bytes(),filename=str(p))
    sys.path.insert(0,str(HERE.parent/'phase2'));from client import NativeDecoder
    sys.path.insert(0,str(OLD/'a1-prep'));from reference_policy import NativeCap,reference_body
    cap=NativeCap(OLD/'a1-prep/zig-out/bin/native-cap',HERE.parent/'phase2/references/tokenizer.json',HERE.parent/'decl-control/dispatch-high',OLD/'a1-prep/native-build-receipt.json')
    dec=NativeDecoder(HERE.parent/'phase2/references/tokenizer.json','c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b')
    historical_path=OLD/'a2-report/final-audit-v1/audited-rows.jsonl';selected={}
    with historical_path.open() as f:
        for line in f:
            row=json.loads(line)
            if row.get('arm')!='lane':continue
            for ref in row['safety']['references']:
                call=next(c for c in row['tensorfold']['compute']['calls'] if c['id']==ref['call_id'])
                reference=reference_body(ref['result']);r=cap.cap([reference])[0];clipped=r['token_count']>512
                if clipped in selected:continue
                assert call['output']==r['output'] and call['status']==ref['result']['status']
                row['study_arm']='L-old'
                p={'identity':['L-old',row['suite'],row['id'],row['seed'],row['boot']],'call_id':call['id'],'qualified':True,
                   'source_sha256':hashlib.sha256(call['code'].encode()).hexdigest(),'landed_body_sha256':hashlib.sha256(call['output'].encode()).hexdigest(),
                   'full_reference_sha256':hashlib.sha256(reference).hexdigest(),'full_reference_bytes':len(reference),'landed_body_bytes':len(call['output'].encode()),
                   'full_reference_tokens':r['token_count'],'retained_prefix_tokens':r['retained_prefix_tokens'],'clipped':clipped,'provenance':ref}
                selected[clipped]=(copy.deepcopy(row),p)
            if len(selected)==2:break
    assert len(selected)==2,'actual known clipped/unclipped receipts required'
    reporter=publish.module(HERE/'analysis/reporter.py','_format_real_presentation_check')
    controls=[]
    for clipped,(row,p) in selected.items():
        call=next(c for c in row['tensorfold']['compute']['calls'] if c['id']==p['call_id'])
        result=publish.native_receipt(row,p,dec,{'fixture':'Historical actual A2 receipts, never new study evidence'})
        assert reporter.native_receipt_verified(result,call) and result['complete_value'] is (not clipped)
        controls.append('actual clipped policy' if clipped else 'actual complete value')
        for label,mutate in [('changed landed bytes',lambda r,q:next(c for c in r['tensorfold']['compute']['calls'] if c['id']==q['call_id']).update(output='altered')),
                             ('unqualified presentation',lambda r,q:q.update(qualified=False)),
                             ('status mismatch',lambda r,q:q['provenance']['result'].update(status='exit_error'))]:
            damaged=copy.deepcopy(row);other=copy.deepcopy(p);mutate(damaged,other)
            try:publish.native_receipt(damaged,other,dec,{'fixture':True})
            except (ValueError,KeyError):controls.append(label+' rejected')
            else:raise AssertionError('accepted '+label)
    with tempfile.TemporaryDirectory(prefix='format-output-binding-',dir=HERE) as folder:
        root=Path(folder)
        for name in ('rows','diagnostics','presentations'):(root/(name+'.jsonl')).write_text('{"content":"7"}\n')
        raw={'qualified':True,'outputs':{n:{'path':str(root/(n+'.jsonl')),'sha256':sha(root/(n+'.jsonl'))} for n in ('rows','diagnostics','presentations')}}
        assert set(publish.validate_audit_outputs(raw,root))=={'rows','diagnostics','presentations'}
        for name in ('rows','presentations'):
            path=root/(name+'.jsonl');before=path.read_bytes();path.write_text('{"content":"8"}\n')
            try:publish.validate_audit_outputs(raw,root)
            except ValueError:controls.append('changed audited '+name+' rejected')
            else:raise AssertionError('unbound audit output '+name)
            path.write_bytes(before)
    proof={'passed':True,'spec_sha256':publish.SPEC,'GPU_requests':0,'historical_fixture_only':True,'controls':controls,
           'actual_historical_rows_sha256':sha(historical_path),'actual_historical_rows_path':str(historical_path),
           'source_sha256':{str(p):sha(p) for p in (Path(__file__),HERE/'publish.py',HERE/'audit_collection.py',HERE/'analysis/reporter.py')}}
    with (HERE/'publication-check.json').open('x') as f:json.dump(proof,f,indent=2)
    print(json.dumps({'passed':True,'actual_native_cases':2,'mutation_controls':8,'model_requests':0}))
if __name__=='__main__':main()
