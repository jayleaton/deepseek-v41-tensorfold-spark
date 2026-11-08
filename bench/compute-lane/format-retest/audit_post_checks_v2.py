#!/usr/bin/env python3
"""Qualify only the postcollection manifest-lookup correction; no inference."""
import ast
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
import audit_collection_post_v2 as auditor

HERE=Path(__file__).resolve().parent
def save(path,value):path.write_text(json.dumps(value)+'\n')
def main():
    ast.parse(Path(auditor.__file__).read_text())
    cfg=auditor.load(HERE/'stdbench-format-20261107-v1/campaign-v1.json')
    byid=auditor.public_tasks();freeze=Path(cfg['freeze']);manifest=auditor.load(freeze/'manifest.json')
    templates=[t for b in manifest['boots'] for t in auditor.load(freeze/b['file'])['requests']]
    assert len(templates)==360 and all('task' not in t and t['id'] in byid for t in templates)
    assert all(byid[t['id']]['family'] in ('integer','modular','division','dp','graph','aggregation') for t in templates)
    selected=[t for t in templates if byid[t['id']]['family']=='integer'][:3]
    with tempfile.TemporaryDirectory(prefix='format-postaudit-controls-',dir=HERE) as folder:
        root=Path(folder);f=root/'freeze';f.mkdir();(root/'dataset-v1').mkdir()
        shutil.copyfile(HERE/'dataset-v1/tasks.jsonl',root/'dataset-v1/tasks.jsonl')
        rt=root/'runtime/runs/prefix-format/format-L';rt.mkdir(parents=True)
        save(f/'manifest.json',{'turns':[{'name':'format','boots':['format-L']}],'boots':[{'boot':'format-L','file':'L.json'}]})
        save(f/'L.json',{'requests':selected});save(root/'campaign.json',{'freeze':str(f),'prefix':'prefix'})
        raw={'status':'ok','content':'7','native_accounting':{'complete':True}}
        save(rt/'raw-observations.jsonl',{'diagnostic':False,'study_arm':selected[0]['study_arm'],'template_key':selected[0]['key'],'response':raw})
        t=selected[1];save(rt/'responses.jsonl',dict(study_arm=t['study_arm'],suite=t['suite'],id=t['id'],seed=t['seed'],status='transport_error',boot='format-L'))
        markers=root/'runtime/attempted/prefix';markers.mkdir(parents=True)
        save(markers/'ambiguous.json',dict(study_arm=selected[2]['study_arm'],key=selected[2]['key'],boot='format-L'))
        with patch.object(auditor,'HERE',root),patch.object(auditor,'audit',side_effect=ValueError('controlled failure')):
            result=auditor.audit_or_block(root/'campaign.json',root/'out')
        observed=auditor.rows(root/'out/rows.jsonl')
        assert result['status']=='BLOCKED' and not result['qualified'] and result['observed']==2
        assert len(result['missing_keys'])==len(result['ambiguous_attempts'])==1 and result['no_retry_authorization']
        assert all(r['family']=='integer' for r in observed)
        assert observed[0]['unqualified_raw_observation'] and not observed[0]['answer_eligible'] and 'native_accounting' not in observed[0]
        assert observed[1]['status']=='transport_error'
    receipt=dict(passed=True,actual_templates_without_nested_task=360,blocked_fallback_controls=3,model_requests=0,
        source_sha256={str(p):auditor.sha(p) for p in (Path(__file__),Path(auditor.__file__),HERE/'audit_collection.py')})
    with (HERE/'audit-post-check-v2.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(dict(passed=True,actual_templates=360,fallback_preserved=True,model_requests=0)))
if __name__=='__main__':main()
