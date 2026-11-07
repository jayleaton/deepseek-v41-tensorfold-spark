#!/usr/bin/env python3
"""Actual bounded CPU qualifications, never a model or queue action."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'compute-dl'
SPEC='b47ede697eed3a38d244ad5ff7447690732bd12945ff4592554d45bc2d8527ec'
GRADER='1a2249a1b5e0b06f03a4300995d0c320e5a21630d261b3791501f7c83740fb9f'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def save(path,value):
    with Path(path).open('x') as out:json.dump(value,out,indent=2)
def runcheck(script,args=()):
    process=subprocess.run([sys.executable,'-B',str(script),*map(str,args)],capture_output=True,text=True,timeout=7200)
    if process.returncode:raise RuntimeError(process.stdout+'\n'+process.stderr)
    return {'passed':True,'stdout':process.stdout,'stderr':process.stderr,'source_sha256':sha(script),'returncode':process.returncode}
def refs():
    ds=HERE/'dataset-v1';tasks=[json.loads(l) for l in (ds/'tasks.jsonl').read_text().splitlines()];pin=load(HERE/'generation-pin-v1.json')
    refs=[load(ds/n) for n in ('reference-a.json','reference-b.json','process-reference-a.json','process-reference-b.json')]
    assert all(r==refs[0] for r in refs)
    for letter in ('a','b'):
        record=load(ds/f'process-reference-{letter}.json.receipt.json')
        assert record['status']=='PASS' and record['count']==60 and record['input_sha256']==sha(ds/'tasks.jsonl')
        assert record['source_sha256']==pin['legacy_source_pins']['reference_'+letter]
        assert record['output_sha256']==sha(ds/f'process-reference-{letter}.json') and record['python_version']==pin['python_version']
    save(ds/'independent-process-verification.json',{'passed':True,'spec_sha256':SPEC,'tasks':60,'separate_nodes':['rank0-host:A','rank1-host:B'],
        'tasks_sha256':sha(ds/'tasks.jsonl'),'source_sha256':{str(OLD/f'reference_{l}.py'):pin['legacy_source_pins']['reference_'+l] for l in ('a','b')},
        'artifact_sha256':{str(p):sha(p) for p in ds.glob('*.json')}})
    return {'passed':True,'tasks':len(tasks)}
def audit_checks():
    spec=importlib.util.spec_from_file_location('_format_audit_controls',HERE/'audit_collection.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.path.insert(0,str(HERE/'analysis'))
    import handwritten_checks,reporter
    tasks,answers,_=handwritten_checks.fixture()
    integer_tasks=[t for t in tasks if t['family']=='integer'][:3]
    with tempfile.TemporaryDirectory(prefix='format-audit-fixtures-',dir=HERE) as folder:
        root=Path(folder);freeze=root/'freeze';freeze.mkdir();rt=root/'runtime/runs/prefix-format/format-L';rt.mkdir(parents=True)
        templates=[dict(study_arm='L-old',key=['compute_dl',t['id'],101],suite='compute_dl',id=t['id'],seed=101,arm='lane',task=t) for t in integer_tasks]
        save(freeze/'manifest.json',{'turns':[{'name':'format','boots':['format-L']}],'boots':[{'boot':'format-L','file':'L.json'}]})
        save(freeze/'L.json',{'requests':templates})
        save(root/'campaign.json',{'freeze':str(freeze),'prefix':'prefix'})
        raw={'status':'ok','content':'7','native_accounting':{'complete':True,'returned_tokens':10}}
        (rt/'raw-observations.jsonl').write_text(json.dumps({'diagnostic':False,'study_arm':'L-old','template_key':templates[0]['key'],'response':raw})+'\n')
        (rt/'responses.jsonl').write_text(json.dumps(dict(templates[1],status='transport_error',content='',boot='format-L'))+'\n')
        marker=root/'runtime/attempted/prefix';marker.mkdir(parents=True)
        save(marker/'fake.json',dict(study_arm='L-old',key=templates[2]['key'],boot='format-L'))
        with patch.object(module,'HERE',root),patch.object(module,'audit',side_effect=ValueError('handwritten accounting failure')):
            result=module.audit_or_block(root/'campaign.json',root/'result')
        assert result['qualified'] is False and result['status']=='BLOCKED' and result['observed']==2
        assert len(result['ambiguous_attempts'])==1 and len(result['missing_keys'])==1 and result['no_retry_authorization'] is True
        observed=module.rows(root/'result/rows.jsonl')
        assert observed[0]['unqualified_raw_observation'] and observed[0]['answer_eligible'] is False and 'native_accounting' not in observed[0]
        assert observed[1]['status']=='transport_error'
        assert all(r['family']=='integer' for r in observed)
        indexed=reporter.normalize(observed,tasks,answers,reporter.load_grader(OLD/'grade.py'))
        assert len(indexed)==2 and len(reporter.observation_ledger(tasks,indexed))==360
        assert sum(not r['observed'] for r in reporter.observation_ledger(tasks,indexed))==358
        assert raw['native_accounting']['complete'] is True  # Original raw evidence unchanged.
    return {'passed':True,'controls':['raw-only response retained unqualified','processed transport error retained','marker-only attempt stays ambiguous/missing','no retry authorization','fixtures cleaned']}
def basics():
    for p in HERE.rglob('*.py'):
        if 'dataset-v1' not in p.parts:ast.parse(p.read_bytes(),filename=str(p))
    assert sha(HERE/'T5-FORMAT-RETEST-SPEC.md')==SPEC and sha(OLD/'grade.py')==GRADER
    grader=runcheck(OLD/'outcome_check.py')
    save(HERE/'grader-check-v2.json',dict(passed=True,spec_sha256=SPEC,positive_negative_fixtures=True,result=grader,
         source_sha256={str(p):sha(p) for p in (OLD/'grade.py',OLD/'outcome_check.py',OLD/'analysis.py',Path(__file__))}))
    analysis=runcheck(HERE/'analysis/handwritten_checks.py',['--grader',OLD/'grade.py'])
    proof=dict(passed=True,spec_sha256=SPEC,scheduled_replies=360,strict_grader_sha256=GRADER,result=analysis,
               source_sha256={str(p):sha(p) for p in [*(HERE/'analysis').glob('*.py'),Path(__file__)]})
    save(HERE/'analysis-check-v2.json',proof)
    audit=audit_checks();save(HERE/'audit-check-v2.json',dict(passed=True,spec_sha256=SPEC,controls=audit,
        source_sha256={str(p):sha(p) for p in (HERE/'audit_collection.py',Path(__file__))}))
    print(json.dumps({'passed':True,'grader':True,'analysis':True,'audit_failures':True,'model_requests':0}))
def inherited():
    dispatch=HERE.parent/'decl-control/dispatch-high'
    proof=dispatch/'dispatch-proof.json'
    assert sha(proof)=='77e43ebc13afdc16a8304836869953f8ceadfc8c6775b8f551aceb3b15af0436'
    d=load(proof);assert d['verdict']=='PASS' and d['observed_fixtures']==8 and d['whole_stream_noncall_and_wrong_arm_controls'] is True
    pins={str(proof):sha(proof)}
    for rel,h in d['sha256'].items():
        path=dispatch/rel;assert path.resolve().is_relative_to(dispatch.resolve()) and sha(path)==h;pins[str(path)]=h
    path=dispatch/'unmodified-source-sha256.json';pins[str(path)]=sha(path);sources=load(path);assert len(sources)==3436
    for rel,h in sources.items():
        path=dispatch/rel;assert path.resolve().is_relative_to(dispatch.resolve()) and sha(path)==h;pins[str(path)]=h
    r7=load(HERE/'r7-adapter-check.json');assert r7['passed'] and r7['oracle_sha256']=='35f2bdf0243df7917b2b61098a1ff4a91bf1044854634912ab4800495db853fc'
    pins[str(HERE/'r7-adapter-check.json')]=sha(HERE/'r7-adapter-check.json')
    for p,h in r7['source_sha256'].items():
        assert sha(p)==h
        target=HERE/'r7/oracle.py' if p.endswith('/owned-oracle/tool/oracle.py') else Path(p)
        assert sha(target)==h
        pins[str(target)]=h
    save(HERE/'dispatch-check.json',dict(passed=True,spec_sha256=SPEC,dispatch_proof_sha256=sha(proof),image_tag='tensorfold-dsv41-imr:r-3a429d6f122d',
        image_ids=['sha256:5be7d7d40f267590b9e7b41410f48d40048af24e004203629a347d3d7c16a8cd','sha256:80beaba732d1ed9439e012b007d8dbe19ec2f8962f50fbac29a18c8cd6fda306'],
        engine_revision='3a429d6f122d9eb2900c43e622933d311e4be44f',artifact_sha256=pins,source_sha256={str(Path(__file__)):sha(__file__)},
        r7_coverage='CPU adapter controls only before collection; actual pinned r7 owned-cell audit after collection, distinct from native compute replay'))
    print(json.dumps({'passed':True,'dispatch_fixtures':8,'production_sources':len(sources),'model_requests':0}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['references','basics','inherited']);a=p.parse_args()
    if a.action=='references':print(json.dumps(refs()))
    elif a.action=='basics':basics()
    else:inherited()
