#!/usr/bin/env python3
"""Focused checks for final review corrections; no generation or model request."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import qualify_cpu
import publish
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    for p in HERE.rglob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
    proof=qualify_cpu.audit_checks()
    sys.path.insert(0,str(HERE/'analysis'))
    import handwritten_checks,reporter
    tasks,answers,_=handwritten_checks.fixture()
    grader=reporter.load_grader(HERE.parent/'compute-dl/grade.py')
    result=reporter.analyze([],tasks,answers,{},grader)
    assert result['status']=='BLOCKED' and result['collection_status']=='INCOMPLETE'
    assert len(result['missing_keys'])==len(result['ledger'])==360 and result['observed']==0
    assert all(r['returned_ids'] is None for r in result['ledger'])
    with tempfile.TemporaryDirectory(prefix='format-empty-report-',dir=HERE) as folder:
        reporter.write_report(result,Path(folder)/'report')
        assert 'INCOMPLETE' in (Path(folder)/'report/RESULTS.md').read_text()
    receipt={'passed':True,'spec_sha256':publish.SPEC,'controls':proof,
             'zero_observations':{'reported':'BLOCKED/INCOMPLETE','missing':360,'unknown_costs':True,'fixture_outputs_cleaned':True},
             'source_sha256':{str(p):sha(p) for p in (Path(__file__),HERE/'audit_collection.py',HERE/'publish.py',HERE/'qualify_cpu.py',HERE/'analysis/reporter.py')}}
    with (HERE/'audit-check-v3.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({'passed':True,'retained_failure_report':True,'zero_observation_report':True,'model_requests':0}))
if __name__=='__main__':main()
