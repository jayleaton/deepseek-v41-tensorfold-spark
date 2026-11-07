#!/usr/bin/env python3
"""Bound controls and real retained-row compatibility; no collection APIs."""
import contextlib
import hashlib
import json
from pathlib import Path
import fakecheck
import collect_report
from pins import SPEC_SHA256
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    before={str(p):sha(p) for p in HERE.glob('*.py')}
    with (HERE/'qualification-check.log').open('x') as log,contextlib.redirect_stdout(log):fakecheck.main()
    vr=ROOT/'a1-prep/revalidation-v2'
    rows=[json.loads(l) for l in (vr/'audited-rows.jsonl').read_text().splitlines()]
    presentations={(tuple(p['identity']),p['call_id']):p for p in map(json.loads,(vr/'presentations.jsonl').read_text().splitlines())}
    for r in rows:
        key=tuple(r[k] for k in ('suite','id','seed','arm','boot'))
        for ref in r['safety']['references']:ref['presentation']=presentations[key,ref['call_id']]
    tasks=[json.loads(l) for l in (ROOT/'dataset-v1/tasks-public.jsonl').read_text().splitlines()]
    answers=json.loads((ROOT/'dataset-v1/answers-a.json').read_text())
    report=collect_report.collect(rows,tasks,answers,metadata={'integrity':{
        'qualified':True,'spec_sha256':SPEC_SHA256,'provenance':{'qualification_sha256':sha(vr/'qualification.json')}}})
    analysis=report['analysis'];assert analysis['observed']==342 and analysis['status']=='INCOMPLETE'
    assert analysis['arms']['L']['solved']==169 and analysis['arms']['D']['solved']==1
    assert analysis['activation']['distinct_tasks']==112
    assert analysis['contrasts']['L_minus_P']['overall']['sign_flip'] is None
    assert before=={str(p):sha(p) for p in HERE.glob('*.py')}
    result={'passed':True,'spec_sha256':SPEC_SHA256,'full_20000_draw_fake_controls':True,
            'real_retained_observed':342,'strict_original_solve_counts_preserved':{'D':1,'L':169},
            'retained_activation':112,'incomplete_primary_withheld':True,'source_sha256':before,
            'log_sha256':sha(HERE/'qualification-check.log'),'GPU_requests':0}
    (HERE/'qualification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'passed':True,'real_retained_observed':342,'draws':20000,'GPU_requests':0}))
if __name__=='__main__':main()
