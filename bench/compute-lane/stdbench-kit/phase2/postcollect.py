#!/usr/bin/env python3
"""Offline orchestration of frozen graders and review sealing; no inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import activation_report
import analysis
import client
import grade_all
import math_review

HERE=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines()]
def private_jsonl(p,values):
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:
        for row in values: f.write(client.canonical(row)+'\n')
def save(p,value):
    with Path(p).open('x') as f: json.dump(value,f,indent=2); f.write('\n')

def collect(data,out):
    receipt=load(HERE/'cpu-final-v3/verification.json')
    assert receipt['passed']
    for path,digest in receipt['code_sha256'].items():
        local=HERE.parent/Path(path).relative_to('<rank1-work>/stdbench')
        assert sha(local)==digest, 'frozen source drift: '+str(local)
    assert sha(HERE/'frozen-v3/manifest.json')==receipt['freeze_sha256']
    manifest=load(HERE/'frozen-v3/analysis-plain.json')
    dirs=[data/'stdbench-20261105-v1-pilot/pilot-plain',data/'stdbench-20261105-v2-pilot/pilot-lane']
    for turn in range(1,7):
        dirs += [data/f'stdbench-20261105-v2-t{turn:02}/t{turn:02}-{arm}' for arm in ('plain','lane')]
    observations=[row for d in dirs for row in rows(d/'responses.jsonl')]
    safety=[row for d in dirs for row in rows(d/'safety.jsonl')]
    grade_all.validate_responses(observations,manifest)
    assert len(observations)==368
    for row in observations:
        assert row['native_accounting']==client.accounting(row), 'native-accounting attestation changed'
    out.mkdir(exist_ok=False)
    save(out/'inputs.json',{'spec_sha256':receipt['spec_sha256'],'files_sha256':{str(d/n):sha(d/n) for d in dirs for n in ('responses.jsonl','diagnostics.jsonl','safety.jsonl')},'wrapper_sha256':sha(__file__)})
    activated=activation_report.report(observations,safety,manifest)
    save(out/'activation.json',activated)
    graded=grade_all.grade_all(observations,manifest,'sha256:80beaba732d1ed9439e012b007d8dbe19ec2f8962f50fbac29a18c8cd6fda306')
    private_jsonl(out/'all-provisional.jsonl',graded)
    private_jsonl(out/'nonmath-graded.jsonl',[r for r in graded if r['suite']!='math500'])
    sealed=math_review.seal(rows(HERE.parent/'math500-selected.jsonl'),graded,out/'math-public.json',out/'math-private.json',preseal_path=HERE/'MATH-PRESEAL-v1.json',preseal_sha256=receipt['math_preseal']['preseal_sha256'])
    save(out/'seal-receipt.json',sealed)
    print(client.canonical({'postcollection':'GRADED_AND_SEALED','cohort_replies':368,'math_candidates':sealed['candidates'],'public_packet_sha256':sealed['packet_sha256']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();collect(a.data_root,a.out)
