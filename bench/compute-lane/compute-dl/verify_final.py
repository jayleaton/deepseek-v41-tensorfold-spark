#!/usr/bin/env python3
"""Final real-artifact/liveness check; no inference or GPU queue mutation."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import urllib.request

HERE=Path(__file__).resolve().parent
PREFIX='stdbench-compute-dl-20261105-v1'
SPEC='c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('rank',type=int,choices=(0,1));args=parser.parse_args()
    root=HERE/PREFIX;cfg=load(root/'campaign-v1.json')
    for source,expected in cfg['code_sha256'].items():
        path=Path(source.replace('<rank0-home>/',f'<rank{args.rank}-home>/',1))
        if sha(path)!=expected:raise ValueError('sealed source/data drift: '+str(path))
    if (root/'GO').exists() or (root/'GO.REVOKED-safety').read_text().strip()!=SPEC:raise ValueError('safety GO not revoked')
    for source in HERE.glob('*.py'):ast.parse(source.read_bytes(),filename=str(source))
    bench=HERE.parent.parent/'dsv41-bench'
    expected=load(bench/'arms-gx.json')['arms']['D']['ranks'][args.rank]['image_id']
    name='tensorfold-zig-r'+str(args.rank)
    container=json.loads(subprocess.check_output(['docker','container','inspect',name],text=True,timeout=30))[0]
    if container['Image']!=expected or container['State']['Running'] is not True:raise ValueError('G image/running mismatch')
    module=importlib.util.spec_from_file_location('compute_rank_final',HERE/'rank_ops.py')
    rank_ops=importlib.util.module_from_spec(module);module.loader.exec_module(rank_ops)
    native=rank_ops.engine_environment(name,args.rank)
    if native['pid_namespace']<=1:raise ValueError('native G process absent')
    names=subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True,timeout=30).splitlines()
    if any(n.startswith(PREFIX+'-') for n in names):raise ValueError('owned study container remains')
    result={'passed':True,'rank':args.rank,'spec_sha256':SPEC,'sealed_files_unchanged':len(cfg['code_sha256']),
            'G_image_id':container['Image'],'G_native_pid_namespace':native['pid_namespace'],
            'G_native_role':'serve' if args.rank==0 else 'follow','owned_study_containers':0,
            'collection_GO_revoked':True,'source_sha256':sha(__file__)}
    if args.rank==0:
        if Path('<rank0-work>/queue/ORDER').read_text().strip():raise ValueError('queue not empty')
        budget=load(root/'budget.json')
        if budget['active'] is not None or budget['overrun_seconds']!=0:raise ValueError('budget interval remains active/overrun')
        for turn,want in (('t01','QUALIFIED'),('t02','STOPPED')):
            steps=[json.loads(line) for line in (HERE/'runs'/(PREFIX+'-'+turn)/'steps.jsonl').read_text().splitlines()]
            if steps[-1]['step']!='turn' or steps[-1]['verdict']!=want:raise ValueError('actual turn log verdict differs')
            if sum(s['step']=='G-restore' and s['verdict']=='SERVING_VERIFIED' for s in steps)!=1:raise ValueError('G restore proof absent')
        publication=load(HERE/'publication-proof.json')
        for key,path in (('results_sha256','RESULTS.md'),('per_problem_sha256','per-problem.csv'),
                         ('supplementary_telemetry_sha256','supplementary-telemetry.json')):
            if sha(HERE/path)!=publication[key]:raise ValueError('published artifact drift')
        with (HERE/'per-problem.csv').open() as source:rows=list(csv.DictReader(source))
        if len(rows)!=480 or sum(r['missing']=='True' for r in rows)!=138:raise ValueError('CSV missingness grid differs')
        counts={arm:(sum(r['solved']=='1' for r in rows if r['arm']==arm),
                     sum(r['missing']=='False' for r in rows if r['arm']==arm)) for arm in ('D','L')}
        if counts!={'D':(1,120),'L':(169,222)}:raise ValueError('published observed solve counts differ')
        clipping=load(HERE/'diagnosis-v1/truncation-proof.json')
        if clipping['passed'] is not True or clipping['native_output_sha256']!=clipping['expected_clipped_output_sha256']:
            raise ValueError('actual clipping proof failed')
        with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=5) as response:
            result['models_http_status']=response.status
            result['served_model_id']=json.load(response)['data'][0]['id']
        if result['models_http_status']!=200:raise ValueError('G models endpoint failed')
        result.update(observed=342,missing=138,observed_solves=counts,held_seconds=budget['held_seconds'],
                      results_sha256=publication['results_sha256'],primary_claim='WITHHELD_INCOMPLETE')
    with (HERE/f'final-verification-r{args.rank}.json').open('x') as output:json.dump(result,output,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
