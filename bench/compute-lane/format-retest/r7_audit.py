#!/usr/bin/env python3
"""Run unchanged pinned r7 in a CPU-only no-network container; retain its coverage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid
HERE=Path(__file__).resolve().parent
ORACLE_SHA='35f2bdf0243df7917b2b61098a1ff4a91bf1044854634912ab4800495db853fc'
IMAGE='sha256:5be7d7d40f267590b9e7b41410f48d40048af24e004203629a347d3d7c16a8cd'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,why):
    if not ok:raise ValueError(why)
def run(rows,oracle,out):
    require(sha(oracle)==ORACLE_SHA,'agreed r7 bytes differ')
    out=Path(out);out.mkdir(exist_ok=False)
    inputs=out/'receipts';inputs.mkdir();result=out/'oracle-output';result.mkdir()
    original_sha=sha(rows);count=0;owned=0
    with Path(rows).open() as source:
        for count,line in enumerate(source,1):
            row=json.loads(line)
            runner=(row.get('tensorfold') or {}).get('runner') or {}
            owned+=sum((runner.get(k) or {}).get('owned',0) for k in ('inline','trace','fence','repl'))
            with (inputs/f'{count:04d}.json').open('x') as f:json.dump(row,f)
    if count==0:
        proof={'qualified':False,'executed':False,'oracle_sha256':ORACLE_SHA,'input_rows_sha256':original_sha,
               'observations':0,'runner_owned_count':0,'verdict':{'verdict':'UNEXERCISED','records':0},
               'wrong_values':None,'wrong_bytes':None,'not_in_text':None,'cuts':None,'unknowns':None,
               'provenance':{'oracle_path':str(oracle),'source_sha256':sha(__file__),'reason':'No observations: r7 was not executed.'},
               'limitation':'No owned-cell or native-compute safety result can be inferred; publish BLOCKED/INCOMPLETE with all scheduled keys retained.'}
        with (out/'r7.json').open('x') as f:json.dump(proof,f,indent=2)
        return proof
    name='stdbench-format-r7-'+uuid.uuid4().hex
    command=['docker','run','--rm','--pull','never','--name',name,'--runtime','runc',
        '--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
        '--user',f'{os.getuid()}:{os.getgid()}','--pids-limit','128','--memory','768m','--memory-swap','768m','--cpus','1',
        '--tmpfs','/tmp:rw,nosuid,size=64m','--env','NVIDIA_VISIBLE_DEVICES=void','--env','CUDA_VISIBLE_DEVICES=',
        '--mount',f'type=bind,src={Path(oracle).resolve().parent},dst=/tool,readonly',
        '--mount',f'type=bind,src={inputs.resolve()},dst=/receipts,readonly',
        '--mount',f'type=bind,src={result.resolve()},dst=/out']
    if os.environ.get('STDBENCH_CPU_JOB'):command+=['--label','stdbench.cpu_job='+os.environ['STDBENCH_CPU_JOB']]
    script='import subprocess,sys\nfor a in [("extract","/out","/receipts"),("evaluate","/out"),("report","/out"),("gate","/out")]:\n r=subprocess.run([sys.executable,"-I","/tool/oracle.py",*a]);\n if r.returncode:sys.exit(r.returncode)\n'
    command+=['--entrypoint','/usr/bin/nice',IMAGE,'-n','19','ionice','-c3','timeout','7200','python3','-I','-c',script]
    with (out/'run.log').open('x') as log:
        process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
    gate=result/'gate.json'
    require(gate.exists(),'r7 produced no verdict; inspect retained run.log')
    verdict=json.loads(gate.read_text())
    proof={'qualified':verdict.get('verdict')=='PASS','oracle_sha256':ORACLE_SHA,'input_rows_sha256':original_sha,
        'observations':count,'runner_owned_count':owned,'verdict':verdict,
        'wrong_values':verdict.get('wrong-value',0),'wrong_bytes':verdict.get('wrong-span',0),
        'not_in_text':verdict.get('not-in-text',0),'cuts':verdict.get('cut',0),
        'unknowns':sum(verdict.get(k,0) for k in ('unknown','unparsed','inconclusive','short')),
        'provenance':{'gate_path':str(gate),'gate_sha256':sha(gate),'source_sha256':sha(__file__),
                      'oracle_path':str(oracle),'container_name':name,'image_id':IMAGE,'subprocess_rc':process.returncode},
        'limitation':'r7 covers owned runner cells; zero owned cells is unexercised coverage. Native compute results are separately source/frame/isolated-CPython/presentation audited.'}
    with (out/'r7.json').open('x') as f:json.dump(proof,f,indent=2)
    require(sha(rows)==original_sha,'rows changed during r7')
    return proof
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rows',type=Path,required=True);p.add_argument('--oracle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=run(a.rows,a.oracle,a.out);print(json.dumps({'qualified':r['qualified'],'observations':r['observations'],'runner_owned_count':r['runner_owned_count']}))
