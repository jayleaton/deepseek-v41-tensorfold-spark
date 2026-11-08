#!/usr/bin/env python3
"""Actual diagnostic method with recorded/fake replies; no network or GPU calls."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import controller
import cpu_check
import protocol
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
load=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(l) for l in Path(p).read_text().splitlines() if l]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    for p in [*HERE.glob('*.py'),*(ROOT/'a1-prep').glob('*.py')]:ast.parse(p.read_text(),filename=str(p))
    fake=cpu_check.check()
    cfg=load(ROOT/protocol.PREFIX/'stage-campaign.json')
    original=ROOT/'runs/stdbench-compute-dl-20261105-v1-t01/t01-L'
    actual=rows(original/'diagnostics.jsonl');safeties=rows(original/'safety.jsonl')
    bykey={(r['suite'],r['id'],r['seed'],r['arm'],r['boot']):r for r in safeties}
    templates=load(ROOT/'stdbench-compute-dl-20261105-v1/frozen-v1/boots/t01-L.json')['diagnostics']
    controls=[]
    for missing in (None,0,1):
        turn=object.__new__(controller.Turn);turn.cfg=cfg
        turn.event=lambda *a,**k:None
        replies=copy.deepcopy(actual)
        safety=[copy.deepcopy(bykey[tuple(r[k] for k in ('suite','id','seed','arm','boot'))]) for r in actual]
        if missing is not None:
            replies[missing]['tensorfold']['compute']['calls']=[]
            safety[missing]['binding']['delivered_successes']=0
        it=iter(zip(replies,safety));turn.measure_one=lambda *a:next(it)
        try:turn.diagnostics({'arm':'lane','boot':'fake-only','diagnostics':templates})
        except RuntimeError:
            assert missing is not None
            controls.append('missing '+str(templates[missing]['reference'])+' delivery refused')
        else:
            assert missing is None
            controls.append('both323and55 independently delivered')
    lifecycle=[]
    for mode in ('success','failure'):
        receipt=load(HERE/f'lifecycle-{mode}.json')
        assert receipt['guarded'] is True
        assert receipt['expected_exit']==(0 if mode=='success' else 73)
        assert all(not Path(f'/proc/{pid}').exists() for pid in receipt['owned_pids']), 'settled verifier producer survived'
        lifecycle.append({'mode':mode,'owned_exact_pids_exited':True,'receipt_sha256':sha(HERE/f'lifecycle-{mode}.json')})
    shells=[HERE/n for n in ('launch.sh','turn.sh','cpu_gate.sh','fence.sh','lifecycle-launch.sh')]
    import subprocess
    result=subprocess.run(['bash','<rank0-work>/queue/turn-preflight.sh',*map(str,shells)],capture_output=True,text=True,timeout=7200)
    assert result.returncode==0,(result.stdout,result.stderr)
    proof={'passed':True,'spec_sha256':protocol.QUALIFIED_SPEC_SHA256,'fake':fake,'diagnostic_controls':controls,
           'sigpipe_lifecycle':lifecycle,'preflight':{'passed':True,'stdout':result.stdout,'stderr':result.stderr},
           'code_sha256':{str(p):sha(p) for p in [*HERE.glob('*.py'),*HERE.glob('*.sh')]},'GPU_requests':0}
    (ROOT/protocol.PREFIX/'control-check.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps({'passed':True,'diagnostic_controls':controls,'sigpipe_lifecycle':True,'GPU_requests':0}))
if __name__=='__main__':main()
