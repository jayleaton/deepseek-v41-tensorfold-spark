#!/usr/bin/env python3
"""CPU-only budget-refusal and partial-boot report controls; never starts a turn."""
import copy
import hashlib
import json
from pathlib import Path
from audit_lifecycle import classify,allow_partial_boot
HERE=Path(__file__).resolve().parent

def main():
    e=lambda s,v,**kw:dict(step=s,verdict=v,**kw)
    refusal=[e('preflight','PASS'),e('collection','INCOMPLETE',reason='4h25m cumulative admission cutoff: no new request or boot'),e('turn','INCOMPLETE')]
    state=classify(refusal);assert not state['held'] and state['budget_stop']
    controls=['unheld prequeue budget refusal remains incomplete without fake restoration/charge']
    held=[e('queue','APPENDED_AT_TAIL'),e('queue','AT_HEAD'),e('G-stop','REQUESTED'),e('gpu-locks','HELD'),
          e('owned-containers','DRAIN_VERIFIED',boot='P'),e('G-restore','SERVING_VERIFIED'),
          e('budget','RECORDED',held_seconds=15901),*refusal[-2:]]
    state=classify(held);assert state['held']
    assert allow_partial_boot(state,'P',0,0,0)
    controls.append('cooling/pre-ready cutoff needs zero attempts and drain')
    ready=held[:-2]+[e('boot','READY_EFFECTIVE_ENV_VERIFIED',boot='P')]+held[-2:]
    state=classify(ready)
    assert allow_partial_boot(state,'P',0,0,0) and allow_partial_boot(state,'P',0,1,0)
    controls.append('ready partial diagnostics cutoff keeps missingness')
    ready.insert(-1,e('diagnostics','QUALIFIED',boot='P'))
    assert not allow_partial_boot(classify(ready),'P',1,2,1)
    controls.append('qualified ready boot may retain paired request prefix')
    def reject(name,fn):
        try:fn()
        except ValueError:controls.append(name)
        else:raise AssertionError('accepted '+name)
    reject('missing restoration refused',lambda:classify([s for s in held if s['step']!='G-restore']))
    reject('unheld GPU activity refused',lambda:classify(refusal+[e('response','RECORDED')]))
    reject('unready dataset attempt refused',lambda:allow_partial_boot(classify(held),'P',0,0,1))
    reject('unready lacks drain refused',lambda:allow_partial_boot(classify([s for s in held if s['step']!='owned-containers']),'P',0,0,0))
    reject('dataset before full diagnostic gate refused',lambda:allow_partial_boot(classify(ready),'P',1,1,1))
    bad=copy.deepcopy(ready)
    for s in bad:
        if s['step']=='collection':s['reason']='something else'
    reject('partial diagnostics without budget cutoff refused',lambda:allow_partial_boot(classify(bad),'P',0,1,0))
    proof={'passed':True,'controls':controls,'GPU_requests':0,'source_sha256':{
        str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),HERE/'audit_lifecycle.py',HERE/'audit_collection.py')}}
    (HERE/'lifecycle-check-receipt.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps({'passed':True,'controls':len(controls),'GPU_requests':0}))
if __name__=='__main__':main()
