#!/usr/bin/env python3
"""Reuse the pinned native CPU renderer; never contact a server."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from runtime_stage import PREFIX,SPEC_SHA256,TOKENIZER_SHA256

HERE=Path(__file__).resolve().parent
ROOT=HERE/PREFIX
LEGACY=HERE.parent/'decl-control'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    old=json.loads((LEGACY/'cpu-proof.json').read_text())
    binary=LEGACY/'prompt-gate'
    tokenizer=HERE.parent/'phase2/references/tokenizer.json'
    if sha(binary)!=old['binary_sha256'] or sha(tokenizer)!=TOKENIZER_SHA256:
        raise ValueError('CPU renderer/tokenizer pin drift')
    inputs=json.loads((ROOT/'prompt-inputs.json').read_text())
    if len(inputs)!=242 or len({tuple(r['key']) for r in inputs})!=242:
        raise ValueError('CPU renderer grid differs')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void',OMP_NUM_THREADS='1',MAKEFLAGS='-j1')
    arrays=ROOT/'prompt-arrays.jsonl'
    with arrays.open('x') as output,(ROOT/'render.log').open('x') as log:
        result=subprocess.run([str(binary),str(ROOT/'prompt-inputs.json'),str(tokenizer)],
                              stdout=output,stderr=log,env=env,timeout=7200)
    if result.returncode: raise RuntimeError('native CPU render failed; retain log')
    rows=[json.loads(line) for line in arrays.read_text().splitlines()]
    if len(rows)!=242: raise ValueError('native CPU render coverage differs')
    for row,request in zip(rows,inputs):
        if (row['key']!=request['key'] or not row['declared_off_ids']
                or row['declared_off_ids']!=row['declared_lane_ids']
                or row['declared_lane_ids']!=row['original_builtin_lane_ids']
                or row['held_off'] is not False or row['held_lane'] is not True):
            raise ValueError('native CPU prompt/arming mismatch')
    proof={'passed':True,'spec_sha256':SPEC_SHA256,'pairs':242,'dataset_pairs':240,
           'diagnostic_pairs':2,'arrays_sha256':sha(arrays),'input_sha256':sha(ROOT/'prompt-inputs.json'),
           'binary_sha256':sha(binary),'tokenizer_sha256':sha(tokenizer),
           'source_sha256':sha(__file__),'actual_served_array_equality_observed':False,
           'limitation':'CPU-matched/count-checked prompts; same-length server rendering differences remain unobserved.'}
    with (ROOT/'prompt-proof.json').open('x') as output:json.dump(proof,output,indent=2)
    print(json.dumps(proof))

if __name__=='__main__': main()
