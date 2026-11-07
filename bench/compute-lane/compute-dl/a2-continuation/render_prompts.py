#!/usr/bin/env python3
"""Bind already-qualified native prompt arrays to identical fresh-stage inputs."""
import hashlib
import json
from pathlib import Path
from runtime_stage import PREFIX,SPEC_SHA256,TOKENIZER_SHA256

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/PREFIX
PRIOR=HERE.parent/'stdbench-compute-a1-20261105-v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())

def main():
    campaign=load(PRIOR/'campaign-v1.json')
    old=load(PRIOR/'prompt-proof.json')
    prior_receipt=Path(campaign['cpu_receipt'])
    cpu=load(prior_receipt)
    def require(ok,reason):
        if not ok:raise ValueError(reason)
    require(sha(prior_receipt)==campaign['cpu_receipt_sha256']
            and cpu['passed'] is True and cpu['campaign_sha256']==__import__('protocol').campaign_digest(campaign), 'prior CPU seal drift')
    require(cpu['preparation_file_sha256'][str(PRIOR/'prompt-proof.json')]==sha(PRIOR/'prompt-proof.json'),'qualified native prompt proof drift')
    require(old['passed'] is True and old['spec_sha256']=='0dca1fa52838388ea1e7afbc3a4c5e9747775f64e871640df7a66496369e5453' and old['pairs']==242,'prior native proof differs')
    require(sha(PRIOR/'prompt-arrays.jsonl')==old['arrays_sha256']==campaign['cpu_prompt_arrays_sha256']
            and sha(ROOT/'prompt-inputs.json')==sha(PRIOR/'prompt-inputs.json')==old['input_sha256'],'prompt inputs or qualified arrays changed')
    legacy=HERE.parent.parent/'decl-control'
    require(sha(legacy/'prompt-gate')==old['binary_sha256']
            and sha(HERE.parent.parent/'phase2/references/tokenizer.json')==old['tokenizer_sha256']==TOKENIZER_SHA256,'native renderer/tokenizer drift')
    inputs=load(ROOT/'prompt-inputs.json')
    rows=[json.loads(line) for line in (PRIOR/'prompt-arrays.jsonl').read_text().splitlines()]
    require(len(inputs)==len(rows)==len({tuple(r['key']) for r in rows})==242,'prompt coverage differs')
    for row,request in zip(rows,inputs,strict=True):
        require(row['key']==request['key'] and row['plain_ids'] and row['declared_off_ids']
                and row['declared_off_ids']==row['declared_lane_ids']==row['original_builtin_lane_ids']
                and row['held_off'] is False and row['held_lane'] is True,'prompt qualification differs')
    with (ROOT/'prompt-arrays.jsonl').open('xb') as out:out.write((PRIOR/'prompt-arrays.jsonl').read_bytes())
    proof=dict(old,spec_sha256=SPEC_SHA256,source_sha256=sha(__file__),reuse=True,
               prior_proof_path=str(PRIOR/'prompt-proof.json'),prior_proof_sha256=sha(PRIOR/'prompt-proof.json'),
               prior_cpu_receipt_path=str(prior_receipt),prior_cpu_receipt_sha256=sha(prior_receipt))
    with (ROOT/'prompt-proof.json').open('x') as out:json.dump(proof,out,indent=2)
    print(json.dumps({'passed':True,'pairs':242,'reused_qualified_arrays':True,'GPU_requests':0}))

if __name__=='__main__':main()
