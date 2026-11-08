#!/usr/bin/env python3
"""Actual native-adapter dry runs, fresh names and effective env on one CPU rank."""

import sys as _archive_sys
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import protocol
import rank_ops
from coordination import check

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)
HERE=Path(__file__).resolve().parent
rank=int(sys.argv[1]);root=HERE.parent/protocol.PREFIX
cfg=json.loads((root/'campaign-v1.json').read_text())
assert not (root/"GO").exists(),"GO must be absent during rank qualification"
names=subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True,timeout=7200).splitlines()
assert not any(n.startswith(protocol.PREFIX+'-') for n in names),'new namespace clash'
coordination=check()
proofs=[]
for boot in cfg['layout']['boots']:
    manifest=root/f'rank{rank}/boots'/boot['boot']/'manifest.json'
    before=set((root/'proof'/boot['boot']).glob('verify-*.json')) if (root/'proof'/boot['boot']).exists() else set()
    run=subprocess.run([sys.executable,str(HERE/'rank_ops.py'),'verify',str(manifest),str(rank),'--dry-run'],capture_output=True,text=True,timeout=7200)
    if run.returncode:raise RuntimeError(run.stderr or run.stdout)
    records=[json.loads(l) for l in run.stdout.splitlines() if l.startswith('{')]
    assert len(records)==1
    p=Path(records[0]['proof']);assert p not in before
    proof=json.loads(p.read_text());rm=json.loads(manifest.read_text())
    expected=rm['phase2']['ranks'][rank]['effective_env']
    actual=proof['container_env']
    selected=lambda e:{k:v for k,v in e.items() if k.startswith(('TF_','GLM53_TF','NCCL'))}
    assert selected(actual)==selected(expected)
    assert proof['image_id']==cfg['image_ids'][rank] and proof['dry_run'] is True
    assert actual['TF_DSV41_COMPUTE_LANE']==str(int(boot['arm']=='L'))
    for k,v in rm['phase2']['ranks'][rank]['expected_off'].items():assert actual.get(k)==v=='0'
    proofs.append({'boot':boot['boot'],'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                   'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'effective_env':actual})
result={'passed':True,'rank':rank,'fresh_namespace_clear':True,'actual_native_adapter_dry_runs':len(proofs),
        'coordination':coordination,'proofs':proofs,'GPU_requests':0,'spec_sha256':protocol.QUALIFIED_SPEC_SHA256,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root/f'rank-check-r{rank}.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'passed':True,'rank':rank,'native_dry_runs':len(proofs),'GPU_requests':0}))
