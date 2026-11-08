#!/usr/bin/env python3
"""Read-only effective process-environment audit; no container/process operations."""

import sys as _archive_sys
import hashlib
import json
from pathlib import Path

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)
root=Path(__file__).resolve().parent
proofs=[]
for turn in ('pilot','t01','t02','t03','t04','t05','t06'):
    for arm in ('plain','lane'):
        version='v1' if turn=='pilot' and arm=='plain' else 'v2'
        prefix='stdbench-20261105-'+version
        boot=turn+'-'+arm
        manifest=root/prefix/'boots'/boot/'manifest.json'
        data=json.loads(manifest.read_text())
        for rank in (0,1):
            directory=(root/'post-proof-v1/rank0'/boot if rank==0 else root/prefix/'proof'/boot)
            records=[]
            for path in directory.glob('*.json'):
                record=json.loads(path.read_text())
                if record.get('command')=='snapshot' and record.get('rank')==rank and record.get('dry_run') is False:
                    records.append((path,record))
            assert len(records)==1, 'one actual snapshot required: '+boot+' rank'+str(rank)
            path,record=records[0]
            assert record['present'] is True and record['boot']==boot
            assert record['manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
            assert record['image_id']==data['arms']['D']['ranks'][rank]['image_id']
            env=record['native_process']['env']
            assert record['native_process']['pid_namespace']>1
            expected=data['phase2']['ranks'][rank]['effective_env']
            for key,value in expected.items():
                if key.startswith(('TF_','GLM53_TF','NCCL')):
                    assert env.get(key)==value, 'effective drift: '+boot+' '+key
            assert env['TF_DSV41_COMPUTE_LANE']==str(int(arm=='lane'))
            for key,value in env.items():
                assert not any(word in key for word in ('KV_SPLIT_DEDUP','KV_SPLIT_EARLY','TF_TP_ROCE_CHUNKED_KB'))
                if any(word in key for word in ('RUNNER','TRACE','GUIDE','STATED','COMMIT','SYNC_GRAPH')):
                    assert key in expected, 'unsealed optional flag: '+key
            proofs.append({'boot':boot,'arm':arm,'rank':rank,'snapshot_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'manifest_sha256':record['manifest_sha256'],'image_id':record['image_id'],'qualified':True})
assert len(proofs)==28
with (root/'post-v1/environment-audit.json').open('x') as f: json.dump({'qualified':True,'rank_snapshots':28,'proofs':proofs},f,indent=2);f.write('\n')
print(json.dumps({'effective_environment_audit':'PASS','rank_snapshots':28}))
