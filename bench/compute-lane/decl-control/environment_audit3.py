#!/usr/bin/env python3
"""Postcollection audit of actual native-process snapshots; no live operations."""
import hashlib
import json
from pathlib import Path
import sys
from generate_envs3 import PREFIX, ORDERS, ZERO_FLAGS, IMAGE_IDS

root=Path(__file__).resolve().parent
evidence=Path(sys.argv[1])
out=Path(sys.argv[2])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proofs=[]
for turn,arms in ORDERS:
    for arm in arms:
        boot=turn+'-'+arm
        manifest=root/PREFIX/'boots'/boot/'manifest.json'
        data=json.loads(manifest.read_text())
        for rank in (0,1):
            records=[]
            for path in (evidence/f'rank{rank}'/boot).glob('*.json'):
                record=json.loads(path.read_text())
                if record.get('command')=='snapshot' and record.get('rank')==rank and record.get('dry_run') is False:
                    records.append((path,record))
            assert len(records)==1, 'one actual snapshot required per boot/rank'
            path,record=records[0]
            assert record['present'] is True and record['boot']==boot
            assert record['manifest_sha256']==sha(manifest)
            assert record['image_id']==IMAGE_IDS[rank]
            native=record['native_process']
            assert native['pid_namespace']>1
            env=native['env']; expected=data['phase2']['ranks'][rank]['effective_env']
            for key,value in expected.items():
                if key.startswith(('TF_','GLM53_TF','NCCL')): assert env.get(key)==value, 'effective flag drift'
            assert env['TF_DSV41_COMPUTE_LANE']==str(int(arm=='lane'))
            for flag in ZERO_FLAGS: assert env['TF_DSV41_'+flag]=='0'
            for key,value in env.items():
                assert not any(word in key for word in ('KV_SPLIT_DEDUP','KV_SPLIT_EARLY','TF_TP_ROCE_CHUNKED_KB'))
                if any(word in key for word in ('RUNNER','TRACE','GUIDE','S3','STATED','COMMIT','SYNC_GRAPH')):
                    assert key in expected, 'unsealed optional feature'
            proofs.append({'boot':boot,'arm':arm,'rank':rank,'snapshot_sha256':sha(path),
                           'manifest_sha256':sha(manifest),'image_id':record['image_id'],
                           'pool_policy':'source-qualified absent' if arm!='lane' else 'source-qualified enabled',
                           'qualified':True})
assert len(proofs)==18
with out.open('x') as f:json.dump({'qualified':True,'rank_snapshots':18,'proofs':proofs},f,indent=2);f.write('\n')
print(json.dumps({'environment_audit':'PASS','actual_native_snapshots':18}))
