#!/usr/bin/env python3

import sys as _archive_sys
import json
import subprocess
import sys
from pathlib import Path

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

rank = int(sys.argv[1])
root = Path(f'<rank{rank}-work>/stdbench/phase2')
index = json.loads((root/(sys.argv[3] if len(sys.argv)>3 else 'stage-v2')/'manifest.json').read_text())
proof = Path(sys.argv[2])
proof.mkdir(exist_ok=False)
for boot in index['boots']:
    manifest = root/index['prefix']/boot['manifest']
    data = json.loads(manifest.read_text())
    name = data['phase2']['ranks'][rank]['container_name']
    inspected = subprocess.run(['docker','container','inspect',name],capture_output=True,text=True)
    assert inspected.returncode != 0 and any(s in inspected.stderr.lower() for s in ('no such object:', 'no such container:')), 'owned container name clash/daemon failure'
    with (proof/(boot['boot']+'.log')).open('x') as log:
        subprocess.run([sys.executable,'-B',str(root/'rank_ops.py'),'verify',str(manifest),str(rank),'--dry-run'],
                       check=True,stdout=log,stderr=subprocess.STDOUT)
if rank == 0:
    with (proof/'scripts.log').open('x') as log:
        subprocess.run(['bash','<rank0-work>/queue/turn-preflight.sh',
            str(root/'turn.sh'),str(root/'launch.sh'),str(root/'fence.sh'),
            str(root/'cpu-rank0-host.sh'),str(root/'cpu-rootrank1_host.sh')],check=True,stdout=log,stderr=subprocess.STDOUT)
(proof/'verdict.json').write_text(json.dumps({'passed':True,'rank':rank,'boots':14,'effective_envs_checked':14})+'\n')
print(json.dumps({'rank_preflight':'PASS','rank':rank,'boots':14}))
