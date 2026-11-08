#!/usr/bin/env python3
"""CPU dry-run against the node's real original adapter, assets and Docker."""

import sys as _archive_sys
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from generate_envs3 import PREFIX

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

root = Path(__file__).resolve().parent
rank = int(sys.argv[1])
assert rank in (0,1)
index = json.loads((root/PREFIX/'manifest.json').read_text())
receipts = []
for boot in index['boots']:
    manifest = root/PREFIX/boot['manifest']
    result = subprocess.run([sys.executable,'-B',str(root/'rank_ops3.py'),'verify',str(manifest),str(rank),'--dry-run'],
                            capture_output=True,text=True,timeout=7200)
    log = root/f'preflight-{boot["boot"]}-r{rank}.log'
    with log.open('x') as f: f.write(result.stdout); f.write(result.stderr)
    assert result.returncode==0, f'rank dry-run failed: {log.name}'
    receipts.append({'boot':boot['boot'],'rank':rank,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                     'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
proof = {'verdict':'PASS','rank':rank,'real_adapter_dry_runs':len(receipts),'GPU':False,'receipts':receipts}
with (root/f'rank-preflight-r{rank}.json').open('x') as f: json.dump(proof,f,indent=2); f.write('\n')
print(json.dumps({'rank_preflight':'PASS','rank':rank,'boots':len(receipts),'GPU':False}))
