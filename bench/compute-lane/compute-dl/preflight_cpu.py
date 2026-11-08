#!/usr/bin/env python3
"""Retain actual shell preflight output without queue or GPU actions."""

import sys as _archive_sys
import hashlib
import json
from pathlib import Path
import subprocess
from runtime_stage import PREFIX,SPEC_SHA256

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)
HERE=Path(__file__).resolve().parent
ROOT=HERE/PREFIX
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
for name in ('launch.sh','turn.sh','fence.sh'):
    source=HERE/name
    run=subprocess.run(['bash','<rank0-work>/queue/turn-preflight.sh',str(source)],
                       capture_output=True,text=True,timeout=7200)
    check={'script':str(source),'sha256':sha(source),'returncode':run.returncode,
           'stdout':run.stdout,'stderr':run.stderr}
    checks.append(check)
with (ROOT/'cpu-preflight.json').open('x') as output:
    json.dump({'passed':all(c['returncode']==0 for c in checks),'spec_sha256':SPEC_SHA256,
               'checks':checks,'source_sha256':sha(__file__)},output,indent=2)
if any(c['returncode'] for c in checks):raise RuntimeError('preflight failed; inspect receipt')
print(json.dumps({'preflight':'PASS','scripts':len(checks)}))
