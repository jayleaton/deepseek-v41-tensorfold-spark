#!/usr/bin/env python3
"""Exercise actual entrypoints on CPU, including unapproved-entry refusal."""

import sys as _archive_sys
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import stage_control

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)
HERE=Path(__file__).resolve().parent
for path in HERE.glob('*.py'): ast.parse(path.read_text(),filename=str(path))
pf=subprocess.run(['bash','<rank0-work>/queue/turn-preflight.sh',str(HERE/'launch.sh'),str(HERE/'turn.sh')],
                  text=True,capture_output=True,timeout=7200)
assert pf.returncode==0,(pf.stdout,pf.stderr)
pre=stage_control.preflight()
for name in ('launch.sh','turn.sh'):
    staged=subprocess.run(['bash',str(HERE/name),'--preflight-only'],capture_output=True,text=True,timeout=7200)
    assert staged.returncode==0 and json.loads(staged.stdout)['collection_authorized'] is False
    stopped=subprocess.run(['bash',str(HERE/name)],capture_output=True,text=True,timeout=7200)
    assert stopped.returncode!=0 and 'SPEC_PENDING' in stopped.stderr
result={'passed':True,'GPU_requests':0,'preflight_stdout':pf.stdout,'preflight_stderr':pf.stderr,
        'preparation':pre,'unapproved_entries_refused':2,
        'checked_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in ('.py','.zig','.sh')}}
(HERE/'preflight-receipt.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'passed':True,'GPU_requests':0,'unapproved_entries_refused':2}))
