#!/usr/bin/env python3
"""Read-only exact-name admission check; never stops or removes a container."""

import sys as _archive_sys
import json
from pathlib import Path
import sys
from generate_envs3 import PREFIX, IMAGE_IDS
from rank_ops3 import inspect_container

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)
root=Path(__file__).resolve().parent
rank=int(sys.argv[1]); assert rank in (0,1)
index=json.loads((root/PREFIX/'manifest.json').read_text())
names=[]
for entry in index['boots']:
    manifest=json.loads((root/PREFIX/entry['manifest']).read_text())
    name=manifest['phase2']['ranks'][rank]['container_name']
    assert inspect_container(name,IMAGE_IDS[rank]) is None, 'fresh boot name already exists'
    names.append(name)
with (root/f'clashes-r{rank}.json').open('x') as f:
    json.dump({'verdict':'PASS','rank':rank,'exact_names_absent':names},f,indent=2); f.write('\n')
print(json.dumps({'clashes':'PASS','rank':rank,'exact_names_absent':len(names),'GPU':False}))
