#!/usr/bin/env python3
"""Read-only exclusion discovery, run only in a guarded Spark CPU scope."""

import sys as _archive_sys
from collections import Counter
import hashlib
import json
from pathlib import Path

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('<rank0-work>')
w9=base/'imr-t5-w9/frozen'
manifest=json.loads((w9/'manifest.json').read_text())
path=w9/'tasks-public.json'
assert sha(path)==manifest['files_sha256']['tasks-public.json']
tasks=json.loads(path.read_text());assert len(tasks)==450
ab=base/'ab3-20261104'
freeze=json.loads((ab/'AB3-FREEZE.json').read_text())
units=ab/'kit2/ab3-units.jsonl'
assert sha(units)==freeze['units_sha256']
rows=[json.loads(line) for line in units.read_text().splitlines() if line]
assert len(rows)==60
print(json.dumps({'W9':{'source':str(path),'sha256':sha(path),'count':len(tasks),
                       'family_counts':Counter(t['set'] for t in tasks)},
                  'AB3':{'source':str(units),'sha256':sha(units),'count':len(rows)},
                  'spec_shapes':Counter(json.dumps(sorted(t.get('spec',{}))) for t in tasks)}))
for family in sorted({t['set'] for t in tasks}):
    selected=[t for t in tasks if t['set']==family]
    if family in ('algo','bignum'):
        for t in selected:
            print(json.dumps({'source_task':{k:t[k] for k in ('id','set','prompt','spec') if k in t}}))
    else:
        t=selected[0]
        print(json.dumps({'sample_source_task':{k:t[k] for k in ('id','set','prompt','spec') if k in t}}))
cfg=json.loads((ab/'pilot-ab-b-configs.json').read_text())
def paths(value,key=''):
    if isinstance(value,dict):
        for k,v in value.items(): paths(v,key+'/'+k)
    elif isinstance(value,list):
        for i,v in enumerate(value): paths(v,key+'/'+str(i))
    elif isinstance(value,str) and value.startswith('/') and value.endswith(('.json','.jsonl')):
        print(json.dumps({'pilot_source_path':value,'field':key}))
paths(cfg)
