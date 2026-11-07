#!/usr/bin/env python3
"""Pin original pilot requests in place; never read generated replies as tasks."""
import hashlib
import json
from pathlib import Path
from exclusion_sources import PILOT_IDS

root=Path(__file__).resolve().parent
base=Path('<rank0-work>/dsv41-bench/out/imr-measure-20261104T074954Z/ab-a-plain-lane/a-s4')
items=[]
def requests(value,pointer=''):
    found=[]
    if isinstance(value,dict):
        if isinstance(value.get('messages'),list) and any(m.get('role')=='user' for m in value['messages'] if isinstance(m,dict)):
            found.append((pointer,value))
        for key,item in value.items():
            found.extend(requests(item,pointer+'/'+key.replace('~','~0').replace('/','~1')))
    elif isinstance(value,list):
        for index,item in enumerate(value): found.extend(requests(item,pointer+'/'+str(index)))
    return found
for identifier in sorted(PILOT_IDS):
    path=base/(identifier+'.json');raw=path.read_bytes()
    found=requests(json.loads(raw))
    if len(found)!=1: raise ValueError('ambiguous original pilot request: '+identifier)
    pointer,body=found[0]
    assert len([m for m in body['messages'] if m.get('role')=='user'])==1
    items.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),
                  'id':identifier,'request_pointer':pointer})
with (root/'exclusions-config.json').open('x') as output:
    json.dump({'pilot_receipts':items},output,indent=2);output.write('\n')
print(json.dumps({'original_pilot_requests_pinned':len(items),'generated_reply_text_used':False}))
