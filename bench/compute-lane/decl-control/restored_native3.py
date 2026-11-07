#!/usr/bin/env python3
"""Direct read-only G liveness probe; run through the guarded Spark CPU scope."""
import json
import os
from pathlib import Path
import sys
import urllib.request
from rank_ops3 import engine_environment

rank=int(sys.argv[1]); assert rank in (0,1)
root=Path(__file__).resolve().parent
native=engine_environment('tensorfold-zig-r'+str(rank),rank)
assert native['pid_namespace']>1
proof={'passed':True,'rank':rank,'native_pid_namespace':native['pid_namespace'],
       'binary':native['argv'][0],'role':'serve' if rank==0 else 'follow',
       'endpoint_completion_requested':False}
if rank==0:
    with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=30) as response:
        assert response.status==200
        data=json.load(response)
    assert data.get('data')
    proof['models_http_status']=200
    proof['model_ids']=[row['id'] for row in data['data']]
out=root/('G-native-alive-r'+str(rank)+'-a1-v1.json')
fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as output:
    json.dump(proof,output,indent=2);output.write('\n')
print(json.dumps({'G_native_liveness':'PASS','rank':rank,'pid_namespace':native['pid_namespace'],
                  'model_endpoint':proof.get('models_http_status')}))
