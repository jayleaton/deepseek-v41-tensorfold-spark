#!/usr/bin/env python3
"""Source one fresh rank config in a clean CPU shell; never launches a rank."""

import sys as _archive_sys
import json
import os
from pathlib import Path
import subprocess
import sys

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)
HERE=Path(__file__).resolve().parent
rank=int(sys.argv[1]); stage=HERE/'staged-P'
record=json.loads((stage/f'rank{rank}/effective-env.json').read_text())
config=stage/record['stage_config']
run=subprocess.run(['bash','-c','source "$1" || exit; env -0','cpu-env-check',str(config)],
                   env={'PATH':os.environ['PATH'],'HOME':os.environ['HOME']},capture_output=True,check=True,timeout=7200)
allenv=dict(x.decode().split('=',1) for x in run.stdout.split(b'\0') if x)
expected=record['effective_env']
assert all(allenv.get(k)==v for k,v in expected.items())
serving=lambda k: k.startswith(('TF_','GLM53_TF','NCCL','GUARD','SPARK'))
actual={k:v for k,v in allenv.items() if k in expected or serving(k)}
assert not [k for k in actual if k not in expected and not k.startswith(('GUARD','SPARK'))], 'unexpected serving flags'
assert all(actual.get(k)=='0' for k in record['expected_off'])
assert record['guard_source']=='/etc/spark/thermal.env'
image=subprocess.run(['docker','image','inspect','--format','{{.Id}}',record['image']],capture_output=True,text=True,check=True,timeout=7200).stdout.strip()
assert image==record['image_id']
containers=subprocess.run(['docker','ps','-a','--format','{{.Names}}'],capture_output=True,text=True,check=True,timeout=7200).stdout.splitlines()
assert not any(record['container_name'] in name for name in containers)
assert not Path(record['effective_env']['STATE_DIR']).exists()
assert not Path(record['effective_env']['CACHE_DIR']).exists()
result={'passed':True,'rank':rank,'image_id':image,'effective_env':actual,'thermal_source':record['guard_source'],
        'new_namespace_clear':True,'GPU_requests':0,'collection_authorized':False}
(HERE/f'env-check-r{rank}.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'passed':True,'rank':rank,'GPU_requests':0,'new_namespace_clear':True}))
