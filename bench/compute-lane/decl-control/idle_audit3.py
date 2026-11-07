#!/usr/bin/env python3
"""Read-only final restoration/owned-job audit, through a guarded CPU scope."""
import argparse
import json
from pathlib import Path
import subprocess
from generate_envs3 import PREFIX, IMAGE_IDS
from rank_ops3 import inspect_container

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--rank',type=int,choices=(0,1),required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
root=Path(__file__).resolve().parent
if a.rank==0:
    queue=Path('<rank0-work>/queue')
    assert not (queue/'ORDER').read_text().strip() and not (queue/'QUIET').exists()
    profile=json.loads(Path('<rank0-work>/dsv41-bench/arms-gx.json').read_text())
    expected=[rank['image_id'] for rank in profile['arms']['D']['ranks']]
    image_file=root/'G-restored-image-ids-a1.json'
    with image_file.open('x') as output:
        json.dump(expected,output)
    current=Path('<rank0-work>/r5/cand-e/EFAST2-RUN').read_text().strip()
    status=json.loads((Path('<rank0-work>/dsv41-bench/out')/current/'status.json').read_text())
    assert status['serving'] and status['engine_sha']=='58bece1d3184269059fd3f7f755c52f4f295113d'
else:
    expected=json.loads((root/'G-restored-image-ids-a1.json').read_text())
    current=None
index=json.loads((root/PREFIX/'manifest.json').read_text())
absent=[]
for entry in index['boots']:
    manifest=json.loads((root/PREFIX/entry['manifest']).read_text())
    name=manifest['phase2']['ranks'][a.rank]['container_name']
    assert inspect_container(name,IMAGE_IDS[a.rank]) is None
    absent.append(name)
actual=subprocess.check_output(['docker','inspect','--format','{{.Image}} {{.State.Running}}',
                               'tensorfold-zig-r'+str(a.rank)],text=True,timeout=30).strip()
assert actual==expected[a.rank]+' true'
units=subprocess.check_output(['systemctl','--user','list-units','--all','--state=active,activating,deactivating',
                               '--no-legend','--plain'],text=True,timeout=30).splitlines()
assert not any(line.split()[0].startswith(PREFIX) for line in units if line.split())
with a.out.open('x') as output:
    json.dump({'qualified':True,'rank':a.rank,'G_running_image_id':expected[a.rank],
               'G_serving_run':current,'owned_container_names_absent':absent,
               'owned_turn_services_pending':False,'GPU_operation':False},output,indent=2)
    output.write('\n')
print(json.dumps({'final_live_audit':'PASS','rank':a.rank,'G_running':True,'owned_containers_absent':len(absent)}))
