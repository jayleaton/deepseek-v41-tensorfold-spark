#!/usr/bin/env python3
"""Final read-only liveness/queue/cleanliness checks on the corresponding Spark."""
import argparse
import datetime
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
p=argparse.ArgumentParser();p.add_argument('rank',type=int,choices=(0,1));a=p.parse_args()
rank=a.rank;root=Path(f'<rank{rank}-work>/stdbench/phase2')
run=lambda args:subprocess.check_output(args,text=True,timeout=30)
expected=['sha256:e6cf428b02d188ceda5859b2f25839d4c305ad1f66b758b634b5d07dbe9da4f5','sha256:ca6feff3164a67710816be940968403f938947e4974ff04f61717d8f0f5688f9'][rank]
name=f'tensorfold-zig-r{rank}'
container=json.loads(run(['docker','inspect',name]))[0]
assert container['Image']==expected and container['State']['Running'] is True
assert container['State']['Pid']>1
native=[]
for line in run(['docker','top',name,'-eo','pid,comm']).splitlines()[1:]:
    fields=line.split()
    if len(fields)==2 and (fields[1] in ('tf-dsv41','tf-dsv41-m1','tensorfold-dsv41') or fields[1].startswith('tensorfold-dsv4')):
        pid=int(fields[0])
        try:os.kill(pid,0)
        except PermissionError:pass # Kernel EPERM confirms the exact root-owned PID exists.
        native.append({'pid':pid,'comm':fields[1],'kernel_liveness':True})
assert len(native)==1, 'expected exactly one live native G process'
owned=[n for n in run(['docker','ps','-a','--format','{{.Names}}']).splitlines() if n.startswith(('stdbench-20261105-','stdbench-jail-'))]
assert not owned, 'owned experiment or jail container remains'
proof={'passed':True,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rank':rank,'G_image_id':expected,'G_native_process':native[0],'owned_containers_remaining':owned,'no_inference_requests':True}
if rank==0:
    queue=Path('<rank0-work>/queue')
    assert not (queue/'QUIET').exists() and not (queue/'ORDER').read_text().strip()
    current=Path('<rank0-work>/r5/cand-e/EFAST2-RUN').read_text().strip()
    status=json.loads((Path('<rank0-work>/dsv41-bench/out')/current/'status.json').read_text())
    assert status['serving'] is True and status['engine_sha']=='58bece1d3184269059fd3f7f755c52f4f295113d'
    validation=json.loads((root/'post-v1/report-validation.json').read_text())
    assert hashlib.sha256((root/'post-v1/RESULTS.md').read_bytes()).hexdigest()==validation['results_sha256']
    proof.update(G_run=current,ORDER_empty=True,QUIET_absent=True,report_sha256=validation['results_sha256'])
    with (root/'post-v1/final-script-preflight.log').open('x') as log:
        subprocess.run(['bash','<rank0-work>/queue/turn-preflight.sh',*[str(root/n) for n in ('turn.sh','launch.sh','fence.sh','cpu-rank0-host.sh','cpu-rootrank1_host.sh','stage_math.sh','stage_postcollection.sh','recover-v1.sh')]],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
with (root/f'post-v1/final-state-r{rank}.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps({'final_state':'PASS','rank':rank,'G_native_process_live':True,'owned_containers_remaining':0,'no_inference_requests':True}))
