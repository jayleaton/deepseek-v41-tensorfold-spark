#!/usr/bin/env python3
"""Read actual G status, model endpoint and rank containers after a settled turn."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

G='58bece1d3184269059fd3f7f755c52f4f295113d'
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,why):
    if not ok:raise ValueError(why)
def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    queue=Path('<rank0-work>/queue/ORDER')
    require(not queue.read_text().strip() and not queue.with_name('QUIET').exists(),'queue/restoration not settled')
    steps=[json.loads(l) for l in a.steps.read_text().splitlines()]
    require(steps[-1]['step']=='turn' and steps[-1]['verdict'] in ('QUALIFIED','INCOMPLETE','STOPPED'),'turn not settled')
    restored=[s for s in steps if s['step']=='G-restore' and s['verdict']=='SERVING_VERIFIED']
    require(len(restored)==1,'verified normal G restoration missing')
    r=restored[0];log=Path(r['actual_step_path']);require(sha(log)==r['actual_step_sha256'],'restoration log drift')
    text=log.read_text();require(all(v in text for v in ('SERVING ','/v1/models 200','POPPED; ORDER:',G)),'normal restoration evidence incomplete')
    current=Path('<rank0-work>/r5/cand-e/EFAST2-RUN').read_text().strip()
    status_path=Path('<rank0-work>/dsv41-bench/out')/current/'status.json';status=load(status_path)
    require(status.get('serving') is True and status.get('engine_sha')==G and current==r['serving_run'],'actual G status mismatch')
    with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=20) as response:
        http=response.status;models=json.load(response)
    require(http==200 and models['data'][0]['id']=='DeepSeek-V4.1-Flash-TF','actual model endpoint mismatch')
    expected=load('<rank0-work>/dsv41-bench/arms-gx.json')['arms']['D']['ranks'];ranks=[]
    for rank in (0,1):
        command=['docker','container','inspect',f'tensorfold-zig-r{rank}']
        if rank:command=['ssh','-o','BatchMode=yes','root@<rank1-link-ip>',*command]
        c=json.loads(subprocess.check_output(command,text=True,timeout=7200))[0]
        require(c['Image']==expected[rank]['image_id'] and c['State']['Running'] is True,'actual G rank image/running mismatch')
        ranks.append({'rank':rank,'image_id':c['Image'],'host_pid':c['State']['Pid'],'running':True})
    receipt={'passed':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'models_http_status':http,'model_id':models['data'][0]['id'],'G_engine_sha':G,'G_run':current,
             'ranks':ranks,'steps_path':str(a.steps),'steps_sha256':sha(a.steps),'restore_log_path':str(log),
             'restore_log_sha256':sha(log),'status_path':str(status_path),'status_sha256':sha(status_path),
             'source_sha256':sha(__file__),'inference_requests':0}
    with a.out.open('x') as out:json.dump(receipt,out,indent=2)
    print(json.dumps({'passed':True,'models_http_status':http,'G_run':current,'ranks_running':2}))
if __name__=='__main__':main()
