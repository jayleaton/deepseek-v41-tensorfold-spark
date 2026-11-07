#!/usr/bin/env python3
"""After normal G restoration: prove live service and close a derived prior ledger.

No queue, GPU lock, engine, historical ledger or inference mutation. CPU scope only.
"""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
A1='0dca1fa52838388ea1e7afbc3a4c5e9747775f64e871640df7a66496369e5453'
BASE='c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141'
G='58bece1d3184269059fd3f7f755c52f4f295113d'
OLD='stdbench-compute-a1-20261105-v2'
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,why):
    if not ok:raise ValueError(why)
def save(name,value):
    with (HERE/name).open('x') as out:json.dump(value,out,indent=2,allow_nan=False)
def main():
    queue=Path('<rank0-work>/queue/ORDER');require(not queue.read_text().strip(),'G recovery must be settled and queue released first')
    require(not queue.with_name('QUIET').exists(),'QUIET window active')
    restore=ROOT/'a1-runtime/G-recovery-20261106-v3.log';text=restore.read_text()
    steps=[]
    for line in text.splitlines():
        if re.match(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ SERVING ',line):steps.append(('SERVING',line))
        elif re.match(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ /v1/models 200$',line):steps.append(('MODELS200',line))
        elif re.match(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ POPPED; ORDER:',line):steps.append(('POPPED',line))
    require([s[0] for s in steps]==['SERVING','MODELS200','POPPED'],'normal G recovery step evidence incomplete')
    require('EF_SHA='+G in steps[0][1],'restored G source mismatch')
    current=Path('<rank0-work>/r5/cand-e/EFAST2-RUN').read_text().strip()
    status_path=Path('<rank0-work>/dsv41-bench/out')/current/'status.json';status=load(status_path)
    require(status.get('serving') is True and status.get('engine_sha')==G and current in steps[0][1],'actual serving status mismatch')
    with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=10) as response:
        http=response.status;models=json.load(response)
    require(http==200 and models['data'][0]['id']=='DeepSeek-V4.1-Flash-TF','actual models endpoint/model mismatch')
    expected=load('<rank0-work>/dsv41-bench/arms-gx.json')['arms']['D']['ranks']
    ranks=[]
    for rank in (0,1):
        command=['docker','container','inspect',f'tensorfold-zig-r{rank}']
        if rank:command=['ssh','-o','BatchMode=yes','root@<rank1-link-ip>',*command]
        container=json.loads(subprocess.check_output(command,text=True,timeout=7200))[0]
        require(container['Image']==expected[rank]['image_id'] and container['State']['Running'] is True,'actual G rank image/running mismatch')
        ranks.append({'rank':rank,'image_id':container['Image'],'host_pid':container['State']['Pid'],'running':True})
    raw=ROOT/OLD/'budget.json';original=load(raw);raw_hash=sha(raw)
    require(original['schema']==2 and original['spec_sha256']==A1 and original['held_seconds']==3413.519101628801,'prior A1 budget binding differs')
    active=original['active'];require(active and active['turn']=='A1-resume','failed active interval absent')
    require(not (ROOT/'a1-runtime/attempted'/OLD).exists(),'failed attempt has dataset markers; reconcile keys before resume')
    run=ROOT/'a1-runtime/runs'/(OLD+'-A1-resume')
    require(not list(run.rglob('responses.jsonl')) and not list(run.rglob('raw-observations.jsonl')),'failed attempt has new observations')
    failures=[json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()]
    require(any(s['step']=='collection' and s['verdict']=='STOPPED' and 'read-only lease failed' in s['reason'] for s in failures),'failed collection stop evidence differs')
    start=datetime.datetime.fromisoformat(active['started_utc'])
    end=datetime.datetime.now(datetime.timezone.utc);duration=(end-start).total_seconds()
    require(duration>=0,'negative recovery interval')
    reconciled=copy.deepcopy(original);total=original['held_seconds']+duration
    interval=dict(active,finished_utc=end.isoformat(),held_seconds=duration,cumulative_seconds=total,
                  run_prefix=OLD,clock='conservative UTC wall interval after controller loss',
                  recovery_log_sha256=sha(restore),restored_G_run=current,no_new_requests=True)
    reconciled['turns'].append(interval);reconciled.update(active=None,held_seconds=total,overrun_seconds=max(0,total-18000))
    save('raw-failed-budget.json',original);save('reconciled-budget.json',reconciled)
    receipt={'passed':True,'spec_sha256':A1,'parent_spec_sha256':BASE,'raw_budget_path':str(raw),
             'raw_budget_sha256':raw_hash,'reconciled_budget_path':str(HERE/'reconciled-budget.json'),
             'reconciled_budget_sha256':sha(HERE/'reconciled-budget.json'),'held_seconds':total,
             'original_carry_seconds':original['held_seconds'],'charged_recovery_seconds':duration,
             'interval_start_utc':start.isoformat(),'interval_end_utc':end.isoformat(),
             'no_new_requests':True,'models_http_status':http,'served_model_id':models['data'][0]['id'],
             'restored_G_engine_sha':G,'restored_G_run':current,'ranks':ranks,
             'restore_log_path':str(restore),'restore_log_sha256':sha(restore),'status_sha256':sha(status_path),
             'failed_steps_sha256':sha(run/'steps.jsonl'),'outage_utc':'2026-11-05 ~07:30Z (JJ confirmed)',
             'policy':'conservative full UTC interval; no infrastructure-wait subtraction',
             'limit_seconds':18000,'admission_seconds':15900,'remaining_until_admission_seconds':max(0,15900-total),
             'source_sha256':sha(__file__)}
    save('recovery.json',receipt)
    require(sha(raw)==raw_hash,'raw failed ledger changed')
    print(json.dumps({k:receipt[k] for k in ('passed','models_http_status','held_seconds','charged_recovery_seconds','no_new_requests')}))
if __name__=='__main__':main()
