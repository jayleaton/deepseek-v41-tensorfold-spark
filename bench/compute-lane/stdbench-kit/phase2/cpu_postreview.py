#!/usr/bin/env python3

import sys as _archive_sys
import json
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

root = Path('<rank1-work>/stdbench/phase2')
sys.path.insert(0,str(root))
import controller
import rank_ops
from jail import Jail

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

subprocess.run([sys.executable,str(root/'controller_selftest.py'),str(root/'control-recovery-v2.json')],check=True)
assert json.loads((root/'control-recovery-v2.json').read_text())['passed']
with tempfile.TemporaryDirectory(prefix='stdbench-pending-') as temporary:
    manifest = Path(temporary)/'manifest.json'
    manifest.write_text(json.dumps({'phase2':{'ranks':[{'container_name':'fake-r0'},{'container_name':'fake-r1'}]}}))
    boot = {'boot':'fake','rank_manifest':{'0':str(manifest)}}
    child = subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
    fake = SimpleNamespace(ranks=[(1,child)],event=lambda *a,**k:None)
    try:
        try:
            controller.Turn.require_drained(fake,boot)
        except RuntimeError:
            pass
        else:
            raise AssertionError('pending launcher must block drain/release')
    finally:
        child.terminate()
        child.wait()
    with patch.object(controller.subprocess,'run',return_value=SimpleNamespace(returncode=1,stderr='No such object',stdout='')):
        controller.Turn.require_drained(fake,boot)
    assert not fake.ranks
signals=[]
previous=signal.signal(signal.SIGUSR1,lambda *_:signals.append('lease-lost'))
lease=subprocess.Popen([sys.executable,'-c','import time; time.sleep(.05)'])
watched=SimpleNamespace(lease_expected=True,lease_lost=False,cleaning=False,lease=lease)
try:
    controller.Turn.watch_lease(watched)
    lease.wait()
    assert watched.lease_lost and signals
    try:
        controller.Turn.health(watched)
    except RuntimeError:
        pass
    else:
        raise AssertionError('lost lease must block collection')
finally:
    signal.signal(signal.SIGUSR1,previous)
jail=Jail('sha256:80beaba732d1ed9439e012b007d8dbe19ec2f8962f50fbac29a18c8cd6fda306')
for code,want in [('print(17*19)','323'),('print(sum(range(11)))','55')]:
    result=jail.run_reference(code)
    assert controller.reference_matches({'status':'ok','output':want},result)
    assert not controller.reference_matches({'status':'ok','output':'1'+want},result)
# Read-only probe of the existing steady G rank; no model/API request.
process=rank_ops.engine_environment('tensorfold-zig-r1',1)
assert process['pid_namespace'] > 1 and process['env']['TF_DSV41_COMPUTE_LANE']=='1'
(root/'postreview-v1.json').write_text(json.dumps({'passed':True,'pending_launcher_blocks_release':True,
    'lease_loss_stops_collection':True,'normalized_reference_323_55_integration':True,
    'live_native_process_environment_probe':True,'recovery_cases':6})+'\n')
print(json.dumps({'postreview':'PASS','no_GPU_or_API_requests':True}))
