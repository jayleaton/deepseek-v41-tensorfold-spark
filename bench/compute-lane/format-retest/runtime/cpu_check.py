"""Parent Spark CPU fixtures. No transport, engine, task generation or queue mutation."""
import ast
import copy
import fcntl
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import protocol
from budget import HeldBudget,BudgetExhausted,initialize
from coordination import check as coordination
from manifest import validate_boots
HERE=Path(__file__).resolve().parent

def check():
    controls=[]
    def reject(label,fn):
        try:fn()
        except (ValueError,RuntimeError,KeyError,OSError):controls.append(label);return
        raise AssertionError('accepted '+label)
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    cfg=protocol.read_config(HERE.parent/protocol.PREFIX/'stage-campaign.json')
    protocol.require_staging_ready(cfg)
    frozen=Path(cfg['freeze'])
    index=json.loads((frozen/'manifest.json').read_text())
    boots=[json.loads((frozen/b['file']).read_text()) for b in index['boots']]
    validate_boots(boots)
    for label,mutate in (
        ('missing request',lambda b:b[0]['requests'].pop()),
        ('duplicate identity',lambda b:b[0]['requests'].__setitem__(1,copy.deepcopy(b[0]['requests'][0]))),
        ('wording order',lambda b:b[0]['requests'].reverse()),
        ('wire mutation',lambda b:b[0]['requests'][0].__setitem__('wire','bad')),
        ('P lane',lambda b:b[1]['requests'][0].__setitem__('arm','lane')),
        ('diagnostic reference',lambda b:b[0]['diagnostics'][0].__setitem__('reference','0'))):
        damaged=copy.deepcopy(boots);mutate(damaged);reject(label,lambda:validate_boots(damaged))
    reject('pending qualification/GO',lambda:protocol.require_collection_ready(cfg))
    with tempfile.TemporaryDirectory(prefix='format-cpu-',dir=HERE) as folder:
        tmp=Path(folder);clock=[0.0];initialize(tmp/'budget.json')
        reject('exclusive budget',lambda:initialize(tmp/'budget.json'))
        b=HeldBudget(tmp/'budget.json',protocol.QUALIFIED_SPEC_SHA256,'format',clock=lambda:clock[0])
        assert b.data['held_seconds']==0 and b.data['turns']==[]
        b.begin();clock[0]=5879.9;b.ensure_can_start();clock[0]=5880
        reject('exact5880 admission',b.ensure_can_start)
        reject('active restart',lambda:HeldBudget(tmp/'budget.json',protocol.QUALIFIED_SPEC_SHA256,'format'))
        clock[0]=6301;b.finish_after_verified_restore()
        assert b.data['held_seconds']==6301 and b.data['overrun_seconds']==1
        b=HeldBudget(tmp/'budget.json',protocol.QUALIFIED_SPEC_SHA256,'format')
        reject('no second turn',b.ensure_can_start)
        import importlib.util
        source=HERE.parent.parent/'compute-dl/a1-prep/audit_lifecycle.py'
        spec=importlib.util.spec_from_file_location('_format_cutoff_lifecycle',source)
        lifecycle=importlib.util.module_from_spec(spec);spec.loader.exec_module(lifecycle)
        for where in ('before readiness','between diagnostics'):
            steps=[dict(step='G-stop',verdict='REQUESTED'),dict(step='collection',verdict='INCOMPLETE',reason='format admission cutoff reached'),
                   dict(step='owned-containers',verdict='DRAIN_VERIFIED',boot='format-L'),dict(step='G-restore',verdict='SERVING_VERIFIED'),
                   dict(step='budget',verdict='RECORDED',held_seconds=6301),dict(step='turn',verdict='INCOMPLETE')]
            if where=='between diagnostics':steps.insert(1,dict(step='boot',verdict='READY_EFFECTIVE_ENV_VERIFIED',boot='format-L'))
            state=lifecycle.classify(steps)
            assert state['budget_stop'] is True
            assert lifecycle.allow_partial_boot(state,'format-L',0,1 if where=='between diagnostics' else 0,0) is True
            controls.append('budget cutoff '+where+' classified INCOMPLETE')
        lock=tmp/'lock';lock.write_bytes(b'UNCHANGED')
        with lock.open('r') as held:
            fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert coordination(lock)['flock_attempted'] is False
        assert lock.read_bytes()==b'UNCHANGED'
        reject('missing lock',lambda:coordination(tmp/'absent'))
        reject('directory lock',lambda:coordination(tmp))
        os.mkfifo(tmp/'fifo');reject('FIFO lock',lambda:coordination(tmp/'fifo'))
        with patch('coordination.os.open',side_effect=PermissionError('fixture')):reject('unreadable lock',lambda:coordination(lock))
    import subprocess,sys
    result=subprocess.run([sys.executable,'-B',str(HERE.parent/'prep/checks.py')],capture_output=True,text=True,timeout=7200)
    assert result.returncode==0,(result.stdout,result.stderr)
    return dict(passed=True,controls=controls,coordination_fixtures=True,prep_native_fake_controls=dict(stdout=result.stdout,stderr=result.stderr),GPU_requests=0)
if __name__=='__main__':print(json.dumps(check()))
