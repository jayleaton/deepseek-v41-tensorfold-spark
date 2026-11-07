"""Fresh study budget. Every second STOP through verified G restore is charged."""
import datetime
import json
import math
import os
from pathlib import Path
import time
from protocol import PREFIX, QUALIFIED_SPEC_SHA256, require
LIMIT_SECONDS=6300
ADMISSION_SECONDS=5880
class BudgetExhausted(RuntimeError): pass

def campaign_kwargs(c): return {}
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def validate_history(d):
    require(d['schema']==1 and d['spec_sha256']==QUALIFIED_SPEC_SHA256 and d['run_prefix']==PREFIX,'budget identity')
    require(d['limit_seconds']==LIMIT_SECONDS and d['admission_seconds']==ADMISSION_SECONDS,'budget limits')
    total=0
    require(len(d['turns'])<=1,'one ORDER turn only')
    for interval in d['turns']:
        duration=interval['held_seconds']
        require(type(duration) in (int,float) and math.isfinite(duration) and duration>=0,'invalid held time')
        require(interval['turn']=='format' and interval['run_prefix']==PREFIX,'foreign interval')
        total+=duration
        require(interval['cumulative_seconds']==total,'cumulative sum differs')
    require(d['held_seconds']==total and d['overrun_seconds']==max(0,total-LIMIT_SECONDS),'budget sum/overrun differs')
    return d

def fresh():
    return dict(schema=1,spec_sha256=QUALIFIED_SPEC_SHA256,run_prefix=PREFIX,
                limit_seconds=LIMIT_SECONDS,admission_seconds=ADMISSION_SECONDS,
                held_seconds=0,overrun_seconds=0,turns=[],active=None)

def initialize(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as out: json.dump(fresh(),out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())

class HeldBudget:
    def __init__(self,path,spec_sha,turn,*,clock=time.monotonic):
        require(spec_sha==QUALIFIED_SPEC_SHA256 and turn=='format','single fresh format turn only')
        self.path=Path(path)
        require(self.path.is_absolute() and '..' not in self.path.parts and not any(p.is_symlink() for p in (self.path,*self.path.parents)),'unsafe ledger')
        self.data=validate_history(json.loads(self.path.read_text()))
        require(self.data['active'] is None,'unfinished outage; parent must reconcile actual full interval')
        self.clock,self.turn,self.started=clock,turn,None
        self.completed=bool(self.data['turns'])
    def elapsed(self): return self.data['held_seconds']+(self.clock()-self.started if self.started is not None else 0)
    def ensure_can_start(self):
        if self.completed or self.elapsed()>=ADMISSION_SECONDS: raise BudgetExhausted('format admission cutoff reached')
    def save(self):
        temporary=self.path.with_name(self.path.name+'.tmp-'+str(os.getpid()))
        with temporary.open('x') as out:
            json.dump(self.data,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
        os.replace(temporary,self.path)
    def begin(self):
        self.ensure_can_start();require(self.started is None,'already active')
        self.started=self.clock()
        self.data['active']=dict(turn=self.turn,run_prefix=PREFIX,started_utc=utc(),pid=os.getpid())
        self.save()
    def finish_after_verified_restore(self):
        if self.started is None:return
        duration=self.clock()-self.started;require(duration>=0,'clock reversed')
        self.data['held_seconds']+=duration
        self.data['turns'].append(dict(self.data['active'],finished_utc=utc(),held_seconds=duration,cumulative_seconds=self.data['held_seconds']))
        self.data['active']=None;self.data['overrun_seconds']=max(0,self.data['held_seconds']-LIMIT_SECONDS)
        self.started=None;self.completed=True;self.save()
