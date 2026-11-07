"""Schema4 full A1 carry; one new P turn, never mutates schema3 sources."""
import copy
import datetime
import json
import os
from pathlib import Path
import time
from a1_budget import (require, local, bound_json, seconds, utc, validate_history,
                       validate_original, CARRY, BASE_SHA, A1_SHA, BudgetExhausted)
from adoption import A2_SHA, validate
LIMIT_SECONDS=36000
ADMISSION_SECONDS=33900
RUN_PREFIX='stdbench-compute-a2-20261106-v1'

def campaign_kwargs(c):
    return dict(original_path=local(c['original_budget']),original_sha=c['original_budget_sha256'],
                prior_path=local(c['prior_budget']),prior_sha=c['prior_budget_sha256'],
                audit_path=local(c['a1_audit']),audit_sha=c['a1_audit_sha256'])

def validate_prior(*, original_path, original_sha, prior_path, prior_sha, audit_path, audit_sha):
    validate_original(original_path,original_sha)
    prior,_,_,_=validate(Path(original_path).parent.parent,prior_path,prior_sha,audit_path,audit_sha)
    require(prior['original_ledger_sha256']==original_sha,'original carry authority differs')
    return prior

class HeldBudget:
    def __init__(self,path,spec_sha,turn,*,original_path,original_sha,prior_path,prior_sha,audit_path,audit_sha,clock=time.monotonic):
        require(spec_sha==A2_SHA and turn=='A2-plain','single A2 continuation turn only')
        prior=validate_prior(original_path=original_path,original_sha=original_sha,prior_path=prior_path,
                             prior_sha=prior_sha,audit_path=audit_path,audit_sha=audit_sha)
        path=Path(path)
        require(path.is_absolute() and '..' not in path.parts and not any(p.is_symlink() for p in (path,*path.parents)),'unsafe ledger path')
        require(all(path.resolve()!=local(p).resolve() for p in (original_path,prior_path,audit_path)),'immutable A1 source')
        self.path,self.spec_sha,self.turn,self.clock=path,spec_sha,turn,clock
        self.started=None
        baseline=dict(schema=4,spec_sha256=A2_SHA,parent_spec_sha256=BASE_SHA,a1_spec_sha256=A1_SHA,
            run_prefix=RUN_PREFIX,original_ledger_sha256=original_sha,original_carry_seconds=CARRY,
            prior_ledger_sha256=prior_sha,a1_audit_sha256=audit_sha,carry_seconds=prior['held_seconds'],
            baseline_interval_count=len(prior['turns']),limit_seconds=LIMIT_SECONDS,admission_seconds=ADMISSION_SECONDS)
        if not path.exists():
            self.data=copy.deepcopy(prior);self.data.update(baseline)
            self.data['overrun_seconds']=max(0,prior['held_seconds']-LIMIT_SECONDS)
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x') as out:
                json.dump(self.data,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
        self.data=json.loads(path.read_text());d=self.data
        require(all(d.get(k)==v for k,v in baseline.items()),'A2 carry/limits/binding drift')
        require(d['turns'][:len(prior['turns'])]==prior['turns'],'A1 history changed')
        validate_history(d)
        require(d['held_seconds']>=prior['held_seconds'],'A1 carry decreased')
        appended=d['turns'][len(prior['turns']):]
        require(len(appended)<=1 and all(t.get('run_prefix')==RUN_PREFIX and t['turn']=='A2-plain' for t in appended),'single continuation interval only')
        require(d['active'] is None,'unfinished interval; reconcile before proceeding')
        self.completed=bool(appended)

    def elapsed(self):
        return self.data['held_seconds']+(self.clock()-self.started if self.started is not None else 0)

    def ensure_can_start(self):
        if self.completed or self.elapsed()>=ADMISSION_SECONDS:
            raise BudgetExhausted('A2 admission closed: no new request or boot')

    def save(self):
        temporary=self.path.with_name(self.path.name+'.tmp-'+str(os.getpid()))
        with temporary.open('x') as out:
            json.dump(self.data,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
        os.replace(temporary,self.path)

    def begin(self):
        self.ensure_can_start()
        require(self.started is None,'held interval already active')
        self.started=self.clock()
        self.data['active']=dict(turn=self.turn,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            run_prefix=RUN_PREFIX,pid=os.getpid(),clock='monotonic within foreground controller')
        self.save()

    def finish_after_verified_restore(self):
        if self.started is None:return
        duration=self.clock()-self.started;require(duration>=0,'clock reversed')
        self.data['held_seconds']+=duration
        self.data['turns'].append(dict(self.data['active'],finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            held_seconds=duration,cumulative_seconds=self.data['held_seconds']))
        self.data['active']=None;self.data['overrun_seconds']=max(0,self.data['held_seconds']-LIMIT_SECONDS)
        self.started=None;self.completed=True;self.save()
