#!/usr/bin/env python3
"""Immutable recovery baseline plus fresh v3 held-time intervals; no reset."""
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

LIMIT_SECONDS=18000
ADMISSION_SECONDS=15900
BASE_SHA='c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141'
A1_SHA='0dca1fa52838388ea1e7afbc3a4c5e9747775f64e871640df7a66496369e5453'
CARRY=3413.519101628801
RUN_PREFIX='stdbench-compute-a1-20261106-v3'
FAILED_PREFIX='stdbench-compute-a1-20261105-v2'
ENGINE_SHA='58bece1d3184269059fd3f7f755c52f4f295113d'

class BudgetExhausted(RuntimeError):
    pass

def require(ok, reason):
    if not ok: raise ValueError(reason)

def local(path):
    return Path(str(path).replace('<rank0-home>/','<rank1-home>/',1)) if str(Path(__file__)).startswith('<rank1-home>/') else Path(path)

def bound_json(path, digest):
    path=local(path)
    require(isinstance(digest,str) and len(digest)==64 and hashlib.sha256(path.read_bytes()).hexdigest()==digest, 'recovery artifact drift: '+str(path))
    return json.loads(path.read_text())

def seconds(value):
    require(type(value) in (float,int) and math.isfinite(value) and value>=0,'invalid held seconds')
    return value

def utc(value):
    result=datetime.datetime.fromisoformat(value.replace('Z','+00:00'))
    require(result.utcoffset()==datetime.timedelta(0),'recovery timestamp must be UTC')
    return result

def validate_history(data):
    total=0.0
    for interval in data['turns']:
        total+=seconds(interval['held_seconds'])
        require(math.isclose(total,seconds(interval['cumulative_seconds']),rel_tol=0,abs_tol=1e-6),'interval cumulative mismatch')
    require(math.isclose(total,seconds(data['held_seconds']),rel_tol=0,abs_tol=1e-6),'history total mismatch')

def validate_prior(*, original_path, original_sha, prior_path, prior_sha, recovery_path, recovery_sha):
    original=validate_original(local(original_path),original_sha)
    prior=bound_json(prior_path,prior_sha); receipt=bound_json(recovery_path,recovery_sha)
    require(prior['schema']==2 and prior['spec_sha256']==A1_SHA and prior['parent_spec_sha256']==BASE_SHA
        and prior['original_ledger_sha256']==original_sha and prior['carry_seconds']==CARRY
        and prior['active'] is None and prior['limit_seconds']==LIMIT_SECONDS
        and prior['admission_seconds']==ADMISSION_SECONDS,'reconciled prior protocol differs')
    validate_history(prior)
    require(prior['turns'][:len(original['turns'])]==original['turns'] and prior['held_seconds']>=CARRY,'original history changed')
    require(receipt.get('passed') is True and receipt['spec_sha256']==A1_SHA and receipt['parent_spec_sha256']==BASE_SHA
        and local(receipt['reconciled_budget_path'])==local(prior_path) and receipt['reconciled_budget_sha256']==prior_sha
        and receipt['held_seconds']==prior['held_seconds'] and receipt['original_carry_seconds']==CARRY
        and receipt.get('no_new_requests') is True and receipt['models_http_status']==200
        and receipt['restored_G_engine_sha']==ENGINE_SHA
        and receipt['policy']=='conservative full UTC interval; no infrastructure-wait subtraction','recovery receipt differs')
    raw=bound_json(receipt['raw_budget_path'],receipt['raw_budget_sha256'])
    require(raw['schema']==2 and raw['spec_sha256']==A1_SHA and raw['parent_spec_sha256']==BASE_SHA
        and raw['original_ledger_sha256']==original_sha and raw['carry_seconds']==CARRY
        and raw['limit_seconds']==LIMIT_SECONDS and raw['admission_seconds']==ADMISSION_SECONDS,'raw prior protocol differs')
    validate_history(raw)
    require(raw['active'] is not None and utc(raw['active']['started_utc'])==utc(receipt['interval_start_utc'])
        and utc(receipt['interval_start_utc'])==utc('2026-11-06T08:53:34.922223Z'),'failed active interval differs')
    duration=(utc(receipt['interval_end_utc'])-utc(receipt['interval_start_utc'])).total_seconds()
    require(duration>=0 and math.isclose(seconds(receipt['charged_recovery_seconds']),duration,rel_tol=0,abs_tol=1e-6),'recovery UTC charge differs')
    require(len(prior['turns'])==len(raw['turns'])+1 and prior['turns'][:-1]==raw['turns'],'raw turn histories changed')
    last=prior['turns'][-1]
    require(last['turn']==raw['active']['turn'] and last['run_prefix']==FAILED_PREFIX
        and utc(last['started_utc'])==utc(receipt['interval_start_utc'])
        and utc(last['finished_utc'])==utc(receipt['interval_end_utc'])
        and math.isclose(last['held_seconds'],duration,rel_tol=0,abs_tol=1e-6)
        and math.isclose(prior['held_seconds'],raw['held_seconds']+duration,rel_tol=0,abs_tol=1e-6),'reconciled interval differs')
    log=local(receipt['restore_log_path'])
    require(hashlib.sha256(log.read_bytes()).hexdigest()==receipt['restore_log_sha256'],'actual G restoration log drift')
    return prior

def campaign_kwargs(config):
    return dict(original_path=local(config['original_budget']),original_sha=config['original_budget_sha256'],
                prior_path=local(config['prior_budget']),prior_sha=config['prior_budget_sha256'],
                recovery_path=local(config['recovery_receipt']),recovery_sha=config['recovery_receipt_sha256'])

class HeldBudget:
    def __init__(self,path,spec_sha,turn,*,original_path,original_sha,prior_path,prior_sha,recovery_path,recovery_sha,clock=time.monotonic):
        require(spec_sha==A1_SHA,'continuation spec differs')
        prior=validate_prior(original_path=original_path,original_sha=original_sha,prior_path=prior_path,
                             prior_sha=prior_sha,recovery_path=recovery_path,recovery_sha=recovery_sha)
        path=Path(path)
        require(path.is_absolute() and '..' not in path.parts and not any(p.is_symlink() for p in (path,*path.parents)),'unsafe budget ledger path')
        require(all(path.resolve()!=local(p).resolve() for p in (original_path,prior_path,recovery_path)),'cannot mutate recovery sources')
        self.path,self.spec_sha,self.turn,self.clock=path,spec_sha,turn,clock
        self.started=None
        baseline=dict(schema=3,spec_sha256=A1_SHA,parent_spec_sha256=BASE_SHA,run_prefix=RUN_PREFIX,
            original_ledger_sha256=original_sha,original_carry_seconds=CARRY,
            prior_ledger_sha256=prior_sha,recovery_receipt_sha256=recovery_sha,
            carry_seconds=prior['held_seconds'],baseline_interval_count=len(prior['turns']),
            limit_seconds=LIMIT_SECONDS,admission_seconds=ADMISSION_SECONDS)
        if not path.exists():
            self.data=copy.deepcopy(prior);self.data.update(baseline)
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x') as output:
                json.dump(self.data,output,indent=2,allow_nan=False);output.write('\n');output.flush();os.fsync(output.fileno())
        self.data=json.loads(path.read_text());d=self.data
        require(all(d.get(k)==v for k,v in baseline.items()),'continuation recovery binding changed')
        require(d['turns'][:len(prior['turns'])]==prior['turns'],'baseline histories changed')
        validate_history(d)
        require(d['held_seconds']>=prior['held_seconds'],'reconciled carry decreased')
        require(all(t.get('run_prefix')==RUN_PREFIX for t in d['turns'][len(prior['turns']):]),'new interval namespace differs')
        if d['active'] is not None: raise RuntimeError('unfinished interval: reconcile actual restoration first')

    def elapsed(self):
        return self.data['held_seconds']+(self.clock()-self.started if self.started is not None else 0)

    def ensure_can_start(self):
        if self.elapsed()>=ADMISSION_SECONDS:
            raise BudgetExhausted('4h25m cumulative admission cutoff: no new request or boot')

    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temporary=self.path.with_name(self.path.name+'.tmp-'+str(os.getpid()))
        with temporary.open('x') as output:
            json.dump(self.data,output,indent=2,allow_nan=False);output.write('\n')
            output.flush();os.fsync(output.fileno())
        os.replace(temporary,self.path)

    def begin(self):
        self.ensure_can_start()
        if self.started is not None: raise RuntimeError('held interval already active')
        self.started=self.clock()
        self.data['active']={'turn':self.turn,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                             'run_prefix':RUN_PREFIX,'pid':os.getpid(),'clock':'monotonic within foreground controller'}
        self.save()

    def finish_after_verified_restore(self):
        if self.started is None: return
        duration=self.clock()-self.started
        if duration<0: raise ValueError('monotonic clock reversed')
        active=self.data['active']
        self.data['held_seconds']+=duration
        self.data['turns'].append(dict(active,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                      held_seconds=duration,cumulative_seconds=self.data['held_seconds']))
        self.data['active']=None
        self.data['overrun_seconds']=max(0,self.data['held_seconds']-LIMIT_SECONDS)
        self.started=None
        self.save()

def validate_original(path, expected):
    import hashlib
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise ValueError('original budget drift')
    data=json.loads(path.read_text())
    if (data['schema']!=1 or data['spec_sha256']!=BASE_SHA or data['active'] is not None
        or data['held_seconds']!=CARRY or data['limit_seconds']!=LIMIT_SECONDS
        or data['admission_seconds']!=ADMISSION_SECONDS): raise ValueError('original carry binding differs')
    return data

