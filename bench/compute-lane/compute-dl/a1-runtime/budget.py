#!/usr/bin/env python3
"""Persistent operator-held time; never interrupt safety drain/G restoration."""
import datetime
import json
import math
import os
from pathlib import Path
import time

LIMIT_SECONDS=18000
ADMISSION_SECONDS=15900


class BudgetExhausted(RuntimeError):
    pass


class HeldBudget:
    def __init__(self,path,spec_sha,turn,*,clock=time.monotonic):
        self.path=Path(path)
        if not self.path.is_absolute() or '..' in self.path.parts or any(p.is_symlink() for p in (self.path,*self.path.parents)):
            raise ValueError('unsafe budget ledger path')
        self.spec_sha,self.turn,self.clock=spec_sha,turn,clock
        self.started=None
        self.data=json.loads(self.path.read_text()) if self.path.exists() else {
            'schema':1,'spec_sha256':spec_sha,'limit_seconds':LIMIT_SECONDS,
            'admission_seconds':ADMISSION_SECONDS,'held_seconds':0.0,'active':None,'turns':[]}
        if (self.data['schema']!=1 or self.data['spec_sha256']!=spec_sha
                or self.data['limit_seconds']!=LIMIT_SECONDS or self.data['admission_seconds']!=ADMISSION_SECONDS):
            raise ValueError('budget protocol changed')
        if not isinstance(self.data['held_seconds'],(int,float)) or not math.isfinite(self.data['held_seconds']) or self.data['held_seconds']<0:
            raise ValueError('invalid accumulated GPU time')
        if self.data['active'] is not None:
            raise RuntimeError('unfinished held-time interval: restore/reconcile its actual G evidence before resuming')

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
                             'pid':os.getpid(),'clock':'monotonic within foreground controller'}
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

BASE_SHA='c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141'
A1_SHA='0dca1fa52838388ea1e7afbc3a4c5e9747775f64e871640df7a66496369e5453'
CARRY=3413.519101628801

def validate_original(path, expected):
    import hashlib
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise ValueError('original budget drift')
    data=json.loads(path.read_text())
    if (data['schema']!=1 or data['spec_sha256']!=BASE_SHA or data['active'] is not None
        or data['held_seconds']!=CARRY or data['limit_seconds']!=LIMIT_SECONDS
        or data['admission_seconds']!=ADMISSION_SECONDS): raise ValueError('original carry binding differs')
    return data

_LegacyHeldBudget=HeldBudget
class HeldBudget(_LegacyHeldBudget):
    def __init__(self,path,spec_sha,turn,*,original_path,original_sha,clock=time.monotonic):
        import copy
        if spec_sha!=A1_SHA: raise ValueError('A1 budget spec differs')
        original=validate_original(original_path,original_sha)
        path=Path(path)
        if not path.is_absolute() or '..' in path.parts or any(p.is_symlink() for p in (path,*path.parents)): raise ValueError('unsafe budget path')
        if path==Path(original_path) or path.resolve()==Path(original_path).resolve(): raise ValueError('cannot mutate original budget')
        if not path.exists():
            data=copy.deepcopy(original)
            data.update(schema=2,spec_sha256=A1_SHA,parent_spec_sha256=BASE_SHA,
                        original_ledger_sha256=original_sha,carry_seconds=CARRY)
            if not path.is_absolute() or '..' in path.parts or any(p.is_symlink() for p in (path,*path.parents)):
                raise ValueError('unsafe budget path')
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x') as output:
                json.dump(data,output,indent=2); output.write('\n'); output.flush(); os.fsync(output.fileno())
        # Validate schema2 directly; all time/admission/save/restore methods inherited.
        self.path,self.spec_sha,self.turn,self.clock=path,spec_sha,turn,clock
        self.started=None
        self.data=json.loads(path.read_text())
        d=self.data
        if (d['schema']!=2 or d['spec_sha256']!=A1_SHA or d['parent_spec_sha256']!=BASE_SHA
            or d['original_ledger_sha256']!=original_sha or d['carry_seconds']!=CARRY
            or d['limit_seconds']!=LIMIT_SECONDS or d['admission_seconds']!=ADMISSION_SECONDS
            or not isinstance(d['held_seconds'],(int,float)) or not math.isfinite(d['held_seconds'])
            or d['held_seconds']<CARRY): raise ValueError('A1 ledger/carry changed')
        if d['active'] is not None: raise RuntimeError('unfinished interval: reconcile actual restoration first')
