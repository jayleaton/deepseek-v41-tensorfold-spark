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
