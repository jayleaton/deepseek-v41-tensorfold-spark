#!/usr/bin/env python3
"""Fake-clock checks for cumulative held time and restoration reserve."""
import json
from pathlib import Path
import tempfile
from budget import HeldBudget,BudgetExhausted


def main():
    tick=[0.0]
    with tempfile.TemporaryDirectory(prefix='FAKE_BUDGET-') as directory:
        path=Path(directory)/'ledger.json';clock=lambda:tick[0]
        first=HeldBudget(path,'f'*64,'FAKE-t01',clock=clock)
        first.begin();tick[0]+=100;first.finish_after_verified_restore()
        tick[0]+=9999  # Offline work/queue waiting with G restored is excluded.
        second=HeldBudget(path,'f'*64,'FAKE-t02',clock=clock)
        assert second.elapsed()==100
        second.begin()
        try: HeldBudget(path,'f'*64,'FAKE-t03',clock=clock)
        except RuntimeError: pass
        else: raise AssertionError('unreconciled active interval accepted')
        tick[0]+=15800
        try: second.ensure_can_start()
        except BudgetExhausted: pass
        else: raise AssertionError('4h25 admission cutoff not enforced')
        # A required safety restoration may overrun; never terminate it here.
        tick[0]+=2200;second.finish_after_verified_restore()
        stored=json.loads(path.read_text())
        assert stored['held_seconds']==18100 and stored['overrun_seconds']==100
        assert stored['active'] is None and len(stored['turns'])==2
        third=HeldBudget(path,'f'*64,'FAKE-t03',clock=clock)
        try: third.begin()
        except BudgetExhausted: pass
        else: raise AssertionError('budget reset across turns')
    result={'passed':True,'status':'FAKE_NOT_COLLECTION','cross_turn_counter':True,
            'offline_time_excluded':True,'admission_4h25m':True,'safety_overrun_retained':True}
    with (Path(__file__).parent/'budget-check.json').open('x') as output:
        json.dump(result,output,indent=2);output.write('\n')
    print(json.dumps(result))


if __name__=='__main__': main()
