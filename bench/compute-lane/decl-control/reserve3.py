#!/usr/bin/env python3
"""Pre-registered reserve selection: presence only, never correctness or costs."""
import json
from pathlib import Path


def select_missing(boots, observations):
    expected={}
    for boot in boots:
        for row in boot['requests']:
            key=(row['suite'],row['id'],row['seed'],boot['arm'])
            if key in expected: raise ValueError('duplicate frozen request')
            expected[key]=row
    seen=set()
    for row in observations:
        key=(row['suite'],row['id'],row['seed'],row['arm'])
        if key in seen or key not in expected: raise ValueError('duplicate/foreign observed unit')
        if row.get('diagnostic') or row.get('actual_request_wire')!=expected[key]['wire']:
            raise ValueError('observed unit not bound to frozen wire')
        # Every persisted request counts, including errors, caps and bad answers.
        seen.add(key)
    selected=[]
    for boot in boots:
        rows=[r for r in boot['requests'] if (r['suite'],r['id'],r['seed'],boot['arm']) not in seen]
        if rows: selected.append(dict(boot,requests=rows))
    return selected


def from_settled_parent(parent_cfg, parent_runs, frozen):
    parent_runs, frozen=Path(parent_runs),Path(frozen)
    index=json.loads((frozen/'manifest.json').read_text())
    boots=[json.loads((frozen/e['file']).read_text()) for e in index['boots']]
    observations=[]
    for turn in ('t01','t02','t03'):
        run=parent_runs/(parent_cfg['prefix']+'-'+turn)
        verdict=json.loads((run/'verdict.json').read_text())
        if verdict['verdict'] not in ('QUALIFIED','STOPPED'): raise ValueError('parent turn is not settled')
        steps=[json.loads(s) for s in (run/'steps.jsonl').read_text().splitlines()]
        if not any(s['step']=='G-restore' and s['verdict']=='SERVING_VERIFIED' for s in steps):
            raise ValueError('parent restoration not verified')
        for boot in next(t['boots'] for t in index['turns'] if t['name']==turn):
            path=run/boot/'responses.jsonl'
            if path.exists(): observations += [json.loads(s) for s in path.read_text().splitlines()]
    result=select_missing(boots,observations)
    if not result: raise ValueError('no unmeasured units; reserve forbidden')
    return result


def selfcheck():
    row=lambda i:{'suite':'s','id':i,'seed':101,'wire':i}
    boots=[{'arm':'plain','requests':[row('a'),row('b')]}]
    for status in ('ok','error','timeout','policy_terminated'):
        seen=[{'suite':'s','id':'a','seed':101,'arm':'plain','actual_request_wire':'a',
               'status':status,'grade':'incorrect'}]
        result=select_missing(boots,seen)
        assert [r['id'] for r in result[0]['requests']]==['b']
    try: select_missing(boots,seen+seen)
    except ValueError: pass
    else: raise AssertionError('duplicate observed response accepted')
    return {'reserve_selection':'PASS','outcome_independent':True,'observed_errors_never_replaced':True}
