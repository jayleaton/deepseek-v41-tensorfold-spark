#!/usr/bin/env python3
"""Generated-task request/record adapter; no generator and no network operation."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
LEGACY=ROOT.parent.parent/'decl-control'
sys.path.insert(0,str(LEGACY))
import client3
from reconciliation import reconcile,wire

ARMS=('plain','declared_off','lane')


def build_request(task,seed,arm,settings,tools):
    """Pure adapter. Sampling/answer prompt/seeds are supplied by the future spec."""
    if arm not in ARMS or type(seed) is not int: raise ValueError('invalid arm/seed')
    if not isinstance(task.get('prompt'),str) or not task['prompt'].strip():
        raise ValueError('ordinary task prompt required')
    if any(k in settings for k in ('tools','messages','seed')):
        raise ValueError('body identity fields cannot be overridden by settings')
    body=copy.deepcopy(settings)
    if body.get('stream') is not True or body.get('return_token_ids') is not True:
        raise ValueError('stream and native returned-ID evidence required')
    if not isinstance(body.get('model'),str) or not body['model']:
        raise ValueError('explicit model required')
    body.update(messages=[{'role':'user','content':task['prompt']}],seed=seed)
    if arm!='plain': body['tools']=copy.deepcopy(tools)
    sent=wire(body)
    return {'key':[task['suite'],task['id'],seed],'arm':arm,
            'body_template':body,'wire':sent,'wire_sha256':hashlib.sha256(sent.encode()).hexdigest(),
            'execution_policy':{'TF_DSV41_COMPUTE_LANE':str(int(arm=='lane'))}}


def require_declared_pair(d,l):
    if d['arm']!='declared_off' or l['arm']!='lane' or d['key']!=l['key']:
        raise ValueError('D/L pair identity mismatch')
    for row in (d,l):
        if wire(row['body_template'])!=row['wire'] or hashlib.sha256(row['wire'].encode()).hexdigest()!=row['wire_sha256']:
            raise ValueError('wire/body drift')
    if d['wire'].encode()!=l['wire'].encode(): raise ValueError('D/L HTTP bytes differ')
    return True


def classify_observation(row,template,cpu_prompt,decoder):
    """Same A1 semantics: no D execution/follow-up; missing input stays unknown."""
    row=copy.deepcopy(row)
    arm=template['arm']
    row['arm']=arm
    row['initial_prompt_reconciliation']=reconcile(row,dict(template,body=template['body_template']),cpu_prompt)
    row['attempts']=client3.assemble_attempts(row.get('sse_frames') or [])
    row['native_accounting']=client3.accounting(row,template['body_template']['max_tokens'])
    if arm=='declared_off' and (row.get('external_tool_calls') or row['attempts']['malformed']):
        if row['status']=='ok': row['status']='policy_terminated'
        row['answer_eligible']=False
        row['policy_termination']='client-visible call retained; no executor, fabricated result or follow-up'
    else:
        row['answer_eligible']=row['status']=='ok' and not row.get('external_tool_calls')
    binding=client3.authenticate_calls(row,decoder)
    row['native_binding']=binding
    row['input_accounting']=client3.input_accounting(row,binding)
    row['call_accounting']=client3.call_accounting(row,decoder)
    if arm in ('plain','declared_off') and ((row.get('tensorfold') or {}).get('compute') or {}).get('calls'):
        raise ValueError('effective-OFF assertion failed; caller must retain raw observation before stopping')
    return row
