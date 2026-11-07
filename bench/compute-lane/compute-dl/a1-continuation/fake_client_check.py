#!/usr/bin/env python3
"""Hand-written FAKE fixtures only; never study task generation or model access."""
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import protocol
import request_adapter as adapter


def rejected(call):
    try: call()
    except (ValueError,FileNotFoundError,KeyError): return
    raise AssertionError('invalid input accepted')


def main():
    root=Path(__file__).resolve().parent
    tools=json.loads((root.parent.parent/'decl-control/canonical-python-tools.json').read_text())
    # A single trivial handwritten fixture is not part of the future120-task set.
    task={'suite':'TEST_ONLY','id':'not-a-study-unit','prompt':'What is 2 plus 2?',
          'answer':'4','payload':{'left':2,'right':2}}
    settings={'model':'FAKE-NOT-SERVED','stream':True,'return_token_ids':True,
              'max_tokens':8192,'enable_thinking':True,'reasoning_effort':'high'}
    d=adapter.build_request(task,101,'declared_off',settings,tools)
    l=adapter.build_request(task,101,'lane',settings,tools)
    assert adapter.require_declared_pair(d,l)
    assert d['execution_policy']!=l['execution_policy']
    assert 'answer' not in d['body_template'] and 'payload' not in d['body_template']
    assert d['body_template']['messages']==[{'role':'user','content':task['prompt']}]
    mutated=dict(l,wire=l['wire']+' ')
    rejected(lambda:adapter.require_declared_pair(d,mutated))
    rejected(lambda:adapter.build_request(task,101,'lane',dict(settings,seed=0),tools))
    rejected(lambda:protocol.require_collection_ready({}))
    rejected(lambda:protocol.require_bound_spec({}))
    with tempfile.TemporaryDirectory(prefix='fake-spec-',dir=root) as directory:
        spec=Path(directory)/'spec';spec.write_text('FAKE_NOT_COLLECTION')
        cfg={'spec':str(spec),'spec_sha256':hashlib.sha256(spec.read_bytes()).hexdigest()}
        rejected(lambda:protocol.require_bound_spec(cfg))
        rejected(lambda:protocol.require_collection_ready(cfg))
        spec.write_text('changed')
        rejected(lambda:protocol.require_bound_spec(cfg))
    ids=[1,2,3]
    frame={'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'id':'call-fixture','type':'function',
            'function':{'name':'python','arguments':'{"code":"print(4)"}'}}]},'finish_reason':'tool_calls'}],
           'usage':{'prompt_tokens':2,'completion_tokens':3},
           'tensorfold':{'token_ids':ids,'token_sha256':adapter.client3.old.sha_ids(ids)}}
    parsed=adapter.client3.old.parse_stream(['data: '+json.dumps(frame)])
    row=dict(parsed,status='ok',transport_status='ok',sse_frames=[frame],sse_done=True,
             actual_request_wire=d['wire'])
    cpu={'key':d['key'],'declared_off_ids':[6,7]}
    binding={'qualified':True,'native_frame_count':0,'delivered_successes':0}
    with patch.object(adapter.client3,'authenticate_calls',return_value=binding):
        observed=adapter.classify_observation(row,d,cpu,None)
    assert observed['status']=='policy_terminated' and not observed['answer_eligible']
    assert observed['native_accounting']['complete'] and observed['initial_prompt_reconciliation']['qualified']
    assert observed['call_accounting']['blocked_dispatches']==1
    assert observed['call_accounting']['successful_executions']==0
    unknown=dict(row,tensorfold=dict(frame['tensorfold'],compute={'calls':[{'inserted':True}]}))
    with patch.object(adapter.client3,'authenticate_calls',return_value=binding):
        rejected(lambda:adapter.classify_observation(unknown,d,cpu,None))
    result={'passed':True,'status':'FAKE_NOT_COLLECTION','real_network':False,
            'real_tasks_generated':0,'D_call_no_external_execution':True,
            'spec_drift_and_missing_gate_rejected':True,'declared_wire_identity':True}
    return result


if __name__=='__main__': main()
