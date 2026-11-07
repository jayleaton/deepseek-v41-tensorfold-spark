#!/usr/bin/env python3
"""Read actual native fixture receipts without displaying prompts or token IDs."""
import json
from pathlib import Path
from client3 import assemble_attempts


def audit(root):
    root=Path(root)
    rows={r['case']:r for r in map(json.loads,(root/'fixture-receipts.jsonl').read_text().splitlines())}
    assert len(rows)==8
    for form in ('whole','stream'):
        d,l=rows['D-'+form],rows['L-'+form]
        assert d['dispatch_attempts']==d['dispatch_forwarded']==d['resume_count']==0
        assert d['held'] is False and d['submits']==1 and d['http_status']==200
        assert l['dispatch_attempts']==l['dispatch_forwarded']==1
        assert l['held'] is True and l['submits']==1 and l['resume_count']==2
        assert d['submitted_prompt_ids']==l['submitted_prompt_ids']
        assert d['request_wire']==l['request_wire']
        request=json.loads(d['request_wire'])
        assert request['reasoning_effort']=='high' and request['max_tokens']==8192
        assert request['deadline_s']==270 and request['temperature']==1
        assert request['top_p']==0.95 and request['top_k']==20 and request['seed']==101
        assert not d['resume_input_ids'] and l['resume_input_ids'][0] and not l['resume_input_ids'][1]
        if form=='whole':
            assert d['response']['choices'][0]['finish_reason']=='tool_calls'
            call=d['response']['choices'][0]['message']['tool_calls'][0]['function']
            assert call['name']=='python' and json.loads(call['arguments'])=={'code':'print(17*19)'}
            assert l['response']['choices'][0]['finish_reason']=='stop'
            compute=l['response']['tensorfold']['compute']
        else:
            attempts=assemble_attempts(d['sse_payloads'])
            assert len(attempts['calls'])==1 and not attempts['malformed']
            assert attempts['calls'][0]['name']=='python'
            assert json.loads(attempts['calls'][0]['arguments'])=={'code':'print(17*19)'}
            assert d['sse_done']==l['sse_done']==1
            compute=next(f['tensorfold']['compute'] for f in reversed(l['sse_payloads'])
                         if 'compute' in f.get('tensorfold',{}))
        assert len(compute['calls'])==1
        assert compute['calls'][0]['status']=='ok' and compute['calls'][0]['output']=='323'
        assert compute['calls'][0]['inserted'] is True
        assert rows['noncall-'+form]['dispatch_attempts']==0
        assert rows['wrongly-armed-D-'+form]['dispatch_attempts']==1
    return {'verdict':'PASS','D_dispatches_each':0,'L_dispatches_each':1,'whole_and_stream':True,
            'D_semantics':'HTTP200 tool_calls; no insertion or resume, no client follow-up',
            'proof_scope':'pinned-source routing, not instrumentation of deployed binary'}


if __name__=='__main__': print(json.dumps(audit(Path(__file__).resolve().parent/'dispatch-high')))
