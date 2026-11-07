#!/usr/bin/env python3
"""Fake HTTP/SSE and collector persistence checks; no socket or GPU operation."""
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import client3
import controller3
from reconciliation import wire


def main():
    root = Path(__file__).resolve().parent
    remote=controller3.remote_rank(['python3','-B','fixture.py'])[-1]
    for required in ('systemd-run','--scope','MemoryMax=2G','MemorySwapMax=0','nice -n 19','ionice -c3','timeout 7200','MAKEFLAGS=-j1'):
        assert required in remote
    draft = json.loads((root/'DRAFT-requests-NOT-QUALIFIED.json').read_text())
    prompts = {tuple(r['key']):r for r in map(json.loads,(root/'prompt-arrays.jsonl').read_text().splitlines())}
    template = next(r for r in draft['requests'] if r['arm']=='declared_off')
    template = dict(template, suite=template['key'][0], id=template['key'][1], seed=template['key'][2],
                    task_sha256='fixture-only', body_template=template['body'])
    n = len(prompts[tuple(template['key'])]['declared_off_ids'])
    ids = [1,2,3]
    tf = {'token_ids':ids,'token_sha256':client3.old.sha_ids(ids),'enable_thinking':True,'reasoning_effort':'high'}
    frames = [
        {'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'id':'call-1','type':'function','function':{'name':'py','arguments':'{"code":'}}]}}]},
        {'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'function':{'name':'thon','arguments':'"print(323)"}'}}]}}]},
        {'choices':[{'index':0,'delta':{},'finish_reason':'tool_calls'}],
         'usage':{'prompt_tokens':n,'completion_tokens':3},'tensorfold':tf}]
    payload = [('data: '+json.dumps(f)+'\n').encode() for f in frames]+[b'data: [DONE]\n']
    sends = []
    class FakeResponse:
        status=200
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def __iter__(self): return iter(payload)
    def fake_urlopen(request, **kwargs):
        sends.append(request.data)
        assert request.data == template['wire'].encode()
        assert kwargs['timeout']>=330
        return FakeResponse()
    with tempfile.TemporaryDirectory(prefix='decl-client3-check-',dir=root) as tmp:
        tmp = Path(tmp)
        go = tmp/'GO'; go.write_text(controller3.SPEC_SHA)
        transport = client3.Transport('http://127.0.0.1:8000/v1/chat/completions',go,
            root/'T5-DECL-CONTROL-SPEC-A1.md',controller3.SPEC_SHA)
        with patch.object(client3.urllib.request,'urlopen',side_effect=fake_urlopen):
            row = transport(template['body'])
            assert row['native_accounting']['complete']
            assert row['attempts']['calls'][0]['name']=='python'
            assert json.loads(row['attempts']['calls'][0]['arguments'])=={'code':'print(323)'}
            assert row['actual_request_wire']==template['wire']
            turn = controller3.Turn.__new__(controller3.Turn)
            turn.me = 'stdbench-fake-client3'
            turn.out = tmp/'run'; (turn.out/'t01-declared_off').mkdir(parents=True)
            turn.health = lambda:None
            turn.cfg = {'go_file':str(go),'spec':str(root/'T5-DECL-CONTROL-SPEC-A1.md'),
                        'tokenizer':'NOT_USED_FAKE_DECODER','tokenizer_sha256':'NOT_USED'}
            turn.prompts = prompts
            binding = {'qualified':True,'delivered_successes':0,'native_frame_count':0}
            with patch.object(controller3,'QUIET',tmp/'QUIET'), \
                 patch.object(client3,'NativeDecoder',return_value=None), \
                 patch.object(client3,'authenticate_calls',return_value=binding):
                observed, safety = turn.measure_one({'arm':'declared_off','boot':'t01-declared_off'},template)
            assert observed['status']=='policy_terminated' and observed['transport_status']=='ok'
            assert observed['answer_eligible'] is False and observed['native_accounting']['complete']
            assert observed['call_accounting']['attempted']==1
            assert observed['call_accounting']['blocked_dispatches']==1
            assert observed['call_accounting']['successful_executions']==0
            assert observed['initial_prompt_reconciliation']['qualified']
            assert observed['input_accounting']['prompt_plus_returned_tokens']==n+3
            saved=json.loads((turn.out/'t01-declared_off/responses.jsonl').read_text())
            assert saved['actual_request_wire']==template['wire']
            assert saved['status']=='policy_terminated'
            assert not (tmp/'QUIET').exists()
            payload = payload[:-1]+[b'data: {"error":{"code":"loop_stop"}}\n',b'data: [DONE]\n']
            failed = transport(template['body'])
            assert failed['status']=='error' and failed['transport_status']=='ok'
            assert failed['native_accounting']['complete']
            payload = [b'data: {"error":{"code":"engine_failed"}}\n',b'data: [DONE]\n']
            failed = transport(template['body'])
            assert failed['status']=='error' and not failed['native_accounting']['complete']
        go.write_text('NOT_GO')
        with patch.object(client3.urllib.request,'urlopen',side_effect=AssertionError('network must be unreachable')):
            try: transport(template['body'])
            except ValueError: pass
            else: raise AssertionError('missing GO accepted')
    assert len(sends)==4
    assert client3.generation_errors({'sse_frames':[],'sse_done':False,'finish_reason':'stop'})
    unknown = dict(observed, tensorfold=dict(tf,compute={'calls':[{'inserted':True}], 'inserted_tokens':2}))
    assert client3.input_accounting(unknown,binding)['processed_input_tokens'] is None
    result = {'verdict':'PASS','fake_sends':len(sends),'D_attempt_retained_incorrect_no_execution':True,
              'wire_bytes_exact':True,'multi_round_totals_withheld':True,'real_network':False,
              'dispatch_proof':'separate mandatory native harness',
              'sha256': {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [root/'client3.py',root/'controller3.py',root/'reconciliation.py',Path(__file__)]}}
    with (root/'client3-check.json').open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
