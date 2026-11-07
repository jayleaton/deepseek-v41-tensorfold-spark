#!/usr/bin/env python3
"""Diagnostic-only native tokenizer proof. No gate relaxation or model requests."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE=Path(__file__).resolve().parent
LEGACY=HERE.parent/'decl-control'
OUT=HERE/'diagnosis-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest=lambda value:hashlib.sha256(value).hexdigest()
load=lambda p:json.loads(Path(p).read_text())

def save(path,value):
    with path.open('x') as output:
        path.chmod(0o600);json.dump(value,output,indent=2)

def main():
    cfg=load(HERE/'stdbench-compute-dl-20261105-v1/campaign-v1.json')
    for path,expected in cfg['code_sha256'].items():
        if sha(path)!=expected:raise ValueError('precollection source/data drift')
    case=load(OUT/'retained-case.json')
    refs={r['call_id']:r for r in case['safety']['references']}
    calls=[c for c in case['calls'] if c['inserted'] is True and refs[c['id']]['matches'] is False]
    if len(calls)!=1:raise ValueError('one actual mismatch required')
    call=calls[0];reference=refs[call['id']]['result']
    body=base64.b64decode(reference['normalized_body_base64'],validate=True)
    native=call['output'].encode()
    cap=re.search(rb'\n\[output truncated at ([0-9]+) tokens\]$',native)
    if not cap or int(cap.group(1))!=512:raise ValueError('actual 512-token marker required')
    if call['status']!='ok' or reference['status']!='ok':raise ValueError('status mismatch is not clipping')
    source=LEGACY/'dispatch-high/zig/src/server/compute_segments.zig'
    expected_source=cfg['code_sha256'][str(source)]
    if sha(source)!=expected_source:raise ValueError('production clipping source drift')
    text=source.read_text()
    if 'fn capTokens(' not in text or 'srv.text.encodePlain(a, body)' not in text:raise ValueError('native clipping policy absent')
    old=load(LEGACY/'cpu-proof.json');binary=LEGACY/'prompt-gate'
    if sha(binary)!=old['binary_sha256']:raise ValueError('native renderer binary drift')
    tokenizer=Path(cfg['tokenizer'])
    if sha(tokenizer)!=cfg['tokenizer_sha256']:raise ValueError('tokenizer drift')
    plain=body.decode('utf-8')
    if '<｜User｜>' in plain or '<｜Assistant｜>' in plain:raise ValueError('diagnostic needs literal plain stdout')
    inputs=OUT/'truncation-render-input.json'
    save(inputs,[{'key':['diagnostic-clipping','full-stdout',0],
                  'messages':[{'role':'user','content':plain}]}])
    arrays=OUT/'truncation-render.jsonl'
    with arrays.open('x') as output,(OUT/'truncation-render.log').open('x') as log:
        run=subprocess.run([str(binary),str(inputs),str(tokenizer)],stdout=output,stderr=log,timeout=7200)
    if run.returncode:raise RuntimeError('retained native CPU renderer failure')
    rendered=[json.loads(line) for line in arrays.read_text().splitlines()]
    if len(rendered)!=1:raise ValueError('native renderer coverage differs')
    ids=rendered[0]['plain_ids']
    added={t['content']:t['id'] for t in load(tokenizer)['added_tokens']}
    user,assistant=added['<｜User｜>'],added['<｜Assistant｜>']
    if ids.count(user)!=1 or ids.count(assistant)!=1:raise ValueError('ambiguous native role boundary')
    body_ids=ids[ids.index(user)+1:ids.index(assistant)]
    sys.path.insert(0,str(HERE.parent/'phase2'))
    from client import NativeDecoder
    decoder=NativeDecoder(tokenizer,cfg['tokenizer_sha256'])
    if decoder.decode(body_ids)!=body:raise ValueError('native plain body token boundary differs')
    if len(body_ids)<=512:raise ValueError('reference does not exceed native result cap')
    n=512
    while n:
        prefix=decoder.decode(body_ids[:n])
        try:prefix.decode('utf-8');break
        except UnicodeDecodeError:n-=1
    expected=(prefix+b'\n[output truncated at 512 tokens]') if n else b'[output truncated at 512 tokens]'
    if expected!=native:raise ValueError('native 512-token policy does not explain delivered bytes')
    native_value=ast.literal_eval(native.splitlines()[0].decode())
    firstline_correct=native_value==load(HERE/'dataset-v1/answers-a.json')[case['identity']['id']]
    proof={'passed':True,'diagnosis_only':True,'frozen_gate_still_failed':True,'GPU_requests':0,
           'case':case['identity'],'classification':'NATIVE_RESULT_CAP_NOT_APPLIED_BY_FROZEN_REPLAY_COMPARISON',
           'full_reference_body_tokens':len(body_ids),'retained_prefix_tokens':n,'native_cap':512,
           'exact_source_policy_reconciles_delivered_bytes':True,'complete_first_line_matches_task_reference':firstline_correct,
           'native_output_sha256':digest(native),'full_reference_sha256':digest(body),
           'expected_clipped_output_sha256':digest(expected),'production_source_sha256':expected_source,
           'production_source_path':str(source),'native_renderer_sha256':sha(binary),
           'tokenizer_sha256':sha(tokenizer),'native_arrays_sha256':sha(arrays),'source_sha256':sha(__file__),
           'limitation':'Diagnostic confirmation only. No observed reply was replaced, gate changed, or collection resumed.'}
    save(OUT/'truncation-proof.json',proof)
    print(json.dumps({k:proof[k] for k in ('passed','classification','full_reference_body_tokens',
        'retained_prefix_tokens','exact_source_policy_reconciles_delivered_bytes','complete_first_line_matches_task_reference','frozen_gate_still_failed','GPU_requests')}))

if __name__=='__main__':main()
