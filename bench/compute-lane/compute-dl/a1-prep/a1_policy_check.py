#!/usr/bin/env python3
"""Prospective A1 presentation-policy controls against actual pinned native functions."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from reference_policy import NativeCap,compare_pairs,reference_body,sha
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def main():
    cap=NativeCap(HERE/'zig-out/bin/native-cap',ROOT.parent/'phase2/references/tokenizer.json',ROOT.parent/'decl-control/dispatch-high',HERE/'native-build-receipt.json')
    source=ROOT.parent/'phase2/jail_worker.py'
    spec=importlib.util.spec_from_file_location('pinned_jail_worker_controls',source)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    def reference(out=b'',err=b'',status='ok',code=0):
        body=worker.normalized(out,err,status,code)
        return dict(status=status,exit_code=code,stdout_base64=base64.b64encode(out).decode(),
                    stderr_base64=base64.b64encode(err).decode(),normalized_body_base64=base64.b64encode(body).decode(),
                    normalized_body_sha256=hashlib.sha256(body).hexdigest(),elapsed_seconds=.01,
                    output_limited=status=='output_limit',native_result_token_cap=512)
    samples=[reference(),reference(b'42\n'),reference(b'out\n',b'err\n'),reference(err=b'err\n'),
             reference(b'out',b'error',status='exit_error',code=1),reference(status='exit_error',code=-9),
             reference(b'prefix',status='timeout',code=-9),reference(b'x'*65536,status='output_limit',code=-9)]
    records=[]
    for i,r in enumerate(samples):
        records.append({'key':i,'stdout':base64.b64decode(r['stdout_base64']).decode(),
                        'stderr':base64.b64decode(r['stderr_base64']).decode(),'status':r['status'],'exit_code':r['exit_code']})
    native=cap.run_records(records)
    for ref,actual in zip(samples,native,strict=True):
        assert reference_body(ref)==actual['normalized_body'].encode()
        assert compare_pairs(cap,[({'status':ref['status'],'output':actual['output']},ref)])==[True]
    boundaries=['1 '*255+'1','1 '*256,'1 '*256+'1']
    values=cap.cap([b.encode() for b in boundaries])
    assert [r['token_count'] for r in values]==[511,512,513]
    assert values[0]['output']==boundaries[0] and values[1]['output']==boundaries[1]
    assert values[2]['output'].endswith('\n[output truncated at 512 tokens]')
    # Native decoding handles 2-,3-,4-byte character seams at the actual token cap.
    seams=[('1 '*n+u*40).encode() for u in ('é','界','🙂') for n in range(250,257)]
    seam_rows=cap.cap(seams)
    assert all(r['token_count']>512 for r in seam_rows)
    assert all(r['output'].encode().decode()==r['output'] for r in seam_rows)
    literal=b'42\n[output truncated at 512 tokens]'
    assert cap.cap([literal])[0]['output'].encode()==literal
    short=reference(b'42')
    badoutputs=['42\n[output truncated at 512 tokens]','42\n[output truncated at 511 tokens]',' 42','42\n']
    assert compare_pairs(cap,[({'status':'ok','output':s},short) for s in badoutputs])==[False]*4
    for key,value in [('normalized_body_sha256','0'*64),('normalized_body_base64','!'),('status','unavailable'),('native_result_token_cap',511),('output_limited',True)]:
        bad=copy.deepcopy(short);bad[key]=value
        try:reference_body(bad)
        except (ValueError,KeyError):pass
        else:raise AssertionError('bad reference policy accepted: '+key)
    case=json.loads((ROOT/'diagnosis-v1/retained-case.json').read_text())
    ref=next(r['result'] for r in case['safety']['references'] if r['matches'] is False)
    full=reference_body(ref);changed=full[:-1]+b'Z'
    original,tail=cap.cap([full,changed])
    assert full!=changed and original['output']==tail['output']
    # Source/code/status and missing-frame adversarial checks are performed on
    # real receipts by revalidate.py; their evidence is required at resumption.
    result={'passed':True,'spec_sha256':'0dca1fa52838388ea1e7afbc3a4c5e9747775f64e871640df7a66496369e5453',
            'boundary_tokens':[r['token_count'] for r in values],'native_body_branches':len(samples),
            'UTF8_seams':len(seam_rows),'false_marker_and_byte_mutations_rejected':4,
            'reference_policy_mutations_rejected':5,'invisible_tail_change':'observationally indistinguishable after clipping; not detected',
            'native_build_receipt_sha256':sha(HERE/'native-build-receipt.json'),'worker_sha256':sha(source),
            'code_sha256':{str(p):sha(p) for p in (Path(__file__),HERE/'reference_policy.py')},'GPU_requests':0}
    (HERE/'a1-policy-check-receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'passed':True,'boundary_tokens':result['boundary_tokens'],'GPU_requests':0}))
if __name__=='__main__':main()
