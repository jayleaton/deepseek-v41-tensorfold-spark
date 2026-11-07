#!/usr/bin/env python3
"""Saved-receipt replay and adversarial body controls; CPU only, no engine calls."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
from reference_policy import NativeCap,compare_pairs,sha
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE=ROOT.parent/'decl-control/dispatch-high'
def main():
    receipt=json.loads((HERE/'native-build-receipt.json').read_text())
    cap=NativeCap(HERE/'zig-out/bin/native-cap',ROOT.parent/'phase2/references/tokenizer.json',SOURCE,HERE/'native-build-receipt.json')
    case=json.loads((ROOT/'diagnosis-v1/retained-case.json').read_text())
    refs={r['call_id']:r for r in case['safety']['references']}
    calls=[c for c in case['calls'] if c['inserted'] is True and refs[c['id']]['matches'] is False]
    assert len(calls)==1
    call=calls[0]; ref=refs[call['id']]['result']; full=base64.b64decode(ref['normalized_body_base64'],validate=True)
    assert full!=call['output'].encode()
    actual=cap.cap([full])[0]
    assert actual['token_count']==554
    assert actual['output']==call['output']
    assert compare_pairs(cap,[(call,ref)])==[True]
    controls=[]
    for field,value in [('output','X'+call['output'][1:]),('output',call['output']+'x'),
                         ('output',call['output'].replace('512 tokens','511 tokens')),('status','timeout')]:
        mutated=copy.deepcopy(call); mutated[field]=value; controls.append((mutated,ref))
    assert compare_pairs(cap,controls)==[False]*len(controls)
    # Ordinary short output, literal marker, empty-normalized output, and Unicode.
    short=['42','[no output]','answer\n[output truncated at 512 tokens]','<｜User｜> literal','é🙂世界']
    outputs=cap.cap([v.encode() for v in short])
    assert [r['output'] for r in outputs]==short
    # Numerals are single native tokens here: verify boundary without assuming it.
    boundaries=['1 '*n for n in range(250, 515)]
    rows=cap.cap([v.encode() for v in boundaries])
    assert any(r['token_count']==512 for r in rows)
    assert any(r['token_count']>512 for r in rows)
    for body,row in zip(boundaries,rows):
        if row['token_count']<=512: assert row['output']==body
        else: assert row['output'].endswith('\n[output truncated at 512 tokens]') and row['output']!=body
    unicode_rows=cap.cap([('🙂é世界 '*600).encode()])
    assert unicode_rows[0]['token_count']>512
    unicode_rows[0]['output'].encode('utf-8').decode('utf-8')
    bad=copy.deepcopy(ref); bad['normalized_body_base64']='!'
    try: compare_pairs(cap,[(call,bad)])
    except ValueError: pass
    else: raise AssertionError('malformed reference accepted')
    result={'passed':True,'GPU_requests':0,'policy':'native encodePlain -> capTokens(512) -> decode(false)',
            'saved_case':case.get('case',call.get('id')),'full_tokens':actual['token_count'],
            'full_body_sha256':hashlib.sha256(full).hexdigest(),
            'delivered_body_sha256':hashlib.sha256(call['output'].encode()).hexdigest(),
            'old_full_comparison':False,'new_exact_comparison':True,'negative_controls':len(controls)+1,
            'boundary_token_counts':[r['token_count'] for r in rows if 510 <= r['token_count'] <= 515],
            'binary_sha256':receipt['binary_sha256'],'source_pins':receipt['source_pins'],
            'scope':'body comparison only; existing provenance gates remain mandatory',
            'collection_authorized':False,'amended_spec_sha256':None}
    (HERE/'clip-check-receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('passed','GPU_requests','full_tokens','new_exact_comparison')}))
if __name__=='__main__': main()
