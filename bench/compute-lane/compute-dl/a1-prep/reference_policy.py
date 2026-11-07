#!/usr/bin/env python3
"""Exact delivered-body comparison, after existing native/jail provenance gates.

This module replaces only body comparison. It grants no collection approval and
must not replace the dispatch, call-ID, payload, worker or image audits.
"""
import base64
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
TOKENIZER_SHA='c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class NativeCap:
    def __init__(self, binary, tokenizer, source, receipt):
        self.binary=Path(binary); self.tokenizer=Path(tokenizer)
        self.source=Path(source); self.receipt=Path(receipt)
        self.pins=json.loads(self.receipt.read_text())
        self.verify()
    def verify(self):
        if sha(self.binary)!=self.pins['binary_sha256']: raise ValueError('native cap binary drift')
        if sha(self.tokenizer)!=TOKENIZER_SHA: raise ValueError('tokenizer drift')
        for rel,expected in self.pins['harness_sha256'].items():
            if sha(HERE/rel)!=expected: raise ValueError('native cap harness drift: '+rel)
        for rel,expected in self.pins['source_pins'].items():
            if sha(self.source/rel)!=expected: raise ValueError('native cap source drift: '+rel)
        source=(self.source/'zig/src/server/compute_segments.zig').read_text()
        start=source.index('fn capTokens('); end=source.index('\n}\n',start)+2
        header='const std = @import("std");\nconst Allocator = @import("root").CapAllocator;\nconst Server = @import("root").CapServer;\n'
        if (HERE/'cap_function.zig').read_text()!=header+'pub '+source[start:end]+'\n':
            raise ValueError('generated function differs from pinned production policy')
    def cap(self, bodies):
        self.verify()
        rows=[{'key':i,'body':b.decode('utf-8','strict')} for i,b in enumerate(bodies)]
        return self.run_records(rows)
    def run_records(self, rows):
        self.verify()
        with tempfile.TemporaryDirectory(prefix='native-cap-',dir=HERE) as directory:
            path=Path(directory)/'input.json'; path.write_text(json.dumps(rows))
            result=subprocess.run([str(self.binary),str(path),str(self.tokenizer)],
                                  capture_output=True,timeout=7200,check=True)
        output=[json.loads(line) for line in result.stdout.splitlines()]
        if len(output)!=len(rows) or [r['key'] for r in output]!=[r['key'] for r in rows]:
            raise ValueError('native cap coverage mismatch')
        return output

def compare_pairs(capper, pairs):
    """Return exact booleans; unavailable, malformed and status mismatch fail closed."""
    bodies=[]
    for call,result in pairs:
        if not isinstance(result.get('normalized_body_base64'),str): raise ValueError('missing reference body')
        bodies.append(reference_body(result))
    expected=capper.cap(bodies)
    return [isinstance(call.get('status'),str) and call['status']==result.get('status')
            and isinstance(call.get('output'),str)
            and call['output'].encode('utf-8')==row['output'].encode('utf-8')
            for (call,result),row in zip(pairs,expected,strict=True)]

def reference_body(result):
    """Check the pinned jail/native normalization, byte limit and supported status."""
    status=result.get('status'); code=result.get('exit_code')
    if status not in ('ok','exit_error','timeout','output_limit'):
        raise ValueError('unsupported independent reference status')
    if type(code) is not int or (status=='ok' and code!=0):
        raise ValueError('reference exit status unavailable/inconsistent')
    out=base64.b64decode(result['stdout_base64'],validate=True)
    err=base64.b64decode(result['stderr_base64'],validate=True)
    if len(out)+len(err)>65536 or result.get('output_limited') is not (status=='output_limit'):
        raise ValueError('reference byte-limit policy mismatch')
    if result.get('native_result_token_cap')!=512:
        raise ValueError('reference token policy mismatch')
    elapsed=result.get('elapsed_seconds')
    if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0:
        raise ValueError('reference timing unavailable')
    body=out
    if err: body+=(b'\n[stderr]\n' if body else b'[stderr]\n')+err
    if status!='ok':
        if body and not body.endswith(b'\n'): body+=b'\n'
        if status=='timeout': body+=b'[timed out after 3000 ms]'
        elif status=='output_limit': body+=b'[output truncated at 65536 bytes]'
        elif code<0: body+=('[killed by signal %d]' % -code).encode()
        else: body+=('[exit status %d]' % code).encode()
    if body.endswith(b'\n'): body=body[:-1]
    body=body or b'[no output]'
    stored=base64.b64decode(result['normalized_body_base64'],validate=True)
    if body!=stored or hashlib.sha256(body).hexdigest()!=result.get('normalized_body_sha256'):
        raise ValueError('reference body normalization/hash mismatch')
    return body

def presentation(capper, call, result):
    body=reference_body(result); row=capper.cap([body])[0]
    landed=call['output'].encode()
    exact=(call.get('status')==result.get('status') and landed==row['output'].encode())
    clipped=row['token_count']>512
    return {'qualified':exact,'classification':('presentation-verified clipped' if clipped else 'presentation-verified unclipped') if exact else 'unqualified',
            'full_reference_tokens':row['token_count'],'retained_prefix_tokens':row['retained_prefix_tokens'],
            'full_reference_bytes':len(body),'landed_body_bytes':len(landed),
            'full_reference_sha256':hashlib.sha256(body).hexdigest(),'landed_body_sha256':hashlib.sha256(landed).hexdigest(),
            'marker_bytes':list(b'\n[output truncated at 512 tokens]') if clipped else [],
            'authenticates':'landed presentation; invisible tail is not authenticated'}
