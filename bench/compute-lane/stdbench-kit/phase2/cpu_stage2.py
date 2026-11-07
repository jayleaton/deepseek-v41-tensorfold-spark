#!/usr/bin/env python3
import hashlib
import json
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path('<rank1-work>/stdbench/phase2')
sys.path.insert(0, str(root))
import client
from jail import Jail

with tempfile.TemporaryDirectory(prefix='stdbench-bytecode-') as tmp:
    for name in ('generate_envs.py','rank_ops.py','client.py','freeze.py','cpu_guard.py','jail.py','jail_worker.py','controller.py'):
        py_compile.compile(str(root/name), cfile=str(Path(tmp)/(name+'c')), doraise=True)
lock_sha = hashlib.sha256((root.parent/'kit/datasets.lock.json').read_bytes()).hexdigest()
subprocess.run([sys.executable, str(root/'freeze.py'), '--dry-run', '--lock-sha', lock_sha, '--out', str(root/'freeze-dry-v4')], check=True)
subprocess.run([sys.executable, str(root/'generate_envs.py'), '--out', str(root/'stage-v2')], check=True)
subprocess.run([sys.executable, str(root/'client.py'), '--selftest', str(root/'client-selftest-v2.json')], check=True)
image = 'sha256:80beaba732d1ed9439e012b007d8dbe19ec2f8962f50fbac29a18c8cd6fda306'
subprocess.run([sys.executable, str(root/'jail.py'), '--qualify', '--image-id', image,
                '--source-gz', str(root.parent/'kit/sources/humaneval.jsonl.gz'),
                '--selected', str(root.parent/'humaneval-selected.jsonl'), '--out', str(root/'jail-qual-v1')], check=True)
receipt = json.loads((root/'jail-qual-v1/qualification.json').read_text())
assert receipt['qualified'], 'jail qualification failed; inspect retained cases'
original = json.loads((root/'references/native-sample.json').read_text())
record = {'status':'ok', 'sse_done':True, 'usage':original['usage'], 'tensorfold':original['extra']['tensorfold']}
record['native_accounting'] = client.accounting(record)
assert record['native_accounting']['complete']
tokenizer = root/'references/tokenizer.json'
decoder = client.NativeDecoder(tokenizer, hashlib.sha256(tokenizer.read_bytes()).hexdigest())
auth = client.authenticate_calls(record, decoder)
assert auth['qualified'] and auth['delivered_successes'] > 0
jail = Jail(image)
for call in record['tensorfold']['compute']['calls']:
    if call.get('inserted'):
        result = jail.run_reference(call['code'])
        assert result['status'] == call['status'] and result['normalized_body'] == call['output']
proof = {'passed':True, 'native_fixture':record['native_accounting'],
         'native_source_output_binding':auth['qualified'], 'jail_canonical_and_negative_controls':True,
         'tokenizer_sha256':hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
         'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}}
(root/'cpu-stage2.json').write_text(json.dumps(proof, indent=2)+'\n')
print(json.dumps({'stage2':'PASS', 'jail_cases':len(receipt['results']), 'native_fixture_accounting':True}))
