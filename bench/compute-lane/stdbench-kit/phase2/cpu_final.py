#!/usr/bin/env python3
"""Final immutable precollection gate; execute only in guarded Spark CPU scope."""

import sys as _archive_sys
import hashlib
import json
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path
root = Path('<rank1-work>/stdbench/phase2')
sys.path[:0] = [str(root),str(root.parent/'kit')]
import activation_report, analysis, bench, client, controller, freeze, grade_all, math_review, native_frames
from jail import Jail

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = root/'cpu-final-v3'
out.mkdir(exist_ok=False)
for error in ('error: no such object: owned', 'Error: No such object: owned', 'No such container: owned'):
    assert controller.container_absent(error)
assert not controller.container_absent('permission denied')
assert not controller.container_absent('Cannot connect to the Docker daemon')
absent=subprocess.run(['docker','inspect','stdbench-never-created-absence-control'],capture_output=True,text=True)
assert absent.returncode and controller.container_absent(absent.stderr)
prior=root/'retained-v1/pilot-plain'
old_manifest=json.loads((root/'frozen-v2/manifest.json').read_text())
entry=next(b for b in old_manifest['boots'] if b['boot']=='pilot-plain')
boot=json.loads((root/'frozen-v2'/entry['file']).read_text())
retained={'directory':str(prior),'sha256':{n:sha(prior/n) for n in ('responses.jsonl','diagnostics.jsonl','safety.jsonl')}}
assert controller.completed_boot(boot,retained)
bad=json.loads(json.dumps(boot)); bad['requests'].pop()
try:
    controller.completed_boot(bad,retained)
except ValueError:
    pass
else:
    raise AssertionError('must reject incomplete retained cohort')
with tempfile.TemporaryDirectory(prefix='stdbench-final-syntax-') as tmp:
    for p in sorted(root.glob('*.py')):
        py_compile.compile(str(p),cfile=str(Path(tmp)/(p.name+'c')),doraise=True)
checks = {'analysis':analysis.selftest(),'grader':grade_all.selftest(),'activation':activation_report.selftest()}
math_review.selftest()
client.selftest(out/'client.json')
native_frames.selftest(out/'native-frames.json')
original = json.loads((root/'references/native-sample.json').read_text())
record = {'status':'ok','sse_done':True,'usage':original['usage'],'tensorfold':original['extra']['tensorfold']}
record['native_accounting'] = client.accounting(record)
decoder = client.NativeDecoder(root/'references/tokenizer.json',sha(root/'references/tokenizer.json'))
auth = client.authenticate_calls(record,decoder)
assert auth['qualified'], auth['reason']
assert auth['delivered_successes'] > 0
jail = Jail('sha256:80beaba732d1ed9439e012b007d8dbe19ec2f8962f50fbac29a18c8cd6fda306')
for c in record['tensorfold']['compute']['calls']:
    if c.get('inserted'):
        assert controller.reference_matches(c,jail.run_reference(c['code']))
qualified = json.loads((root/'jail-qual-v1/qualification.json').read_text())
assert qualified['qualified']
for name,expected in qualified['sources_sha256'].items():
    assert sha(root/name)==expected, 'frozen jail/extraction source changed'
assert json.loads((root/'postreview-v1.json').read_text())['passed']
subprocess.run([sys.executable,str(root/'controller_selftest.py'),str(out/'controller.json')],check=True)
assert sha(root/'references/tokenizer.json') == sha('<home>/.cache/dsv41-flash17-k6-mtp/model/tokenizer.json')
old=json.loads((root/'cpu-final-v1/verification.json').read_text())
for n in ('analysis.py','grade_all.py','math_review.py','activation_report.py','jail.py','jail_worker.py'):
    assert sha(root/n)==old['code_sha256'][str(root/n)], 'no grader/statistics changes after collection'
assert sha(root/'MATH-PRESEAL-v1.json') == old['math_preseal_file_sha256']
inputs = {s:root.parent/(s+'-selected.jsonl') for s in ('gsm8k','math500','humaneval')}
code = {str(p):sha(p) for pattern in ('*.py','*.sh') for p in sorted(root.glob(pattern))}
code.update({str(root.parent/'kit'/n):sha(root.parent/'kit'/n) for n in ('bench.py','grader_worker.py','datasets.lock.json')})
settings = {'model':'DeepSeek-V4.1-Flash-TF','reasoning_effort':'high','temperature':1.0,'top_p':.95,'top_k':20,'max_tokens':8192,'deadline_s':270,'request_timeout':360}
config = {'spec':str(root/'T5-STDBENCH-SPEC.md'),'spec_sha':'9c7327007cfd15825a608f2f6b06f2b39e33b88548c5bb7deb1b209b7395d240','kit':str(root.parent/'kit'),'lock':str(root.parent/'kit/datasets.lock.json'),'lock_sha':sha(root.parent/'kit/datasets.lock.json'),'inputs':{s:{'path':str(p),'sha256':sha(p)} for s,p in inputs.items()},'counts':{'gsm8k':48,'math500':20,'humaneval':24},'settings':settings,'code_sha_files':code}
manifest=freeze.freeze(config,root/'frozen-v3')
assert sum(x['pairs_per_arm'] for x in manifest['suites'].values())==184
receipt={'passed':True,'spec_sha256':config['spec_sha'],'checks':checks,'cohort_replies':368,'math_preseal':old['math_preseal'],'math_preseal_file_sha256':sha(root/'MATH-PRESEAL-v1.json'),'code_sha256':code,'freeze_sha256':sha(root/'frozen-v3/manifest.json'),'tokenizer_sha256':sha(root/'references/tokenizer.json'),'jail_cases':len(qualified['results']),'jail_receipt_sha256':sha(root/'jail-qual-v1/qualification.json'),'postreview_receipt_sha256':sha(root/'postreview-v1.json'),'native_exact_frame_binding':auth['qualified'],'native_fixture_sha256':sha(root/'references/native-sample.json')}
(out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'final_CPU_gate':'PASS','cohort_replies':368,'native_exact_frames':True,'jail_cases':len(qualified['results'])}))
