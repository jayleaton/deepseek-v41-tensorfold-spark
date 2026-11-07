#!/usr/bin/env python3
"""Final CPU gate and immutable collection freeze, only inside guarded Spark scope."""

import sys as _archive_sys
import hashlib
import json
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path('<rank1-work>/stdbench/phase2')
sys.path.insert(0, str(root))
sys.path.insert(0, str(root.parent/'kit'))
import analysis
import activation_report
import bench
import freeze
import grade_all
import math_review

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def call(*args):
    subprocess.run([sys.executable, *map(str,args)], check=True)


out = root/'cpu-final-v1'
out.mkdir(exist_ok=False)
with tempfile.TemporaryDirectory(prefix='stdbench-syntax-') as tmp:
    for path in sorted(root.glob('*.py')):
        py_compile.compile(str(path), cfile=str(Path(tmp)/(path.name+'c')), doraise=True)
checks = {'analysis':analysis.selftest(), 'grader':grade_all.selftest(), 'activation':activation_report.selftest()}
math_review.selftest()
call(root/'controller_selftest.py', out/'controller.json')
assert json.loads((out/'controller.json').read_text())['passed']
qualified = json.loads((root/'jail-qual-v1/qualification.json').read_text())
assert qualified['qualified']
for name, expected in qualified['sources_sha256'].items():
    assert digest(root/name) == expected, 'jail extraction/normalization drift after qualification'
for name in ('freeze.py','generate_envs.py','client.py'):
    assert digest(root/name) == json.loads((root/'cpu-stage2.json').read_text())['source_sha256'][name]
assert digest(root/'references/tokenizer.json') == digest('<home>/.cache/dsv41-flash17-k6-mtp/model/tokenizer.json')
inputs = {suite:root.parent/(suite+'-selected.jsonl') for suite in ('gsm8k','math500','humaneval')}
all_tasks = [row for path in inputs.values() for row in bench.read_jsonl(path)]
preseal = math_review.preseal(all_tasks, root/'MATH-PRESEAL-v1.json')
settings = {'model':'DeepSeek-V4.1-Flash-TF','reasoning_effort':'high','temperature':1.0,
            'top_p':.95,'top_k':20,'max_tokens':8192,'deadline_s':270,'request_timeout':360}
code = {str(path):digest(path) for path in sorted(root.glob('*.py'))}
code.update({str(path):digest(path) for path in sorted(root.glob('*.sh'))})
config = {'spec':str(root/'T5-STDBENCH-SPEC.md'),
          'spec_sha':'9c7327007cfd15825a608f2f6b06f2b39e33b88548c5bb7deb1b209b7395d240',
          'kit':str(root.parent/'kit'),'lock':str(root.parent/'kit/datasets.lock.json'),
          'lock_sha':digest(root.parent/'kit/datasets.lock.json'),
          'inputs':{suite:{'path':str(path),'sha256':digest(path)} for suite,path in inputs.items()},
          'counts':{'gsm8k':48,'math500':20,'humaneval':24},'settings':settings,'code_sha_files':code}
manifest = freeze.freeze(config, root/'frozen-v1')
assert sum(x['pairs_per_arm'] for x in manifest['suites'].values()) == 184
# Refuse any clash with a planned owned container before the pilot is queued.
index = json.loads((root/'stage-v2/manifest.json').read_text())
for boot in index['boots']:
    record = json.loads((root/'stage-v2'/boot['manifest']).read_text())
    name = record['phase2']['ranks'][1]['container_name']
    result = subprocess.run(['docker','container','inspect',name],capture_output=True,text=True)
    assert result.returncode != 0 and 'No such' in result.stderr, 'planned rank1-host boot name clash or daemon error'
receipt = {'passed':True,'spec_sha256':config['spec_sha'],'checks':checks,
           'cohort_replies':368,'math_preseal':preseal,'math_preseal_file_sha256':digest(root/'MATH-PRESEAL-v1.json'),
           'code_sha256':code,'freeze_sha256':digest(root/'frozen-v1/manifest.json'),
           'tokenizer_sha256':digest(root/'references/tokenizer.json'), 'jail_cases':len(qualified['results']),
           'rank1_host_boot_names_absent':True}
(out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'final_CPU_gate':'PASS','cohort_replies':368,'spec_sha256':config['spec_sha'],'jail_cases':len(qualified['results'])}))
