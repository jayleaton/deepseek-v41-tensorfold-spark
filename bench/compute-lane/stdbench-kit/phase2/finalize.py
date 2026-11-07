#!/usr/bin/env python3
"""Merge sealed judgments and run the unchanged registered paired analysis."""
import hashlib
import json
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path
import analysis
import math_review
root=Path(__file__).resolve().parent
out=root/'post-v1'
load=lambda p: json.loads(Path(p).read_text())
rows=lambda p: [json.loads(x) for x in Path(p).read_text().splitlines()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x') as f: json.dump(v,f,indent=2); f.write('\n')
def save_rows(p,values):
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:
        for r in values:f.write(analysis.canonical(r)+'\n')
with tempfile.TemporaryDirectory(prefix='stdbench-final-syntax-') as temp:
    for p in sorted(root.glob('*.py')): py_compile.compile(str(p),cfile=str(Path(temp)/(p.name+'c')),doraise=True)
receipt=load(root/'cpu-final-v3/verification.json')
for name in ('analysis.py','math_review.py','grade_all.py','activation_report.py','jail.py','jail_worker.py'):
    assert sha(root/name)==receipt['code_sha256']['<rank1-work>/stdbench/phase2/'+name]
subprocess.run([sys.executable,str(root/'audit_envs.py')],check=True)
for args in (["--selftest","--out",str(out/'audit-selftest.json')],["--data-root",str(root/'post-data-v1'),"--freeze",str(root/'frozen-v3'),"--tokenizer",str(root/'references/tokenizer.json'),"--tokenizer-sha",receipt['tokenizer_sha256'],"--out",str(out/'native-audit.json')]):
    subprocess.run([sys.executable,str(root/'audit_records.py'),*args],check=False)
assert load(out/'audit-selftest.json').get('passed') is True
sealed=load(out/'seal-receipt.json')
assert sealed['packet_sha256']==load(root/'math-review-v1/seal-receipt.json')['packet_sha256']
packet=load(out/'math-public.json')
private=load(out/'math-private.json')
first=math_review.load_labels(root/'math-review-v1/reviewer-a.json')
second=math_review.load_labels(root/'math-review-v1/reviewer-b.json')
reviewed,audit=math_review.merge(packet,private,first,second,'reviewer-A','reviewer-B',key_sha256=sealed['private_key_sha256'])
assert not audit['disputed_candidates']
math_review.write_private(out/'math-review-audit.json',audit)
save_rows(out/'math-reviewed.jsonl',reviewed)
allrows=rows(out/'nonmath-graded.jsonl')+reviewed
report=analysis.analyze(allrows,load(root/'frozen-v3/analysis-plain.json'))
save(out/'analysis.json',report)
# No model text or native IDs in terminal summary.
print(json.dumps({'analysis':'COMPLETE','dataset_rows':len(allrows),'math_unresolved':audit['bounds']['unresolved'],'suites':{s:{'plain_solved':v['plain']['solved'],'lane_solved':v['lane']['solved'],'per_arm':v['scheduled_per_arm'],'tokens_ratio':v['tokens']['paired_sums_ratio_lane_over_plain'],'wall_ratio':v['time']['paired_sums_ratio_lane_over_plain'],'holm_p':v['sign_flip']['holm_adjusted_p']} for s,v in report['suites'].items()},'native_audit_qualified':load(out/'native-audit.json').get('qualified'),'environment_audit_qualified':load(out/'environment-audit.json')['qualified']}))
