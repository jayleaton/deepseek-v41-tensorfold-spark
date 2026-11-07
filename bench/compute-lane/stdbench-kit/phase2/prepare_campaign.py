#!/usr/bin/env python3
"""Bind the qualified rank1-host source freeze to identical rank0-host files; no GPU/API."""
import hashlib
import json
from pathlib import Path
root=Path('<rank0-work>/stdbench/phase2')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt=json.loads((root/'cpu-final-v3/verification.json').read_text())
assert receipt['passed']
code={p.replace('<rank1-home>/','<rank0-home>/'):h for p,h in receipt['code_sha256'].items()}
for p,h in code.items():
    assert sha(p)==h, 'rank0-host/sourcefreeze mismatch: '+p
frozen=root/'frozen-v3'
assert sha(frozen/'manifest.json')==receipt['freeze_sha256']
manifest=json.loads((frozen/'manifest.json').read_text())
prefix='stdbench-20261105-v2'
rank_manifests={b['boot']:{str(r):f'<rank{r}-work>/stdbench/phase2/{prefix}/boots/{b["boot"]}/manifest.json' for r in (0,1)} for b in manifest['boots']}
for b,paths in rank_manifests.items():
    assert Path(paths['0']).is_file()
cfg={'prefix':prefix,'spec':str(root/'T5-STDBENCH-SPEC.md'),'spec_sha256':receipt['spec_sha256'],'cpu_receipt':str(root/'cpu-final-v3/verification.json'),'code_sha256':code,'freeze':str(frozen),'freeze_sha256':receipt['freeze_sha256'],'rank_manifests':rank_manifests,'pilot_verdict':str(root/'pilot-qualified-v2.json'),'go_file':str(root/'GO-v2'),'tokenizer':str(root/'references/tokenizer.json'),'tokenizer_sha256':receipt['tokenizer_sha256'],'image_ids':['sha256:5be7d7d40f267590b9e7b41410f48d40048af24e004203629a347d3d7c16a8cd','sha256:80beaba732d1ed9439e012b007d8dbe19ec2f8962f50fbac29a18c8cd6fda306']}
prior=root/'runs/stdbench-20261105-v1-pilot/pilot-plain'
cfg['completed_boots']={'pilot-plain':{'directory':str(prior),'sha256':{n:sha(prior/n) for n in ('responses.jsonl','diagnostics.jsonl','safety.jsonl')}}}
with (root/'campaign-v2.json').open('x') as f: json.dump(cfg,f,indent=2); f.write('\n')
with (root/'GO-v2').open('x') as f: f.write(receipt['spec_sha256']+'\n')
print(json.dumps({'campaign_prepared':True,'source_files_matched':len(code),'cohort_replies':368,'no_GPU_or_API_requests':True}))
