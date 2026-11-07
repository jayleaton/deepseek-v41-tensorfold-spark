#!/usr/bin/env python3
"""Bind the completed rank1-host A1 CPU seal to identical rank0-host artifacts."""

import sys as _archive_sys
import hashlib
import json
from pathlib import Path
import sys
from generate_envs3 import PREFIX, IMAGE_IDS

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt_path=root/sys.argv[1]/'verification.json'
receipt=json.loads(receipt_path.read_text())
assert receipt['passed'] and receipt['fresh_replies']==552
code={p.replace('<rank1-home>/','<rank0-home>/',1):h for p,h in receipt['code_sha256'].items()}
for p,digest in code.items(): assert sha(p)==digest, 'rank0-host/source freeze mismatch: '+p
frozen=root/'frozen-a1-v1'
assert sha(frozen/'manifest.json')==receipt['freeze_sha256']
manifest=json.loads((frozen/'manifest.json').read_text())
ranks={b['boot']:{str(r):f'<rank{r}-work>/stdbench/decl-control/{PREFIX}/boots/{b["boot"]}/manifest.json'
                 for r in (0,1)} for b in manifest['boots']}
assert sha(root/'prompt-arrays.jsonl')==receipt['cpu_prompt_arrays_sha256']
cfg={'prefix':PREFIX,'base_spec':str(root/'T5-DECL-CONTROL-SPEC.md'),
     'spec':str(root/'T5-DECL-CONTROL-SPEC-A1.md'),'spec_sha256':receipt['spec_sha256'],
     'cpu_receipt':str(receipt_path),'cpu_receipt_sha256':sha(receipt_path),
     'code_sha256':code,'freeze':str(frozen),'freeze_sha256':receipt['freeze_sha256'],
     'rank_manifests':ranks,'go_file':str(root/'GO-A1-v1'),
     'cpu_prompt_arrays':str(root/'prompt-arrays.jsonl'),
     'cpu_prompt_arrays_sha256':receipt['cpu_prompt_arrays_sha256'],
     'tokenizer':str(root.parent/'phase2/references/tokenizer.json'),
     'tokenizer_sha256':'c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b',
     'image_ids':IMAGE_IDS,'max_turns':4,'scheduled_turns':['t01','t02','t03'],
     'reserve_selection':'reserve3.select_missing: presence only; no observed-outcome replacement'}
with (root/'campaign-a1-v1.json').open('x') as f:json.dump(cfg,f,indent=2);f.write('\n')
with (root/'GO-A1-v1').open('x') as f:f.write(receipt['spec_sha256']+'\n')
print(json.dumps({'campaign_binding':'PASS','source_files_matched':len(code),'fresh_replies':552,'GPU':False}))
