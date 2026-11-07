#!/usr/bin/env python3
"""Seal the complete MATH cohort using the frozen mapping; no grading/inference."""
import hashlib
import json
from pathlib import Path
import math_review
root=Path(__file__).resolve().parent
receipt=json.loads((root/'cpu-final-v3/verification.json').read_text())
for name in ('math_review.py','analysis.py'):
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==receipt['code_sha256']['<rank1-work>/stdbench/phase2/'+name]
data=root/'math-data-v1'
paths=sorted(data.glob('*/responses.jsonl'))
assert len(paths)==6
rows=[json.loads(line) for p in paths for line in p.read_text().splitlines() if json.loads(line)['suite']=='math500']
assert len(rows)==80
out=root/'math-review-v1'
out.mkdir(exist_ok=False)
tasks=[json.loads(line) for line in (root.parent/'math500-selected.jsonl').read_text().splitlines()]
sealed=math_review.seal(tasks,rows,out/'public.json',out/'private.json',preseal_path=root/'MATH-PRESEAL-v1.json',preseal_sha256=receipt['math_preseal']['preseal_sha256'])
(out/'seal-receipt.json').write_text(json.dumps(sealed,indent=2)+'\n')
(out/'inputs.json').write_text(json.dumps({'files_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
print(json.dumps({'MATH_candidates_sealed':80,'public_packet_sha256':sealed['packet_sha256']}))
