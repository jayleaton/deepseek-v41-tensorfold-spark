#!/usr/bin/env python3
"""Validate sealed independent labels and emit only disputed public candidates."""

import sys as _archive_sys
import hashlib
import json
from pathlib import Path
import math_review

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)
root=Path(__file__).resolve().parent
review=root/'math-review-v1'
receipt=json.loads((root/'cpu-final-v3/verification.json').read_text())
assert hashlib.sha256((root/'math_review.py').read_bytes()).hexdigest()==receipt['code_sha256']['<rank1-work>/stdbench/phase2/math_review.py']
for line in (review/'labels-premerge.sha256').read_text().splitlines():
    digest,name=line.split(maxsplit=1)
    assert hashlib.sha256((review/Path(name).name).read_bytes()).hexdigest()==digest
packet=json.loads((review/'public.json').read_text())
sealed=json.loads((review/'seal-receipt.json').read_text())
assert math_review.digest(packet)==sealed['packet_sha256']
allowed={r['candidate_id'] for r in packet}
a=math_review.validate_labels(math_review.load_labels(review/'reviewer-a.json'),'reviewer-A',allowed,require_all=True)
b=math_review.validate_labels(math_review.load_labels(review/'reviewer-b.json'),'reviewer-B',allowed,require_all=True)
disputes={k for k in allowed if a[k]['label']!=b[k]['label']}
with (review/'disputed-public.json').open('x') as f: json.dump([r for r in packet if r['candidate_id'] in disputes],f,ensure_ascii=False,indent=2);f.write('\n')
proof={'passed':True,'candidates':80,'initial_reviewers':2,'disputed':len(disputes),'public_packet_sha256':sealed['packet_sha256'],'first_sha256':hashlib.sha256((review/'reviewer-a.json').read_bytes()).hexdigest(),'second_sha256':hashlib.sha256((review/'reviewer-b.json').read_bytes()).hexdigest()}
with (review/'review-validation.json').open('x') as f: json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(proof))
