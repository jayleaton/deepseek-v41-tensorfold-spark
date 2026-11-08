#!/usr/bin/env python3
"""I/O adapter for the unchanged sealed math3 merge; no labels are generated."""
import argparse
import json
from pathlib import Path
import math3

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prepared',type=Path,required=True)
parser.add_argument('--first',required=True)
parser.add_argument('--second',required=True)
parser.add_argument('--third')
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--audit',type=Path,required=True)
args=parser.parse_args()
receipt=json.loads((args.prepared/'seal-receipt.json').read_text())
packet=json.loads((args.prepared/'math-public.json').read_text())
private=json.loads((args.prepared/'math-private.json').read_text())
assert (args.prepared/'math-private.json').stat().st_mode & 0o777 == 0o600
assert not args.out.exists() and not args.audit.exists() and args.out.resolve()!=args.audit.resolve()
merged,audit=math3.merge(packet,private,math3.load_labels(args.first),math3.load_labels(args.second),
    'a1-math-independent-a','a1-math-independent-b',key_sha256=receipt['private_key_sha256'],
    third=math3.load_labels(args.third) if args.third else None,
    third_id='a1-math-independent-c' if args.third else None)
math3.write_private(args.audit,audit)
descriptor=args.out.open('x')
try:
    args.out.chmod(0o600)
    for row in merged:
        descriptor.write(math3.canonical(row)+'\n')
finally:
    descriptor.close()
print(json.dumps({'blinded_merge':'PASS','reviewed_answers':len(merged),
                  'disputed_candidates':len(audit['disputed_candidates']),
                  'third_reviewed':audit['third_reviewed']}))
