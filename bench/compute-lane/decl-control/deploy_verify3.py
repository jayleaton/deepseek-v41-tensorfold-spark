#!/usr/bin/env python3
"""Hash-only verification of the CPU-sealed rank1-host deployment; no model calls."""
import hashlib
import json
from pathlib import Path
import sys

path=Path(sys.argv[1])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(path)==sys.argv[2]
receipt=json.loads(path.read_text())
assert receipt['passed']
for name,digest in receipt['code_sha256'].items(): assert sha(Path(name))==digest, 'sealed deployment drift'
print(json.dumps({'rank1_host_deployment':'PASS','files':len(receipt['code_sha256']),'GPU':False}))
