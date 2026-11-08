#!/usr/bin/env python3
"""Hash-only verification of the CPU-sealed rank1-host deployment; no model calls."""

import sys as _archive_sys
import hashlib
import json
from pathlib import Path
import sys

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

path=Path(sys.argv[1])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(path)==sys.argv[2]
receipt=json.loads(path.read_text())
assert receipt['passed']
for name,digest in receipt['code_sha256'].items(): assert sha(Path(name))==digest, 'sealed deployment drift'
print(json.dumps({'rank1_host_deployment':'PASS','files':len(receipt['code_sha256']),'GPU':False}))
