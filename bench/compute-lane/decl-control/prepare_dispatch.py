#!/usr/bin/env python3
"""Copy read-only pinned CPU source into a separate host-harness tree."""

import sys as _archive_sys
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

root = Path(__file__).resolve().parent
source = Path('<rank0-work>/imr-te/build-3a429d6f122d')
out = root/'dispatch'
out.mkdir(exist_ok=True)
assert not (out/'zig').exists()
(out/'zig').mkdir()
for name in ('src', 'build'):
    shutil.copytree(source/'zig'/name, out/'zig'/name)
hashes = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted((out/'zig').rglob('*')) if p.is_file()}
with (out/'unmodified-source-sha256.json').open('x') as f:
    json.dump(hashes, f, indent=2); f.write('\n')
subprocess.run(['scp', '-r', str(out), 'root@<rank1-link-ip>:<rank1-work>/stdbench/decl-control/'], check=True, timeout=7200)
print(json.dumps({'source_copy':'PASS', 'files':len(hashes), 'engine_edited':False}))
