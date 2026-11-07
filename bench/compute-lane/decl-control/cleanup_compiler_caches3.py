#!/usr/bin/env python3
"""After all jobs settle, remove exact owned compiler caches, retaining proofs."""

import sys as _archive_sys
import json
from pathlib import Path
import shutil

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

root=Path(__file__).resolve().parent
assert not Path('<rank0-work>/queue/QUIET').exists() if str(root).startswith('<rank0-home>/') else True
removed=[]
for fixture in ('dispatch','dispatch-high'):
    base=root/fixture
    assert json.loads((base/'dispatch-proof.json').read_text())['verdict']=='PASS'
    for relative in ('.zig-cache','.zig-global-cache','guard-mutant/.zig-cache','guard-mutant/.zig-global-cache'):
        path=base/relative
        assert not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path.relative_to(root)))
with (root/'compiler-cache-cleanup.json').open('x') as f:
    json.dump({'qualified_proofs_retained':True,'removed_exact_owned_directories':removed},f,indent=2);f.write('\n')
print(json.dumps({'compiler_caches_cleaned':len(removed),'proofs_retained':True}))
