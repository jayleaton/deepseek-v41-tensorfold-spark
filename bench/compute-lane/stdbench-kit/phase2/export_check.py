#!/usr/bin/env python3
"""Final standalone kit checks; constrained Spark CPU scope only."""

import sys as _archive_sys
import hashlib
import json
import py_compile
import subprocess
import tempfile
from pathlib import Path

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)
root=Path(__file__).resolve().parent
sources=sorted(root.glob('*.py'))+sorted(root.glob('*.sh'))
with tempfile.TemporaryDirectory(prefix='stdbench-export-check-') as temp:
    for path in sources:
        if path.suffix=='.py':py_compile.compile(str(path),cfile=str(Path(temp)/(path.name+'c')),doraise=True)
        else:subprocess.run(['bash','-n',str(path)],check=True,timeout=7200)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
validation=json.loads((root/'post-v1/report-validation.json').read_text())
assert sha(root/'post-v1/RESULTS.md')==validation['results_sha256']
assert json.loads((root/'post-v1/native-audit.json').read_text())['qualified'] is True
assert json.loads((root/'post-v1/environment-audit.json').read_text())['qualified'] is True
assert json.loads((root/'post-v1/request-order-audit.json').read_text())['qualified'] is True
proof={'passed':True,'python_files':sum(p.suffix=='.py' for p in sources),'shell_files':sum(p.suffix=='.sh' for p in sources),'source_sha256':{p.name:sha(p) for p in sources},'report_sha256':validation['results_sha256'],'temporary_bytecode_cleaned':True}
with (root/'post-v1/export-check.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps({'final_kit_checks':'PASS','python_files':proof['python_files'],'shell_files':proof['shell_files'],'report_hash_matches':True}))
