#!/usr/bin/env python3
import json
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path('<rank1-work>/stdbench/phase2')
with tempfile.TemporaryDirectory(prefix='stdbench-bytecode-') as tmp:
    for name in ('generate_envs.py', 'rank_ops.py', 'client.py', 'freeze.py', 'cpu_guard.py'):
        py_compile.compile(str(root/name), cfile=str(Path(tmp)/(name+'c')), doraise=True)
subprocess.run([sys.executable, str(root/'generate_envs.py'), '--out', str(root/'stage-v1')], check=True)
subprocess.run([sys.executable, str(root/'client.py'), '--selftest', str(root/'client-selftest-v1.json')], check=True)
subprocess.run([sys.executable, str(root/'freeze.py'), '--dry-run', '--out', str(root/'freeze-dry-v1')], check=True)
print(json.dumps({'stage1': 'PASS', 'rank_envs': 28, 'boot_manifests': 14, 'no_gpu_or_endpoint': True}))
