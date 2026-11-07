#!/usr/bin/env python3
"""Fresh namespace and recovery regression gate, guarded CPU execution only."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
root=Path('<rank1-work>/stdbench/phase2')
sys.path.insert(0,str(root))
import generate_envs
stage=root/'stage-v3'
index=generate_envs.generate(root/'references',stage,'stdbench-20261105-v2')
private=root/index['prefix']
private.mkdir(exist_ok=False)
shutil.copytree(stage/'rank1/configs',private/'configs')
shutil.copytree(stage/'boots',private/'boots')
subprocess.run([sys.executable,str(root/'rank_preflight.py'),'1',str(root/'rank-preflight-v2'),'stage-v3'],check=True)
subprocess.run([sys.executable,str(root/'cpu_final.py')],check=True)
print(json.dumps({'recovery_cpu_gate':'PASS','fresh_namespace':index['prefix'],'no_outcome_retries':True}))
