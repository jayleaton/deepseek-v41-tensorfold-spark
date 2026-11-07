#!/usr/bin/env python3
"""Stage sealed requests and rank configs on CPU; no queue or endpoint."""

import sys as _archive_sys
import json
from pathlib import Path
import shutil
import subprocess
import sys
import generate_envs3
from freeze3 import freeze

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

root = Path(__file__).resolve().parent
subprocess.run([sys.executable,'-B',str(root/'rank_check3.py')],check=True,timeout=7200)
out = root/generate_envs3.PREFIX
index = generate_envs3.generate(root.parent/'phase2/references',out)
freeze(root/'frozen-a1-v1')
shutil.copytree(out/'rank1/configs',out/'configs')
print(json.dumps({'CPU_stage':'PASS','boots':len(index['boots']),'envs':18,'GPU':False}))
