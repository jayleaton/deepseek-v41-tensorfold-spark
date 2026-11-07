#!/usr/bin/env python3
"""Stage a new high-thinking fixture without changing the completed first proof."""

import sys as _archive_sys
import json
from pathlib import Path
import shutil
import subprocess

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

root=Path(__file__).resolve().parent
old=root/'dispatch'; new=root/'dispatch-high'
assert not (new/'unmodified-source-sha256.json').exists()
for folder in ('src','build'):
    shutil.copytree(old/'zig'/folder,new/'zig'/folder,dirs_exist_ok=True,
                    ignore=lambda path,names:[n for n in names if n in ('decl_control_probe.zig','decl_control_tools.json')])
shutil.copy2(old/'unmodified-source-sha256.json',new/'unmodified-source-sha256.json')
subprocess.run(['scp','-r',str(new),'root@<rank1-link-ip>:<rank1-work>/stdbench/decl-control/'],check=True,timeout=7200)
print(json.dumps({'high_fixture_stage':'PASS','engine_edits':False}))
