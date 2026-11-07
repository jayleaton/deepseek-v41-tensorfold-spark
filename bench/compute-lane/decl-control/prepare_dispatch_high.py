#!/usr/bin/env python3
"""Stage a new high-thinking fixture without changing the completed first proof."""
import json
from pathlib import Path
import shutil
import subprocess

root=Path(__file__).resolve().parent
old=root/'dispatch'; new=root/'dispatch-high'
assert not (new/'unmodified-source-sha256.json').exists()
for folder in ('src','build'):
    shutil.copytree(old/'zig'/folder,new/'zig'/folder,dirs_exist_ok=True,
                    ignore=lambda path,names:[n for n in names if n in ('decl_control_probe.zig','decl_control_tools.json')])
shutil.copy2(old/'unmodified-source-sha256.json',new/'unmodified-source-sha256.json')
subprocess.run(['scp','-r',str(new),'root@<rank1-link-ip>:<rank1-work>/stdbench/decl-control/'],check=True,timeout=7200)
print(json.dumps({'high_fixture_stage':'PASS','engine_edits':False}))
