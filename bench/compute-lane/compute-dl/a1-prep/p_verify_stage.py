#!/usr/bin/env python3
"""Run only on a Spark in a bounded CPU scope."""

import sys as _archive_sys
import ast
import json
from pathlib import Path
import p_stage
import p_check

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)
HERE=Path(__file__).resolve().parent
for p in HERE.glob('*.py'): ast.parse(p.read_text(),filename=str(p))
fake=p_check.check()
index=p_stage.stage()
staged=p_check.check_stage(HERE/'staged-P')
(HERE/'p-check-receipt.json').write_text(json.dumps({'fake':fake,'staged':staged,
    'requests':index['requests'],'collection_authorized':False,'GPU_requests':0},indent=2))
print(json.dumps({'passed':True,'requests':index['requests'],'GPU_requests':0,'collection_authorized':False}))
