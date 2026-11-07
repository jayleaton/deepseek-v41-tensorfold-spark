#!/usr/bin/env python3
"""Run only on a Spark in a bounded CPU scope."""
import ast
import json
from pathlib import Path
import p_stage
import p_check
HERE=Path(__file__).resolve().parent
for p in HERE.glob('*.py'): ast.parse(p.read_text(),filename=str(p))
fake=p_check.check()
index=p_stage.stage()
staged=p_check.check_stage(HERE/'staged-P')
(HERE/'p-check-receipt.json').write_text(json.dumps({'fake':fake,'staged':staged,
    'requests':index['requests'],'collection_authorized':False,'GPU_requests':0},indent=2))
print(json.dumps({'passed':True,'requests':index['requests'],'GPU_requests':0,'collection_authorized':False}))
