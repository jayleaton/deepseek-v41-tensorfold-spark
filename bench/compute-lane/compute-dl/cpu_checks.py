#!/usr/bin/env python3
"""Small serial precollection tests; every result is retained, never GPU authority."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import re

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('name',choices=('generation','outcome','disjoint','lifecycle','budget','controller','runtime','report'))
p.add_argument('--version',default='v1')
args=p.parse_args()
if not re.fullmatch(r'v[0-9]+',args.version): raise ValueError('invalid receipt version')
names={'generation':'generation_check.py','outcome':'outcome_check.py','disjoint':'disjoint_check.py',
       'lifecycle':'lifecycle_check.py','budget':'budget_check.py','controller':'controller_safety_check.py',
       'runtime':'runtime_check.py','report':'report_check.py'}
source=names[args.name]
hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in root.glob('*.py')}
for path in root.glob('*.py'): ast.parse(path.read_bytes(),filename=str(path))
run=subprocess.run([sys.executable,'-B',str(root/source)],capture_output=True,text=True,timeout=7200)
receipt={'name':args.name,'source_sha256':hashes,'returncode':run.returncode,
         'stdout':run.stdout,'stderr':run.stderr,'GPU_authorized':False}
with (root/('cpu-'+args.name+'-'+args.version+'.json')).open('x') as output:
    json.dump(receipt,output,indent=2);output.write('\n')
print(json.dumps({'check':args.name,'passed':run.returncode==0,'stdout_tail':run.stdout[-2000:],
                  'stderr_tail':run.stderr[-1500:]}))
if run.returncode: raise RuntimeError('retained failing CPU check: '+args.name)
