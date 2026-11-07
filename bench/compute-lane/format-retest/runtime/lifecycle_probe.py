#!/usr/bin/env python3
"""Run inside the real bounded CPU guard; captures exact owned process identities."""
import json
import os
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
mode=sys.argv[1];assert mode in ('success','failure')
unit=os.environ.get('STDBENCH_CPU_JOB');assert unit and unit.startswith('stdbench-cpu-')
pid=os.getpid();wrapper=None
while pid>1:
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if '/phase2/cpu-rank0-host.sh' in cmd:
        wrapper=pid;break
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();pid=int(fields[1])
assert wrapper,'real admission wrapper not found'
children=[int(x) for x in Path(f'/proc/{wrapper}/task/{wrapper}/children').read_text().split()]
result={'mode':mode,'scope':unit,'wrapper_pid':wrapper,'owned_pids':[wrapper,*children],
        'expected_exit':0 if mode=='success' else 73,'guarded':True,'GPU_requests':0}
with (HERE/f'lifecycle-{mode}.json').open('x') as output:json.dump(result,output,indent=2)
print(json.dumps({'mode':mode,'guarded':True,'expected_exit':result['expected_exit']}),flush=True)
raise SystemExit(result['expected_exit'])
