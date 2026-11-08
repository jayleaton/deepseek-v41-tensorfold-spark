#!/usr/bin/env python3
"""Serial rank dry runs only, in the caller's bounded CPU scope."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from runtime_stage import PREFIX,SPEC_SHA256
HERE=Path(__file__).resolve().parent
ROOT=HERE/PREFIX
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('rank',type=int,choices=(0,1));a=p.parse_args()
    campaign=ROOT/'campaign-v1.json'
    if not campaign.exists():
        with campaign.open('xb') as out:out.write((ROOT/'stage-campaign.json').read_bytes())
    entries=[]
    for boot in ('t01-D','t01-L','t02-L','t02-D'):
        manifest=ROOT/f'rank{a.rank}/boots/{boot}/manifest.json'
        log=ROOT/f'cpu-verify-{boot}-r{a.rank}.log'
        result=subprocess.run([sys.executable,'-B',str(HERE/'rank_ops.py'),'verify',str(manifest),str(a.rank),'--dry-run'],
                              capture_output=True,text=True,timeout=7200)
        with log.open('x') as output:output.write(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('rank dry run failed; inspect '+str(log))
        event=json.loads(result.stdout.strip().splitlines()[-1])
        proof=Path(event['proof']);data=json.loads(proof.read_text())
        if data['command']!='verify' or data['dry_run'] is not True or data['rank']!=a.rank or data['boot']!=boot:
            raise ValueError('rank verification receipt differs')
        entries.append({'boot':boot,'rank':a.rank,'proof':str(proof),'sha256':sha(proof),
                        'manifest_sha256':sha(manifest),'log_sha256':sha(log)})
    receipt={'passed':True,'rank':a.rank,'spec_sha256':SPEC_SHA256,'checks':entries,
             'source_sha256':sha(HERE/'rank_ops.py'),'runner_sha256':sha(__file__)}
    with (ROOT/f'cpu-rank-{a.rank}.json').open('x') as output:json.dump(receipt,output,indent=2)
    print(json.dumps({'rank':a.rank,'dry_runs':len(entries),'passed':True}))

if __name__=='__main__':main()
