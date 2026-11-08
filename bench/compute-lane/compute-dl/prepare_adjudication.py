#!/usr/bin/env python3
"""Inventory uncollected ORIGINAL requests; never generate tasks or collect replies."""
import hashlib
import json
from pathlib import Path

# Export adapter: help must not execute this historical host script.
import sys as _archive_sys
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__)
    raise SystemExit(0)
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
report=load(HERE/'report-v1/report.json')
cfg=load(HERE/'stdbench-compute-dl-20261105-v1/campaign-v1.json')
missing={(r['id'],r['seed'],r['arm']) for r in report['planned_observations'] if r['classification']=='missing'}
remaining=[]
for entry in cfg['layout']['boots']:
    path=Path(cfg['freeze'])/entry['file']
    if sha(path)!=entry['sha256']:raise ValueError('original request freeze drift')
    for row in load(path)['requests']:
        if (row['id'],row['seed'],entry['arm']) in missing:
            remaining.append({'id':row['id'],'seed':row['seed'],'arm':entry['arm'],
                'original_boot':entry['boot'],'key':row['key'],'wire_sha256':row['wire_sha256'],
                'task_sha256':row['task_sha256'],'original_boot_file':str(path),'original_boot_sha256':entry['sha256']})
counts={arm:sum(r['arm']==arm for r in remaining) for arm in ('D','L')}
if len(remaining)!=len(missing)!=138:raise ValueError('remaining identity census differs')
if len(remaining)!=138 or counts!={'D':120,'L':18}:raise ValueError('actual stop inventory differs')
result={'spec_sha256':cfg['spec_sha256'],'frozen_report_sha256':sha(HERE/'report-v1/report.json'),
    'source_sha256':sha(__file__),'collection_authorized':False,'existing_observations_may_be_replaced':False,
    'count':len(remaining),'counts':counts,'remaining_original_requests':remaining}
with (HERE/'remaining-collection.json').open('x') as output:json.dump(result,output,indent=2)
print(json.dumps({'remaining':len(remaining),'counts':counts,'collection_authorized':False,
                  'manifest_sha256':sha(HERE/'remaining-collection.json')}))
