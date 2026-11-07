#!/usr/bin/env python3
"""Read-only bounded original-manifest discovery on rank0-host; no inference/generation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path('<rank0-work>')
NAMES={'tasks.jsonl','tasks-public.json','tasks-public.jsonl','prompts.jsonl',
       'ab3-units.jsonl','bench_prompts.json','t4-handoff-tasks.jsonl'}
SKIP={'raw','state','cache','.git','node_modules','executions','sessions','proof',
      'reports','out','inputs','receipts','request-evidence'}
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(exist_ok=False)
    roots=[x for x in ROOT.iterdir() if x.is_dir() and not x.is_symlink()
           and x.name.startswith(('imr-','implementer-b-','t6-census','stdbench'))]
    sources=[];rejected=[];records=[]
    for root in sorted(roots):
        for base,dirs,files in os.walk(root,followlinks=False):
            here=Path(base);depth=len(here.relative_to(root).parts)
            dirs[:]=sorted(d for d in dirs if d not in SKIP and not (here/d).is_symlink() and depth<5)
            for name in sorted(files):
                if name not in NAMES:continue
                path=here/name
                if path.is_symlink():raise ValueError('symlink source: '+str(path))
                data=path.read_bytes()
                if len(data)>25*1024*1024:raise ValueError('unexpectedly large original manifest')
                try:
                    rows=json.loads(data) if data.lstrip().startswith((b'[',b'{')) and path.suffix!='.jsonl' else [json.loads(l) for l in data.splitlines() if l.strip()]
                    if isinstance(rows,dict):
                        rows=rows.get('tasks',rows.get('prompts',rows))
                        if isinstance(rows,dict):rows=[dict(id=k,**v) if isinstance(v,dict) else {'id':k,'prompt':v} for k,v in rows.items()]
                    if not isinstance(rows,list) or not rows:raise ValueError('not a nonempty task array')
                    rows=[dict(r,id='engine-bench-'+r['name']) if isinstance(r,dict) and 'id' not in r and isinstance(r.get('name'),str) else r for r in rows]
                    if not all(isinstance(r,dict) and isinstance(r.get('id'),str) and isinstance(r.get('prompt'),str) for r in rows):raise ValueError('not ID/prompt originals')
                except (ValueError,TypeError) as e:
                    rejected.append({'path':str(path),'sha256':sha(data),'reason':str(e)});continue
                source={'path':str(path),'sha256':sha(data),'count':len(rows),'mtime_ns':path.stat().st_mtime_ns}
                sources.append(source)
                for row in rows:
                    record={k:row[k] for k in ('id','prompt','family','inputs','spec','set','suite') if k in row}
                    record.update(source_path=str(path),source_sha256=source['sha256'],source=str(path))
                    records.append(record)
    for name,value in [('sources.json',sources),('records.json',records),('rejected.json',rejected)]:
        with (a.out/name).open('x') as out:json.dump(value,out,indent=2)
    print(json.dumps({'source_files':len(sources),'original_records_before_dedup':len(records),'rejected_sources':len(rejected),'model_requests':0}))
if __name__=='__main__':main()
