#!/usr/bin/env python3
"""Re-read/hash authoritative exclusions and export unresolved originals for review."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'compute-dl'
sys.path.insert(0,str(OLD))
from exclusion_sources import load_exclusions
from disjointness import normalize_prompt,source_review_inventory,prompt_sha256
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
encode=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--discovery',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    sources=json.loads((a.discovery/'sources.json').read_text())
    rejected=json.loads((a.discovery/'rejected.json').read_text())
    if rejected:raise ValueError('unresolved original source schema')
    for s in sources:
        if sha(s['path'])!=s['sha256']:raise ValueError('discovered source changed')
    paths={s['path'] for s in sources}
    for mandatory in ('<rank0-work>/imr-t4/measure/tasks.jsonl','<rank0-work>/t6-census-20261101/prompts.jsonl','<rank0-work>/imr-t5-w9/frozen/tasks-public.json','<rank0-work>/stdbench/compute-dl/dataset-v1/tasks-public.jsonl','<rank0-work>/imr-s2/turn/prompts.jsonl','<rank0-work>/imr-s1-guide/prompts.jsonl'):
        if mandatory not in paths:raise ValueError('missing earlier study '+mandatory)
    config=OLD/'exclusions-config.json'
    originals,legacy_audit=load_exclusions(json.loads(config.read_text()))
    discovered=json.loads((a.discovery/'records.json').read_text())
    combined=originals+discovered
    unique=[];dedup=[];byid={};seen={};aliases=[]
    for original in combined:
        if original['id'].startswith('compute-dl-v1-'):continue
        row=dict(original)
        evidence={k:row[k] for k in ('prompt','family','inputs','spec') if k in row}
        key=encode(evidence)
        if key in seen:
            dedup.append({'id':row['id'],'source_path':row['source_path'],'source_sha256':row['source_sha256'],'same_as':seen[key]});continue
        ident=row['id']
        if ident in byid:
            row['source_original_id']=ident
            row['id']='historical-'+hashlib.sha256((row['source_path']+'|'+ident+'|'+key).encode()).hexdigest()[:24]
            aliases.append({'original_id':ident,'variant_id':row['id'],'source_path':row['source_path'],'reason':'retain distinct original prompt/typed evidence, never drop a changed descendant'})
        seen[key]=row['id'];byid[row['id']]=row;unique.append(row)
    # Previously reviewed prompt-bound resolutions remain valid only on exact bytes.
    oldres=json.loads((OLD/'manual-resolutions-v1.json').read_text())
    resolutions={}
    for row in unique:
        original_id=row.get('source_original_id',row['id'])
        r=oldres.get(original_id)
        if r and r['prompt_sha256']==prompt_sha256(row['prompt']):
            ev=r['original_evidence']
            if sha(ev['path'])!=ev['sha256']:raise ValueError('old semantic evidence drift')
            resolutions[row['id']]=r
    review=[r for r in source_review_inventory(unique) if r['classification']=='unresolved-original' and r['id'] not in resolutions]
    a.out.mkdir(exist_ok=False)
    values={'historical-records.json':unique,'source-audit.json':{'legacy':legacy_audit,'discovered_sources':sources,'discovery_sources_sha256':sha(a.discovery/'sources.json'),'exclusion_config_sha256':sha(config),'deduplications':dedup,'retained_variants':aliases,'records':len(unique),'parent_source_sha256':sha(__file__)},'review-pending.json':review,'resolutions-inherited.json':resolutions}
    for name,value in values.items():
        with (a.out/name).open('x') as out:json.dump(value,out,indent=2)
    print(json.dumps({'unique_historical_records':len(unique),'deduplicated':len(dedup),'retained_variants':len(aliases),'unresolved_originals':len(review),'model_requests':0}))
if __name__=='__main__':main()
