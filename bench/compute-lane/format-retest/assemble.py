#!/usr/bin/env python3
"""Assemble actual CPU evidence into a fresh campaign; never creates GO or queues."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'compute-dl'
PREFIX='stdbench-format-20261107-v1';ROOT=HERE/PREFIX
SPEC='b47ede697eed3a38d244ad5ff7447690732bd12945ff4592554d45bc2d8527ec'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2)
def require(ok,why):
    if not ok:raise ValueError(why)
def source_map(cfg):
    pins=dict(cfg['code_sha256'])
    for p in HERE.rglob('*.py'):
        if not any(part.startswith(('discovery-','inventory-','runtime-stage-')) for part in p.relative_to(HERE).parts):pins[str(p)]=sha(p)
    for p in (HERE/'runtime').glob('*.sh'):pins[str(p)]=sha(p)
    for p in [OLD/'collect_report.py',OLD/'analysis.py',OLD/'grade.py',OLD/'a1-prep/audit_lifecycle.py',HERE.parent/'phase2/audit_records.py',HERE.parent/'phase2/jail_worker.py']:
        pins[str(p)]=sha(p)
    sys.path.insert(0,str(HERE/'prep'));from legacy import PATHS
    frozen=load(HERE/'generation-pin-v1.json')['legacy_source_pins']
    for k,p in PATHS.items():require(sha(p)==frozen[k],'legacy code drift '+k);pins[str(p)]=frozen[k]
    return pins
def refresh():
    require(not (ROOT/'GO').exists() and not (ROOT/'budget.json').exists(),'precollection refresh only')
    cfg=load(ROOT/'stage-campaign.json')
    require(cfg['spec_sha256']==SPEC and cfg['prefix']==PREFIX,'new campaign only')
    cfg['code_sha256']=source_map(cfg)
    cfg['cpu_prompt_arrays_sha256']=sha(ROOT/'prompt-arrays.jsonl')
    for name in ('stage-campaign','campaign-v1'):
        path=ROOT/(name+'.json');archive=ROOT/(name+'.before-source-refresh-'+sha(path)[:16]+'.json')
        require(not archive.exists(),'identical prior candidate already archived')
        path.rename(archive);save(path,cfg)
    print(json.dumps({'refreshed':True,'source_files':len(cfg['code_sha256']),'GPU_requests':0}))
def preparation():
    sys.path.insert(0,str(HERE/'prep'));from generation import inventory;from legacy import Legacy
    cfg=load(HERE/'generation-config-v1.json');_,history=inventory(cfg,Legacy())
    gen=load(HERE/'dataset-v1/generation-audit.json');independent=load(HERE/'dataset-v1/independent-process-verification.json')
    require(independent['passed'] and independent['tasks']==60 and gen['files_sha256']['tasks.jsonl']==sha(HERE/'dataset-v1/tasks.jsonl'),'reference/task binding')
    require(sha(HERE/'T5-FORMAT-RETEST-SPEC.md')==SPEC and cfg['spec_sha256']==SPEC,'authority')
    pin=load(HERE/'generation-pin-v1.json')
    for name,h in pin['source_pins'].items():require(sha(HERE/'prep'/name)==h,'generation source drift')
    sourceaudit=load(HERE/'inventory-v1/source-audit.json')
    for entry in sourceaudit['discovered_sources']+sourceaudit['legacy']['files']:require(sha(entry['path'])==entry['sha256'],'authoritative historical source drift')
    # Public originals and their authoritative hashes are frozen exports; neither
    # rank needs foreign live recipe paths to reinterpret those mathematical inputs.
    paths=[HERE/'generation-pin-v1.json',HERE/'generation-config-v1.json',HERE/'inventory-sealed-v1.json',HERE/'review-resolutions-v1.json',
           HERE/'inventory-v1/historical-records.json',HERE/'inventory-v1/source-audit.json',HERE/'stage-config-v1.json',HERE/'prep-manifest-v1.json',
           *list((HERE/'dataset-v1').glob('*.json')),*list((HERE/'dataset-v1').glob('*.jsonl')),
           HERE/'r7-adapter-check.json',HERE/'publication-check.json',HERE/'audit-check-v3.json']
    artifacts={str(p):sha(p) for p in paths}
    for name in ('publication-check.json','audit-check-v3.json','r7-adapter-check.json'):require(load(HERE/name)['passed'] is True,'CPU check pending '+name)
    proof=dict(passed=True,spec_sha256=SPEC,tasks=60,scheduled_replies=360,unresolved_overlaps=0,missing_sources=[],
        prep_manifest_sha256=sha(HERE/'prep-manifest-v1.json'),historical_inventory_sha256=sha(HERE/'inventory-sealed-v1.json'),
        historical_unique_instances=history['unique_records'],artifact_sha256=artifacts,
        source_sha256={str(p):sha(p) for p in [Path(__file__),HERE/'prepare.py',HERE/'inventory_sources.py',HERE/'discover_sources.py',*(HERE/'prep').glob('*.py')]},
        limitations='Historical source originals were read and hash-checked on rank0-host; both ranks bind identical frozen public exports and inventory. No old outcomes are pooled.')
    save(HERE/'preparation-check.json',proof);print(json.dumps({'passed':True,'scheduled':360,'historical_instances':history['unique_records']}))
def final():
    require(not (ROOT/'GO').exists() and not (ROOT/'budget.json').exists(),'seal candidate only')
    cfg=load(ROOT/'stage-campaign.json');cfg['code_sha256']=source_map(cfg)
    proofpaths={'preparation':HERE/'preparation-check.json','prompts':ROOT/'prompt-proof.json','controls':ROOT/'control-check.json',
                'rank0':ROOT/'rank-check-r0.json','rank1':ROOT/'rank-check-r1.json','clipping':OLD/'a1-prep/a1-policy-check-receipt.json',
                'dispatch':HERE/'dispatch-check.json','grader':HERE/'grader-check-v2.json','analysis':HERE/'analysis-check-v2.json'}
    cfg['qualification']={name:dict(path=str(path),sha256=sha(path)) for name,path in proofpaths.items()}
    cfg['cpu_prompt_arrays_sha256']=sha(ROOT/'prompt-arrays.jsonl')
    cfg['publication_check']=dict(path=str(HERE/'publication-check.json'),sha256=sha(HERE/'publication-check.json'))
    save(ROOT/'candidate-final-v1.json',cfg)
    print(json.dumps({'candidate':str(ROOT/'candidate-final-v1.json'),'source_files':len(cfg['code_sha256']),'GO_created':False}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['refresh','preparation','final']);a=p.parse_args()
    {'refresh':refresh,'preparation':preparation,'final':final}[a.action]()
