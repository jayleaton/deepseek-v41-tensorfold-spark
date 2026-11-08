#!/usr/bin/env python3
"""Parent CPU-only pins/configs. Generation is a separately invoked pinned operation."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'compute-dl'
sys.path.insert(0,str(HERE/'prep'))
from common import SPEC_SHA256,encoded,create,spec_gate
from legacy import PATHS
from generation import inventory
from legacy import Legacy
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def save(path,value):create(Path(path),encoded(value))
def pin():
    spec_gate(HERE/'T5-FORMAT-RETEST-SPEC.md',SPEC_SHA256)
    receipt={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'python_version':platform.python_version(),'python_full':sys.version,'spec_sha256':SPEC_SHA256,
             'source_pins':{p.name:sha(p) for p in sorted((HERE/'prep').glob('*.py'))},
             'legacy_source_pins':{k:sha(p) for k,p in PATHS.items()},'study_candidates_generated_before_this_pin':0}
    save(HERE/'generation-pin-v1.json',receipt);print(json.dumps({'pre_generation_pin':True,'python_version':receipt['python_version'],'study_candidates':0}))
def generation_config():
    pin=load(HERE/'generation-pin-v1.json')
    for n,h in pin['source_pins'].items():
        if sha(HERE/'prep'/n)!=h:raise ValueError('prep changed after pin: '+n)
    records=load(HERE/'inventory-v1/historical-records.json');byid={r['id']:r for r in records}
    resolutions=load(HERE/'inventory-v1/resolutions-inherited.json')
    new=load(HERE/'review-resolutions-v1.json')
    for ident,r in new.items():
        if ident in resolutions:raise ValueError('duplicate semantic resolution')
        row=byid[ident];e=r['original_evidence']
        from disjointness import prompt_sha256
        if r['prompt_sha256']!=prompt_sha256(row['prompt']):raise ValueError('semantic prompt hash differs')
        if e['path']!=row['source_path'] or e['sha256']!=row['source_sha256'] or sha(e['path'])!=e['sha256']:
            raise ValueError('semantic original evidence differs '+ident)
        resolutions[ident]=r
    sourceaudit=load(HERE/'inventory-v1/source-audit.json')
    for s in sourceaudit['discovered_sources']:
        if sha(s['path'])!=s['sha256']:raise ValueError('historical source drift')
    for s in sourceaudit['legacy']['files']:
        if sha(s['path'])!=s['sha256']:raise ValueError('canonical exclusion source drift')
    tags=['W5','W6','W7','W8','W9','census','guide','trace','pilot-eleven','AB3','guide-S3','GSM8K','MATH','HumanEval','stdbench','declaration-control']
    index={'complete':True,'required_source_ids':['all-earlier-originals','compute-dl-v1'],
           'sources':[{'id':'all-earlier-originals','path':str(HERE/'inventory-v1/historical-records.json'),'sha256':sha(HERE/'inventory-v1/historical-records.json'),'count':len(records),'coverage':tags},
                      {'id':'compute-dl-v1','path':str(OLD/'dataset-v1/tasks-public.jsonl'),'sha256':sha(OLD/'dataset-v1/tasks-public.jsonl'),'count':120,'coverage':['compute-dl-v1']}],
           'resolutions':resolutions}
    save(HERE/'inventory-sealed-v1.json',index)
    cfg={'spec_path':str(HERE/'T5-FORMAT-RETEST-SPEC.md'),'spec_sha256':SPEC_SHA256,'python_version':pin['python_version'],
         'source_pins':pin['source_pins'],'legacy_source_pins':pin['legacy_source_pins'],
         'inventory_path':str(HERE/'inventory-sealed-v1.json'),'inventory_sha256':sha(HERE/'inventory-sealed-v1.json')}
    _,audit=inventory(cfg,Legacy())
    save(HERE/'generation-config-v1.json',cfg)
    print(json.dumps({'generation_config':str(HERE/'generation-config-v1.json'),'sha256':sha(HERE/'generation-config-v1.json'),'unique_exclusion_records':audit['unique_records'],'unresolved':0,'model_requests':0}))
def stage_config():
    from prep import SETTINGS
    pin=load(HERE/'generation-pin-v1.json')
    create(HERE/'settings-v1.json',json.dumps(SETTINGS,ensure_ascii=False,indent=2).encode())
    tools=HERE.parent/'decl-control/canonical-python-tools.json'
    cfg={'spec_path':str(HERE/'T5-FORMAT-RETEST-SPEC.md'),'spec_sha256':SPEC_SHA256,
         'tasks_path':str(HERE/'dataset-v1/tasks.jsonl'),'tasks_sha256':sha(HERE/'dataset-v1/tasks.jsonl'),
         'generation_audit_path':str(HERE/'dataset-v1/generation-audit.json'),'generation_audit_sha256':sha(HERE/'dataset-v1/generation-audit.json'),
         'references':{n:str(HERE/'dataset-v1'/n) for n in ('reference-a.json','reference-b.json')},
         'settings_path':str(HERE/'settings-v1.json'),'settings_sha256':sha(HERE/'settings-v1.json'),
         'tools_path':str(tools),'tools_sha256':sha(tools),'client_wait_s':360,'legacy_source_pins':pin['legacy_source_pins']}
    save(HERE/'stage-config-v1.json',cfg)
    print(json.dumps({'stage_config':str(HERE/'stage-config-v1.json'),'sha256':sha(HERE/'stage-config-v1.json')}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pin','generation-config','stage-config']);a=p.parse_args()
    if a.action=='pin':pin()
    elif a.action=='generation-config':
        sys.path.insert(0,str(OLD));generation_config()
    else:stage_config()
