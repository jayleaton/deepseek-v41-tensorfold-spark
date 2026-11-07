"""Fail-closed format campaign seal. Only the parent may supply actual GO."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE_SHA='c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141'
QUALIFIED_SPEC_SHA256='b47ede697eed3a38d244ad5ff7447690732bd12945ff4592554d45bc2d8527ec'
PREFIX='stdbench-format-20261107-v1'
A1_TURNS=[{'name':'format','boots':['format-L','format-P']}]
GATES=('preparation','prompts','controls','rank0','rank1','clipping','dispatch','grader','analysis')

def require(ok,why):
    if not ok: raise ValueError(why)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def local_path(p):
    canonical=str(p).replace('<rank1-home>/','<rank0-home>/',1)
    return Path(canonical.replace('<rank0-home>/','<rank1-home>/',1)) if str(ROOT).startswith('<rank1-home>/') else Path(canonical)
def read_config(p): return dict(p) if isinstance(p,dict) else json.loads(local_path(p).read_text())
def bound(p,h):
    require(isinstance(h,str) and len(h)==64 and sha(local_path(p))==h,'artifact hash differs: '+str(p))
def campaign_digest(c):
    c=dict(c);c.pop('cpu_receipt_sha256',None)
    return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def require_staging_ready(value):
    c=read_config(value)
    require(c['prefix']==PREFIX and c['spec_sha256']==QUALIFIED_SPEC_SHA256 and c['base_spec_sha256']==BASE_SHA,'protocol identity')
    bound(c['spec'],QUALIFIED_SPEC_SHA256);bound(c['base_spec'],BASE_SHA)
    require(c['arms']==['L','P'] and c['layout']['turns']==A1_TURNS,'physical boot/turn layout')
    require([(b['boot'],b['arm']) for b in c['layout']['boots']]==[('format-L','L'),('format-P','P')],'boot layout')
    require(c['budget']==dict(limit_seconds=6300,admission_seconds=5880),'fresh format budget')
    root=Path('<rank0-work>/stdbench/format-retest')/PREFIX
    require(Path(c['budget_ledger'])==root/'budget.json' and Path(c['go_file'])==root/'GO','namespace')
    require(c['transport_contract']==dict(max_tokens=8192,deadline_s=270),'transport contract')
    require(c['g_restore_step']==dict(format='g-restore-v1',events=['SERVING','MODELS200','POPPED']),'restore contract')
    for key in ('client_wait_s','gpu_parent_s','remote_control_s','turn_s'):
        require(type(c['timeouts'][key]) is int and c['timeouts'][key]>=(330 if key=='client_wait_s' else 7200),'timeout '+key)
    for p,h in c['code_sha256'].items():bound(p,h)
    verify_request_seal(c)
    return c

def verify_request_seal(c):
    frozen=local_path(c['freeze']);bound(frozen/'manifest.json',c['freeze_sha256'])
    index=read_config(frozen/'manifest.json')
    require(index['spec_sha256']==QUALIFIED_SPEC_SHA256 and index['prefix']==PREFIX and index['turns']==A1_TURNS and index['boots']==c['layout']['boots'],'freeze identity')
    from manifest import validate_boots
    boots=[]
    for entry in index['boots']:
        p=frozen/entry['file'];p.resolve().relative_to(frozen.resolve());bound(p,entry['sha256']);boots.append(read_config(p))
    schedule=validate_boots(boots)
    from prep_bridge import prepared
    _,expected_boots,expected_inputs,_=prepared(local_path(c['prep_manifest']),c['prep_manifest_sha256'],local_path(c['prep_config']),c['prep_config_sha256'])
    require(boots==expected_boots,'runtime projection differs from exact prep arrays')
    require(read_config(frozen.parent/'prompt-inputs.json')==expected_inputs,'prompt inputs differ from prepared requests')
    require(index['schedule']==schedule and index['scheduled_replies']==360,'explicit360 schedule differs')
    return index

def receipt(c,name):
    record=c['qualification'][name];bound(record['path'],record['sha256']);p=read_config(record['path'])
    require(p.get('passed') is True,'qualification pending: '+name)
    require(p.get('spec_sha256')==QUALIFIED_SPEC_SHA256 or name in ('clipping','dispatch','grader'),'qualification spec: '+name)
    pins=p.get('source_sha256',p.get('code_sha256',{}))
    require(isinstance(pins,dict) and bool(pins),'qualification source coverage missing: '+name)
    for path,h in pins.items():bound(path,h)
    return p

def require_collection_seal(value):
    c=require_staging_ready(value)
    require(set(c['qualification'])==set(GATES),'all CPU gates required')
    for name in GATES:receipt(c,name)
    from qualification import validate
    validate(c)
    bound(c['cpu_receipt'],c['cpu_receipt_sha256']);proof=read_config(c['cpu_receipt'])
    require(proof['passed'] is True and proof['spec_sha256']==QUALIFIED_SPEC_SHA256 and proof['campaign_sha256']==campaign_digest(c),'final CPU seal pending')
    require(proof['code_sha256']==c['code_sha256'] and proof['qualification']==c['qualification'],'CPU source/evidence differs')
    for field in ('cpu_prompt_arrays','prep_manifest'):
        bound(c[field],c[field+'_sha256']);require(proof[field+'_sha256']==c[field+'_sha256'],'CPU binding '+field)
    from clipping import make_capper
    make_capper(c).verify()
    return c

def require_collection_ready(value):
    c=require_collection_seal(value)
    require(local_path(c['go_file']).read_text().strip()==QUALIFIED_SPEC_SHA256,'parent GO pending')
    return c
