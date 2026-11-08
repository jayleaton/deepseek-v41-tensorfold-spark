#!/usr/bin/env python3
"""A1-only fail-closed collection authority; staging never creates GO."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE_SHA = 'c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141'
QUALIFIED_SPEC_SHA256 = '0dca1fa52838388ea1e7afbc3a4c5e9747775f64e871640df7a66496369e5453'
PREFIX = 'stdbench-compute-a1-20261105-v2'
CARRY = 3413.519101628801
A1_TURNS = [{'name':'A1-resume','boots':['A1-resume-L','A1-resume-D']},
            {'name':'A1-plain','boots':['A1-plain-P']}]
GATES = {'observed_revalidated':342, 'remaining_original':138, 'P_requests':240,
         'scheduled_replies':720, 'native_clipping':True, 'native_prompt_arrays':True,
         'served_count_controls':True, 'dispatch_source_image':True, 'rank_checks':2,
         'fresh_rank1_host_clash':True, 'fake_controls':True, 'preflight':True,
         'sigpipe_lifecycle':True, 'strict_grader_pinned':True, 'analysis_secondary_pinned':True,
         'native_build_pinned':True, 'exact_optional_env':True}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def local_path(path):
    return Path(str(path).replace('<rank0-home>/','<rank1-home>/',1)) if str(ROOT).startswith('<rank1-home>/') else Path(path)
def read_config(value):
    result = dict(value) if isinstance(value,dict) else json.loads(local_path(value).read_text())
    if not isinstance(result,dict): raise ValueError('configuration must be an object')
    return result

def require(condition, reason):
    if not condition: raise ValueError(reason)

def bound(path, expected):
    require(isinstance(expected,str) and len(expected)==64 and sha(local_path(path))==expected, 'sealed artifact drift: '+str(path))

def require_bound_spec(value):
    c=read_config(value)
    require(c.get('spec_sha256')==QUALIFIED_SPEC_SHA256 and c.get('base_spec_sha256')==BASE_SHA, 'A1/base protocol binding differs')
    bound(c['spec'],QUALIFIED_SPEC_SHA256); bound(c['base_spec'],BASE_SHA)
    return c

def require_staging_ready(value):
    c=require_bound_spec(value)
    require(c.get('prefix')==PREFIX and c.get('arms')==['D','L','P'], 'fresh A1 namespace/arms required')
    require(c.get('layout',{}).get('turns')==A1_TURNS, 'fixed A1 turn order differs')
    require([(b['boot'],b['arm']) for b in c['layout']['boots']]==[
        ('A1-resume-L','L'),('A1-resume-D','D'),('A1-plain-P','P')], 'fixed boots differ')
    require(c.get('budget')=={'limit_seconds':18000,'admission_seconds':15900}, 'budget differs')
    root=Path('<rank0-work>/stdbench/compute-dl')/PREFIX
    require(Path(c['budget_ledger'])==root/'budget.json' and Path(c['go_file'])==root/'GO', 'A1 ledger/GO namespace differs')
    require(c.get('transport_contract')=={'max_tokens':8192,'deadline_s':270}, 'transport contract differs')
    require(c.get('g_restore_step')=={'format':'g-restore-v1','events':['SERVING','MODELS200','POPPED']}, 'restoration contract differs')
    for k in ('client_wait_s','gpu_parent_s','remote_control_s','turn_s'):
        require(type(c['timeouts'][k]) is int and c['timeouts'][k]>=(360 if k=='client_wait_s' else 7200), 'timeout floor: '+k)
    for path,digest in c.get('code_sha256',{}).items(): bound(path,digest)
    return c

def require_collection_seal(value):
    c=require_staging_ready(value)
    bound(c['cpu_receipt'],c['cpu_receipt_sha256'])
    proof=read_config(c['cpu_receipt'])
    require(proof.get('passed') is True and proof.get('spec_sha256')==QUALIFIED_SPEC_SHA256 and proof.get('parent_spec_sha256')==BASE_SHA,'CPU_GATES_PENDING')
    require(proof.get('gates')==GATES and proof.get('code_sha256')==c['code_sha256'] and len(c['code_sha256'])>=12, 'CPU gate/source coverage differs')
    for field in ('freeze','cpu_prompt_arrays','revalidation_receipt','original_budget'):
        actual=Path(c[field])/'manifest.json' if field=='freeze' else c[field]
        bound(actual,c[field+'_sha256'])
        require(proof.get(field+'_sha256')==c[field+'_sha256'], 'CPU artifact binding differs: '+field)
    revalidation=read_config(c['revalidation_receipt'])
    require(revalidation.get('passed') is True and revalidation.get('spec_sha256')==QUALIFIED_SPEC_SHA256
        and revalidation.get('parent_spec_sha256')==BASE_SHA and revalidation.get('observed')==342
        and revalidation.get('remaining')==138 and revalidation.get('counts')=={'D':120,'L':222}
        and revalidation.get('activation_distinct_L',0)>=30 and revalidation.get('preserved_original_seal') is True,
        'uniform original revalidation pending')
    require(bool(revalidation.get('raw_source_sha256')), 'original raw source seal missing')
    for path,digest in revalidation['raw_source_sha256'].items(): bound(path,digest)
    for path,digest in proof.get('preparation_file_sha256',{}).items(): bound(path,digest)
    require(bool(proof.get('preparation_file_sha256')), 'native/build/env evidence not sealed')
    policy_receipt=ROOT.parent/'a1-prep/a1-policy-check-receipt.json'
    require(str(policy_receipt) in {str(local_path(path)) for path in proof['preparation_file_sha256']}, 'A1 native policy controls not sealed')
    require(read_config(policy_receipt).get('passed') is True, 'A1 native policy controls pending')
    from budget import validate_original
    validate_original(local_path(c['original_budget']),c['original_budget_sha256'])
    from clipping import make_capper
    make_capper(c).verify()
    require(proof.get('campaign_sha256')==campaign_digest(c), 'final campaign seal differs')
    verify_request_seal(c)
    return c

def require_collection_ready(value):
    c=require_collection_seal(value)
    require(local_path(c['go_file']).read_text().strip()==QUALIFIED_SPEC_SHA256, 'new A1 GO absent')
    return c

def campaign_digest(config):
    c=dict(config); c.pop('cpu_receipt_sha256',None)
    return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def verify_request_seal(c):
    """Rebuild the request/mask relation from immutable original receipts."""
    import runtime_stage as stage
    root=ROOT.parent
    originals=stage.old_boots(root)
    retained,missing=stage.retention(root,originals)
    frozen=local_path(c['freeze']); index=read_config(frozen/'manifest.json')
    retention=read_config(frozen.parent/'retention.json')
    require(retention==retained, 'retained raw342 mask/source relation changed')
    bound(frozen.parent/'retention.json',index['retention_sha256'])
    require(index['scheduled_replies']==720 and index['fresh_replies']==378, 'A1 scheduled/fresh counts differ')
    expected=[('A1-resume-L','L',[r for r in originals['t02-L']['requests'] if ('L',*r['key']) in missing]),
              ('A1-resume-D','D',[r for r in originals['t02-D']['requests'] if ('D',*r['key']) in missing]),
              ('A1-plain-P','P',[stage.plain(r) for boot in ('t01-D','t02-D') for r in originals[boot]['requests']])]
    schedule=[]
    for b in originals.values():
        schedule.extend({'key':[b['arm'],*r['key']],'observed':(b['arm'],*r['key']) not in missing,
                         'wire_sha256':r['wire_sha256'],'task_sha256':r['task_sha256']} for r in b['requests'])
    for (name,arm,rows),entry in zip(expected,index['boots'],strict=True):
        require((entry['boot'],entry['arm'])==(name,arm), 'A1 boot/order differs')
        bound(frozen/entry['file'],entry['sha256']); boot=read_config(frozen/entry['file'])
        require(boot['requests']==rows and boot['boot']==name and boot['arm']==arm, 'exact original/P requests differ')
        diags=originals['t01-'+('L' if arm=='L' else 'D')]['diagnostics']
        if arm=='P': diags=[stage.plain(r) for r in diags]
        require(boot['diagnostics']==diags, 'fixed diagnostics differ')
        if arm=='P': schedule.extend({'key':['P',*r['key']],'observed':False,'wire_sha256':r['wire_sha256'],'task_sha256':r['task_sha256']} for r in rows)
    require(index['schedule']==schedule, '720-key derived manifest differs')
    revalidation=read_config(c['revalidation_receipt'])
    normalized=lambda m:{str(local_path(p)):h for p,h in m.items()}
    require(all(normalized(revalidation['raw_source_sha256']).get(path)==digest for path,digest in normalized(retained['raw_source_sha256']).items()), 'revalidation raw source coverage differs')
