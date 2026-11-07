#!/usr/bin/env python3
"""Parent-only Spark CPU checks. No network, model, engine, queue or Docker work."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import protocol as p
import runtime_stage as s
from budget import HeldBudget, BudgetExhausted

HERE=Path(__file__).resolve().parent

def check(root=HERE.parent):
    root=Path(root); controls=[]
    def require(value,label):
        if not value: raise AssertionError(label)
        controls.append(label)
    def reject(label,fn):
        try: fn()
        except (ValueError,RuntimeError,KeyError,FileExistsError,FileNotFoundError): controls.append(label); return
        raise AssertionError('accepted '+label)
    for source in HERE.glob('*.py'): ast.parse(source.read_text(),filename=str(source))
    original=s.old_boots(root); retained,missing=s.retention(root,original)
    require(retained['observed']==342 and len(missing)==138,'actual raw342 mask')
    require(('L','compute_dl','compute-dl-v1-aggregation-015',101) not in missing,'stopped reply never rescheduled')
    changed=copy.deepcopy(original); changed['t02-D']['requests'][0]['wire_sha256']='0'*64
    reject('remaining wire mutation',lambda:s.retention(root,changed))
    for arm in ('D','L'):
        d=copy.deepcopy(original['t01-'+arm]['requests'][0]); plain=s.plain(d)
        expected=copy.deepcopy(d['body_template']); del expected['tools']
        require(plain['body_template']==expected and plain['wire']==s.wire(expected),'P only removes tools '+arm)
        require(d['body_template'].get('tools') is not None,'original declaration retained '+arm)
    # Exact staging/order and exclusive local materialization under a temporary tree.
    with tempfile.TemporaryDirectory(prefix='a1-fake-',dir=HERE) as tmp:
        tmp=Path(tmp)
        index=s.stage(tmp/'stage',root=root)
        staged=json.loads((tmp/'stage/stage-campaign.json').read_text())
        require(index['turns']==p.A1_TURNS and index['fresh_replies']==378,'actual A1 stage order')
        rows=json.loads((tmp/'stage/frozen-v1/boots/A1-plain-P.json').read_text())['requests']
        require(rows==[s.plain(r) for b in ('t01-D','t02-D') for r in original[b]['requests']],'P exact original D order')
        require(len(index['schedule'])==720 and sum(r['observed'] for r in index['schedule'])==342,'720 schedule mask')
        for rank in (0,1):
            receipt=s.materialize(tmp/'stage',rank,tmp/f'rank{rank}')
            require(receipt['collection_qualified'] is False,'materialization grants no GO '+str(rank))
        reject('no stage overwrite',lambda:s.stage(tmp/'stage',root=root))
        # Change config control fields against actual spec paths, before any collection IO.
        staged['spec']=str(HERE/'T5-COMPUTE-DL-SPEC-A1.md'); staged['base_spec']=str(root/'T5-COMPUTE-DL-SPEC.md')
        p.require_staging_ready(staged)
        for field,value in [('prefix',s.OLD_PREFIX),('arms',['D','L']),('budget',{'limit_seconds':18001,'admission_seconds':15900}),('transport_contract',{'max_tokens':8191,'deadline_s':270}),('go_file',str(root/'GO'))]:
            changed=copy.deepcopy(staged); changed[field]=value
            reject('config mutation '+field,lambda:p.require_staging_ready(changed))
        changed=copy.deepcopy(staged); changed['layout']['turns'].reverse()
        reject('turn reversal',lambda:p.require_staging_ready(changed))
        reject('pending CPU qualification refuses collection',lambda:p.require_collection_ready(staged))
        old=root/s.OLD_PREFIX/'budget.json'; oldhash=s.sha(old); clock=[100.0]
        budget=HeldBudget(tmp/'budget.json',p.QUALIFIED_SPEC_SHA256,'A1-resume',original_path=old,original_sha=oldhash,clock=lambda:clock[0])
        require(budget.elapsed()==p.CARRY,'budget original carry')
        budget.begin(); clock[0]+=25
        reject('active ledger refuses restart',lambda:HeldBudget(tmp/'budget.json',p.QUALIFIED_SPEC_SHA256,'A1-resume',original_path=old,original_sha=oldhash))
        budget.finish_after_verified_restore()
        require(budget.elapsed()==p.CARRY+25 and s.sha(old)==oldhash,'restore charged and original ledger immutable')
        budget.data['held_seconds']=15900
        try: budget.ensure_can_start()
        except BudgetExhausted: controls.append('admission cutoff')
        else: raise AssertionError('admission cutoff missing')
    # Fake native cap checks comparator control flow only; real tokenizer proof parent-owned.
    from clipping import policy
    policy=policy()
    class FakeCap:
        def cap(self,bodies):
            return [{'output':b.decode(),'token_count':1,'retained_prefix_tokens':1} for b in bodies]
    body=b'323'; b64=lambda b:base64.b64encode(b).decode()
    result={'status':'ok','exit_code':0,'stdout_base64':b64(body),'stderr_base64':'',
            'output_limited':False,'native_result_token_cap':512,'elapsed_seconds':0.01,
            'normalized_body_base64':b64(body),'normalized_body_sha256':hashlib.sha256(body).hexdigest()}
    call={'status':'ok','output':'323'}
    require(policy.compare_pairs(FakeCap(),[(call,result)])==[True],'exact fake reference')
    for label,mutate in [('landed byte',lambda x:x.update(output='324')),('status swap',lambda x:x.update(status='exit_error')),('false marker',lambda x:x.update(output='323\n[output truncated at 512 tokens]'))]:
        altered=copy.deepcopy(call); mutate(altered)
        require(policy.compare_pairs(FakeCap(),[(altered,result)])==[False],label+' refused')
    import importlib.util
    spec=importlib.util.spec_from_file_location('_a1_cpu_rank_ops',HERE/'rank_ops.py')
    rank=importlib.util.module_from_spec(spec);spec.loader.exec_module(rank)
    exact_optional,cleanup_identity=rank.exact_optional,rank.cleanup_identity
    exact_optional({'TF_DSV41_COMPUTE_LANE':'0'},{'TF_DSV41_COMPUTE_LANE':'0','PATH':'fake'})
    reject('unexpected optional flag',lambda:exact_optional({'TF_DSV41_COMPUTE_LANE':'0'},{'TF_DSV41_COMPUTE_LANE':'0','TF_DSV41_GUIDE':'0'}))
    for rank in (0,1):
        for boot in ('A1-resume-L','A1-resume-D','A1-plain-P'):
            path=f'<rank{rank}-work>/stdbench/compute-dl/{p.PREFIX}/rank{rank}/boots/{boot}/manifest.json'
            require(cleanup_identity(path,rank)[2]==boot,'cleanup survives GO revocation '+boot+str(rank))
    import fake_client_check
    client_result=fake_client_check.main()
    require(client_result['D_call_no_external_execution'], 'fake D native client accounting')
    return {'passed':True,'status':'FAKE_NOT_COLLECTION','GPU_requests':0,'controls':controls,
            'pending':'actual native CPU prompts/clipping/rank/env/dispatch/preflight/SIGPIPE and parent qualification'}

if __name__=='__main__': print(json.dumps(check(),indent=2))
