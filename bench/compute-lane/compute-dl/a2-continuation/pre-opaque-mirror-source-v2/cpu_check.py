#!/usr/bin/env python3
"""Parent Spark CPU controls only; no native model or queue calls."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import protocol as p
import runtime_stage as s
import adoption
from budget import HeldBudget, BudgetExhausted, campaign_kwargs, validate_prior
HERE=Path(__file__).resolve().parent

def check(root=HERE.parent):
    root=Path(root);controls=[]
    def require(ok,label):
        if not ok:raise AssertionError(label)
        controls.append(label)
    def reject(label,fn):
        try:fn()
        except (ValueError,RuntimeError,KeyError,OSError):controls.append(label);return
        raise AssertionError('accepted '+label)
    for source in HERE.glob('*.py'):ast.parse(source.read_text(),filename=str(source))
    original=s.old_boots(root);retained,missing=s.retention(root,original)
    require(retained['observed']==342 and len(missing)==138,'original342 immutable')
    with tempfile.TemporaryDirectory(prefix='a2-fake-',dir=HERE) as tmp:
        tmp=Path(tmp);index=s.stage(tmp/'stage',root=root)
        cfg=json.loads((tmp/'stage/stage-campaign.json').read_text())
        audit=json.loads(Path(cfg['a1_audit']).read_text())
        rows=[json.loads(l) for l in (Path(cfg['a1_audit']).parent/'audited-rows.jsonl').read_text().splitlines()]
        remaining,observed,full=adoption.suffix(original,rows)
        require(index['fresh_replies']==len(remaining) and len(index['schedule'])==720,'actual suffix counts')
        for label,mutated in [('missing DL',[r for r in rows if r is not next(r for r in rows if r['arm']=='lane')]),
                              ('duplicate',rows+[rows[-1]])]:
            reject('suffix '+label,lambda:adoption.suffix(original,mutated))
        # Synthetic prefix independent of actual count; errors/caps retained by identity.
        dl=[r for r in rows if r['arm']!='plain']
        prows=[dict(arm='plain',suite=r['key'][0],id=r['key'][1],seed=r['key'][2],status='error') for r in full]
        require(adoption.suffix(original,dl+prows[:2])[0]==full[2:],'error replies never retried')
        reject('P gap',lambda:adoption.suffix(original,dl+[prows[1]]))
        reject('P reorder',lambda:adoption.suffix(original,dl+prows[1::-1]))
        for rank in (0,1):
            s.materialize(tmp/'stage',rank,tmp/f'rank{rank}')
            require(not (tmp/f'rank{rank}'/'GO').exists(),'no materialized GO')
        reject('stage overwrite',lambda:s.stage(tmp/'stage',root=root))
        cfg['spec']=str(HERE/'T5-COMPUTE-DL-SPEC-A2.md');cfg['a1_spec']=str(HERE/'T5-COMPUTE-DL-SPEC-A1.md')
        cfg['base_spec']=str(root/'T5-COMPUTE-DL-SPEC.md')
        p.require_staging_ready(cfg)
        for field,value in [('prefix',adoption.A1_PREFIX),('budget',{'limit_seconds':18000,'admission_seconds':15900}),('transport_contract',{'max_tokens':8191,'deadline_s':270})]:
            changed=copy.deepcopy(cfg);changed[field]=value
            reject('config '+field,lambda:p.require_staging_ready(changed))
        reject('pending qualification',lambda:p.require_collection_ready(cfg))
        fakecfg=dict(cfg,freeze=str(tmp/"stage/frozen-v1"),adoption_receipt=str(tmp/"stage/adoption.json"))
        p.verify_request_seal(fakecfg)
        bootpath=tmp/"stage/frozen-v1/boots/A2-plain-P.json"
        saved=bootpath.read_bytes();damaged=json.loads(saved)
        damaged["requests"][0]["wire"]+=" "
        bootpath.write_text(json.dumps(damaged))
        reject("suffix wire mutation",lambda:p.verify_request_seal(fakecfg))
        bootpath.write_bytes(saved)
        receiptpath=tmp/"stage/adoption.json";saved=receiptpath.read_bytes();damaged=json.loads(saved)
        damaged["unchanged_held_seconds"]=0;receiptpath.write_text(json.dumps(damaged))
        reject("adoption carry mutation",lambda:p.verify_request_seal(fakecfg))
        receiptpath.write_bytes(saved)
        kwargs=campaign_kwargs(cfg);prior=validate_prior(**kwargs);clock=[10.0]
        hashes={str(path):s.sha(path) for path in (kwargs['prior_path'],kwargs['audit_path'],kwargs['original_path'])}
        budget=HeldBudget(tmp/'budget.json',p.QUALIFIED_SPEC_SHA256,'A2-plain',**kwargs,clock=lambda:clock[0])
        require(budget.data['schema']==4 and budget.elapsed()==prior['held_seconds'] and budget.data['turns']==prior['turns'],'full schema4 carry')
        for field,value in [('limit_seconds',18000),('held_seconds',0),('carry_seconds',0),('prior_ledger_sha256','0'*64)]:
            changed=copy.deepcopy(budget.data);changed[field]=value;(tmp/'bad.json').write_text(json.dumps(changed))
            reject('budget mutation '+field,lambda:HeldBudget(tmp/'bad.json',p.QUALIFIED_SPEC_SHA256,'A2-plain',**kwargs))
        reject('audit drift',lambda:validate_prior(**dict(kwargs,audit_sha='0'*64)))
        reject('ledger drift',lambda:validate_prior(**dict(kwargs,prior_sha='0'*64)))
        budget.begin();clock[0]+=25
        reject('active restart',lambda:HeldBudget(tmp/'budget.json',p.QUALIFIED_SPEC_SHA256,'A2-plain',**kwargs))
        budget.finish_after_verified_restore()
        require(budget.elapsed()==prior['held_seconds']+25,'full restoration interval charged')
        loaded=HeldBudget(tmp/'budget.json',p.QUALIFIED_SPEC_SHA256,'A2-plain',**kwargs)
        try:loaded.ensure_can_start()
        except BudgetExhausted:controls.append('second A2 interval prohibited')
        else:raise AssertionError('second continuation allowed')
        require(all(s.sha(path)==digest for path,digest in hashes.items()),'all carry sources immutable')
        fakeprior=copy.deepcopy(prior)
        fakeprior.update(held_seconds=33899,overrun_seconds=15899,
                         turns=[dict(turn="FAKE_ONLY",held_seconds=33899,cumulative_seconds=33899)])
        with patch("budget.validate_prior",return_value=fakeprior):
            ticks=[0.0]
            edge=HeldBudget(tmp/"edge.json",p.QUALIFIED_SPEC_SHA256,"A2-plain",**kwargs,clock=lambda:ticks[0])
            require(edge.data['held_seconds']==33899 and edge.data['overrun_seconds']==0
                    and fakeprior['overrun_seconds']==15899,'old overrun retained only under old authority; no held-time reset')
            edge.begin();ticks[0]=1.0
            try:edge.ensure_can_start()
            except BudgetExhausted:controls.append("exact33900 fake-clock admission boundary")
            else:raise AssertionError("33900 admission allowed")
            ticks[0]=2102.0;edge.finish_after_verified_restore()
            require(edge.data["held_seconds"]==36001 and edge.data["overrun_seconds"]==1,"safety restore charged beyond36000")
            damaged=copy.deepcopy(edge.data);damaged["turns"][0]["cumulative_seconds"]-=1
            (tmp/"edge.json").write_text(json.dumps(damaged))
            reject("standalone history mutation",lambda:HeldBudget(tmp/"edge.json",p.QUALIFIED_SPEC_SHA256,"A2-plain",**kwargs))
        from coordination import check
        import fcntl
        import os
        lock=tmp/'coordination.lock'; lock.write_bytes(b'')
        with lock.open('rb') as holder:
            fcntl.flock(holder,fcntl.LOCK_EX)
            require(check(lock)['passed'],'regular readable coordination passes while G holds flock')
        reject('missing coordination file',lambda:check(tmp/'missing.lock'))
        reject('nonregular coordination file',lambda:check(tmp))
        fifo=tmp/'fifo';os.mkfifo(fifo)
        reject('coordination FIFO refused without blocking',lambda:check(fifo))
        with patch('coordination.os.open',side_effect=PermissionError('fixture unreadable')):
            reject('unreadable coordination file',lambda:check(lock))
        require(lock.read_bytes()==b'','coordination file unchanged')
        import evidence
        fake_source='<rank0-work>/HISTORICAL_FIXTURE_ONLY.json'
        direct=tmp/'primary-evidence.json';base=tmp/'evidence-mirrors'
        digest=hashlib.sha256(b'qualified bytes').hexdigest()
        mirrored=evidence.mirror_path(fake_source,base);mirrored.parent.mkdir(parents=True)
        mirrored.write_bytes(b'qualified bytes')
        with patch.object(evidence,'ROOT',Path('<rank1-work>/stdbench/compute-dl')), \
             patch.object(evidence,'MIRRORS',base),patch.object(evidence,'primary',return_value=direct):
            require(evidence.qualified_path(fake_source,digest)==mirrored,'historical mirror exact digest')
            direct.write_bytes(b'wrong live bytes')
            with patch('protocol.local_path',return_value=direct):
                reject('live code cannot use historical evidence fallback',lambda:p.bound(fake_source,digest))
            mirrored.write_bytes(b'corrupted mirror')
            require(evidence.qualified_path(fake_source,digest)==direct,'corrupt historical mirror refused')
            reject('corrupt historical bytes refuse qualification',lambda:p.bound_preparation(fake_source,digest))
            direct.write_bytes(b'qualified bytes')
            require(evidence.qualified_path(fake_source,digest)==direct,'matching primary preferred')
        reject('relative historical path refused',lambda:evidence.mirror_path('../escape',base))
        linked=tmp/'linked-mirror';linked.symlink_to(base,target_is_directory=True)
        reject('symlink historical mirror refused',lambda:evidence.mirror_path(fake_source,linked))
        budget.completed=False;budget.data['held_seconds']=33900

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
        for boot in ('A2-plain-P',):
            path=f'<rank{rank}-work>/stdbench/compute-dl/{p.PREFIX}/rank{rank}/boots/{boot}/manifest.json'
            require(cleanup_identity(path,rank)[2]==boot,'cleanup survives GO revocation '+boot+str(rank))
    import fake_client_check
    client_result=fake_client_check.main()
    require(client_result['D_call_no_external_execution'], 'fake D native client accounting')
    return {'passed':True,'status':'FAKE_NOT_COLLECTION','GPU_requests':0,'coordination_fixtures':True,'controls':controls,
            'pending':'actual native CPU prompts/clipping/rank/env/dispatch/preflight/SIGPIPE and parent qualification'}

if __name__=='__main__': print(json.dumps(check(),indent=2))
