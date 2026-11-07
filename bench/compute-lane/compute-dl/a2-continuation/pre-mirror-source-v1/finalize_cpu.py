#!/usr/bin/env python3
"""Seal actual CPU gates and carried ledger. Never creates GO or queues a turn."""
import hashlib
import json
from pathlib import Path
import protocol
from budget import HeldBudget, campaign_kwargs
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;TARGET=ROOT/protocol.PREFIX
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,reason):
    if not ok:raise ValueError(reason)
def canonical(path):return Path(str(path).replace('<rank1-home>/','<rank0-home>/',1))
def main():
    require(not (TARGET/'GO').exists(),'GO must remain absent during qualification')
    cfg=load(TARGET/'stage-campaign.json');protocol.require_staging_ready(cfg)
    old=load(ROOT/'stdbench-compute-dl-20261105-v1/campaign-v1.json')
    code=dict(old['code_sha256'])
    for directory in (HERE,ROOT/'a1-analysis',ROOT/'a1-prep',ROOT/'a1-recovery-20261106',ROOT/'a2-report'):
        for p in directory.iterdir():
            if p.suffix in ('.py','.sh','.zig'):code[str(p)]=sha(p)
    code[str(ROOT/'T5-COMPUTE-DL-SPEC-A1.md')]=sha(ROOT/'T5-COMPUTE-DL-SPEC-A1.md')
    for spec in ("T5-COMPUTE-DL-SPEC-A1.md","T5-COMPUTE-DL-SPEC-A2.md"):code[str(HERE/spec)]=sha(HERE/spec)
    for path,expected in code.items():require(sha(path)==expected,'source drift '+path)
    evidence={}
    def record(p):
        p=Path(p);evidence[str(p)]=sha(p);return load(p) if p.suffix=='.json' else None
    controls=record(TARGET/'control-check.json')
    require(controls['passed'] is True and controls['sigpipe_lifecycle'] and controls['preflight']['passed'],'runtime controls missing')
    require(controls['fake']['coordination_fixtures'] is True,'coordination fixture controls missing')
    require(len(controls['diagnostic_controls'])==3,'each diagnostic control missing')
    for path,expected in controls['code_sha256'].items():require(sha(path)==expected,'tested runtime code drift '+path)
    for mode in ('success','failure'):
        r=record(HERE/f'lifecycle-{mode}.json');record(HERE/f'lifecycle-{mode}.log')
        require(all(not Path(f'/proc/{pid}').exists() for pid in r['owned_pids']),'verifier producer survived')
    analysis=record(ROOT/'a1-analysis/qualification.json');record(ROOT/'a1-analysis/qualification-check.log')
    require(analysis['passed'] is True and analysis['full_20000_draw_fake_controls'] is True
            and analysis['real_retained_observed']==342 and analysis['retained_activation']==112,'amended analysis qualification missing')
    for path,expected in analysis['source_sha256'].items():require(sha(canonical(path))==expected,'tested analysis source drift')
    projection=record(ROOT/'a1-continuation/publish-check-v3/projection-check.json')
    require(projection['passed'] is True and projection['rows']==342
            and projection['source_sha256']==sha(ROOT/'a1-recovery-20261106/publish.py'), 'report projection controls missing')
    report=record(ROOT/"a2-report/qualification.json")
    require(report["passed"] is True and report["spec_sha256"]==protocol.QUALIFIED_SPEC_SHA256,"A2 report qualification pending")
    require(bool(report["source_sha256"]),"A2 report source seals missing")
    report_sources={str(p):sha(p) for p in (ROOT/"a2-report").rglob("*") if p.suffix in (".py",".sh")}
    require(bool(report_sources) and all(report["source_sha256"].get(path)==digest for path,digest in report_sources.items()),"A2 report tested source coverage differs")
    code.update(report_sources)
    for path,expected in report["source_sha256"].items():require(sha(canonical(path))==expected,"qualified A2 report drift")
    for rank in (0,1):
        r=record(TARGET/f'rank-check-r{rank}.json')
        require(r['passed'] is True and r['rank']==rank and r['actual_native_adapter_dry_runs']==len(cfg['layout']['boots'])==1
                and r['fresh_namespace_clear'] is True,'actual rank qualification missing')
        require(r['coordination']['passed'] is True and r['coordination']['path']=='/tmp/rank1-host-gpu.lock'
                and r['coordination']['open_mode']=='O_RDONLY' and r['coordination']['flock_attempted'] is False,'rank coordination path not qualified')
        require([p["boot"] for p in r["proofs"]]==["A2-plain-P"],"rank boot coverage differs")
        require(r["spec_sha256"]==protocol.QUALIFIED_SPEC_SHA256,"rank protocol mismatch")
        for proof in r['proofs']:
            require(proof["manifest_sha256"]==cfg["rank_manifest_sha256"][proof["boot"]][str(rank)],"rank manifest proof mismatch")
            p=canonical(proof['path']);require(sha(p)==proof['sha256'],'native dry-run proof drift');record(p)
        require(r['source_sha256']==sha(HERE/'qualify_rank.py'),'rank checker source drift')
    arrays=TARGET/'prompt-arrays.jsonl';record(arrays)
    rendered=record(TARGET/'prompt-proof.json');record(TARGET/'prompt-inputs.json')
    require(rendered['passed'] is True and rendered['pairs']==242 and rendered['arrays_sha256']==sha(arrays),'native CPU prompt proof missing')
    require(rendered.get('reuse') is True and rendered['input_sha256']==sha(TARGET/'prompt-inputs.json')
            and rendered['source_sha256']==sha(HERE/'render_prompts.py'),'fresh prompt input/reuse binding differs')
    for field in ('prior_proof','prior_cpu_receipt'):
        require(sha(rendered[field+'_path'])==rendered[field+'_sha256'],'reused prompt authority drift')
        record(rendered[field+'_path'])
    prompts=[json.loads(l) for l in arrays.read_text().splitlines()]

    require(len(prompts)==len({tuple(p['key']) for p in prompts})==242,'CPU prompt array key coverage')
    require(all(p['plain_ids'] and p['declared_off_ids']==p['declared_lane_ids']==p['original_builtin_lane_ids'] for p in prompts),'CPU D/L array equality failed')
    native=record(ROOT/'a1-prep/native-build-receipt.json')
    from clipping import make_capper
    make_capper(cfg).verify()
    for name in ('binary','tokenizer'):record(cfg['cap_policy'][name])
    for rel in native['harness_sha256']:record(ROOT/'a1-prep'/rel)
    policy=record(ROOT/'a1-prep/a1-policy-check-receipt.json')
    require(policy['passed'] is True and policy['boundary_tokens']==[511,512,513]
            and policy['native_build_receipt_sha256']==sha(ROOT/'a1-prep/native-build-receipt.json'),'native boundary/status controls missing')
    record(ROOT/'a1-prep/clip-check-receipt.json')
    vr=record(cfg['revalidation_receipt'])
    require(vr['passed'] is True and vr['observed']==342 and vr['remaining']==138 and vr['activation_distinct_L']>=30,'uniform revalidation missing')
    require(vr['policy_controls_sha256']==sha(ROOT/'a1-prep/a1-policy-check-receipt.json'),'revalidation/native controls binding mismatch')
    for path,expected in vr['raw_source_sha256'].items():require(sha(path)==expected,'original raw drift');record(path)
    for rel,expected in vr['files_sha256'].items():
        p=Path(cfg['revalidation_receipt']).parent/rel;require(sha(p)==expected,'revalidated result drift');record(p)
    dispatch=record(ROOT.parent/'decl-control/dispatch-high/dispatch-proof.json')
    require(sha(ROOT.parent/'decl-control/dispatch-high/dispatch-proof.json')=='77e43ebc13afdc16a8304836869953f8ceadfc8c6775b8f551aceb3b15af0436'
            and dispatch['verdict']=='PASS' and dispatch['observed_fixtures']==8,'real-route dispatch proof missing')
    require(sha(ROOT/'grade.py')=='1a2249a1b5e0b06f03a4300995d0c320e5a21630d261b3791501f7c83740fb9f','strict grader changed')
    require(sha(ROOT.parent/'phase2/grade_all.py')=='dcc8ef766247a1b02c993c9a50186b70b68c4fb5b6493e1bb42b3f7727186d36','final selector changed')
    cfg.update(cpu_prompt_arrays_sha256=sha(arrays),revalidation_receipt_sha256=sha(cfg['revalidation_receipt']),
               code_sha256=code,status='CPU_QUALIFIED_A2',collection_qualified=True)
    cfg['freeze_sha256']=sha(Path(cfg['freeze'])/'manifest.json')
    protocol.verify_request_seal(cfg)
    ledger=HeldBudget(cfg['budget_ledger'],cfg['spec_sha256'],'A2-plain',**campaign_kwargs(cfg))
    prior=load(cfg['prior_budget'])
    require(ledger.elapsed()==prior['held_seconds'] and ledger.data['turns']==prior['turns'],'budget baseline changed before collection')
    record(cfg["prior_budget"]);record(cfg["a1_audit"]);record(cfg["adoption_receipt"])
    record(ROOT/"a2-report/freeze.json")
    for path,expected in cfg["a1_audit_files_sha256"].items():
        require(sha(path)==expected,"A1 audit file drift");record(path)
    audit=load(cfg["a1_audit"])
    for path,expected in audit["provenance"]["input_file_sha256"].items():
        require(sha(canonical(path))==expected,"A1 provenance drift");record(canonical(path))
    proof={'passed':True,'spec_sha256':cfg['spec_sha256'],'parent_spec_sha256':cfg['base_spec_sha256'],
           "a1_spec_sha256":protocol.A1_SHA,
           'gates':protocol.GATES,'code_sha256':code,'preparation_file_sha256':evidence,
           'campaign_sha256':protocol.campaign_digest(cfg),
           'freeze_sha256':cfg['freeze_sha256'],'cpu_prompt_arrays_sha256':cfg['cpu_prompt_arrays_sha256'],
           'revalidation_receipt_sha256':cfg['revalidation_receipt_sha256'],'original_budget_sha256':cfg['original_budget_sha256'],
           'prior_budget_sha256':cfg['prior_budget_sha256'],'a1_audit_sha256':cfg['a1_audit_sha256'],'adoption_receipt_sha256':cfg['adoption_receipt_sha256'],
           'budget_carry_seconds':ledger.elapsed(),'dispatch_freshly_rerun':False,'dispatch_source_image_identity_rechecked':True,
           'prompt_limitation':'CPU-matched/count-checked; actual served arrays unobserved, same-length server differences remain possible.',
           'GPU_requests':0,'GO_created':False}
    proofpath=Path(cfg['cpu_receipt']);proofpath.parent.mkdir(exist_ok=True)
    with proofpath.open('x') as f:json.dump(proof,f,indent=2)
    cfg['cpu_receipt_sha256']=sha(proofpath)
    target=TARGET/'campaign-v1.json';temporary=TARGET/'campaign-qualified.json'
    with temporary.open('x') as f:json.dump(cfg,f,indent=2)
    # Only our new, never-started staged candidate is replaced, before GO.
    require(load(target)==load(TARGET/'stage-campaign.json'),'staged candidate was altered or already sealed')
    temporary.replace(target)
    require(protocol.require_collection_seal(target)==cfg,'final offline seal failed')
    print(json.dumps({'passed':True,'source_files':len(code),'evidence_files':len(evidence),'scheduled':720,'fresh':load(Path(cfg['freeze'])/'manifest.json')['fresh_replies'],
                      'carry_seconds':ledger.elapsed(),'campaign_sha256':sha(target),'GO_created':False,'GPU_requests':0}))
if __name__=='__main__':main()
