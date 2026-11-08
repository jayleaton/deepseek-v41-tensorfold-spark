"""Actual evidence checks; parent receipts supplement native pins, never replace them."""
import json
from pathlib import Path
from protocol import receipt,require,bound,local_path,sha,QUALIFIED_SPEC_SHA256
HERE=Path(__file__).resolve().parent

def validate(c):
    prep=receipt(c,'preparation')
    require(prep['prep_manifest_sha256']==c['prep_manifest_sha256'],'prep receipt/request freeze binding')
    require(prep['tasks']==60 and prep['scheduled_replies']==360 and prep['unresolved_overlaps']==0 and prep['missing_sources']==[],'disjointness not qualified')
    require(bool(prep['historical_inventory_sha256']) and bool(prep['artifact_sha256']),'authoritative inventory/reference pins missing')
    for p,h in prep['artifact_sha256'].items():bound(p,h)
    prompts=receipt(c,'prompts')
    require(prompts['pairs']==362 and prompts['arrays_sha256']==c['cpu_prompt_arrays_sha256'],'native prompt coverage')
    bound(Path(c['cpu_prompt_arrays']).parent/'prompt-inputs.json',prompts['input_sha256'])
    bound(HERE.parent.parent/'decl-control/prompt-gate',prompts['binary_sha256'])
    bound(c['tokenizer'],prompts['tokenizer_sha256'])
    controls=receipt(c,'controls')
    require(controls['preflight']['passed'] is True and controls['fake']['passed'] is True and len(controls['sigpipe_lifecycle'])==2,'actual controls/lifecycle required')
    require(len(controls['diagnostic_controls'])==5,'L323/55 and P no-executor controls missing')
    for entry in controls['sigpipe_lifecycle']:
        mode=entry['mode'];require(mode in ('success','failure') and entry['owned_exact_pids_exited'] is True,'lifecycle mode/settlement')
        bound(HERE/f'lifecycle-{mode}.json',entry['receipt_sha256']);bound(HERE/f'lifecycle-{mode}.log',entry['log_sha256'])
    for rank in (0,1):
        r=receipt(c,'rank'+str(rank))
        require(r['rank']==rank and r['actual_native_adapter_dry_runs']==2 and r['fresh_namespace_clear'] is True,'both actual boot dry runs required')
        require(r['coordination']['path']=='/tmp/rank1-host-gpu.lock' and r['coordination']['open_mode']=='O_RDONLY' and r['coordination']['flock_attempted'] is False,'read-only coordination gate')
        require([p['boot'] for p in r['proofs']]==['format-L','format-P'],'rank proof boot coverage')
        for p in r['proofs']:
            bound(p['path'],p['sha256']);proof=json.loads(local_path(p['path']).read_text())
            require(p['manifest_sha256']==c['rank_manifest_sha256'][p['boot']][str(rank)]==proof['manifest_sha256'],'actual manifest binding')
            require(proof['dry_run'] is True and proof['image_id']==c['image_ids'][rank] and proof['container_env']==p['effective_env'],'native launch proof')
    compute=HERE.parent.parent/'compute-dl'
    bound(compute/'grade.py','1a2249a1b5e0b06f03a4300995d0c320e5a21630d261b3791501f7c83740fb9f')
    grader=receipt(c,'grader');require(grader['positive_negative_fixtures'] is True,'node grader fixtures pending')
    clip=receipt(c,'clipping')
    require(clip['boundary_tokens']==[511,512,513] and clip['native_build_receipt_sha256']==sha(local_path(c['cap_policy']['receipt'])),'qualified512 native clipping receipt')
    from clipping import make_capper
    make_capper(c).verify()
    dispatch=HERE.parent.parent/'decl-control/dispatch-high/dispatch-proof.json'
    bound(dispatch,'77e43ebc13afdc16a8304836869953f8ceadfc8c6775b8f551aceb3b15af0436')
    d=json.loads(dispatch.read_text());require(d['verdict']=='PASS' and d['observed_fixtures']==8,'source-frozen dispatch proof')
    dispatch_gate=receipt(c,'dispatch')
    require(dispatch_gate['dispatch_proof_sha256']==sha(dispatch),'dispatch receipt proof binding')
    require(dispatch_gate['image_tag']=='tensorfold-dsv41-imr:r-3a429d6f122d' and dispatch_gate['image_ids']==c['image_ids'],'source/image qualification differs')
    require(dispatch_gate['engine_revision']=='3a429d6f122d9eb2900c43e622933d311e4be44f','engine source pin differs')
    require(bool(dispatch_gate['artifact_sha256']),'baseline dispatch/source/image evidence missing')
    for p,h in dispatch_gate['artifact_sha256'].items():bound(p,h)
    analysis=receipt(c,'analysis')
    require(analysis['scheduled_replies']==360 and analysis['strict_grader_sha256']==sha(compute/'grade.py'),'analysis/grader binding')
    # Every native adapter/protocol/helper used during qualification must remain sealed.
    required=[*HERE.glob('*.py'),*HERE.glob('*.sh')]
    required.append(HERE.parent.parent/'phase2/jail_worker.py')
    required += [compute/'collect_report.py',compute/'analysis.py',compute/'grade.py',compute/'a1-prep/audit_lifecycle.py',HERE.parent.parent/'phase2/audit_records.py']
    for p in required:require(c['code_sha256'].get(str(p).replace('<rank1-home>/','<rank0-home>/',1))==sha(p),'runtime source omitted from seal: '+str(p))
    return True
