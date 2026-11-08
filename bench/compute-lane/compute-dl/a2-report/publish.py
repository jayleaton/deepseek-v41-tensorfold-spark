#!/usr/bin/env python3
"""A2 budget-only collection wrapper around byte-identical A1 science and statistics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'a1-analysis'))
import collect_report
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ARMS=('D','L','P')
A1_SHA='0dca1fa52838388ea1e7afbc3a4c5e9747775f64e871640df7a66496369e5453'
A2_SHA='8eaa9575592f519b504d08022d400602e4b552874c8424b02c59c34f4af174e2'
def compact(row):
    # Only proof-redundant transport bulk is removed after independent raw audit.
    return {k:v for k,v in row.items() if k not in ('sse_frames','actual_request_wire','request','reasoning')}
def fmt(x):return 'unknown' if x is None else f'{x:.3f}'
def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True,type=Path);p.add_argument('--campaign',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--check-only',action='store_true');p.add_argument('--cpu-format-fixture',action='store_true');a=p.parse_args()
    integrity=load(a.audit/'audit.json');cfg=load(a.campaign)
    fixture=integrity.get('CPU_REPORT_FORMAT_FIXTURE_NOT_A2_COLLECTION_AUDIT') is True
    if fixture!=a.cpu_format_fixture:raise ValueError('CPU format fixture and production audit authorities must remain distinct')
    assert integrity['spec_sha256']==cfg['spec_sha256']==A2_SHA
    assert integrity['a1_spec_sha256']==A1_SHA and integrity['qualified'] is True
    tasks=[json.loads(line) for line in (ROOT/'dataset-v1/tasks-public.jsonl').read_text().splitlines()]
    answers=load(ROOT/'dataset-v1/answers-a.json')
    if a.check_only:
        byid={t['id']:t for t in tasks};n=0;kernel=collect_report.analysis.kernel;activated=set()
        for line in (a.audit/'audited-rows.jsonl').open():
            raw=json.loads(line);projected=compact(raw);t=byid[raw['id']];answer=answers[raw['id']]
            assert compact(collect_report.grade.grade(raw,t,answer))==collect_report.grade.grade(projected,t,answer)
            assert kernel.metric_values(raw)==kernel.metric_values(projected)
            assert kernel._row_safe(raw)==kernel._row_safe(projected)
            arm=collect_report.grade.identity(raw)[2]
            active=kernel._verified_activation(dict(raw,arm=arm))
            assert active==kernel._verified_activation(dict(projected,arm=arm))
            if arm=='L' and active:activated.add(raw['id'])
            n+=1
        assert n==integrity['observed'] and len(activated)>=30
        a.out.mkdir(exist_ok=False)
        proof={'passed':True,'rows':n,'projection_preserves':'strict grade, all costs, safety and activation',
               'source_sha256':sha(__file__),'audited_rows_sha256':sha(a.audit/'audited-rows.jsonl'),'GPU_requests':0}
        (a.out/'projection-check.json').write_text(json.dumps(proof,indent=2))
        print(json.dumps(proof));return
    rows=[compact(json.loads(line)) for line in (a.audit/'audited-rows.jsonl').open()]
    # The unchanged scientific kernel expects its A1 science authority, while
    # the independent raw audit records the A2 collection/budget authority.
    # Preserve both explicitly; never relabel the actual raw audit artifact.
    science_integrity=dict(integrity,spec_sha256=A1_SHA,collection_spec_sha256=A2_SHA,
                           authority_mapping='A2 changes only budget and one restoration break; A1 science unchanged')
    metadata={'integrity':science_integrity,'collection_integrity':integrity,'budget':integrity['budget'],
              'campaign_sha256':sha(a.campaign),'audited_rows_sha256':sha(a.audit/'audited-rows.jsonl'),
              'projection':'Transport frames/wire/request/reasoning omitted only from derived grading input; immutable raw files and audit hashes retained.'}
    report=collect_report.collect(rows,tasks,answers,metadata=metadata)
    result=report['analysis'];kernel=collect_report.analysis.kernel
    batches={}
    for batch in ('P-before-break','P-after-break'):
        group=[r for r in report['strict_rows'] if r.get('retention_batch')==batch]
        assert all(collect_report.grade.identity(r)[2]=='P' for r in group)
        batches[batch]={'observed':len(group),'strict_solves':sum(r['grade']=='correct' for r in group),
            'descriptive_only':True,'family_seed':{}}
        for family in collect_report.analysis.FAMILIES:
            for seed in (101,102):
                subset=[r for r in group if r['family']==family and r['seed']==seed]
                costs=[dict(zip(kernel.METRICS,kernel.metric_values(r),strict=True)) for r in subset]
                summary={'observed':len(subset),'strict_solves':sum(r['grade']=='correct' for r in subset)}
                for field in ('wall_seconds','returned_ids'):
                    known=[v[field] for v in costs if v[field] is not None]
                    summary[field]={'known':len(known),'mean':sum(known)/len(known) if known else None}
                batches[batch]['family_seed'][family+'/'+str(seed)]=summary
    assert sum(b['observed'] for b in batches.values())==sum(collect_report.grade.identity(r)[2]=='P' for r in report['strict_rows'])
    report['P_break_batches']=batches
    a.out.mkdir(exist_ok=False)
    for name,value in [('report.json',{k:v for k,v in report.items() if k not in ('strict_rows','secondary_rows')}),
                       ('analysis.json',result),('secondary-analysis.json',report['secondary_analysis'])]:
        with (a.out/name).open('x') as out:json.dump(value,out,indent=2,allow_nan=False)
    for name,values in [('graded.jsonl',report['strict_rows']),('secondary-graded.jsonl',report['secondary_rows'])]:
        with (a.out/name).open('x') as out:
            for row in values:out.write(json.dumps(row,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n')
    indexed={collect_report.grade.identity(r):r for r in report['strict_rows']}
    fields=['id','family','seed','arm','missing','status','strict_grade','sensitivity_grade','returned_ids','initial_plus_returned_SUBTOTAL',
            'processed_input_plus_returned','wall_seconds','cap','deadline','policy_error','retention_batch']
    second={collect_report.grade.identity(r):r for r in report['secondary_rows']}
    with (a.out/'per-problem.csv').open('x',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
        for task in tasks:
            for seed in (101,102):
                for arm in ARMS:
                    key=(task['id'],seed,arm);row=indexed.get(key);metrics=dict(zip(kernel.METRICS,kernel.metric_values(row),strict=True));flags=kernel.flags(row)
                    writer.writerow(dict(id=key[0],family=task['family'],seed=seed,arm=arm,missing=row is None,
                        status=row.get('status','') if row else '',strict_grade=row.get('grade','') if row else '',
                        sensitivity_grade=second.get(key,{}).get('grade',''),returned_ids=metrics.get('returned_ids'),
                        initial_plus_returned_SUBTOTAL=metrics.get('initial_prompt_plus_returned_SUBTOTAL'),processed_input_plus_returned=metrics.get('processed_input_plus_returned_ids'),
                        wall_seconds=metrics.get('wall_seconds'),cap=flags['cap'] if row else '',deadline=flags['deadline'] if row else '',
                        policy_error=flags['policy_error'] if row else '',retention_batch=row.get('retention_batch','') if row else ''))
    lines=['# '+('CPU FORMAT FIXTURE — NOT COLLECTION RESULTS' if fixture else 'Compute study A2 completion — '+result['status']),'',
           ('CPU-only format control; no A2 collection audit or GO.' if fixture else '**NEEDS-T5.** '+str(result['observed'])+'/720 dataset replies observed. See final G verification receipt for independent live status.'),'',
           'Parent SHA `'+cfg['base_spec_sha256']+'`; A1 SHA `'+A1_SHA+'`; A2 SHA `'+A2_SHA+'`.','',result['interpretation'],'',result['historical'],'',
           'The confirmed power outage rebooted both Sparks around2026-11-05T07:30Z. No measured request was interrupted and no exclusions were made. '
           'The failed lock/port/wrapper recovery attempts and conservative UTC restoration charge remain in the ledger. '
           'The idle Flash17 front proxy was stopped to free8000; its unit remains enabled.','',
           'A2 extended cumulative held time to36000s, with33900s admission and2100s restoration reserve, after partial P outcomes were visible. '
           'The sealed A1 controller had no live cutoff reload; one permitted P G-restore break was used. '
           'All old held intervals and actual restoration/boot costs carry forward unchanged. '
           'P-before/after-break batch descriptors appear below; both batches remain in the same primary analysis.','',
           '| Family | Arm | Strict solved/observed | Scheduled | Returned IDs/problem | Wall s/problem | Timeout/deadline replies |','|---|---|---:|---:|---:|---:|---:|']
    for family in ('overall',*collect_report.analysis.FAMILIES):
        subset=[r for r in report['strict_rows'] if family=='overall' or r['family']==family]
        scheduled=240 if family=='overall' else 40
        for arm in ARMS:
            group=[r for r in subset if collect_report.grade.identity(r)[2]==arm]
            values=[dict(zip(kernel.METRICS,kernel.metric_values(r),strict=True)) for r in group]
            mean=lambda field:sum(v[field] for v in values if v[field] is not None)/sum(v[field] is not None for v in values) if any(v[field] is not None for v in values) else None
            lines.append('| %s | %s | %d/%d | %d | %s | %s | %d |'%(family,arm,sum(r['grade']=='correct' for r in group),len(group),scheduled,
                fmt(mean('returned_ids')),fmt(mean('wall_seconds')),sum(kernel.flags(r)['deadline'] for r in group)))
    lines+=['','Means describe known observed rows; scheduled missing costs are unknown, never zero. See JSON for sums/medians/coverage and censoring.','']
    for name,c in result['contrasts'].items():
        x=c['overall'];key='solve_difference_'+c['treatment']+'_minus_'+c['baseline']
        lines+=['## '+name+(' — amended strict primary' if c['primary'] else ' — secondary/descriptive'),' ',
                'Solve difference: '+str(x[key])+'. 95% CI: '+str(x['solve_difference_ci95'])+'. Exact task sign flip: '+json.dumps(x['sign_flip'])+'.',
                'Paired gains/losses: '+json.dumps(x['gains'])+' / '+json.dumps(x['losses'])+'.',
                'Returned-ID cost: '+json.dumps({k:v for k,v in x['costs']['returned_ids'].items() if k in ('ratio_'+c['treatment']+'_over_'+c['baseline'],'ratio_ci95','complete_pairs','full_cohort_claim_withheld')})+'.',
                'Wall cost: '+json.dumps({k:v for k,v in x['costs']['wall_seconds'].items() if k in ('ratio_'+c['treatment']+'_over_'+c['baseline'],'ratio_ci95','complete_pairs','full_cohort_claim_withheld')})+'.','']
    lines+=['## Accounting and sensitivity','',
            'P restoration-break batches (descriptors only; both remain in the primary analysis): '+json.dumps(batches)+'.','',
            '20,000 shared family-stratified task-cluster bootstrap draws, RNG20261105, percentile indices500/19499; both seeds/all arms retained. '
            'Only strict overall L/P is the amended primary. Family/cost/sensitivity intervals are marginal/descriptive. '
            'One-numeral final-span sensitivity is post hoc for retained outcomes, fixed before new collection; arrays/maps retain strict grades. '
            'Scalar-only denominator160 and combined240 per completed arm. All changed and residual rows are in report.json.','',
            'Multi-round processed-input totals remain withheld where unknown. Initial-prompt-plus-returned is only a SUBTOTAL. '
            'Policy terminations are separate from natural caps. More than25% censor contribution or a complete-both reversal blocks an uncensored efficiency claim.','',
            'Activation: '+json.dumps(result['activation'])+'.',
            'Qualification: '+json.dumps(result['qualification'])+'.',
            'Clipped presentation summary: '+json.dumps(report['presentation_summary'])+'.','',
            'D attempted/blocked/policy/final/strict-solve counts by family/seed, native execution counts, all cost coverage and missingness are retained in analysis.json. '
            'Clipping authenticates landed presentation, not invisible tails or complete values; final grading is never rescued by a tool result.','',
            'CPU-matched/count-checked prompts; actual served arrays were not captured and same-length rendering differences remain possible. '
            'Cold boots/cache resets, pause, mostly L-before-P order and outcome-informed amendment remain confounds. '
            'No serving-default promotion or full-token-saving claim.','',
            'DeepSeek published scores in [PUBLISHED.md](../../PUBLISHED.md) are labelled historical references only. '
            'This generated mixture has no matched published vendor score.','',
            '[Complete analysis](analysis.json) · [Sensitivity and changed/residual rows](report.json) · [720-row ledger](per-problem.csv)','']
    (a.out/'RESULTS.md').write_text('\n'.join(lines))
    if not fixture:
        (a.out/'NEEDS-T5.md').write_text('# NEEDS-T5\n\nA1 science with A2 completion budget: '+result['status']+'; '+str(result['observed'])+'/720 observed.\n\nParent '+cfg['base_spec_sha256']+'\nA1 '+A1_SHA+'\nA2 '+A2_SHA+'\n\n[Results](RESULTS.md). Strict L/P is amended primary; D/L availability, D/P and sensitivities are secondary. Outcome-informed extension and one P restoration break disclosed. No default promotion.\n')
    print(json.dumps({'status':result['status'],'observed':result['observed'],'activation':result['activation']['distinct_tasks'],'integrity':result['qualification']['integrity'],'out':str(a.out)}))
if __name__=='__main__':main()
