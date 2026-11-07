#!/usr/bin/env python3
"""Presentation and supplementary telemetry only; consumes the frozen analysis."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())

def main():
    report=load(HERE/'report-v1/report.json');a=report['analysis'];overall=a['overall']
    if a['status']!='INCOMPLETE' or a['observed']!=342:raise ValueError('unexpected report; do not publish assumed outcomes')
    module=importlib.util.spec_from_file_location('frozen_compute_analysis',HERE/'analysis.py')
    analysis=importlib.util.module_from_spec(module);module.loader.exec_module(analysis)
    sealed=report['integrity']['provenance']['input_file_sha256']
    if sha(HERE/'analysis.py')!=sealed[str(HERE/'analysis.py')]:raise ValueError('frozen statistics source drift')
    rows=[json.loads(line) for line in (HERE/'report-v1/graded.jsonl').read_text().splitlines()]
    indexed={analysis.identity(row):row for row in rows}
    tasks=[json.loads(line) for line in (HERE/'dataset-v1/tasks-public.jsonl').read_text().splitlines()]
    audits={tuple(row['identity']):row['qualified'] for row in report['row_audits'] if not row['diagnostic']}
    telemetry={};flat=[]
    def observed_time(row,field):
        if row is None or not row.get('safety',{}).get('qualified'):return None
        calls=((row.get('tensorfold') or {}).get('compute') or {}).get('calls',[])
        values=[c.get(field) for c in calls]
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):return None
        return sum(values)/1000
    for task in tasks:
        for seed in (101,102):
            for arm in ('D','L'):
                row=indexed.get((task['id'],seed,arm));values=analysis.metric_values(row)
                item=dict(id=task['id'],family=task['family'],seed=seed,arm=arm,
                    missing=row is None,status=row['status'] if row else 'missing',
                    solved=analysis.correct(row) if row else None,
                    frozen_row_audit_qualified=audits.get((row['suite'],row['id'],row['seed'],row['arm'],row['boot'])) if row else None,
                    timeout=analysis.flags(row)['deadline'] if row else None,
                    capped=analysis.flags(row)['cap'] if row else None,
                    native_executor_seconds=observed_time(row,'exec_ms'),
                    native_tool_call_seconds=observed_time(row,'call_ms'))
                item.update(zip(analysis.METRICS,values));flat.append(item)
    if len(flat)!=480:raise ValueError('scheduled CSV grid differs')
    for family in ('overall',*analysis.FAMILIES):
        telemetry[family]={}
        for arm in ('D','L'):
            selected=[r for r in flat if r['arm']==arm and (family=='overall' or r['family']==family)]
            telemetry[family][arm]={}
            for field in ('native_executor_seconds','native_tool_call_seconds'):
                known=[r[field] for r in selected if r[field] is not None]
                telemetry[family][arm][field]={'known_rows':len(known),'scheduled_rows':len(selected),
                    'observed_sum':sum(known) if known else None,
                    'observed_mean':statistics.mean(known) if known else None,
                    'observed_median':statistics.median(known) if known else None}
    with (HERE/'per-problem.csv').open('x',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    supplement={'label':'Descriptive supplementary native telemetry; does not replace the frozen analysis',
        'source':'Pinned compute_segments.zig: call_ms is translate/queue/run; exec_ms is native executor run time',
        'coverage':'Only original frozen-qualified rows. Empty call lists have known zero native execution; missing/unqualified data stay unknown.',
        'values':telemetry,'source_sha256':sha(__file__),'frozen_report_sha256':sha(HERE/'report-v1/report.json')}
    with (HERE/'supplementary-telemetry.json').open('x') as output:json.dump(supplement,output,indent=2)
    def f(value):return 'unknown' if value is None else f'{value:.3f}'
    def metric(item,name,arm):
        value=item['costs'][name][arm]
        return f(value['known_mean'])+' (n='+str(value['known_rows'])+')'
    lines=['# Computation-demanding D/L study — INCOMPLETE','',
        '**NEEDS-T5. Collection stopped at 342/480 replies; 138 remain uncollected. G is restored.**',
        'Spec SHA256: `'+a['spec_sha256']+'`.',
        'D = declared tool, executor OFF. L = identical declaration/request, lane ON. No plain arm.',
        '120 deterministic instances, six fixed templates, two seeds; fixed-mixture screen, not a public benchmark or 120 independent task families.','',
        '## Stop and diagnosis','',
        '`compute-dl-v1-aggregation-015`, seed 101, stopped turn 2 at the frozen tool-result/reference gate. '
        'The native result was clipped at 512 tokens; the comparator used full 554-token stdout. '
        'Three isolated CPython replays reproduced the full stdout. A pinned native-tokenizer CPU proof exactly reconciles '
        'the delivered bytes with the production 512-token/UTF-8 clipping policy. The complete first output line matches the independent task reference.',
        'The failing receipt and frozen audit remain unchanged. No observed reply was replaced or regraded, and collection GO is revoked on both ranks. '
        'A T5 amendment is needed for a source-aware verifier and a prospective layout for the remaining 18 L + 120 D replies.',
        '[Truncation proof](diagnosis-v1/truncation-proof.json) · [diagnostic replays](diagnosis-v1/diagnosis.json) · [frozen audit](report-v1/audit.json).','',
        '## Observed results and missingness','',
        'Every family schedules 40 replies per arm. Solved/observed and cost means below describe retained observations only; '
        'different coverage prevents a full-cohort point comparison. Missing costs are unknown, never zero.','',
        '| Family | D solved/observed | L solved/observed | D returned IDs/problem | L returned IDs/problem | D wall s/problem | L wall s/problem | observed timeouts D/L |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for family,item in [('overall',overall),*a['families'].items()]:
        d,l=item['arms']['D'],item['arms']['L']
        timeouts='/'.join(str(round(item['arms'][arm]['rates']['deadline']*item['arms'][arm]['scheduled'])) for arm in ('D','L'))
        lines.append('| '+family+' | '+str(d['solved'])+'/'+str(d['observed'])+' | '+str(l['solved'])+'/'+str(l['observed'])+' | '+
            ' | '.join(metric(item,name,arm) for name in ('returned_ids','wall_seconds') for arm in ('D','L'))+' | '+timeouts+' |')
    lo,hi=overall['difference_bounds']
    lines+=['',f'All 240 scheduled replies per arm: D solve bounds {overall["arms"]["D"]["solved_count_bounds"]}/240; '
        f'L solve bounds {overall["arms"]["L"]["solved_count_bounds"]}/240. L−D logical bounds: {100*lo:.1f} to {100*hi:.1f} percentage points. '
        'These are missing-outcome bounds, not a confidence interval.',
        'The full-cohort primary difference, 95% CI and exact sign-flip p-value are withheld. No completed-study efficacy claim is made. '
        'No serving-default promotion or total-token-saving claim is made.','',
        '## Paired statistics and accounting','',
        'The frozen analysis uses 20,000 shared task-cluster bootstrap draws, stratified by family, Random(20261105), '
        'percentile indices 500/19499; both seeds/arms stay together. The exact task sign-flip is the sole overall primary test. '
        'Family/cost/sensitivity intervals are marginal and descriptive.','']
    for name in ('returned_ids','wall_seconds'):
        cost=overall['costs'][name];selected=cost['paired_accounting_complete_sensitivity']
        lines.append(f'- {name}: full-cohort ratio/CI withheld. Selected complete-pair sensitivity n={selected["pairs"]}: '
            f'L/D={f(selected["ratio_L_over_D"])}; 95% CI={selected["ratio_ci95"]}. This selected sample is not representative of missing cases.')
    lines+=['','Only '+str(overall['costs']['returned_ids']['complete_both_uncensored_sensitivity']['pairs'])+
        ' paired observation is complete-both and uncensored. Censoring blocks an uncensored efficiency claim. '
        'Cap/deadline/policy rates and net/positive censor contributions are retained in the full analysis.',
        'Multi-round processed-input totals remain withheld. Initial-prompt-plus-returned is a separately labelled subtotal, '
        'not full all-round token cost. Full-cohort processed-input-plus-returned ratios/CIs are withheld.',
        '[All sums, means, medians, coverage, censor contributions and paired solve gains/losses](report-v1/analysis.json) · '
        '[480-row per-problem ledger](per-problem.csv).','',
        '## Activation and diagnostics','',
        f'Activation: {a["activation"]["distinct_tasks"]}/120 distinct L task IDs ({a["activation"]["status"]}; threshold 30). '
        'Each credited task has an original successful, delivered, authenticated, replay-matched native execution. '
        'D made zero native executions. Six of eight planned diagnostics were collected; all three completed boots passed their diagnostics.','',
        '| Family | verified L task IDs | seed101 | seed102 | D/L format rejections |',
        '|---|---:|---:|---:|---:|']
    for family,item in a['families'].items():
        activated=set(a['activation']['per_family'][family])
        lines.append('| '+family+' | '+str(len(activated))+' | '+str(len(activated&set(a['activation']['per_seed']['101'])))+
            ' | '+str(len(activated&set(a['activation']['per_seed']['102'])))+' | '+
            str(item['arms']['D']['format_rejections'])+'/'+str(item['arms']['L']['format_rejections'])+' |')
    lines+=['','Per-family/per-seed attempted, blocked, successful/failed execution and result-use counts are in '
        '[the frozen analysis](report-v1/analysis.json). Successful activation does not override incomplete coverage or the failed frozen gate.','',
        '## Supplementary native tool timing','',
        'The native report exposes call_ms (translate/queue/run) and exec_ms (executor run). These descriptive means use '
        'original frozen-qualified rows only; they do not replace the frozen analysis. Complete server end-to-end time is unavailable.','',
        '| Family | D executor s/problem | L executor s/problem | D tool-call s/problem | L tool-call s/problem |',
        '|---|---:|---:|---:|---:|']
    for family in telemetry:
        cells=[]
        for field in ('native_executor_seconds','native_tool_call_seconds'):
            for arm in ('D','L'):
                value=telemetry[family][arm][field];cells.append(f(value['observed_mean'])+' (n='+str(value['known_rows'])+')')
        lines.append('| '+family+' | '+' | '.join(cells)+' |')
    lines+=['','[Timing sums, means, medians and coverage](supplementary-telemetry.json).','',
        '## Operational evidence and limits','',
        f'Cumulative operator-held GPU time: {report["budget"]["held_seconds"]:.3f}s '
        f'({report["budget"]["held_seconds"]/60:.2f}min), including boots, diagnostics, independent replay and both G restorations; overrun0s. '
        'Turn 1 QUALIFIED; turn 2 STOPPED. Budget was not exhausted. No benchmark or CPU check remains pending.',
        'CPU D/L prompt arrays matched for 242 keys; source/image/environment and sent-wire/initial-count checks are retained. '
        'Served initial arrays were not directly observed; same-length server rendering differences remain unobserved. '
        'Full planned 244 wire/count comparisons cannot qualify with 138 replies missing.',
        'Inherited declaration-control base SHA `adb05167fff21469ee7251555cb243f1f109c4d8e3de237fdcaa59ada8eb55e6`; '
        'A1 SHA `e7308621c645aef4402924a1b47c1a52e05baabe00239f915011f8431bbd0e89`.',
        'Public task manifest SHA `3e1e07d7a573eb194c0568afac2ef2112011b2b2cada9012fbbce4e6f47f7548`; '
        'reference A/B agree 120/120, zero disjointness collisions. Frozen grader/statistics and all original receipts are preserved.',
        'DeepSeek published scores in [PUBLISHED.md](../PUBLISHED.md) are labelled historical reference only. '
        'This generated task set has no matched DeepSeek published score; no vendor number is treated as a study arm.','',
        '**NEEDS-T5** — adjudicate the clipping-aware verifier and completion layout before any further collection.','']
    with (HERE/'RESULTS.md').open('x') as output:output.write('\n'.join(lines))
    provenance={'presentation_source_sha256':sha(__file__),'frozen_report_sha256':sha(HERE/'report-v1/report.json'),
        'frozen_analysis_sha256':sha(HERE/'report-v1/analysis.json'),'truncation_proof_sha256':sha(HERE/'diagnosis-v1/truncation-proof.json'),
        'results_sha256':sha(HERE/'RESULTS.md'),'per_problem_sha256':sha(HERE/'per-problem.csv'),
        'supplementary_telemetry_sha256':sha(HERE/'supplementary-telemetry.json'),'primary_analysis_changed':False}
    with (HERE/'publication-proof.json').open('x') as output:json.dump(provenance,output,indent=2)
    print(json.dumps({'status':a['status'],'observed':a['observed'],'missing':len(a['missing_identities']),
        'activation':a['activation']['distinct_tasks'],'results_sha256':provenance['results_sha256'],'primary_analysis_changed':False}))

if __name__=='__main__':main()
