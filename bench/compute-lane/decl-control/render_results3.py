#!/usr/bin/env python3
"""Render only the actual analysis3/audit artifacts to local Markdown, NEEDS-T5.

No regrading, bootstrap, endpoints, publication, tests, or report-selected cohorts.
Authored postcollection; does not change the frozen business calculations.
"""
import argparse
import hashlib
import json
from pathlib import Path


BASE_SHA = 'adb05167fff21469ee7251555cb243f1f109c4d8e3de237fdcaa59ada8eb55e6'
A1_SHA = 'e7308621c645aef4402924a1b47c1a52e05baabe00239f915011f8431bbd0e89'
ARMS = ('plain', 'declared_off', 'lane')
NAMES = {'plain': 'P', 'declared_off': 'D', 'lane': 'L'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fmt(value, scale=1):
    if value is None:
        return 'unavailable'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    return f'{value * scale:.4f}'


def ci(value, scale=1):
    return 'unavailable' if value is None else f'[{fmt(value[0], scale)}, {fmt(value[1], scale)}]'


def table(lines, headers, rows):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines.extend(['', '| ' + ' | '.join(headers) + ' |',
                  '| ' + ' | '.join('---' for _ in headers) + ' |'])
    lines.extend('| ' + ' | '.join(cell(v) for v in row) + ' |' for row in rows)
    lines.append('')


def render(finalized, output):
    require(not output.exists() and not output.with_suffix(output.suffix + '.receipt.json').exists(),
            'report and receipt paths must be new')
    receipt = load(finalized / 'finalize-receipt.json')
    for name, field in (('analysis.json', 'analysis_sha256'), ('audit.json', 'audit_sha256'),
                        ('all-final.jsonl', 'final_rows_sha256'), ('math-review-audit.json', 'review_audit_sha256')):
        require(sha(finalized / name) == receipt[field], 'finalized artifact hash mismatch')
    report = load(finalized / 'analysis.json')
    audit = load(finalized / 'audit.json')
    review = load(finalized / 'math-review-audit.json')
    rows = [json.loads(line) for line in (finalized / 'all-final.jsonl').read_text().splitlines() if line.strip()]
    canonical = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    identities = lambda r: (r['suite'], r['id'], r['seed'], r['arm'])
    require(hashlib.sha256(canonical(sorted(rows, key=identities)).encode()).hexdigest() ==
            report['graded_rows_sha256'], 'analysis/final row binding mismatch')
    require(report['spec_sha256'] == audit['spec_sha256'] == BASE_SHA and
            report['amendment_sha256'] == audit['amendment_sha256'] == A1_SHA, 'protocol mismatch')
    require(len(rows) == audit['dataset_rows'] == 552 and audit['diagnostic_rows'] == 18,
            'full cohort and diagnostic coverage required')
    require(report['pooled_suites'] is False and report['primary_family_size'] == 6,
            'registered analysis contract mismatch')
    lines = ['# Declaration-control A1 results — NEEDS-T5', '',
             f'Base protocol SHA256: `{BASE_SHA}`.', '', f'Amendment A1 SHA256: `{A1_SHA}`.', '',
             'All552 scheduled replies are retained (184 per arm): GSM8K48, MATH-50020 and original HumanEval24, '
             'seeds101/102. These public tasks and seeds were already observed. This fixed-cohort configuration study '
             'has no pooled suite score or population non-inferiority/equivalence verdict.', '',
             'Served model: DeepSeek-V4.1-Flash-1017, revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, '
             'EXL3 2.80bpw quantization. Engine/image revision `3a429d6f122d9eb2900c43e622933d311e4be44f` '
             '(`r-3a429d6f122d`). P has no declaration and execution OFF; D has the explicit Python declaration '
             'and execution OFF; L has the identical declaration/body and execution ON.', '',
             'Thinking/high, temperature1, top_p.95, top_k20, concurrency1; aggregate returned native-ID cap8192, '
             'server deadline270s and client wait360s. Three turns use P→D→L, D→L→P, L→P→D. '
             'Eighteen boot diagnostics are excluded from endpoints. Errors, timeouts, policy terminations and '
             'capped answers remain scheduled observations.', '',
             f'Independent retained-evidence audit qualified: **{fmt(audit["qualified"])}**. '
             f'Environment audit qualified: **{fmt(audit["environment"].get("qualified", False))}**. '
             'Any evidence failure withholds qualified mechanism/efficacy attribution without deleting results.', '',
             'A1 supports only “CPU-matched declaration prompts, identical request bodies and matching served prompt counts” '
             'when the corresponding audit passes. Same-length server substitutions or an unmodelled rendering branch remain '
             'possible. Directly verified served-ID equality and fully isolated execution-only causality are not established.', '',
             'Strict accuracy is primary. Returned native IDs and client end-to-end wall are descriptive secondary endpoints; '
             'wall includes generation and tool rounds. Diagnostics, boot/restoration and offline references are separate. '
             'Input+returned accounting is neither billing nor GPU-work equivalence.']
    summary = []
    for suite, entry in report['suites'].items():
        for arm in ARMS:
            a = entry['arms'][arm]
            summary.append([suite, NAMES[arm], f'{a["solved"]}/{a["scheduled"]}', fmt(a['mean_accuracy'], 100),
                            fmt(a['returned_ids']['known_mean']), f'{a["returned_ids"]["known_rows"]}/{a["scheduled"]}',
                            fmt(a['wall_seconds']['known_mean']), a['timeouts'], a['caps'], a['errors'], a['policy_terminated']])
    table(lines, ['Suite', 'Arm', 'Strict solved/scheduled', 'Accuracy %', 'Returned IDs mean', 'ID coverage',
                  'Wall seconds mean', 'Timeouts', 'Caps', 'Errors', 'Policy terminations'], summary)
    lines.append('GSM strict extraction and original HumanEval official tests plus check(entry_point) use sealed grade3. '
                 'HumanEval candidates run only inside the actual pinned rank1-host Jail. MATH labels come from all120 blinded '
                 'answers in namespace decl-control-v1: two independent reviews, third for disagreements; unresolved answers '
                 'retain adverse/favorable bounds. Reviewer types below explicitly distinguish agents from humans.')
    reviewer_types = sorted({(vote['reviewer_id'], vote['reviewer_type'])
                             for item in review['labels'] for vote in item['labels']})
    table(lines, ['MATH reviewer', 'Reviewer type'], reviewer_types)
    lines.append(f'MATH disputed candidates: {len(review["disputed_candidates"])}; third-reviewed: {review["third_reviewed"]}. '
                 'A reviewer labelled agent is an agent review, with no human-review claim.')
    accuracy = []
    bounds = []
    for suite, entry in report['suites'].items():
        for name in ('D-P', 'L-D', 'L-P'):
            c = entry['contrasts'][name]
            accuracy.append([suite, name, 'planned primary' if c['primary'] else 'secondary/marginal',
                             fmt(c['accuracy_difference_comparison_minus_base'], 100), ci(c['accuracy_ci95'], 100),
                             f'{len(c["gains"])}/{len(c["losses"])}', fmt(c['sign_flip']['p_two_sided']),
                             fmt(c['sign_flip'].get('holm_adjusted_p_family6'))])
            b = c['bounds']
            bounds.append([suite, name, canonical(b['unresolved']), canonical(b['per_arm_rate_bounds']),
                           fmt(b['difference_adverse'], 100), fmt(b['difference_favorable'], 100),
                           fmt(b['adverse_sign_flip'].get('holm_adjusted_p_family6'))])
    lines.append('Paired task clusters retain both seeds. Exact two-sided task sign-flip tests use Holm across the six '
                 'D−P/L−D primary tests. All contrasts/metrics share20000 task bootstrap draws, RNG20261104, sorted IDs, '
                 'percentile indices500/19499. L−P and GSM sensitivity are secondary/marginal. Nonsignificance is not equivalence.')
    table(lines, ['Suite', 'Contrast', 'Role', 'Accuracy Δ pp', 'CI95 pp', 'Gains/losses', 'Exact p', 'Holm family6 p'], accuracy)
    table(lines, ['Suite', 'Contrast', 'Unresolved by arm', 'Per-arm rate bounds', 'Adverse Δ pp',
                  'Favorable Δ pp', 'Adverse Holm p'], bounds)
    metrics, censoring, selected = [], [], []
    for suite, entry in report['suites'].items():
        for name, contrast in entry['contrasts'].items():
            for metric in ('returned_ids', 'initial_prompt_ids', 'processed_input_ids', 'prompt_plus_returned', 'wall_seconds'):
                m = contrast[metric]
                metrics.append([suite, name, metric, f'{m["complete_pairs"]}/{m["scheduled_pairs"]}',
                                'WITHHELD' if m['full_cohort_claim_withheld'] else fmt(m['paired_sums_ratio_comparison_over_base']),
                                ci(m['ratio_ci95']), fmt(m['paired_mean_difference_comparison_minus_base']), ci(m['difference_ci95'])])
                s = m['savings']; u = m['uncensored_complete_both']
                censoring.append([suite, name, metric, u['pairs'], fmt(u['paired_sums_ratio_comparison_over_base']),
                                  fmt(s['net']), fmt(s['positive']), fmt(s['censored_net_fraction'], 100),
                                  fmt(s['censored_positive_fraction'], 100), fmt(m['censoring_reversal']),
                                  fmt(s['over_25_percent_warning'])])
                if m['full_cohort_claim_withheld']:
                    sensitivity = m['paired_accounting_complete_sensitivity']
                    selected.append([suite, name, metric, fmt(m.get('selected_pairs_ratio')), ci(sensitivity['ratio_ci95']),
                                     fmt(m.get('selected_pairs_mean_difference')), ci(sensitivity['difference_ci95'])])
    table(lines, ['Suite', 'Contrast', 'Metric', 'Complete pairs', 'Full-cohort ratio', 'Ratio CI95', 'Mean Δ', 'Δ CI95'], metrics)
    lines.append('Missing all-round input accounting is never zero. Any missing reply withholds that comparison’s full-cohort '
                 'prompt+returned ratio/CI and total-token-saving claim. Known totals and initial counts remain separate. '
                 'Selected accounting-complete sensitivities retain original task clusters and are not representative of missing cases.')
    table(lines, ['Suite', 'Contrast', 'Selected metric', 'Selected ratio', 'Selected CI95', 'Selected mean Δ', 'Selected Δ CI95'], selected)
    table(lines, ['Suite', 'Contrast', 'Metric', 'Uncensored complete pairs', 'Uncensored ratio', 'Net savings',
                  'Positive savings', 'Censored/net %', 'Censored/positive %', 'Censor reversal', '>25% warning'], censoring)
    missingness = []
    seeds = []
    for suite, entry in report['suites'].items():
        for arm in ARMS:
            a = entry['arms'][arm]
            for metric in ('initial_prompt_ids', 'processed_input_ids', 'prompt_plus_returned', 'returned_ids'):
                m = a[metric]
                missingness.append([suite, NAMES[arm], metric, f'{m["known_rows"]}/{m["scheduled_rows"]}',
                                    fmt(m['known_sum']), fmt(m['known_mean']),
                                    'WITHHELD' if m['unknown_identities'] else 'complete'])
            for seed, values in a['per_seed'].items():
                seeds.append([suite, NAMES[arm], seed, f'{values["solved"]}/{values["scheduled"]}',
                              values['errors'], values['timeouts'], values['caps']])
    table(lines, ['Suite', 'Arm', 'Metric', 'Coverage', 'Known sum', 'Known mean', 'Full total'], missingness)
    lines.append('Missing identities and input-accounting classifications are retained in analysis.json and audit.json. '
                 'Full totals are unavailable wherever coverage is incomplete; no original prompt count is multiplied '
                 'by decode batches or imputed as resumed input.')
    table(lines, ['Suite', 'Arm', 'Seed', 'Solved/scheduled', 'Errors', 'Timeouts', 'Caps'], seeds)
    for field, title in (('activation', 'Dataset'), ('diagnostic_activation', 'Excluded diagnostics')):
        activation = []
        for item in audit[field]:
            c = item['counts']
            display = lambda name: f'{c[name]["known_sum"] if c[name]["known_sum"] is not None else "unknown"}; unknown rows={c[name]["unknown_rows"]}'
            activation.append([title, item['suite'], NAMES[item['arm']], item['seed'], item['scheduled'],
                               display('attempted'), display('blocked_dispatches'), display('successful_executions'),
                               display('failed_executions'), display('delivered_authenticated_results'),
                               display('reply_used_result'), item['label']])
        table(lines, ['Scope', 'Suite', 'Arm', 'Seed', 'Replies', 'Attempts', 'Prevented D attempts', 'Successful executions',
                      'Failed executions', 'Authenticated results', 'Replies using result', 'Activation label'], activation)
    lines.append('D prevented-call counts are source-inferred attempts under absent-pool configuration, not deployed '
                 'dispatch-spy counters. Unknown coverage is explicit. DECLARED BUT UNUSED applies only where all L '
                 'execution counts are known and zero; no offloading benefit is demonstrated there. Calls using subgroups '
                 'are exploratory and never replace the scheduled cohort.')
    table(lines, ['Boot', 'Arm', 'Verified native/reference diagnostic calls', 'HTTP audit', 'Boot evidence qualified'],
          [[b['boot'], NAMES[b['arm']], b['verified_diagnostic_calls'], fmt(b['http']['qualified']), fmt(b['qualified'])]
           for b in audit['boots']])
    table(lines, ['Turn', 'Boot', 'Admission/load to ready seconds', 'Separate G restoration seconds', 'G restored run'],
          [[t['turn'], b['boot'], fmt(b['admission_load_seconds']), fmt(t['G_restore_seconds']), t['serving_run']]
           for t in audit['turns'] for b in t['boots']])
    lines.append('Restoration time is one interval per turn, repeated for its three boot rows for context; do not sum '
                 'the repeated column. Times derive from retained steps, with G restoration requiring SERVING_VERIFIED '
                 'after release and POPPED in the restore log, never outer return code alone.')
    secondary = []
    changes = []
    for suite, entry in report['suites'].items():
        for name, c in entry['contrasts'].items():
            s = c.get('gsm_sensitivity_secondary')
            if s:
                secondary.append([suite, name, fmt(s['accuracy_difference_comparison_minus_base'], 100),
                                  ci(s['accuracy_ci95'], 100), fmt(s['sign_flip']['p_two_sided']),
                                  f'{len(s["gains"])}/{len(s["losses"])}'])
    for item in report['gsm_sensitivity_audit']:
        secondary_grade = item['secondary']['grade']
        changes.append([canonical(item['identity']), item['strict_grade'], secondary_grade,
                        'changed' if item['secondary'].get('changed') else 'residual incorrect' if secondary_grade != 'correct' else 'strict preserved'])
    lines.append('GSM ONE-NUMERAL FINAL-SPAN is the preregistered secondary extraction sensitivity; strict accuracy '
                 'remains primary. Every changed and residual grade appears below; the exact extracted spans remain in analysis.json.')
    table(lines, ['Suite', 'Contrast', 'Secondary accuracy Δ pp', 'CI95 pp', 'Marginal exact p', 'Gains/losses'], secondary)
    table(lines, ['GSM identity', 'Strict', 'Secondary', 'Disposition'], changes)
    table(lines, ['Suite', 'Contrast', 'Numerical local-efficiency conditions', 'Numerical total-token conditions', 'Qualified claim'],
          [[suite, name, fmt(c['local_efficiency_numerical_conditions_met']),
            fmt(c['total_token_saving_numerical_conditions_met']), 'withheld / NEEDS-T5']
           for suite, entry in report['suites'].items() for name, c in entry['contrasts'].items()])
    lines.append('Numerical flags are the frozen analysis outputs, not a safety or serving-default verdict. Qualified '
                 'local efficiency also requires full safety/accounting, no net primary solve decrease, returned-ID/wall '
                 'CI uppers<1, no censor reversal and ≤25% net/positive censor contribution. Total-token savings additionally '
                 'require a complete prompt+returned ratio CI upper<1. Where L/D is flat, use “no detected incremental '
                 'execution-enabled effect on this cohort”; neither a flat result nor a nonsignificant test establishes equivalence.')
    published = '../PUBLISHED.md'
    lines.extend(['', 'Published reference comparison — REFERENCE ONLY. Retrieved2026-11-04; '
                  f'see [PUBLISHED.md]({published}), [September Flash card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md), '
                  '[Pro card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/README.md), and '
                  '[exact1017 revision card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/README.md).'])
    table(lines, ['Checkpoint / protocol', 'GSM8K', 'MATH / MATH-500', 'Original HumanEval'], [
        ['Exact Flash-1017 instruct (our3 suites)', 'N/A — not published', 'MATH-500: N/A — not published', 'N/A — not published'],
        ['September V4.1-Flash-BASE — historical REFERENCE ONLY', '93.0 exact match,8-shot', '61.1 exact match,4-shot; MATH, NOT MATH-500', '79.4 pass@1,0-shot'],
        ['V4-Pro-BASE preview-card — historical REFERENCE ONLY', '92.6 exact match,8-shot', '64.5 exact match,4-shot; MATH, NOT MATH-500', '76.8 pass@1,0-shot']])
    lines.append('The historical references use BASE checkpoints/internal evaluation, not Pro GA instruct, not our '
                 'quantization and not matched local baselines. Sampling, generation counts and output caps are not '
                 'published there. No retention target, accuracy-loss inference or token/time baseline is derived from them.')
    lines.extend(['', f'Freeze manifest SHA256: `{audit["freeze_sha256"]}`. '
                  f'CPU receipt SHA256: `{audit["receipt_sha256"]}`. '
                  f'Graded rows canonical SHA256: `{report["graded_rows_sha256"]}`.', '',
                  'The post3/render_results3 wrappers are postcollection I/O/reporting additions, outside the87-file '
                  'precollection seal. Frozen grade3/analysis3/math3 and the outcome mechanism remain unchanged. '
                  'This report is local only; status **NEEDS-T5**.'])
    with output.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    validation = {'status': 'NEEDS-T5', 'renderer_sha256': sha(__file__), 'report_sha256': sha(output),
                  'analysis_sha256': sha(finalized / 'analysis.json'), 'audit_sha256': sha(finalized / 'audit.json'),
                  'base_spec_sha256': BASE_SHA, 'amendment_sha256': A1_SHA,
                  'dataset_rows': 552, 'diagnostic_rows': 18, 'postcollection_wrapper': True}
    with output.with_suffix(output.suffix + '.receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(validation, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalized', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path, help='new local Markdown path')
    args = parser.parse_args()
    render(args.finalized, args.out)
    print(json.dumps({'status': 'NEEDS-T5', 'report_saved': True}))


if __name__ == '__main__':
    main()
