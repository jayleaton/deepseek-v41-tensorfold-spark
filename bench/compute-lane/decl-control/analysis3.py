#!/usr/bin/env python3
"""Offline three-arm analysis. No collection, endpoints, or implicit qualification.

analyze(rows, manifest): exact full scheduled grid only; descriptive outputs never
assert safety. input_accounting schema: complete=true, processed_input_tokens=N,
qualified=true, provenance=<pinned source/control-flow evidence>. Single-input
fallback requires classification='single_input', source_control_flow_verified=true,
no_additional_input=true, provenance, and qualified initial reconciliation.
initial_prompt_reconciliation: qualified=true, initial_prompt_tokens=int.
Native and call evidence are supplied by the parent's independent audited adapter.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'phase2'))
import analysis as phase2
from analysis import (SEEDS, DRAWS, canonical, digest, identity, manifest_tasks,
                      read_jsonl, write_json, number, correct, unresolved, censored)

ARMS = ('plain', 'declared_off', 'lane')
CONTRASTS = {'D-P': ('plain', 'declared_off'), 'L-D': ('declared_off', 'lane'),
             'L-P': ('plain', 'lane')}
SPEC_SHA256 = 'adb05167fff21469ee7251555cb243f1f109c4d8e3de237fdcaa59ada8eb55e6'
A1_SHA256 = 'e7308621c645aef4402924a1b47c1a52e05baabe00239f915011f8431bbd0e89'
PHASE2_SHA256 = '87871688d4aea10d0c8f2f8c7bd2d9ee1f4a759bbbe5e57bb8403127542780df'


def count(value):
    return type(value) is int and value >= 0


def initial_tokens(row):
    evidence = row.get('initial_prompt_reconciliation') or {}
    value = evidence.get('initial_prompt_tokens')
    return value if evidence.get('qualified') is True and count(value) else None


def input_tokens(row):
    evidence = row.get('input_accounting') or {}
    value = evidence.get('processed_input_tokens')
    if (evidence.get('qualified') is True and evidence.get('complete') is True
            and evidence.get('provenance') and count(value)):
        return value
    if (evidence.get('classification') == 'single_input'
            and evidence.get('source_control_flow_verified') is True
            and evidence.get('no_additional_input') is True and evidence.get('provenance')):
        return initial_tokens(row)
    return None


def total_tokens(row):
    initial, returned = input_tokens(row), phase2.native_tokens(row)
    return initial + returned if initial is not None and returned is not None else None


def validate_cohort(rows, manifest, *, _trusted_selftest=False):
    tasks = manifest_tasks(manifest)
    if manifest.get('seeds') != list(SEEDS):
        raise ValueError('seeds must be [101,102]')
    keys = [(t['suite'], t['id']) for t in tasks]
    if not keys or len(keys) != len(set(keys)) or any(
            not isinstance(x, str) or not x for key in keys for x in key):
        raise ValueError('invalid/duplicate tasks')
    if 'tasks_sha256' in manifest and digest(tasks) != manifest['tasks_sha256']:
        raise ValueError('task manifest hash mismatch')
    if not _trusted_selftest:
        kinds = Counter(t.get('kind') for t in tasks)
        if (len(tasks) != 92 or kinds['integer'] + kinds['numeric'] != 48
                or kinds['math_manual'] != 20 or kinds['humaneval'] != 24
                or sorted(Counter(s for s, _ in keys).values()) != [20, 24, 48]):
            raise ValueError('registered 48/20/24 cohort required')
    expected = {(s, i, seed, arm) for s, i in keys for seed in SEEDS for arm in ARMS}
    indexed = {}
    for row in rows:
        key = identity(row)
        if (key not in expected or key in indexed or type(row['seed']) is not int
                or row.get('diagnostic') or row.get('included_in_cohort') is False):
            raise ValueError('unexpected/duplicate cohort row')
        if row.get('grade') not in ('correct', 'incorrect', 'manual_review'):
            raise ValueError('canonical grade required')
        if not isinstance(row.get('status'), str) or not row['status']:
            raise ValueError('observed status required')
        if not number(row.get('elapsed_seconds')) or 'finish_reason' not in row:
            raise ValueError('observed elapsed/finish required even for failures')
        native = row.get('native_accounting')
        if native is not None:
            if not isinstance(native, dict) or type(native.get('complete')) is not bool:
                raise ValueError('invalid native evidence')
            value = native.get('returned_tokens')
            if (value is not None and not count(value)) or (native['complete'] and value is None):
                raise ValueError('native count must be integer; absent never zero')
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError('missing scheduled replies')
    return indexed


def unresolved_bounds(clusters):
    """Pair-local bounds; used by blinded review for each registered contrast."""
    return phase2.unresolved_bounds(clusters)


def metric_report(clusters, value):
    report, totals = phase2.metric_summary(clusters, value)
    ratios, differences = [], []
    # Replaying this exact iterator intentionally shares task draws with suite_report
    # and every other metric/contrast; missing pairs never remove task clusters.
    for indices in phase2.bootstrap_indices(len(clusters)):
        base = sum(totals[i][0] for i in indices)
        comparison = sum(totals[i][1] for i in indices)
        n = sum(totals[i][2] for i in indices)
        ratios.append(comparison / base if base else None)
        differences.append((comparison - base) / n if n else None)
    report['full_cohort_claim_withheld'] = report['complete_pairs'] != report['scheduled_pairs']
    for arm in ('plain', 'lane'):
        report[arm]['coverage'] = 'paired accounting complete observations only'
        if report['complete_pairs'] == 0:
            report[arm]['sum'] = None
    sensitivity = {'selection': 'paired accounting complete; original task clusters retained',
                   'ratio_ci95': phase2.percentile_interval(ratios),
                   'difference_ci95': phase2.percentile_interval(differences)}
    report['paired_accounting_complete_sensitivity'] = sensitivity
    report['ratio_ci95'] = None if report['full_cohort_claim_withheld'] else sensitivity['ratio_ci95']
    report['difference_ci95'] = None if report['full_cohort_claim_withheld'] else sensitivity['difference_ci95']
    if report['full_cohort_claim_withheld']:
        report['selected_pairs_ratio'] = report['paired_sums_ratio_lane_over_plain']
        report['paired_sums_ratio_lane_over_plain'] = None
        report['selected_pairs_mean_difference'] = report['paired_mean_difference_lane_minus_plain']
        report['paired_mean_difference_lane_minus_plain'] = None
    return report


def arm_summary(rows, *, _include_seeds=True):
    result = {'scheduled': len(rows), 'solved': sum(map(correct, rows)),
              'mean_accuracy': sum(map(correct, rows)) / len(rows),
              'statuses': dict(Counter(r['status'] for r in rows)),
              'finish_reasons': dict(Counter(str(r['finish_reason']) for r in rows)),
              'errors': sum(r['status'] != 'ok' for r in rows),
              'timeouts': sum(r['status'] == 'timeout' or r['finish_reason'] in ('timeout', 'deadline') for r in rows),
              'caps': sum(r['finish_reason'] in ('length', 'max_tokens', 'max_output_tokens') for r in rows),
              'policy_terminated': sum(r['status'] == 'policy_terminated' for r in rows)}
    for name, value in [('returned_ids', phase2.native_tokens), ('initial_prompt_ids', initial_tokens),
                        ('processed_input_ids', input_tokens), ('prompt_plus_returned', total_tokens),
                        ('wall_seconds', lambda r: r['elapsed_seconds'])]:
        observed = [(identity(r), value(r)) for r in rows]
        known = [v for _, v in observed if v is not None]
        result[name] = {'known_rows': len(known), 'scheduled_rows': len(rows),
                        'known_sum': sum(known) if known else None,
                        'known_mean': sum(known) / len(known) if known else None,
                        'unknown_identities': [list(k) for k, v in observed if v is None]}
    # No absence-as-zero: each count has independently explicit coverage.
    result['calls'] = {}
    for field in ('attempted', 'blocked_dispatches', 'successful_executions', 'failed_executions',
                  'delivered_authenticated_results', 'reply_used_result'):
        known = [r['call_accounting'][field] for r in rows
                 if isinstance(r.get('call_accounting'), dict)
                 and (r['call_accounting'].get('complete') is True
                      or (r['call_accounting'].get('coverage') or {}).get(field) is True)
                 and count(r['call_accounting'].get(field))]
        result['calls'][field] = {'known_rows': len(known), 'unknown_rows': len(rows) - len(known),
                                 'known_sum': sum(known) if known else None}
    if _include_seeds:
        result['per_seed'] = {str(seed): arm_summary([r for r in rows if r['seed'] == seed],
                                                    _include_seeds=False) for seed in SEEDS}
    return result


def neutral_names(value):
    if isinstance(value, dict):
        return {k.replace('lane', 'comparison').replace('plain', 'base'): neutral_names(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [neutral_names(v) for v in value]
    return value


def analyze(rows, manifest, *, _trusted_selftest=False):
    if hashlib.sha256(Path(phase2.__file__).read_bytes()).hexdigest() != PHASE2_SHA256:
        raise ValueError('frozen phase2 analysis source changed')
    indexed = validate_cohort(rows, manifest, _trusted_selftest=_trusted_selftest)
    tasks = manifest_tasks(manifest)
    reports, primary, adverse = {}, {}, {}
    for suite in sorted({t['suite'] for t in tasks}):
        ids = sorted(t['id'] for t in tasks if t['suite'] == suite)
        suite_rows = [r for r in rows if r['suite'] == suite]
        entry = {'arms': {a: arm_summary([r for r in suite_rows if r['arm'] == a]) for a in ARMS},
                 'contrasts': {}}
        for name, (base, comparison) in CONTRASTS.items():
            clusters = [[(indexed[suite, i, seed, base], indexed[suite, i, seed, comparison])
                         for seed in SEEDS] for i in ids]
            r = phase2.suite_report(clusters, ids)
            for metric, value in [('returned_ids', phase2.native_tokens),
                                  ('initial_prompt_ids', initial_tokens), ('processed_input_ids', input_tokens),
                                  ('prompt_plus_returned', total_tokens),
                                  ('wall_seconds', lambda row: row['elapsed_seconds'])]:
                r[metric] = metric_report(clusters, value)
            r.pop('tokens'); r.pop('time')
            r.pop('plain'); r.pop('lane')
            r.update(base_arm=base, comparison_arm=comparison, primary=name != 'L-P',
                     qualified_efficacy_claim=False,
                     qualification_note='Independent prompt, dispatch, safety and review audit required; A1 claim limits apply.')
            if name != 'L-P':
                primary[suite + '/' + name] = r['sign_flip']['p_two_sided']
                adverse[suite + '/' + name] = r['bounds']['adverse_sign_flip']['p_two_sided']
            if all('gsm_sensitivity' in p and 'gsm_sensitivity' in l for c in clusters for p, l in c):
                secondary = [[(dict(p, grade=p['gsm_sensitivity']['grade']),
                               dict(l, grade=l['gsm_sensitivity']['grade'])) for p, l in c] for c in clusters]
                secondary_report = phase2.suite_report(secondary, ids)
                r['gsm_sensitivity_secondary'] = {
                    key: secondary_report[key] for key in
                    ('accuracy_difference_lane_minus_plain', 'accuracy_ci95', 'sign_flip',
                     'bounds', 'gains', 'losses', 'task_correct_count_differences')}
                r['gsm_sensitivity_secondary']['marginal_secondary'] = True
            entry['contrasts'][name] = neutral_names(r)
        reports[suite] = entry
    adjusted, adverse_adjusted = phase2.holm(primary), phase2.holm(adverse)
    for suite, entry in reports.items():
        for name, r in entry['contrasts'].items():
            if name != 'L-P':
                r['sign_flip']['holm_adjusted_p_family6'] = adjusted[suite + '/' + name]
                r['bounds']['adverse_sign_flip']['holm_adjusted_p_family6'] = adverse_adjusted[suite + '/' + name]
            efficiency = all(not r[m]['full_cohort_claim_withheld']
                             and r[m]['ratio_ci95'] is not None and r[m]['ratio_ci95'][1] < 1
                             and not r[m]['censoring_reversal']
                             and not r[m]['savings']['over_25_percent_warning']
                             for m in ('returned_ids', 'wall_seconds'))
            r['local_efficiency_numerical_conditions_met'] = r['bounds']['difference_adverse'] >= 0 and efficiency
            r['qualified_local_efficiency'] = False
            r['total_token_saving_numerical_conditions_met'] = (
                r['local_efficiency_numerical_conditions_met']
                and not r['prompt_plus_returned']['full_cohort_claim_withheld']
                and r['prompt_plus_returned']['ratio_ci95'] is not None
                and r['prompt_plus_returned']['ratio_ci95'][1] < 1
                and not r['prompt_plus_returned']['censoring_reversal']
                and not r['prompt_plus_returned']['savings']['over_25_percent_warning'])
    return {'spec_sha256': SPEC_SHA256, 'amendment_sha256': A1_SHA256,
            'manifest_sha256': digest(manifest), 'graded_rows_sha256': digest(sorted(rows, key=identity)),
            'bootstrap': {'draws': DRAWS, 'rng': 20261104, 'percentile_indices': [500, 19499],
                          'shared_all_contrasts_metrics': True, 'unit': 'sorted task IDs, both seeds'},
            'primary_family_size': 6, 'pooled_suites': False, 'suites': reports,
            'gsm_sensitivity_audit': [{'identity': list(identity(r)), 'strict_grade': r['grade'],
                                      'secondary': r['gsm_sensitivity']} for r in sorted(rows, key=identity)
                                     if 'gsm_sensitivity' in r],
            'claim_limit': 'CPU-matched declaration prompts, identical request bodies and matching served prompt counts require independent audit; no directly verified served-ID equality.',
            'interpretation': 'No detected incremental execution-enabled effect on this cohort when L/D is flat; nonsignificance is not equivalence.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--inputlist', nargs='+', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    write_json(a.out, analyze([r for path in a.inputlist for r in read_jsonl(path)],
                              json.loads(Path(a.manifest).read_text())))


if __name__ == '__main__':
    main()
