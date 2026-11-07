#!/usr/bin/env python3
"""Offline T5 paired analysis. No requests, model code, or third-party dependencies."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics

SPEC_SHA256 = '9c7327007cfd15825a608f2f6b06f2b39e33b88548c5bb7deb1b209b7395d240'
ARMS = ('plain', 'lane')
SEEDS = (101, 102)
DRAWS = 20000


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def read_jsonl(path):
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    with open(path, 'x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def identity(row):
    return row['suite'], row['id'], row['seed'], row['arm']


def number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def manifest_tasks(manifest):
    """Accept kit flat manifests or freeze.py analysis-{arm}.json manifests."""
    if 'tasks' in manifest:
        return manifest['tasks']
    tasks = []
    for suite, item in manifest['suites'].items():
        if digest(item['tasks']) != item['tasks_sha256']:
            raise ValueError('frozen suite task hash mismatch')
        if any(t['suite'] != suite for t in item['tasks']):
            raise ValueError('frozen suite identity mismatch')
        tasks.extend(item['tasks'])
    return tasks


def validate_cohort(rows, manifest, *, _trusted_selftest=False):
    """Require the exact cross product, including unsuccessful observations."""
    tasks, seeds = manifest_tasks(manifest), manifest['seeds']
    if seeds != list(SEEDS):
        raise ValueError('manifest seeds must be [101, 102]')
    task_keys = []
    for task in tasks:
        key = task['suite'], task['id']
        if not all(isinstance(x, str) and x for x in key):
            raise ValueError('nonempty string suite/id required')
        task_keys.append(key)
    if not task_keys or len(task_keys) != len(set(task_keys)):
        raise ValueError('empty or duplicate manifest tasks')
    if not _trusted_selftest and sorted(Counter(s for s, _ in task_keys).values()) != [20, 24, 48]:
        raise ValueError('registered cohort requires three suites of 20/24/48 tasks')
    if 'tasks_sha256' in manifest and manifest['tasks_sha256'] != digest(tasks):
        raise ValueError('manifest tasks hash mismatch')
    expected = {(s, i, seed, arm) for s, i in task_keys for seed in seeds for arm in ARMS}
    indexed = {}
    for row in rows:
        key = identity(row)
        if type(row['seed']) is not int:
            raise ValueError('integer generation seed required')
        if key not in expected or key in indexed:
            raise ValueError('unexpected/duplicate response identity: ' + repr(key))
        if row.get('grade') not in ('correct', 'incorrect', 'manual_review'):
            raise ValueError('invalid canonical grade')
        if not isinstance(row.get('status'), str) or not row['status']:
            raise ValueError('status required')
        if not number(row.get('elapsed_seconds')):
            raise ValueError('finite observed elapsed_seconds required, including errors')
        if 'finish_reason' not in row or 'tokens' not in row:
            raise ValueError('canonical finish_reason and tokens fields required')
        accounting = row.get('native_accounting')
        if accounting is not None:
            if not isinstance(accounting, dict) or not isinstance(accounting.get('complete'), bool):
                raise ValueError('native_accounting.complete must be boolean')
            value = accounting.get('returned_tokens')
            if value is not None and (not number(value) or not isinstance(value, int)):
                raise ValueError('native returned_tokens must be a nonnegative integer or null')
            if accounting['complete'] and value is None:
                raise ValueError('complete native accounting requires returned_tokens')
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError('missing scheduled replies: ' + repr(sorted(expected - set(indexed))[:8]))
    return indexed


def correct(row):
    # Length censoring does not override an actually correct delivered final answer.
    return int(row['status'] == 'ok' and row['grade'] == 'correct')


def unresolved(row):
    return row['status'] == 'ok' and row['grade'] == 'manual_review'


def censored(row):
    return (row.get('censored') is True or row.get('finish_reason') in
            ('length', 'max_tokens', 'max_output_tokens', 'deadline', 'timeout')
            or row['status'] == 'timeout')


def exact_sign_flip(differences):
    """Integer DP over ±abs(d_i); zero weights cancel from numerator/denominator."""
    if any(not isinstance(d, int) or isinstance(d, bool) or abs(d) > 2 for d in differences):
        raise ValueError('two-seed correctness differences must be integers in [-2, 2]')
    distribution = {0: 1}
    for weight in (abs(d) for d in differences if d):
        updated = Counter()
        for total, count in distribution.items():
            updated[total - weight] += count
            updated[total + weight] += count
        distribution = dict(updated)
    observed = abs(sum(differences))
    numerator = sum(count for total, count in distribution.items() if abs(total) >= observed)
    denominator = sum(distribution.values())
    return {'p_two_sided': numerator / denominator, 'tail_assignments': numerator,
            'assignments': denominator, 'observed_abs_sum': observed,
            'nonzero_tasks': sum(d != 0 for d in differences)}


def holm(pvalues):
    """Return monotone Holm adjusted p-values, with deterministic tie ordering."""
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    previous, adjusted = 0.0, {}
    for index, key in enumerate(ordered):
        value = pvalues[key]
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError('p-value outside [0, 1]')
        previous = max(previous, min(1.0, (len(ordered) - index) * value))
        adjusted[key] = previous
    return adjusted


def bootstrap_indices(n):
    """Fresh RNG per suite; caller orders task IDs before using this iterator."""
    rng = random.Random(20261104)
    for _ in range(DRAWS):
        yield [rng.randrange(n) for _ in range(n)]


def percentile_interval(values):
    if len(values) != DRAWS or any(v is None for v in values):
        return None  # Do not silently condition on nonempty/positive-denominator draws.
    values = sorted(values)
    return [values[500], values[19499]]


def native_tokens(row, complete_only=True):
    accounting = row.get('native_accounting') or {}
    if complete_only and accounting.get('complete') is not True:
        return None
    return accounting.get('returned_tokens')


def metric_summary(clusters, value):
    """Select complete-both task×seed pairs; preserve cluster membership in draws."""
    selected, savings, censored_savings = [], [], []
    for cluster in clusters:
        pairs = []
        for plain, lane in cluster:
            p, l = value(plain), value(lane)
            if p is None or l is None:
                continue
            pairs.append((p, l))
            savings.append(p - l)
            censored_savings.append(p - l if censored(plain) or censored(lane) else 0)
        selected.append(pairs)
    pairs = [pair for cluster in selected for pair in cluster]
    net = sum(savings)
    positive = sum(max(0, saving) for saving in savings)
    censor_net = sum(censored_savings)
    censor_positive = sum(max(0, saving) for saving in censored_savings)
    net_fraction = censor_net / net if net > 0 else None
    positive_fraction = censor_positive / positive if positive > 0 else None
    summary = {
        'complete_pairs': len(pairs), 'scheduled_pairs': sum(map(len, clusters)),
        'selection': 'both arms have observed metric; retain errors and timeouts',
        'savings': {'net': net, 'positive': positive, 'censored_net': censor_net,
                    'censored_positive': censor_positive, 'censored_net_fraction': net_fraction,
                    'censored_positive_fraction': positive_fraction,
                    'over_25_percent_warning': any(x is not None and x > .25
                                                   for x in (net_fraction, positive_fraction))}}
    for index, arm in enumerate(ARMS):
        values = [pair[index] for pair in pairs]
        summary[arm] = {'sum': sum(values), 'mean': statistics.mean(values) if values else None,
                        'median': statistics.median(values) if values else None}
    psum, lsum = summary['plain']['sum'], summary['lane']['sum']
    summary['paired_sums_ratio_lane_over_plain'] = lsum / psum if psum else None
    summary['paired_mean_difference_lane_minus_plain'] = ((lsum - psum) / len(pairs)
                                                         if pairs else None)
    # Uncensored sensitivity uses the same paired completeness selection.
    uncensored = [(value(p), value(l)) for cluster in clusters for p, l in cluster
                  if p['status'] == l['status'] == 'ok' and not censored(p) and not censored(l)
                  and value(p) is not None and value(l) is not None]
    up, ul = sum(p for p, l in uncensored), sum(l for p, l in uncensored)
    summary['uncensored_complete_both'] = {
        'pairs': len(uncensored), 'paired_sums_ratio_lane_over_plain': ul / up if up else None,
        'selection': 'both generations ok, uncensored, and both metric values observed',
        'paired_mean_difference_lane_minus_plain': (ul - up) / len(uncensored) if uncensored else None}
    summary['censoring_reversal'] = bool(psum and lsum < psum and (not up or ul >= up))
    totals = [(sum(p for p, l in cluster), sum(l for p, l in cluster), len(cluster))
              for cluster in selected]
    return summary, totals


def unresolved_bounds(clusters):
    rows = {arm: [pair[index] for cluster in clusters for pair in cluster]
            for index, arm in enumerate(ARMS)}
    n = len(rows['plain'])
    counts = {arm: sum(map(correct, arm_rows)) for arm, arm_rows in rows.items()}
    pending = {arm: sum(map(unresolved, arm_rows)) for arm, arm_rows in rows.items()}
    difference = counts['lane'] - counts['plain']
    return {'label': 'verified-correct rate' if any(pending.values()) else 'resolved correct rate',
            'unresolved': pending,
            'per_arm_rate_bounds': {a: [counts[a] / n, (counts[a] + pending[a]) / n] for a in ARMS},
            'difference_adverse': (difference - pending['plain']) / n,
            'difference_favorable': (difference + pending['lane']) / n}


def suite_report(clusters, task_ids):
    n = len(task_ids) * 2
    differences = [sum(correct(l) - correct(p) for p, l in cluster) for cluster in clusters]
    report = {'tasks': len(task_ids), 'task_ids': task_ids, 'scheduled_per_arm': n,
              'accuracy_difference_lane_minus_plain': sum(differences) / n,
              'task_correct_count_differences': dict(zip(task_ids, differences)),
              'sign_flip': exact_sign_flip(differences), 'bounds': unresolved_bounds(clusters)}
    report['accuracy_label'] = report['bounds']['label']
    report['gains'] = [{'id': task_ids[i], 'seed': p['seed']} for i, cluster in enumerate(clusters)
                       for p, l in cluster if correct(l) > correct(p)]
    report['losses'] = [{'id': task_ids[i], 'seed': p['seed']} for i, cluster in enumerate(clusters)
                        for p, l in cluster if correct(p) > correct(l)]
    for index, arm in enumerate(ARMS):
        rows = [pair[index] for cluster in clusters for pair in cluster]
        report[arm] = {'solved': sum(map(correct, rows)), 'rate': sum(map(correct, rows)) / n,
                       'grade_counts': dict(Counter(r['grade'] for r in rows)),
                       'status_counts': dict(Counter(r['status'] for r in rows)),
                       'errors': sum(r['status'] != 'ok' for r in rows),
                       'error_rate': sum(r['status'] != 'ok' for r in rows) / n,
                       'censored': sum(map(censored, rows)),
                       'censor_rate': sum(map(censored, rows)) / n,
                       'per_seed': {str(seed): {'solved': sum(correct(r) for r in rows if r['seed'] == seed),
                                               'scheduled': len(task_ids),
                                               'errors': sum(r['status'] != 'ok' for r in rows if r['seed'] == seed),
                                               'censored': sum(censored(r) for r in rows if r['seed'] == seed)}
                                    for seed in SEEDS}}
        usage = {}
        for field in ('prompt_tokens', 'completion_tokens', 'reasoning_tokens', 'content_tokens', 'total_tokens'):
            values = [(r.get('tokens') or {}).get(field) for r in rows
                      if isinstance(r.get('tokens'), dict)]
            observed = [v for v in values if number(v)]
            usage[field] = {'observed_rows': len(observed),
                            'observed_sum': sum(observed) if observed else None}
        report[arm]['api_usage_diagnostics_not_native_endpoint'] = usage
    token_report, token_totals = metric_summary(clusters, native_tokens)
    time_report, time_totals = metric_summary(clusters, lambda r: r['elapsed_seconds'])
    for index, arm in enumerate(ARMS):
        rows = [pair[index] for cluster in clusters for pair in cluster]
        observed = [native_tokens(r, False) for r in rows]
        token_report[arm]['native_complete_rows'] = sum(native_tokens(r) is not None for r in rows)
        token_report[arm]['native_complete_coverage'] = sum(native_tokens(r) is not None for r in rows) / n
        token_report[arm]['observed_returned_tokens_lower_bound'] = sum(v for v in observed if v is not None)
        token_report[arm]['observed_rows'] = sum(v is not None for v in observed)
    token_report['full_cohort_claim_withheld'] = token_report['complete_pairs'] != n
    token_report['completion_usage_is_not_native_accounting'] = True
    report['tokens'], report['time'] = token_report, time_report
    sampled = {'accuracy': [], 'tokens_ratio': [], 'tokens_diff': [], 'time_ratio': [], 'time_diff': [],
               'adverse_accuracy': []}
    adverse = [sum(correct(l) - correct(p) - int(unresolved(p)) for p, l in c) for c in clusters]
    # Each draw is shared across accuracy, native tokens, and elapsed time.
    for indices in bootstrap_indices(len(task_ids)):
        sampled['accuracy'].append(sum(differences[i] for i in indices) / n)
        sampled['adverse_accuracy'].append(sum(adverse[i] for i in indices) / n)
        for name, totals in (('tokens', token_totals), ('time', time_totals)):
            p = sum(totals[i][0] for i in indices)
            l = sum(totals[i][1] for i in indices)
            count = sum(totals[i][2] for i in indices)
            sampled[name + '_ratio'].append(l / p if p else None)
            sampled[name + '_diff'].append((l - p) / count if count else None)
    report['accuracy_ci95'] = percentile_interval(sampled['accuracy'])
    report['bounds']['adverse_accuracy_ci95'] = percentile_interval(sampled['adverse_accuracy'])
    for name in ('tokens', 'time'):
        report[name]['ratio_ci95'] = percentile_interval(sampled[name + '_ratio'])
        report[name]['difference_ci95'] = percentile_interval(sampled[name + '_diff'])
    adverse_d = [sum(correct(l) - correct(p) - int(unresolved(p)) for p, l in c) for c in clusters]
    report['bounds']['adverse_sign_flip'] = exact_sign_flip(adverse_d)
    return report


def analyze(rows, manifest, *, _trusted_selftest=False):
    indexed = validate_cohort(rows, manifest, _trusted_selftest=_trusted_selftest)
    tasks = manifest_tasks(manifest)
    suites = {}
    for suite in sorted({t['suite'] for t in tasks}):
        ids = sorted(t['id'] for t in tasks if t['suite'] == suite)
        clusters = [[(indexed[suite, task_id, seed, 'plain'], indexed[suite, task_id, seed, 'lane'])
                     for seed in SEEDS] for task_id in ids]
        suites[suite] = suite_report(clusters, ids)
    adjusted = holm({s: r['sign_flip']['p_two_sided'] for s, r in suites.items()})
    adverse_adjusted = holm({s: r['bounds']['adverse_sign_flip']['p_two_sided'] for s, r in suites.items()})
    for suite, report in suites.items():
        report['sign_flip']['holm_adjusted_p'] = adjusted[suite]
        report['bounds']['adverse_sign_flip']['holm_adjusted_p'] = adverse_adjusted[suite]
        report['primary_statistical_benefit_supported'] = (report['bounds']['difference_adverse'] > 0
                                                          and adverse_adjusted[suite] <= .05)
        report['local_efficiency_numerical_conditions_met'] = (
            report['bounds']['difference_adverse'] >= 0
            and not report['tokens']['full_cohort_claim_withheld']
            and all(r['paired_sums_ratio_lane_over_plain'] is not None
                    and r['paired_sums_ratio_lane_over_plain'] < 1
                    and r['ratio_ci95'] is not None and r['ratio_ci95'][1] < 1
                    and not r['censoring_reversal']
                    and not r['savings']['over_25_percent_warning']
                    for r in (report['tokens'], report['time'])))
        # An analysis file cannot authenticate native execution or adjudication safety itself.
        report['qualified_efficacy_claim'] = False
        report['qualification_note'] = 'Requires independent safety/provenance and sealed MATH review audit.'
    return {'spec_sha256': SPEC_SHA256, 'manifest_sha256': digest(manifest),
            'graded_rows_sha256': digest(sorted(rows, key=identity)), 'pooled_suites': False,
            'bootstrap': {'draws': DRAWS, 'rng': 20261104, 'percentile_indices': [500, 19499],
                          'unit': 'sorted task IDs; both seeds and arms retained'},
            'familywise_alpha': .05, 'suites': suites,
            'interpretation': 'Short public subsets; nonsignificance does not establish non-inferiority.'}


def selftest():
    for weights in ([], [0, 0], [1], [1, -1], [2, -1, 0], [2, 2, -1, -2]):
        totals = [sum(sign * abs(w) for sign, w in zip(signs, weights))
                  for signs in itertools.product((-1, 1), repeat=len(weights))]
        expected = sum(abs(v) >= abs(sum(weights)) for v in totals) / len(totals)
        assert exact_sign_flip(weights)['p_two_sided'] == expected
    assert exact_sign_flip([1, -1])['p_two_sided'] == 1
    assert holm({'a': .01, 'b': .04, 'c': .03}) == {'a': .03, 'c': .06, 'b': .06}
    assert list(itertools.islice(bootstrap_indices(3), 10)) == list(itertools.islice(bootstrap_indices(3), 10))
    assert percentile_interval(list(range(DRAWS))) == [500, 19499]
    tasks = [{'suite': 'fixture', 'id': str(i)} for i in range(2)]
    rows = [{'suite': t['suite'], 'id': t['id'], 'seed': seed, 'arm': arm, 'grade': 'correct',
             'status': 'ok', 'finish_reason': 'length' if arm == 'lane' else 'stop',
             'elapsed_seconds': 2 if arm == 'plain' else 1, 'tokens': {},
             'native_accounting': {'complete': True, 'returned_tokens': 10 if arm == 'plain' else 5}}
            for t in tasks for seed in SEEDS for arm in ARMS]
    manifest = {'tasks': tasks, 'seeds': list(SEEDS)}
    result = analyze(rows, manifest, _trusted_selftest=True)['suites']['fixture']
    assert result['sign_flip']['p_two_sided'] == 1
    assert result['accuracy_ci95'] == [0, 0]
    assert result['tokens']['ratio_ci95'] == [.5, .5]
    assert result['time']['savings']['censored_net_fraction'] == 1
    assert result['time']['savings']['over_25_percent_warning']
    assert result['lane']['solved'] == 4  # capped correct is correct
    assert correct(dict(rows[0], status='timeout')) == 0
    assert correct(dict(rows[0], status='error')) == 0
    # Signed net contribution and positive-only contribution must remain distinct.
    def cost_row(cost, capped=False):
        return {'status': 'ok', 'finish_reason': 'length' if capped else 'stop',
                'elapsed_seconds': cost}
    sensitivity, _ = metric_summary([
        [(cost_row(10), cost_row(5, True)), (cost_row(2), cost_row(4, True))],
        [(cost_row(10), cost_row(6)), (cost_row(10), cost_row(7))]],
        lambda row: row['elapsed_seconds'])
    assert sensitivity['savings']['net'] == 10
    assert sensitivity['savings']['positive'] == 12
    assert sensitivity['savings']['censored_net_fraction'] == .3
    assert sensitivity['savings']['censored_positive_fraction'] == 5 / 12
    for bad in (rows[:-1], rows + [rows[0]]):
        try:
            validate_cohort(bad, manifest, _trusted_selftest=True)
        except ValueError:
            pass
        else:
            raise AssertionError('missing/duplicate not rejected')
    incomplete = [dict(r, native_accounting=None) for r in rows]
    report = metric_summary([[(incomplete[0], incomplete[1])]], native_tokens)[0]
    assert report['complete_pairs'] == 0 and report['plain']['mean'] is None
    from math_review import selftest as math_selftest
    math_selftest()
    return {'selftest': 'passed', 'checks': ['DP/exhaustive', 'ties', 'Holm', 'bootstrap',
            'censor contributions', 'missing/duplicate IDs', 'native missingness', 'MATH seal/merge']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    report = sub.add_parser('report')
    report.add_argument('--graded', nargs='+', required=True, help='canonical JSONL, one or both arms per file')
    report.add_argument('--manifest', required=True, help='JSON with tasks and seeds [101,102]')
    report.add_argument('--out', required=True)
    sub.add_parser('selftest', help='trusted synthetic fixtures only; never executes candidates')
    args = parser.parse_args()
    if args.command == 'selftest':
        print(canonical(selftest()))
    else:
        rows = [row for path in args.graded for row in read_jsonl(path)]
        write_json(args.out, analyze(rows, json.loads(Path(args.manifest).read_text(encoding='utf-8'))))


if __name__ == '__main__':
    main()
