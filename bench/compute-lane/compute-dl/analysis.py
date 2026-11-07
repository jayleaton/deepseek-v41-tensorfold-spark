#!/usr/bin/env python3
"""Offline frozen fixed-mixture D/L analysis; no collection or engine imports.

analyze(graded_rows, public_tasks, integrity=None) accepts the A1 accounting
fields unchanged. Merge the independently retained safety sidecar into each
row's `safety` field. `integrity` must have qualified=True, provenance and the
frozen spec_sha256 after the parent's independent source/prompt/dispatch audit.
Unknown evidence never qualifies a claim. Missing observations remain scheduled.
Optional observed time fields are server_seconds/tool_execution_seconds; absent
times remain unknown. elapsed_seconds is the full client wall observation.
"""
from collections import Counter
import base64
import hashlib
import math
import random
import statistics

SPEC_SHA256 = 'c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141'
FAMILIES = ('integer', 'modular', 'division', 'dp', 'graph', 'aggregation')
SEEDS = (101, 102)
ARMS = ('D', 'L')
DRAWS = 20000


def count(value):
    return type(value) is int and value >= 0


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def identity(row):
    return (row['id'], row['seed'], {'declared_off': 'D', 'lane': 'L'}.get(row['arm'], row['arm']))


def correct(row):
    return int(bool(row) and row.get('status') == 'ok' and row.get('grade') == 'correct')


def flags(row):
    if row is None:
        return {'cap': False, 'deadline': False, 'policy_error': False, 'censored': True}
    finish, status = row.get('finish_reason'), row.get('status')
    cap = finish in ('length', 'max_tokens', 'max_output_tokens')
    deadline = status in ('timeout', 'deadline') or finish in ('timeout', 'deadline')
    error = status != 'ok' or finish in ('error', 'failed', 'cancelled', 'tool_calls', 'function_call')
    return {'cap': cap, 'deadline': deadline, 'policy_error': error,
            'censored': cap or deadline or error or row.get('censored') is True}


def initial_tokens(row):
    evidence = row.get('initial_prompt_reconciliation') or {}
    value = evidence.get('initial_prompt_tokens', evidence.get('server_initial_prompt_tokens'))
    qualified = (evidence.get('qualified') is True
                 or evidence.get('qualified_under_proposed_count_contract') is True)
    return value if qualified and count(value) else None


def returned_tokens(row):
    evidence = row.get('native_accounting') or {}
    value = evidence.get('returned_tokens')
    return value if evidence.get('complete') is True and count(value) else None


def input_tokens(row):
    evidence = row.get('input_accounting') or {}
    value = evidence.get('processed_input_tokens')
    if (evidence.get('qualified') is True and evidence.get('complete') is True
            and evidence.get('provenance') and count(value)):
        return value
    return None


def _sum_known(a, b):
    return a + b if a is not None and b is not None else None


def metric_values(row):
    if row is None:
        return [None] * len(METRICS)
    returned, initial, processed = returned_tokens(row), initial_tokens(row), input_tokens(row)
    times = [row.get(key) if number(row.get(key)) else None
             for key in ('elapsed_seconds', 'server_seconds', 'tool_execution_seconds')]
    return [returned, initial, processed, _sum_known(processed, returned),
            _sum_known(initial, returned), *times]


METRICS = ('returned_ids', 'initial_prompt_ids', 'processed_input_ids',
           'processed_input_plus_returned_ids', 'initial_prompt_plus_returned_SUBTOTAL',
           'wall_seconds', 'server_seconds', 'tool_execution_seconds')
CALL_FIELDS = ('attempted', 'blocked_dispatches', 'successful_executions',
               'failed_executions', 'delivered_authenticated_results', 'reply_used_result')


def exact_sign_flip(differences):
    """Exact two-sided task sign-flip DP, including ties in the tail."""
    if any(type(d) is not int or abs(d) > 2 for d in differences):
        raise ValueError('expected task two-seed correct-count differences')
    distribution = {0: 1}
    for weight in (abs(d) for d in differences if d):
        updated = Counter()
        for total, ways in distribution.items():
            updated[total - weight] += ways
            updated[total + weight] += ways
        distribution = updated
    observed = abs(sum(differences))
    numerator = sum(ways for total, ways in distribution.items() if abs(total) >= observed)
    denominator = sum(distribution.values())
    return {'p_two_sided': numerator / denominator, 'tail_assignments': numerator,
            'assignments': denominator, 'observed_correct_count_difference': sum(differences)}


def percentile_interval(values):
    # Undefined bootstrap ratios cannot be dropped to narrow a reported interval.
    if len(values) != DRAWS or any(v is None for v in values):
        return None
    ordered = sorted(values)
    return [ordered[500], ordered[19499]]


def bootstrap_indices(ids_by_family):
    """One fresh shared stream; order is family, then 20 replacement draws."""
    rng = random.Random(20261105)
    for _ in range(DRAWS):
        yield [[ids[rng.randrange(20)] for _ in range(20)] for ids in ids_by_family]


def _summary(values, scheduled, unknown):
    return {'known_rows': len(values), 'scheduled_rows': scheduled,
            'known_sum_lower_bound': sum(values) if values else None,
            'sum': sum(values) if len(values) == scheduled else None,
            'mean': statistics.mean(values) if len(values) == scheduled else None,
            'median': statistics.median(values) if len(values) == scheduled else None,
            'known_mean': statistics.mean(values) if values else None,
            'known_median': statistics.median(values) if values else None,
            'unknown_identities': unknown}


def _call_summary(rows):
    result = {}
    for field in CALL_FIELDS:
        values = []
        for row in rows:
            evidence = (row or {}).get('call_accounting') or {}
            if ((evidence.get('complete') is True or (evidence.get('coverage') or {}).get(field) is True)
                    and count(evidence.get(field))):
                values.append(evidence[field])
        result[field] = {'known_rows': len(values), 'unknown_rows': len(rows) - len(values),
                         'known_sum': sum(values) if values else None}
    return result


def _verified_delivery(row, call):
    """Join source/output hashes and independent replay bytes by exact call ID."""
    binding, safety = row.get('native_binding') or {}, row.get('safety') or {}
    if (binding.get('qualified') is not True or safety.get('qualified') is not True
            or call.get('id') is None or call.get('inserted') is not True
            or not isinstance(call.get('code'), str) or not isinstance(call.get('output'), str)):
        return False
    bound = [b for b in binding.get('calls', []) if b.get('id') == call['id']]
    replay = [r for r in safety.get('references', []) if r.get('call_id') == call['id']]
    if len(bound) != 1 or len(replay) != 1:
        return False
    proof, receipt = bound[0], replay[0]
    if (proof.get('native_bound') is not True or proof.get('delivered') is not True
            or proof.get('status') != call.get('status') or receipt.get('matches') is not True
            or proof.get('source_sha256') != hashlib.sha256(call['code'].encode()).hexdigest()
            or proof.get('output_sha256') != hashlib.sha256(call['output'].encode()).hexdigest()):
        return False
    result = receipt.get('result') or {}
    try:
        output = base64.b64decode(result['normalized_body_base64'], validate=True)
    except (KeyError, TypeError, ValueError):
        return False
    return result.get('status') == call.get('status') and output == call['output'].encode()


def _verified_activation(row):
    """Require actual joined call/frame/replay receipts, never a call counter."""
    if not row or identity(row)[2] != 'L':
        return False
    calls = ((row.get('tensorfold') or {}).get('compute') or {}).get('calls') or []
    return any(c.get('status') == 'ok' and _verified_delivery(row, c) for c in calls)


def _row_safe(row):
    binding = row.get('native_binding') or {}
    safety = row.get('safety') or {}
    if binding.get('qualified') is not True or safety.get('qualified') is not True:
        return False
    if any(r.get('matches') is not True for r in safety.get('references', [])):
        return False
    calls = ((row.get('tensorfold') or {}).get('compute') or {}).get('calls') or []
    return all(_verified_delivery(row, c) for c in calls if c.get('inserted') is True)


def validate_cohort(rows, tasks):
    tasks = tasks['tasks'] if isinstance(tasks, dict) else tasks
    if len(tasks) != 120 or len({t['id'] for t in tasks}) != 120:
        raise ValueError('frozen 120 distinct public tasks required')
    by_id = {t['id']: t for t in tasks}
    for family in FAMILIES:
        expected = {f'compute-dl-v1-{family}-{slot:03d}' for slot in range(1, 21)}
        actual = {t['id'] for t in tasks if t.get('family') == family}
        if actual != expected:
            raise ValueError('frozen family/slot identities required')
    if any(t.get('suite') != 'compute_dl' or not isinstance(t.get('inputs'), dict)
           or not isinstance(t.get('prompt'), str) or not t['prompt']
           or any(k in t for k in ('answer', 'reference', 'reference_a', 'reference_b')) for t in tasks):
        raise ValueError('public task schema required; private answers prohibited')
    indexed = {}
    for row in rows:
        key = identity(row)
        if (key in indexed or key[0] not in by_id or type(key[1]) is not int or key[1] not in SEEDS
                or key[2] not in ARMS or row.get('suite') != 'compute_dl' or row.get('diagnostic')
                or row.get('grade') not in ('correct', 'incorrect')
                or not isinstance(row.get('status'), str) or not row['status']
                or row.get('included_in_cohort') is False):
            raise ValueError('unexpected/duplicate/ungraded observation')
        indexed[key] = row
    return by_id, indexed


def _cost_report(pairs, metric):
    observed = [(key, metric_values(d)[metric], metric_values(l)[metric], flags(d)['censored'] or flags(l)['censored'])
                for key, d, l in pairs]
    selected = [(d, l, c) for _, d, l, c in observed if d is not None and l is not None]
    uncensored = [(d, l, c) for d, l, c in selected if not c]
    def stats(values):
        d, l = sum(v[0] for v in values), sum(v[1] for v in values)
        return {'pairs': len(values), 'D_sum': d if values else None, 'L_sum': l if values else None,
                'ratio_L_over_D': l / d if d else None,
                'mean_difference_L_minus_D': (l - d) / len(values) if values else None}
    full = len(selected) == len(pairs)
    result = {'full_cohort_claim_withheld': not full, 'scheduled_pairs': len(pairs),
              'complete_pairs': len(selected), 'paired_accounting_complete_sensitivity': stats(selected),
              'complete_both_uncensored_sensitivity': stats(uncensored)}
    for a, position in (('D', 1), ('L', 2)):
        values = [v[position] for v in observed if v[position] is not None]
        result[a] = _summary(values, len(pairs), [[key[0], key[1], a] for key, d, l, _ in observed
                                                if (d if a == 'D' else l) is None])
    result.update(stats(selected) if full else {'ratio_L_over_D': None, 'mean_difference_L_minus_D': None})
    net = sum(d - l for d, l, _ in selected)
    positive = sum(max(d - l, 0) for d, l, _ in selected)
    censored_net = sum(d - l for d, l, c in selected if c)
    censored_positive = sum(max(d - l, 0) for d, l, c in selected if c)
    net_fraction = abs(censored_net) / abs(net) if net else (None if not censored_net else float('inf'))
    positive_fraction = censored_positive / positive if positive else 0
    unc_net = sum(d - l for d, l, _ in uncensored)
    reversal = bool(selected and uncensored and net * unc_net < 0)
    guard = full and bool(uncensored) and not reversal and (net_fraction is None or net_fraction <= .25) and positive_fraction <= .25
    result['censoring'] = {'net_savings': net, 'positive_savings': positive,
                           'censored_net_savings': censored_net, 'censored_positive_savings': censored_positive,
                           'net_contribution_fraction': net_fraction if net_fraction != float('inf') else None,
                           'net_fraction_undefined_with_nonzero_contribution': net == 0 and censored_net != 0,
                           'positive_contribution_fraction': positive_fraction,
                           'over_25_percent': (net_fraction is not None and net_fraction > .25) or positive_fraction > .25,
                           'complete_both_reversal': reversal, 'uncensored_efficiency_guard': guard}
    result['_cluster'] = {}
    for key, d, l, c in observed:
        item = result['_cluster'].setdefault(key[0], [0, 0, 0, 0, 0, 0])
        if d is not None and l is not None:
            item[0] += d; item[1] += l; item[2] += 1
            if not c:
                item[3] += d; item[4] += l; item[5] += 1
    result['_draws'] = [[], [], [], []]
    return result


def analyze(rows, tasks, *, integrity=None):
    by_id, indexed = validate_cohort(rows, tasks)
    ids_by_family = [sorted(i for i, t in by_id.items() if t['family'] == f) for f in FAMILIES]
    all_ids = [i for ids in ids_by_family for i in ids]
    missing = [[i, seed, arm] for i in all_ids for seed in SEEDS for arm in ARMS if (i, seed, arm) not in indexed]
    complete = not missing
    reports = {}
    for name, ids in [('overall', all_ids), *zip(FAMILIES, ids_by_family)]:
        pairs = [((i, seed), indexed.get((i, seed, 'D')), indexed.get((i, seed, 'L'))) for i in ids for seed in SEEDS]
        differences = {i: sum(correct(indexed.get((i, seed, 'L'))) - correct(indexed.get((i, seed, 'D'))) for seed in SEEDS) for i in ids}
        arms = {}
        for arm in ARMS:
            arm_rows = [indexed.get((i, seed, arm)) for i in ids for seed in SEEDS]
            solved = sum(map(correct, arm_rows)); absent = sum(r is None for r in arm_rows)
            arms[arm] = {'scheduled': len(arm_rows), 'observed': len(arm_rows) - absent, 'solved': solved,
                         'solved_count_bounds': [solved, solved + absent],
                         'solve_bounds': [solved / len(arm_rows), (solved + absent) / len(arm_rows)],
                         'statuses': dict(Counter(r['status'] for r in arm_rows if r is not None)),
                         'finish_reasons': dict(Counter(str(r.get('finish_reason')) for r in arm_rows if r is not None)),
                         'format_rejections': sum(bool((r or {}).get('grader', {}).get('format_rejection')) for r in arm_rows),
                         'rates': {flag: sum(flags(r)[flag] for r in arm_rows if r is not None) / len(arm_rows)
                                   for flag in ('cap', 'deadline', 'policy_error')},
                         'rates_are_lower_bounds': bool(absent),
                         'rates_denominator': 'all scheduled replies; missing event status unknown',
                         'rate_upper_bounds': {flag: (sum(flags(r)[flag] for r in arm_rows if r is not None) + absent) / len(arm_rows)
                                               for flag in ('cap', 'deadline', 'policy_error')},
                         'calls': _call_summary(arm_rows),
                         'calls_per_seed': {str(seed): _call_summary([indexed.get((i, seed, arm)) for i in ids]) for seed in SEEDS}}
        bounds = [arms['L']['solve_bounds'][0] - arms['D']['solve_bounds'][1],
                  arms['L']['solve_bounds'][1] - arms['D']['solve_bounds'][0]]
        unc = [(key, d, l) for key, d, l in pairs if d is not None and l is not None and not flags(d)['censored'] and not flags(l)['censored']]
        reports[name] = {'arms': arms, 'scheduled_pairs': len(pairs), 'difference_bounds': bounds,
                         'solve_difference_L_minus_D': sum(differences.values()) / len(pairs) if complete else None,
                         'sign_flip': dict(exact_sign_flip(list(differences.values())), primary=name == 'overall') if complete else None,
                         'task_correct_count_differences': differences,
                         'gains': [list(key) for key, d, l in pairs if d is not None and l is not None and correct(l) > correct(d)],
                         'losses': [list(key) for key, d, l in pairs if d is not None and l is not None and correct(l) < correct(d)],
                         'complete_both_uncensored_sensitivity': {'selection': 'observed paired uncensored replies; descriptive',
                             'pairs': len(unc), 'solve_difference_L_minus_D': sum(correct(l) - correct(d) for _, d, l in unc) / len(unc) if unc else None},
                         'costs': {metric: _cost_report(pairs, m) for m, metric in enumerate(METRICS)},
                         '_accuracy_draws': [], '_uncensored_draws': [],
                         '_uncensored_clusters': {i: [sum(correct(l) - correct(d) for key, d, l in unc if key[0] == i),
                                                       sum(key[0] == i for key, _, _ in unc)] for i in ids}}
    for sampled in bootstrap_indices(ids_by_family):
        for name, ids in [('overall', [i for part in sampled for i in part]), *zip(FAMILIES, sampled)]:
            report = reports[name]
            report['_accuracy_draws'].append(sum(report['task_correct_count_differences'][i] for i in ids) / (2 * len(ids)))
            uc = report['_uncensored_clusters']
            n = sum(uc[i][1] for i in ids)
            report['_uncensored_draws'].append(sum(uc[i][0] for i in ids) / n if n else None)
            for cost in report['costs'].values():
                cluster = cost['_cluster']
                totals = [sum(cluster[i][j] for i in ids) for j in range(6)]
                d, l, n, ud, ul, un = totals
                values = [l / d if d else None, (l - d) / n if n else None,
                          ul / ud if ud else None, (ul - ud) / un if un else None]
                for target, value in zip(cost['_draws'], values):
                    target.append(value)
    for report in reports.values():
        accuracy_draws = report.pop('_accuracy_draws')
        report['solve_difference_ci95'] = percentile_interval(accuracy_draws) if complete else None
        report['complete_both_uncensored_sensitivity']['solve_difference_ci95'] = percentile_interval(report.pop('_uncensored_draws'))
        report.pop('_uncensored_clusters')
        for cost in report['costs'].values():
            intervals = [percentile_interval(v) for v in cost.pop('_draws')]
            cost.pop('_cluster')
            cost['paired_accounting_complete_sensitivity'].update(selection='selected complete pairs; not representative of unknown cases', ratio_ci95=intervals[0], difference_ci95=intervals[1])
            cost['complete_both_uncensored_sensitivity'].update(selection='selected complete paired uncensored replies; descriptive', ratio_ci95=intervals[2], difference_ci95=intervals[3])
            cost['ratio_ci95'] = None if cost['full_cohort_claim_withheld'] else intervals[0]
            cost['difference_ci95'] = None if cost['full_cohort_claim_withheld'] else intervals[1]
    activated = sorted({i for (i, _, _), row in indexed.items() if _verified_activation(row)})
    # Unknown task outcomes are not displayed as zero differences. Internal
    # placeholder totals above never feed a primary point estimate, CI or test.
    for report in reports.values():
        for task_id in report['task_correct_count_differences']:
            if any((task_id, seed, arm) not in indexed for seed in SEEDS for arm in ARMS):
                report['task_correct_count_differences'][task_id] = None
    safety_ok = complete and all(_row_safe(r) for r in rows)
    d_execution = [list(identity(r)) for r in rows if identity(r)[2] == 'D'
                   and (((r.get('tensorfold') or {}).get('compute') or {}).get('calls')
                        or ((r.get('call_accounting') or {}).get('successful_executions') or 0) > 0
                        or ((r.get('call_accounting') or {}).get('failed_executions') or 0) > 0)]
    integrity_ok = bool(integrity and integrity.get('qualified') is True and integrity.get('provenance')
                        and integrity.get('spec_sha256') == SPEC_SHA256)
    overall = reports['overall']; returned = overall['costs']['returned_ids']; wall = overall['costs']['wall_seconds']
    primary_ci = overall['solve_difference_ci95']
    case_a = bool(complete and overall['solve_difference_L_minus_D'] > 0 and primary_ci and primary_ci[0] > 0
                  and overall['sign_flip']['p_two_sided'] <= .05)
    case_b = bool(complete and returned['ratio_L_over_D'] is not None and returned['ratio_L_over_D'] <= .9
                  and returned['ratio_ci95'] and returned['ratio_ci95'][1] < 1
                  and overall['solve_difference_L_minus_D'] >= 0 and wall['ratio_ci95'] and wall['ratio_ci95'][1] <= 1.05
                  and returned['censoring']['uncensored_efficiency_guard'] and wall['censoring']['uncensored_efficiency_guard'])
    qualified = safety_ok and not d_execution and integrity_ok and len(activated) >= 30 and complete
    return {'spec_sha256': SPEC_SHA256, 'status': 'COMPLETE' if complete else 'INCOMPLETE',
            'scheduled': 480, 'observed': len(rows), 'missing_identities': missing,
            'bootstrap': {'draws': DRAWS, 'rng': 20261105, 'percentile_indices': [500, 19499],
                          'shared_all_endpoints': True, 'unit': 'task ID, both seeds and arms', 'strata': list(FAMILIES)},
            'overall': overall, 'families': {f: reports[f] for f in FAMILIES},
            'activation': {'task_ids': activated, 'distinct_tasks': len(activated), 'threshold': 30,
                           'status': 'ADEQUATE' if len(activated) >= 30 else 'INSUFFICIENT EXECUTION EXPOSURE',
                           'per_family': {f: [i for i in activated if by_id[i]['family'] == f] for f in FAMILIES},
                           'per_seed': {str(seed): sorted({i for (i, s, _), row in indexed.items()
                                                          if s == seed and _verified_activation(row)}) for seed in SEEDS}},
            'qualification': {'safety': safety_ok and not d_execution, 'D_execution_identities': d_execution,
                              'integrity': integrity_ok, 'case_A_primary_accuracy': case_a,
                              'case_B_secondary_returned_ID_efficiency': case_b,
                              'qualified_local_execution_enabled_benefit': bool(qualified and (case_a or case_b)),
                              'total_token_saving_claim': False, 'serving_default_promotion': False},
            'interpretation': 'Fixed equal-weight six-template mixture; one overall primary sign-flip test. Family, cost and sensitivity intervals are marginal/descriptive. Nonsignificance is not equivalence. Initial plus returned is a subtotal, not full all-round cost.'}
