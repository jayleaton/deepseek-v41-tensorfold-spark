"""Three-arm amended analysis over the complete fixed scheduled grid.

analyze(rows,tasks,answers=None,*,integrity=None) grades raw rows when answers are
supplied; otherwise requires strict graded rows. Integrity and revalidated safety
receipts are supplied by the parent's independent source/collection audit.
"""
import copy
from collections import Counter

from pins import load_local, SPEC_SHA256, ORIGINAL_SPEC_SHA256

kernel = load_local('pair_analysis')
grading = load_local('grade')
FAMILIES, SEEDS, DRAWS = kernel.FAMILIES, kernel.SEEDS, kernel.DRAWS
ARMS = ('D', 'L', 'P')
METRICS = kernel.METRICS
identity = grading.identity
exact_sign_flip = kernel.exact_sign_flip
bootstrap_indices = kernel.bootstrap_indices
percentile_interval = kernel.percentile_interval


def validate_cohort(rows, tasks):
    # Reuse the full original task/schema validator without its two-arm limit.
    tasks = tasks['tasks'] if isinstance(tasks, dict) else tasks
    indexed = {}
    for row in rows:
        key = identity(row)
        if key in indexed or key[2] not in ARMS:
            raise ValueError('duplicate/unexpected three-arm observation')
        indexed[key] = row
    mapped = [dict(r, arm='D' if identity(r)[2] != 'L' else 'L')
              for r in rows if identity(r)[2] != 'P']
    by_id, _ = kernel.validate_cohort(mapped, tasks)
    # Validate P separately to retain the exact original row-schema rules.
    kernel.validate_cohort([dict(r, arm='D') for r in rows if identity(r)[2] == 'P'], tasks)
    return by_id, indexed


def _translate(value, baseline, treatment):
    replacements = {'D': baseline, 'L': treatment, 'D_sum': baseline + '_sum',
        'L_sum': treatment + '_sum', 'ratio_L_over_D': 'ratio_' + treatment + '_over_' + baseline,
        'mean_difference_L_minus_D': 'mean_difference_' + treatment + '_minus_' + baseline,
        'solve_difference_L_minus_D': 'solve_difference_' + treatment + '_minus_' + baseline}
    if isinstance(value, dict):
        return {replacements.get(k, k): _translate(v, baseline, treatment) for k, v in value.items()}
    if isinstance(value, list):
        return [_translate(v, baseline, treatment) for v in value]
    return value


def _contrast(rows, tasks, baseline, treatment, primary):
    mapped = [dict(r, arm='D' if identity(r)[2] == baseline else 'L') for r in rows
              if identity(r)[2] in (baseline, treatment)]
    # Kernel qualification is discarded: it is a historical two-arm rule, and
    # cannot decide the three-arm amended claim. Only its pure endpoint math is used.
    result = kernel.analyze(mapped, tasks)
    output = _translate({'overall': result['overall'], 'scalar_only': result['scalar_only'], 'families': result['families']}, baseline, treatment)
    for item in [output['overall'], output['scalar_only'], *output['families'].values()]:
        if item['sign_flip']:
            item['sign_flip']['primary'] = primary and item is output['overall']
        # Kernel unknown identities contain internal positional arm strings.
        for cost in item['costs'].values():
            for arm in (baseline, treatment):
                for key in cost[arm]['unknown_identities']:
                    key[2] = baseline if key[2] == 'D' else treatment
    output.update(baseline=baseline, treatment=treatment, primary=primary,
                  interpretation='outcome-informed amended primary' if primary else 'descriptive mechanism/secondary contrast')
    return output


def _native_violation(row):
    if identity(row)[2] == 'L':
        return False
    calls = ((row.get('tensorfold') or {}).get('compute') or {}).get('calls') or []
    binding = row.get('native_binding') or {}
    accounting = row.get('call_accounting') or {}
    return bool(calls or binding.get('calls') or binding.get('native_frame_count', 0)
                or (accounting.get('successful_executions') or 0) > 0
                or (accounting.get('failed_executions') or 0) > 0)


def _classification(rows):
    """Nonexclusive receipt counts: an attempt and a final can coexist."""
    result = {'observed': len(rows), 'attempted_call_replies': 0, 'blocked_call_replies': 0,
              'policy_terminated_replies': 0, 'ordinary_final_answer_replies': 0,
              'strict_solves': sum(kernel.correct(r) for r in rows), 'native_execution_violation_replies': 0}
    for field, target in [('attempted', 'attempted_call_replies'), ('blocked_dispatches', 'blocked_call_replies')]:
        known = [r for r in rows if ((r.get('call_accounting') or {}).get('complete') is True
                 or ((r.get('call_accounting') or {}).get('coverage') or {}).get(field) is True)
                 and kernel.count((r.get('call_accounting') or {}).get(field))]
        result[target] = sum(r['call_accounting'][field] > 0 for r in known)
        result[target + '_unknown_rows'] = len(rows) - len(known)
    result['policy_terminated_replies'] = sum(r.get('status') in ('policy_terminated', 'blocked') for r in rows)
    result['ordinary_final_answer_replies'] = sum(grading.eligible(r) and isinstance(r.get('content'), str)
                                                and bool(r['content'].strip()) for r in rows)
    result['native_execution_violation_replies'] = sum(_native_violation(r) for r in rows)
    result['calls'] = kernel._call_summary(rows)
    return result


def analyze(rows, tasks, answers=None, integrity=None):
    rows = grading.grade_all(rows, tasks, answers) if answers is not None else copy.deepcopy(rows)
    by_id, indexed = validate_cohort(rows, tasks)
    ids = [i for f in FAMILIES for i in sorted(i for i, t in by_id.items() if t['family'] == f)]
    missing = [[i, s, a] for i in ids for s in SEEDS for a in ARMS if (i, s, a) not in indexed]
    complete = not missing
    contrasts = {name: _contrast(rows, tasks, base, treatment, name == 'L_minus_P')
                 for name, base, treatment in [('L_minus_P', 'P', 'L'), ('L_minus_D', 'D', 'L'), ('D_minus_P', 'P', 'D')]}
    primary = contrasts['L_minus_P']['overall']
    # No completed-study primary test/point/CI if ANY of the scheduled720 is missing.
    if not complete:
        for contrast in contrasts.values():
            for item in [contrast['overall'], contrast['scalar_only'], *contrast['families'].values()]:
                item['solve_difference_' + contrast['treatment'] + '_minus_' + contrast['baseline']] = None
                item['solve_difference_ci95'] = None
                item['sign_flip'] = None
    activated = sorted({i for (i, _, a), r in indexed.items() if a == 'L' and kernel._verified_activation(dict(r, arm='L'))})
    violations = [list(identity(r)) for r in rows if _native_violation(r)]
    safety = complete and not violations and all(kernel._row_safe(r) for r in rows)
    integrity_ok = bool(integrity and integrity.get('qualified') is True and integrity.get('provenance')
                        and integrity.get('spec_sha256') == SPEC_SHA256)
    returned, wall = primary['costs']['returned_ids'], primary['costs']['wall_seconds']
    point, ci, test = primary['solve_difference_L_minus_P'], primary['solve_difference_ci95'], primary['sign_flip']
    accuracy = bool(complete and point > 0 and ci and ci[0] > 0 and test['p_two_sided'] <= .05)
    efficiency = bool(complete and returned['ratio_L_over_P'] is not None and returned['ratio_L_over_P'] <= .9
        and returned['ratio_ci95'] and returned['ratio_ci95'][1] < 1 and point >= 0
        and wall['ratio_ci95'] and wall['ratio_ci95'][1] <= 1.05
        and returned['censoring']['uncensored_efficiency_guard'] and wall['censoring']['uncensored_efficiency_guard'])
    qualified = safety and integrity_ok and len(activated) >= 30 and complete
    arms = {}
    for arm in ARMS:
        observed = [r for r in rows if identity(r)[2] == arm]
        solved = sum(kernel.correct(r) for r in observed)
        absent = 240 - len(observed)
        arms[arm] = {'scheduled': 240, 'observed': len(observed), 'solved': solved,
            'solved_count_bounds': [solved, solved + absent],
            'solve_bounds': [solved / 240, (solved + absent) / 240],
            'retention_batches': dict(Counter(r.get('retention_batch', 'unknown') for r in observed)),
            'classifications': _classification(observed),
            'by_family_seed': {f: {str(s): _classification([r for r in observed if by_id[r['id']]['family'] == f and r['seed'] == s])
                                  for s in SEEDS} for f in FAMILIES}}
    return {'spec_sha256': SPEC_SHA256, 'original_spec_sha256': ORIGINAL_SPEC_SHA256,
        'status': 'COMPLETE' if complete else 'INCOMPLETE', 'scheduled': 720, 'observed': len(rows),
        'missing_identities': missing, 'arms': arms, 'contrasts': contrasts,
        'bootstrap': {'draws': DRAWS, 'rng': 20261105, 'percentile_indices': [500, 19499],
                      'shared_all_endpoints_and_contrasts': True, 'unit': 'task ID, both seeds/all arms', 'strata': list(FAMILIES)},
        'activation': {'task_ids': activated, 'distinct_tasks': len(activated), 'threshold': 30,
            'status': 'ADEQUATE' if len(activated) >= 30 else 'INSUFFICIENT EXECUTION EXPOSURE',
            'per_family': {f: [i for i in activated if by_id[i]['family'] == f] for f in FAMILIES},
            'per_seed': {str(s): sorted({i for (i, seed, a), r in indexed.items() if a == 'L' and seed == s
                                       and kernel._verified_activation(dict(r, arm='L'))}) for s in SEEDS}},
        'qualification': {'safety': safety, 'D_P_execution_identities': violations, 'integrity': integrity_ok,
            'case_A_amended_primary_accuracy': accuracy, 'case_B_secondary_returned_ID_efficiency': efficiency,
            'qualified_amended_local_execution_enabled_benefit': bool(qualified and (accuracy or efficiency)),
            'total_token_saving_claim': False, 'serving_default_promotion': False},
        'interpretation': 'Outcome-informed amendment, prospectively frozen completion/P analysis. Strict L/P is the sole amended primary; it is not the originally registered confirmatory test. D/L mechanism and D/P, families, costs and sensitivities are secondary/marginal. Fixed six-template mixture; nonsignificance is not equivalence. Initial plus returned is a subtotal, never full multi-input cost.',
        'historical': 'Original342 outcomes, strict formatting failures, failed audit/stop and original D/L report remain immutable. Additional boots/cache resets, pause and mostly earlier L than P confound interpretation. Before-amendment222 L and resumed18 L are batch descriptors only.'}
