#!/usr/bin/env python3
"""Parent Sparks CPU-only controls. No network, engines, files written or model code execution.

Run this script with python -B under the parent's 2GiB/noSwap/nice19/ionice-c3,
7200s timeout/single worker scope. Do NOT execute on the development machine.
Uses exact real20k bootstrap; no reduced-draw approximation or outcome selection.
"""
import base64
import copy
import hashlib
import json
import itertools
import random
from pathlib import Path

import analysis
import collect_report
import grade
from pins import verify_protocols, pinned_bytes, STRICT_SHA256, ORIGINAL


def rejects(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, KeyError, TypeError):
        return
    raise AssertionError('negative control accepted: ' + function.__name__)


def task(family='integer', slot=1):
    return {'id': 'compute-dl-v1-%s-%03d' % (family, slot), 'suite': 'compute_dl',
            'family': family, 'inputs': {'fixture': True}, 'prompt': 'FAKE ONLY, never collection.'}


def observation(t, arm='plain', seed=101, answer='1'):
    return {'id': t['id'], 'suite': 'compute_dl', 'arm': arm, 'seed': seed,
            'status': 'ok', 'content': answer, 'finish_reason': 'stop', 'answer_eligible': True,
            'elapsed_seconds': 2 if arm == 'lane' else 4,
            'native_binding': {'qualified': True, 'native_frame_count': 0, 'calls': []},
            'safety': {'qualified': True, 'references': []},
            'native_accounting': {'complete': True, 'returned_tokens': 80 if arm == 'lane' else 100},
            'initial_prompt_reconciliation': {'qualified': True, 'initial_prompt_tokens': 10},
            'input_accounting': {'qualified': False, 'complete': False},
            'call_accounting': dict(complete=True, attempted=0, blocked_dispatches=0,
                successful_executions=0, failed_executions=0, delivered_authenticated_results=0, reply_used_result=0),
            'retention_batch': 'fixture'}


def activate(row, clipped=False):
    call = {'id': 'fixture-call', 'code': 'NEVER_EXECUTED_FIXTURE', 'output': '1', 'status': 'ok', 'inserted': True}
    source_hash = hashlib.sha256(call['code'].encode()).hexdigest()
    output_hash = hashlib.sha256(b'1').hexdigest()
    row['tensorfold'] = {'compute': {'calls': [call]}}
    row['native_binding'] = {'qualified': True, 'native_frame_count': 1,
        'calls': [{'id': call['id'], 'native_bound': True, 'delivered': True, 'status': 'ok',
                   'source_sha256': source_hash, 'output_sha256': output_hash}]}
    presentation = {'qualified': True, 'classification': 'presentation-verified clipped' if clipped else 'presentation-verified unclipped',
                    'landed_body_sha256': output_hash, 'landed_body_bytes': 1}
    row['safety'] = {'qualified': True, 'references': [{'call_id': call['id'], 'matches': True,
        'result': {'status': 'ok', 'normalized_body_base64': base64.b64encode(b'full tail').decode()},
        'presentation': presentation}]}
    row['call_accounting'].update(attempted=1, successful_executions=1, delivered_authenticated_results=1, reply_used_result=1)
    return row


def extraction_controls():
    t = task()
    good = ['1', '1.0', '.1 / .1', '1/1', 'The answer is 1.', '1 apples', '$1',
            '\\boxed{1}', 'junk\n#### 1', 'reason\nFinal answer: 1', '<think>wrong 9</think>1',
            '\\boxed{0} then \\boxed{1}']
    # The frozen denominator grammar only permits integer denominators.
    good.remove('.1 / .1')
    bad = ['0', '1 or 2', 'about 1', 'approximately 1', '1 million', '1 percent', '1 2',
           '1/0', '.1 / .1', '1/2', '1e0', 'true', 'NaN', '[1]', '{"x":1}',
           '<think>1', 'reason 1 and 3', '1\n2', '\\boxed{0}\nFinal answer: 1 apples',
           '\\boxed{1', '1\n[output truncated at 512 tokens]']
    for content in good:
        r = observation(t, answer=content)
        assert grade.sensitivity(r, t, 1)['grade'] == 'correct', content
    for content in bad:
        assert grade.sensitivity(observation(t, answer=content), t, 1)['grade'] == 'incorrect', content
    preserved = grade.sensitivity(observation(t, answer='\\boxed{0}\nFinal answer: 1'), t, 1)
    assert preserved['strict_grade'] == preserved['grade'] == 'correct'
    assert preserved['sensitivity']['reason'] == 'strict_preserved'
    for family in grade.SCALARS:
        t = task(family)
        for alias in ('declared_off', 'lane', 'plain'):
            r = observation(t, alias, answer='\\boxed{1}')
            assert grade.grade(r, t, 1)['grade'] == 'incorrect'
            assert grade.sensitivity(r, t, 1)['sensitivity']['changed']
            for patch in [dict(status='policy_terminated'), dict(answer_eligible=False),
                          dict(external_tool_calls=[{'id': 'call'}]), dict(finish_reason='tool_calls'),
                          dict(finish_reason='error'), dict(status='timeout')]:
                assert grade.sensitivity(dict(r, **patch), t, 1)['grade'] == 'incorrect', patch
    for family, answer in [('division', [1, 1]), ('aggregation', dict.fromkeys('ABCDEF', 1))]:
        t = task(family)
        assert grade.sensitivity(observation(t, answer='1'), t, answer)['grade'] == 'incorrect'
        valid = observation(t, answer=json.dumps(answer))
        assert grade.sensitivity(valid, t, answer)['grade'] == 'correct'
        assert grade.sensitivity(valid, t, answer)['sensitivity']['changed'] is False
    t = task()
    strict = grade.grade(observation(t, answer='1'), t, 1)
    assert grade.sensitivity(strict, t, 1)['grade'] == 'correct'
    # All original strict grading outputs, including rejection metadata, identical.
    for text in ['1', '1.0', 'true', 'NaN', '1\n2', '```json\n1\n```', 'Final answer: 1']:
        row = observation(t, answer=text)
        assert grade.grade(row, t, 1) == grade.strict.grade(row, t, 1)


def kernel_controls():
    k = analysis.kernel
    assert k.exact_sign_flip([0] * 120)['p_two_sided'] == 1
    test = k.exact_sign_flip([2] * 120)
    assert test['tail_assignments'] == 2 and test['assignments'] == 2 ** 120
    assert k.exact_sign_flip([2, -1])['p_two_sided'] == 1
    weights = [2, -2, 1, 0, -1, 2]
    brute = [sum(a * b for a, b in zip(signs, weights))
             for signs in itertools.product((-1, 1), repeat=len(weights))]
    expected = sum(abs(v) >= abs(sum(weights)) for v in brute) / len(brute)
    assert k.exact_sign_flip(weights)['p_two_sided'] == expected
    rejects(k.exact_sign_flip, [3])
    assert k.percentile_interval(list(range(20000))) == [500, 19499]
    assert k.percentile_interval([None] + list(range(19999))) is None
    ids = [[f'{f}-{i:02d}' for i in range(20)] for f in k.FAMILIES]
    independent_rng = random.Random(20261105)
    draws = k.bootstrap_indices(ids)
    for _ in range(64):
        expected = [[independent_rng.choice(part) for _ in range(20)] for part in ids]
        assert next(draws) == expected
    first = next(k.bootstrap_indices(ids))
    assert first == next(k.bootstrap_indices(ids))
    assert all(len(part) == 20 and set(part) <= set(source) for part, source in zip(first, ids))
    t = task()
    row = activate(observation(t, 'lane'), clipped=True)
    assert k._verified_activation(row)
    for section, field, value in [('safety', 'qualified', False), ('native_binding', 'qualified', False)]:
        mutant = copy.deepcopy(row); mutant[section][field] = value
        assert not k._verified_activation(mutant)
    for location, field, value in [('call', 'output', '2'), ('call', 'status', 'timeout'),
                                   ('reference', 'matches', False), ('presentation', 'landed_body_sha256', 'bad')]:
        mutant = copy.deepcopy(row)
        target = mutant['tensorfold']['compute']['calls'][0] if location == 'call' else mutant['safety']['references'][0]
        if location == 'presentation': target = target['presentation']
        target[field] = value
        assert not k._verified_activation(mutant)
    d = observation(t, 'declared_off'); l = observation(t, 'lane')
    assert not analysis._native_violation(d)
    assert analysis._native_violation(dict(row, arm='plain'))
    unknown = k._cost_report([(('x', 101), dict(d, native_accounting={}), l)], 0)
    assert unknown['full_cohort_claim_withheld'] and unknown['ratio_L_over_D'] is None
    assert unknown['D']['unknown_identities'] == [['x', 101, 'D']]
    # A cap savings contribution >25% blocks an uncensored efficiency claim.
    censored = k._cost_report([(('x', 101), dict(d, finish_reason='length'), l),
                              (('y', 101), d, l)], 0)
    assert censored['censoring']['over_25_percent'] and not censored['censoring']['uncensored_efficiency_guard']
    reversal = k._cost_report([(('x', 101), dict(d, finish_reason='length', native_accounting={'complete': True, 'returned_tokens': 1000}), l),
                              (('y', 101), l, d)], 0)
    assert reversal['censoring']['complete_both_reversal']


def main():
    verify_protocols()
    before = pinned_bytes(ORIGINAL / 'grade.py', STRICT_SHA256)
    extraction_controls()
    kernel_controls()
    tasks = [task(f, slot) for f in analysis.FAMILIES for slot in range(1, 21)]
    answers = {t['id']: ([1, 1] if t['family'] == 'division' else dict.fromkeys('ABCDEF', 1)
                        if t['family'] == 'aggregation' else 1) for t in tasks}
    rows = []
    for t in tasks:
        for seed in analysis.SEEDS:
            for alias in ('declared_off', 'lane', 'plain'):
                # Strict L:240; strict D/P:0. Sensitivity P scalar:160, non-scalar:0.
                text = json.dumps(answers[t['id']]) if alias == 'lane' else '\\boxed{1}' if alias == 'plain' else '0'
                row = observation(t, alias, seed, text)
                if alias == 'lane': activate(row)
                if alias == 'declared_off':
                    row.update(status='policy_terminated', answer_eligible=False)
                    row['call_accounting'].update(attempted=1, blocked_dispatches=1)
                rows.append(row)
    original = copy.deepcopy(rows)
    integrity = {'qualified': True, 'provenance': 'fake-control-only', 'spec_sha256': analysis.SPEC_SHA256}
    report = collect_report.collect(rows, tasks, answers, metadata={'integrity': integrity})
    assert rows == original
    result = report['analysis']
    assert result['status'] == 'COMPLETE' and result['scheduled'] == 720
    assert result['arms']['L']['solved'] == 240 and result['arms']['P']['solved'] == 0
    assert result['activation']['distinct_tasks'] == 120
    primary = result['contrasts']['L_minus_P']['overall']
    assert primary['solve_difference_L_minus_P'] == 1 and primary['solve_difference_ci95'] == [1, 1]
    assert primary['sign_flip']['primary'] and primary['sign_flip']['assignments'] == 2 ** 120
    assert primary['costs']['returned_ids']['ratio_L_over_P'] == .8
    assert primary['costs']['returned_ids']['ratio_ci95'] == [.8, .8]
    assert primary['costs']['processed_input_plus_returned_ids']['ratio_ci95'] is None
    assert primary['costs']['initial_prompt_plus_returned_SUBTOTAL']['ratio_L_over_P'] == 90 / 110
    assert result['qualification']['qualified_amended_local_execution_enabled_benefit']
    assert not result['contrasts']['L_minus_D']['overall']['sign_flip']['primary']
    assert len(primary['gains']) == 240 and not primary['losses']
    for arm in analysis.ARMS:
        coverage = report['secondary']['coverage'][arm]
        assert coverage['scalar_scheduled'] == 160 and coverage['combined_scheduled'] == 240
        assert coverage['non_scalar_strict_unchanged']
    assert report['secondary']['coverage']['P']['scalar_solved'] == 160
    assert len(report['secondary']['changed_rows']) == 160 and len(report['secondary']['residual_rows']) == 320
    assert report['secondary_analysis']['contrasts']['L_minus_P']['scalar_only']['solve_difference_L_minus_P'] == 0
    assert not report['secondary_analysis']['contrasts']['L_minus_P']['overall']['sign_flip']['primary']
    assert result['arms']['D']['classifications']['attempted_call_replies'] == 240
    assert result['arms']['D']['classifications']['blocked_call_replies'] == 240
    assert result['arms']['D']['classifications']['ordinary_final_answer_replies'] == 0
    assert result['arms']['D']['classifications']['native_execution_violation_replies'] == 0
    graded = grade.grade_all(rows, tasks, answers)
    rejects(analysis.validate_cohort, graded + [graded[0]], tasks)
    rejects(grade.grade_all, [dict(rows[0], arm='Q')], tasks, answers)
    rejects(grade.grade_all, [dict(rows[0], seed=True)], tasks, answers)
    rejects(analysis.validate_cohort, graded, tasks[:-1])
    # Missing-grid statistics use the same real20k kernel, but avoid another
    # whole report/sensitivity pass. Primary is withheld for ANY missing arm.
    missing = analysis.analyze(graded[1:], tasks, integrity=integrity)
    assert missing['status'] == 'INCOMPLETE' and len(missing['missing_identities']) == 1
    assert missing['contrasts']['L_minus_P']['overall']['sign_flip'] is None
    assert missing['contrasts']['L_minus_P']['overall']['solve_difference_ci95'] is None
    assert missing['arms']['D']['solved_count_bounds'] == [0, 1]
    assert not missing['qualification']['qualified_amended_local_execution_enabled_benefit']
    assert '720' in collect_report.render_results(report)
    json.dumps(report, allow_nan=False)
    assert pinned_bytes(ORIGINAL / 'grade.py', STRICT_SHA256) == before
    print(json.dumps({'status': 'PASS', 'scheduled': 720, 'draws': 20000,
                      'controls': 'strict immutability, selector/grammar/eligibility, three-arm mask, shared cluster bootstrap, exact DP, provenance, activation, censor/accounting, row reports'}, sort_keys=True))


if __name__ == '__main__':
    main()
