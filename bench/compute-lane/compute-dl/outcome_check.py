#!/usr/bin/env python3
"""FAKE_NOT_COLLECTION: CPU-only checks, for the parent's bounded Sparks scope.

Run `python3 -B compute-dl/outcome_check.py` on Sparks2G, NEVER the dev box.
No study inputs, model requests, executors, SSH, GPU access or files are produced.
The fake 120-ID grid exercises scheduling only; inputs/prompts are marked fake.
"""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

FAKE = 'FAKE_NOT_COLLECTION'


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_fixture(grader, analysis):
    tasks, answers, rows = [], {}, []
    for family in analysis.FAMILIES:
        for slot in range(1, 21):
            task_id = f'compute-dl-v1-{family}-{slot:03d}'
            tasks.append({'id': task_id, 'family': family, 'suite': 'compute_dl',
                          'inputs': {'fixture': FAKE}, 'prompt': FAKE})
            answers[task_id] = [2, 1] if family == 'division' else (
                dict.fromkeys('ABCDEF', 7) if family == 'aggregation' else 7)
            for seed in analysis.SEEDS:
                for arm in analysis.ARMS:
                    row = {'id': task_id, 'suite': 'compute_dl', 'seed': seed, 'arm': arm,
                           'fixture': FAKE, 'status': 'ok', 'finish_reason': 'stop',
                           'content': json.dumps(answers[task_id]), 'elapsed_seconds': 1,
                           'native_accounting': {'complete': True, 'returned_tokens': 100 if arm == 'D' else 80},
                           'initial_prompt_reconciliation': {'qualified': True, 'initial_prompt_tokens': 10},
                           'input_accounting': {'complete': True, 'qualified': True, 'provenance': FAKE, 'processed_input_tokens': 10},
                           'call_accounting': dict.fromkeys(analysis.CALL_FIELDS, 0),
                           'native_binding': {'qualified': True, 'calls': []},
                           'safety': {'qualified': True, 'references': []}}
                    row['call_accounting']['complete'] = True
                    rows.append(row)
    return tasks, answers, grader.grade_all(rows, tasks, answers)


def attach_verified_call(row):
    code, output, call_id = 'print(7)', '7', FAKE + '-call'
    call = {'id': call_id, 'status': 'ok', 'inserted': True, 'code': code, 'output': output}
    row['tensorfold'] = {'compute': {'calls': [call]}}
    row['native_binding']['calls'] = [{'id': call_id, 'status': 'ok', 'delivered': True,
        'native_bound': True, 'source_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'output_sha256': hashlib.sha256(output.encode()).hexdigest()}]
    row['safety']['references'] = [{'call_id': call_id, 'matches': True,
        'result': {'status': 'ok', 'normalized_body_base64': base64.b64encode(output.encode()).decode()}}]


def run_checks():
    grader = _load('_compute_dl_check_grade', 'grade.py')
    analysis = _load('_compute_dl_check_analysis', 'analysis.py')
    task = {'id': FAKE, 'family': 'integer', 'suite': 'compute_dl', 'inputs': {}, 'prompt': FAKE}
    def grade_text(text, answer=7, **fields):
        row = dict(status='ok', content=text, finish_reason='stop')
        row.update(fields)
        return grader.grade(row, task, answer)
    for text in ('7', ' 7\n', 'reason\n```json\n7\n``` \n', 'reason\n```\n7\n```',
                 'reason\nFiNaL AnSwEr: 7', '```json\n2\n```\n```json\n7\n```',
                 '```python\nignored\n```\nreason\n```json\n7\n```'):
        assert grade_text(text)['grade'] == 'correct', text
    for text in ('true', '7.0', '7e0', '1e999', 'NaN', 'Infinity', '-Infinity', '07', '+7',
                 '7 8', '[7', '7 trailing', '```json\n7\n``` trailing',
                 '```python\n7\n```', 'reason 7', '{"A":7,"A":7}', '"7"', 'null',
                 '```json\n[7,8]\n```', '```json\n8\n```'):
        assert grade_text(text)['grade'] == 'incorrect', text
    # A syntactically selected but wrong value must not fall through to 7.
    wrong = '[\n7\n]'
    assert grade_text(wrong)['grader']['extraction'] == 'whole'
    assert grade_text(wrong)['grade'] == 'incorrect'
    assert grade_text('```json\n8\n```')['grader']['extraction'] == 'last_fence'
    assert grade_text('7', finish_reason='length')['grade'] == 'correct'
    for fields in ({'status': 'error'}, {'status': 'policy_terminated'}, {'external_tool_calls': [{}]},
                   {'answer_eligible': False}, {'finish_reason': 'tool_calls'}):
        assert grade_text('7', **fields)['grade'] == 'incorrect'
    assert grade_text(None)['grade'] == 'incorrect'
    division = dict(task, family='division')
    assert grader.grade({'status': 'ok', 'content': '[2,1]'}, division, [2, 1])['grade'] == 'correct'
    for text in ('[2,true]', '[2,1.0]', '[2,1,0]', '{"q":2,"r":1}'):
        assert grader.grade({'status': 'ok', 'content': text}, division, [2, 1])['grade'] == 'incorrect'
    aggregation = dict(task, family='aggregation')
    answer = dict.fromkeys('ABCDEF', 7)
    assert grader.grade({'status': 'ok', 'content': json.dumps(dict(reversed(list(answer.items()))))}, aggregation, answer)['grade'] == 'correct'
    for text in ('{"A":7,"B":7,"C":7,"D":7,"E":7,"F":7,"A":7}',
                 '{"A":7,"B":7,"C":7,"D":7,"E":7}',
                 '{"A":7,"B":7,"C":7,"D":7,"E":7,"F":7.0}'):
        assert grader.grade({'status': 'ok', 'content': text}, aggregation, answer)['grade'] == 'incorrect'
    for family in ('integer', 'modular', 'dp', 'graph'):
        assert grader.grade({'status': 'ok', 'content': '7'}, dict(task, family=family), 7)['grade'] == 'correct'
    assert analysis.exact_sign_flip([0, 0])['p_two_sided'] == 1
    assert analysis.exact_sign_flip([2, 2, 2])['p_two_sided'] == .25
    assert analysis.exact_sign_flip([2] * 6)['p_two_sided'] == .03125
    assert analysis.exact_sign_flip([1, -2])['p_two_sided'] == 1
    assert analysis.percentile_interval(list(range(20000))) == [500, 19499]
    assert analysis.percentile_interval([None] * 20000) is None
    tasks, answers, rows = fake_fixture(grader, analysis)
    original = copy.deepcopy(rows)
    # Sixty fake L replies across both seeds cover only thirty distinct tasks.
    for row in rows:
        if row['arm'] == 'L' and row['id'] in {t['id'] for t in tasks[:30]}:
            attach_verified_call(row)
    assert len({r['id'] for r in rows if analysis._verified_activation(r)}) == 30
    forged = copy.deepcopy(next(r for r in rows if analysis._verified_activation(r)))
    forged['safety']['references'][0]['result']['normalized_body_base64'] = 'OA=='
    assert not analysis._verified_activation(forged)
    assert not analysis._row_safe(forged)
    fake_counter = copy.deepcopy(original[1])
    fake_counter['native_binding']['delivered_successes'] = 999
    fake_counter['call_accounting']['successful_executions'] = 999
    assert not analysis._verified_activation(fake_counter)
    # Genuine call-bearing A1 rows lack a full input ledger: remove that evidence.
    rows[1]['input_accounting'] = {'complete': False, 'qualified': False}
    before = copy.deepcopy(rows)
    integrity = {'qualified': True, 'provenance': FAKE, 'spec_sha256': analysis.SPEC_SHA256}
    report = analysis.analyze(rows, tasks, integrity=integrity)
    assert rows == before
    assert report['status'] == 'COMPLETE' and report['scheduled'] == 480
    assert report['activation']['distinct_tasks'] == 30
    assert report['qualification']['qualified_local_execution_enabled_benefit']
    assert report['qualification']['case_B_secondary_returned_ID_efficiency']
    assert not report['qualification']['case_A_primary_accuracy']
    overall = report['overall']
    assert overall['solve_difference_ci95'] == [0, 0]
    assert overall['sign_flip']['p_two_sided'] == 1
    returned = overall['costs']['returned_ids']
    assert returned['D']['sum'] == 24000 and returned['L']['sum'] == 19200
    assert returned['ratio_ci95'] == [.8, .8]
    assert returned['difference_ci95'] == [-20, -20]
    assert returned['D']['median'] == 100
    full_input = overall['costs']['processed_input_plus_returned_ids']
    assert full_input['full_cohort_claim_withheld'] and full_input['ratio_ci95'] is None
    assert full_input['ratio_L_over_D'] is None and full_input['L']['sum'] is None
    assert full_input['paired_accounting_complete_sensitivity']['pairs'] == 239
    assert overall['costs']['initial_prompt_plus_returned_SUBTOTAL']['L']['sum'] == 21600
    assert not report['qualification']['total_token_saving_claim']
    assert not report['qualification']['serving_default_promotion']
    json.dumps(report, allow_nan=False)
    incomplete = analysis.analyze(rows[:-1], tasks, integrity=integrity)
    assert incomplete['status'] == 'INCOMPLETE' and len(incomplete['missing_identities']) == 1
    assert incomplete['overall']['solve_difference_ci95'] is None
    assert incomplete['overall']['sign_flip'] is None
    assert not incomplete['qualification']['qualified_local_execution_enabled_benefit']
    assert incomplete['overall']['arms']['L']['solve_bounds'] == [239 / 240, 1]
    assert incomplete['overall']['costs']['returned_ids']['ratio_ci95'] is None
    assert incomplete['overall']['costs']['returned_ids']['L']['sum'] is None
    json.dumps(incomplete, allow_nan=False)
    # Known sign flips and all gain/loss IDs, using the full two-seed grid.
    changed = copy.deepcopy(original)
    changed[0]['grade'] = 'incorrect'  # gain in seed101
    changed[3]['grade'] = 'incorrect'  # loss in seed102, same task cluster
    task_id = changed[0]['id']
    # Censor savings dominate: all returned savings arise on capped L rows.
    for row in changed:
        if row['arm'] == 'L':
            row['finish_reason'] = 'length'
    censored = analysis.analyze(changed, tasks, integrity=integrity)
    assert censored['overall']['gains'] == [[task_id, 101]]
    assert censored['overall']['losses'] == [[task_id, 102]]
    assert censored['overall']['task_correct_count_differences'][task_id] == 0
    assert censored['overall']['solve_difference_ci95'] == [0, 0]  # both seeds stay together
    assert censored['overall']['costs']['returned_ids']['censoring']['over_25_percent']
    assert not censored['qualification']['qualified_local_execution_enabled_benefit']
    # Cost sensitivity reversal with exact, small paired sums.
    d, l = copy.deepcopy(original[0]), copy.deepcopy(original[1])
    l['native_accounting']['returned_tokens'] = 110
    cd, cl = copy.deepcopy(d), copy.deepcopy(l)
    cl['native_accounting']['returned_tokens'] = 1
    cl['finish_reason'] = 'length'
    cost = analysis._cost_report([(('unc', 101), d, l), (('cap', 101), cd, cl)], 0)
    assert cost['censoring']['complete_both_reversal']
    assert not cost['censoring']['uncensored_efficiency_guard']
    # Separate positive primary-accuracy case: six task clusters gain both seeds.
    # L costs more here, so case B cannot be the reason qualification succeeds.
    accuracy_rows = copy.deepcopy(rows)
    gain_ids = {t['id'] for t in tasks[:6]}
    for row in accuracy_rows:
        if row['arm'] == 'D' and row['id'] in gain_ids:
            row['grade'] = 'incorrect'
        if row['arm'] == 'L':
            row['native_accounting']['returned_tokens'] = 120
    accuracy = analysis.analyze(accuracy_rows, tasks, integrity=integrity)
    assert accuracy['overall']['solve_difference_L_minus_D'] == .05
    assert accuracy['overall']['sign_flip']['p_two_sided'] == .03125
    assert accuracy['overall']['solve_difference_ci95'][0] > 0
    assert accuracy['qualification']['case_A_primary_accuracy']
    assert not accuracy['qualification']['case_B_secondary_returned_ID_efficiency']
    assert accuracy['qualification']['qualified_local_execution_enabled_benefit']
    return {'fixture': FAKE, 'passed': True, 'study_generation': False, 'collection': False,
            'checks': ['strict_extraction', 'typed_answers', 'exact_task_DP', 'shared_cluster_CIs',
                       'incomplete_bounds', 'missing_input_cost', 'activation_receipt_join',
                       'all_gain_loss_IDs', 'censor_contributions_and_reversal', 'positive_case_A']}


if __name__ == '__main__':
    print(json.dumps(run_checks(), sort_keys=True, allow_nan=False))
