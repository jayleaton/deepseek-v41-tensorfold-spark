#!/usr/bin/env python3
"""Handwritten synthetic controls; never real T5 evidence. Parent runs on node.

Exactly one analyze() / 20,000-draw stream. No endpoints, generation or replay.
No artifacts are written. All fixtures here are explicitly synthetic.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import reporter as r


def native_contract_controls():
    """Synthetic receipt validation only; never claims actual codec qualification."""
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
    full = 'a' * 600
    prefix = full[:512]
    marker = '\n[output truncated at 512 tokens]'
    call = {'id': 'fake-native', 'code': 'print("synthetic")', 'status': 'ok',
            'output': prefix + marker}
    receipt = {
        'qualified': True, 'provenance': 'handwritten fake metadata, not real evidence',
        'presentation_verified': True, 'full_reference_verified': True,
        'complete_value': False, 'clipped': True, 'matches_bytes': True, 'in_text': True,
        'policy_transform_verified': True, 'native_prefix_exact': True,
        'frame_binding_verified': True, 'source_binding_verified': True,
        'native_status': 'ok', 'reference_status': 'ok', 'native_frame_sha256': 'f' * 64,
        'source_sha256': sha(call['code']), 'landed_output_sha256': sha(call['output']),
        'expected_presentation_sha256': sha(call['output']),
        'full_reference_sha256': sha(full), 'full_reference_bytes': len(full),
        'full_reference_body_tokens': 600, 'retained_prefix_tokens': 512,
        'retained_prefix_bytes': len(prefix), 'landed_output_bytes': len(call['output']),
        'marker_bytes': marker,
    }
    assert r.native_receipt_verified(receipt, call)
    # A policy clip qualifies presentation, never a fragment as a complete value.
    for change in ({'complete_value': True}, {'full_reference_verified': False},
                   {'presentation_verified': False}, {'policy_transform_verified': False},
                   {'native_prefix_exact': False}, {'frame_binding_verified': False},
                   {'source_binding_verified': False}, {'reference_status': 'error'},
                   {'native_status': 'error'}, {'native_frame_sha256': None},
                   {'landed_output_sha256': '0' * 64}, {'source_sha256': '0' * 64},
                   {'expected_presentation_sha256': '0' * 64},
                   {'marker_bytes': '[output truncated at 512 tokens]'},
                   {'full_reference_body_tokens': 512}, {'retained_prefix_tokens': 513}):
        assert not r.native_receipt_verified(dict(receipt, **change), call), change
    assert not r.native_receipt_verified(receipt, dict(call, output='b' + call['output'][1:]))
    # An independent-reference tail change can leave presentation unchanged.
    # This contract must not claim to detect invisible native tail differences.
    tail_change = dict(receipt, full_reference_sha256=sha(prefix + 'b' * 88))
    assert r.native_receipt_verified(tail_change, call)
    # A naturally printed marker below the cap is complete, not policy clipping.
    text = '[output truncated at 512 tokens]'
    complete_call = dict(call, output=text)
    complete = dict(receipt, clipped=False, complete_value=True, marker_bytes='',
                    full_reference_sha256=sha(text), landed_output_sha256=sha(text),
                    expected_presentation_sha256=sha(text), full_reference_bytes=len(text),
                    full_reference_body_tokens=20, retained_prefix_tokens=20,
                    retained_prefix_bytes=len(text), landed_output_bytes=len(text))
    assert r.native_receipt_verified(complete, complete_call)
    assert not r.native_receipt_verified(dict(complete, clipped=True), complete_call)
    return call, receipt


def fixture():
    tasks, answers = [], {}
    for family in r.FAMILIES:
        for slot in range(1, 11):
            task_id = f'compute-format-v1-{family}-{slot:03d}'
            tasks.append({'id': task_id, 'family': family, 'suite': 'compute_dl'})
            answers[task_id] = ([7, 2] if family == 'division' else
                                dict.fromkeys('ABCDEF', 7) if family == 'aggregation' else 7)
    by_id = {t['id']: t for t in tasks}
    rows = []
    for task_id, seed, arm in r.expected_grid(tasks):
        rows.append({'taskid': task_id, 'family': by_id[task_id]['family'], 'seed': seed,
                     'study_arm': arm, 'arm': 'lane' if arm.startswith('L') else 'plain',
                     'status': 'ok', 'content': json.dumps(answers[task_id]),
                     'native_accounting': {'complete': True, 'returned_tokens': 2},
                     'initial_prompt_reconciliation': {'qualified': True, 'initial_prompt_tokens': 10},
                     'elapsed_seconds': 1,
                     'call_accounting': {'complete': True, **dict.fromkeys(r.CALL_FIELDS, 0)}})
    return tasks, answers, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grader', type=Path, required=True)
    args = parser.parse_args()
    fake_call, fake_receipt = native_contract_controls()
    before = set(sys.modules)
    grader = r.load_grader(args.grader)
    grader2 = r.load_grader(args.grader)
    assert grader is not grader2 and grader.__name__ != grader2.__name__
    assert grader.__name__ not in sys.modules and grader2.__name__ not in sys.modules
    assert not any(name.startswith('_format_retest_grade_') for name in set(sys.modules) - before)
    tasks, answers, rows = fixture()
    task = next(t for t in tasks if t['family'] == 'integer')
    row = {'status': 'ok', 'content': '7'}
    def check(changes, expected):
        result = r.classify(dict(row, **changes), task, 7, grader)
        for field, value in expected.items():
            assert result[field] == value, (changes, field, result)
    check({'content': '{"answer":7}'}, {'strict_correct': False, 'format_failure': True,
                                       'selected_span_classification': 'wrong_shape'})
    check({'status': 'deadline', 'answer_eligible': False},
          {'strict_correct': False, 'format_failure': True, 'selected_span_classification': 'typed_exact_value'})
    check({'external_tool_calls': [{'name': 'python'}]}, {'strict_correct': False, 'format_failure': True})
    check({'content': '8'}, {'strict_correct': False, 'format_failure': False, 'typed_wrong_value': True})
    check({'content': '"7"'}, {'strict_correct': False, 'format_failure': True, 'typed_wrong_value': False})
    check({'content': ''}, {'strict_correct': False, 'format_failure': True, 'selected_span_classification': 'absent'})
    check({'content': 'not json'}, {'strict_correct': False, 'format_failure': True})
    check({'content': '7.0'}, {'strict_correct': False, 'format_failure': True, 'selected_span_error': 'json_float'})
    check({'content': '{"answer":7,"answer":7}'}, {'strict_correct': False, 'format_failure': True, 'selected_span_error': 'duplicate_key'})
    check({'content': '{"answer":7}\n7'}, {'strict_correct': True, 'format_failure': False, 'selected_span_source': 'last_line'})
    check({'content': '7\n8'}, {'strict_correct': False, 'format_failure': False, 'typed_wrong_value': True})
    check({'finish_reason': 'length'}, {'strict_correct': True, 'format_failure': False})
    check({'finish_reason': 'length', 'content': '{"answer":7}'}, {'strict_correct': False, 'format_failure': True})
    check({'finish_reason': 'length', 'answer_eligible': False}, {'strict_correct': False, 'format_failure': True})
    # Physical lane/plain aliases cannot become study arms or hide mislabelling.
    idx = r.normalize(rows, tasks, answers, grader)
    assert len(idx) == 360
    for bad in (dict(rows[0], arm='plain'), dict(rows[0], id='different'),
                dict(rows[0], study_arm='L'), dict(rows[0], seed=True)):
        try:
            r.normalize([bad], tasks, answers, grader)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid identity accepted')
    try:
        r.normalize([rows[0], rows[0]], tasks, answers, grader)
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate accepted')
    # A P loss plus a P gain leaves the aggregate unchanged and MUST fail.
    changed = copy.deepcopy(rows)
    pold = next(x for x in changed if x['study_arm'] == 'P-old' and x['family'] == 'dp' and x['seed'] == 101)
    pnew_loss = next(x for x in changed if x['taskid'] == pold['taskid'] and x['seed'] == 101 and x['study_arm'] == 'P-new')
    pnew_loss['content'] = '8'
    pold_gain = next(x for x in changed if x['study_arm'] == 'P-old' and x['family'] == 'graph' and x['seed'] == 101)
    pold_gain['content'] = '8'
    changed_idx = r.normalize(changed, tasks, answers, grader)
    p_tasks = [t for t in tasks if t['family'] in r.P_FAMILIES]
    pairs = r.paired(changed_idx, p_tasks, 'P-old', 'P-new', 'strict_correct')
    assert len(pairs['losses']) == len(pairs['gains']) == 1
    grid = r.expected_grid(tasks)
    arms = {a: r.summarize([k for k in grid if k[2] == a], changed_idx) for a in r.ARMS}
    assert arms['P-old']['strict_solves'] == arms['P-new']['strict_solves']
    assert r.acceptance(True, True, arms, pairs['losses']) == 'FAIL'
    assert r.acceptance(True, True, arms, []) == 'PASS'
    arms['L-new']['strict_solves'] -= 1
    assert r.acceptance(True, True, arms, []) == 'FAIL'
    arms['L-new']['strict_solves'] += 1
    for n, status in ((2, 'PASS'), (3, 'INCONCLUSIVE'), (6, 'INCONCLUSIVE'), (7, 'FAIL')):
        arms['L-new']['format_failures'] = n
        assert r.acceptance(True, True, arms, []) == status
    assert r.acceptance(False, True, arms, []) == 'INCOMPLETE'
    assert r.acceptance(False, False, arms, []) == 'BLOCKED'
    assert r.acceptance(True, False, arms, []) == 'BLOCKED'
    # A capped format failure is retained; no filtering from the denominator.
    capped = dict(rows[0], finish_reason='length', content='{"answer":7}')
    capped_idx = r.normalize([capped], tasks, answers, grader)
    s = r.summarize([r.canonical_key(capped)], capped_idx)
    assert s['scheduled'] == s['observed'] == s['caps'] == s['format_failures'] == 1
    # Every delivered result needs a separate clipping-aware replay receipt.
    safety_row = copy.deepcopy(next(x for x in rows if x['study_arm'] == 'L-new'))
    safety_row['call_accounting']['delivered_authenticated_results'] = 1
    safety_row['tensorfold'] = {'compute': {'calls': [{'id': 'fake-call', 'inserted': True, 'code': 'print(7)', 'output': '7'}]}}
    safety_row['safety'] = {'qualified': True, 'safe': True}
    q = r.qualification({}, r.normalize([safety_row], tasks, answers, grader), tasks)
    assert not q['qualified'] and q['delivered_results'] == 1 and q['replay_receipts'] == 0
    # Qualified native clip is distinct from unresolved cuts; overall fake audit
    # stays BLOCKED. r7 scope is zero owned runner cells, not native code coverage.
    safety_row['tensorfold']['compute']['calls'] = [dict(fake_call, inserted=True)]
    fake_receipt = dict(fake_receipt, **r.key_json(r.canonical_key(safety_row)), call_id=fake_call['id'])
    fake_audit = {
        'kind': 'handwritten-fake-controls',
        'r7': {'qualified': True, 'provenance': 'fake zero-owned oracle metadata',
               'scope': 'owned_runner_cells', 'all_optional_runners_off': True,
               'oracle_pinned': True, 'oracle_sha256': 'a' * 64,
               **dict.fromkeys(('owned_cell_count', 'checked_owned_cell_count', 'wrong_values',
                                'wrong_bytes', 'not_in_text', 'cuts', 'unknowns'), 0)},
        'native_replay': {'qualified': True, 'provenance': 'fake native metadata',
                          'independently_isolated_cpython': True, 'presentation_token_limit': 512,
                          'clipping_policy_qualified': True, 'policy_source_sha256': 'b' * 64,
                          'tokenizer_sha256': 'c' * 64, 'policy_clipped_results': 1,
                          'receipts': [fake_receipt],
                          **dict.fromkeys(('wrong_values', 'wrong_bytes', 'not_in_text',
                                           'unresolved_cuts', 'unknowns'), 0)},
    }
    safety_idx = r.normalize([safety_row], tasks, answers, grader)
    q = r.qualification(fake_audit, safety_idx, tasks)
    assert not q['qualified'] and q['presentation_verified_policy_clips'] == 1
    assert q['verified_complete_values'] == q['unresolved_native_results'] == 0
    assert q['native_results'][0]['classification'] == 'presentation-verified clipped'
    assert not any('r7 ' in reason or 'native replay' in reason or 'delivered replay' in reason
                   for reason in q['blocking_reasons'])
    for field in ('unresolved_cuts', 'unknowns'):
        bad = copy.deepcopy(fake_audit)
        bad['native_replay'][field] = 1
        q = r.qualification(bad, safety_idx, tasks)
        assert any('native replay' in reason for reason in q['blocking_reasons'])
    bad = copy.deepcopy(fake_audit)
    bad['r7']['scope'] = 'native_compute_code'
    assert any('r7 ' in reason for reason in r.qualification(bad, safety_idx, tasks)['blocking_reasons'])
    # Exactly one full 20k stream. Complete synthetic cohort still BLOCKED.
    report = r.analyze(rows, tasks, answers, {'kind': 'handwritten-fake-controls'}, grader)
    assert report['status'] == 'BLOCKED' and report['collection_status'] == 'COMPLETE'
    assert len(report['ledger']) == 360 and report['missing_keys'] == []
    boot = report['bootstrap']
    assert boot['primary_L_wording']['ci95'] == [0, 0]
    assert boot['primary_L_wording']['cost_ratios_right_over_left']['returned_ids']['ci95'] == [1, 1]
    assert boot['primary_L_wording']['cost_ratios_right_over_left']['full_input_ids']['ci95'] is None
    # Missing-key ledger and unknown cost behavior without another bootstrap.
    missing = grid[0]
    incomplete = dict(idx)
    del incomplete[missing]
    ledger = r.observation_ledger(tasks, incomplete)
    absent = next(x for x in ledger if (x['id'], x['seed'], x['study_arm']) == missing)
    assert len(ledger) == 360 and not absent['observed'] and absent['format_failure']
    assert absent['strict_reason'] == 'scheduled_missing' and absent['returned_ids'] is None
    s = r.summarize([missing], incomplete)
    assert s['missing'] == s['format_failures'] == 1 and s['strict_solves'] == 0
    assert s['costs']['returned_ids']['sum'] is None
    assert s['costs']['returned_ids']['known_sum'] is None
    assert s['costs']['returned_ids']['unknown_keys'] == [r.key_json(missing)]
    assert r.interval([None] * r.DRAWS) is None
    assert r.interval(list(range(r.DRAWS))) == [500, 19499]
    assert 'NEEDS-T5' in r.markdown(report)
    print('Handwritten synthetic controls passed. NEEDS-T5: no production or safety evidence generated.')


if __name__ == '__main__':
    main()
