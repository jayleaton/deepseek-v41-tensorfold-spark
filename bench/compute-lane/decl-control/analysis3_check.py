#!/usr/bin/env python3
"""CPU-only fake checks. Parent runs on admitted Spark, never on the dev box.

No subprocess, model/network requests, sandbox, external attempt execution or GPU.
Run: PYTHONDONTWRITEBYTECODE=1 python3 analysis3_check.py
"""
import copy
import itertools

import analysis3 as a
import grade3 as g
import math3 as m


def rejects(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('invalid input accepted')


def selfcheck():
    hashes = g.verify_sources()
    rejects(lambda: g.verify_sources(dict(hashes, wrapper='changed')))
    for text, reference in [('Sam gets 20 feet.', '20'), ('9 eggs per child', '9'),
                            ('Final answer: .5 dollars', '.5'), ('20', '20')]:
        assert g.gsm_sensitivity({'answer': reference if reference != '.5' else '0.5'}, text)['grade'] == 'correct'
    for text in ('20 or 30', '20 million', 'about 20 cookies', '18 hours 48 minutes',
                 '6 scoops (buy 4, get 2 free)', '20 percent', '1/0', '#### 20\nMaybe 30'):
        assert g.gsm_sensitivity({'answer': '20'}, text)['grade'] == 'incorrect'
    assert g.gsm_sensitivity({'answer': '20'}, 'Sam gets 20 feet.')['changed']
    assert not g.gsm_sensitivity({'answer': '20'}, '20')['changed']
    for differences in ([], [0, 0], [1], [2, -1], [2, -2, 1, 0]):
        totals = [sum(s * abs(w) for s, w in zip(signs, differences))
                  for signs in itertools.product((-1, 1), repeat=len(differences))]
        expected = sum(abs(t) >= abs(sum(differences)) for t in totals) / len(totals)
        assert a.phase2.exact_sign_flip(differences)['p_two_sided'] == expected
    assert a.phase2.percentile_interval(list(range(20000))) == [500, 19499]
    tasks = [{'suite': suite, 'id': str(i), 'kind': kind, 'answer': '20', 'prompt': 'fixture'}
             for suite, kind in [('gsm', 'numeric'), ('math', 'math_manual'), ('he', 'humaneval')]
             for i in range(2)]
    rows = [dict(suite=t['suite'], id=t['id'], seed=seed, arm=arm, status='ok',
                 content='20', grade='correct', elapsed_seconds=3, finish_reason='stop',
                 native_accounting={'complete': True, 'returned_tokens': 10},
                 initial_prompt_reconciliation={'qualified': True, 'initial_prompt_tokens': 5},
                 input_accounting={'classification': 'single_input', 'source_control_flow_verified': True,
                                   'no_additional_input': True, 'provenance': 'CPU FAKE ONLY'})
            for t in tasks for seed in a.SEEDS for arm in a.ARMS]
    manifest = {'seeds': list(a.SEEDS), 'tasks': tasks}
    class FakeJail:
        calls = 0
        def grade_humaneval(self, task, content):
            self.calls += 1
            return {'grade': 'correct', 'status': 'ok', 'harness_receipt': True}
    jail = FakeJail()
    graded = g.grade_all(rows, manifest, 'fake', _trusted_selftest=True, _trusted_jail=jail)
    assert jail.calls == 12
    terminated = [dict(r, status='policy_terminated') for r in rows]
    failed = g.grade_all(terminated, manifest, 'fake', _trusted_selftest=True, _trusted_jail=jail)
    assert all(r['grade'] == 'incorrect' for r in failed if r['suite'] != 'math')
    assert jail.calls == 12  # never run code in an undelivered answer
    for bad in (rows[:-1], rows + [rows[0]]):
        rejects(lambda: a.validate_cohort(bad, manifest, _trusted_selftest=True))
    result = a.analyze(rows, manifest, _trusted_selftest=True)
    assert result['primary_family_size'] == 6
    for suite in result['suites'].values():
        for name, contrast in suite['contrasts'].items():
            assert contrast['accuracy_ci95'] == [0, 0]
            assert contrast['returned_ids']['ratio_ci95'] == [1, 1]
            assert contrast['prompt_plus_returned']['ratio_ci95'] == [1, 1]
            assert ('holm_adjusted_p_family6' in contrast['sign_flip']) == (name != 'L-P')
            assert not contrast['qualified_efficacy_claim']
    assert a.input_tokens(dict(rows[0], input_accounting={})) is None
    assert a.input_tokens(dict(rows[0], input_accounting={'classification': 'single_input'})) is None
    assert a.input_tokens(dict(rows[0], input_accounting={'classification': 'multi_input'})) is None
    assert a.phase2.native_tokens(dict(rows[0], native_accounting=None)) is None
    # Partial metric selection keeps the original two task clusters and its missing IDs.
    partial = copy.deepcopy(rows)
    partial[0]['input_accounting'] = {'classification': 'MULTI-ROUND/INPUT-ACCOUNTING-UNKNOWN'}
    partial[0]['native_accounting'] = {'complete': False, 'returned_tokens': None}
    report = a.analyze(partial, manifest, _trusted_selftest=True)
    r = report['suites']['gsm']['contrasts']['D-P']
    assert r['prompt_plus_returned']['full_cohort_claim_withheld']
    assert r['prompt_plus_returned']['ratio_ci95'] is None
    assert r['returned_ids']['ratio_ci95'] is None
    assert r['wall_seconds']['complete_pairs'] == 4
    assert r['accuracy_ci95'] == [0, 0]
    assert report['suites']['gsm']['arms']['plain']['calls']['attempted']['known_sum'] is None
    censored = [[(dict(rows[0], elapsed_seconds=10),
                   dict(rows[0], elapsed_seconds=5, finish_reason='length'))]]
    censor_report = a.metric_report(censored, lambda row: row['elapsed_seconds'])
    assert censor_report['savings']['over_25_percent_warning']
    assert censor_report['censoring_reversal']
    assert m.selftest()['fixture_candidates'] == 12
    preseal = m.build_preseal([t for t in tasks if t['suite'] == 'math'], _trusted_selftest=True)
    assert len(preseal['mapping']) == 12 and preseal['namespace'] == 'decl-control-v1'
    assert all(k.startswith('decl-control-v1-') for k in preseal['mapping'])
    assert preseal == m.build_preseal([t for t in reversed(tasks) if t['suite'] == 'math'], _trusted_selftest=True)
    return {'fake_checks': 'passed', 'external_execution': False,
            'qualification': 'synthetic controls only; no collection or safety qualification',
            'grader_hashes': hashes}


if __name__ == '__main__':
    print(a.canonical(selfcheck()))
