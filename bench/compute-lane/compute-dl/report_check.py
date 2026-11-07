#!/usr/bin/env python3
"""Parent CPU-scope fake controls only. No network, inference or real task generation.

Run via cpu_checks.py report under the parent's node CPU guard. Never auto-runs
on import. Temporary fixture artifacts are automatically removed.
"""
import base64
import copy
import json
from pathlib import Path
from unittest.mock import patch
import collect_report as report


def main():
    root = Path(__file__).resolve().parent
    assert Path(report.analysis.__file__).resolve() == root / 'analysis.py'
    assert Path(report.grade.__file__).resolve() == root / 'grade.py'
    assert report.analysis.SPEC_SHA256 == report.grade.SPEC_SHA256 == report.SPEC_SHA256
    task = {'suite': 'compute_dl', 'id': 'FAKE_NOT_COLLECTION', 'family': 'integer',
            'inputs': {'a': 2, 'b': 2, 'c': 0}, 'prompt': 'Fixture only: 2*2.'}
    body = {'stream': True, 'return_token_ids': True, 'max_tokens': 8192}
    wire = report.reconciliation.wire(body)
    template = {'suite': task['suite'], 'id': task['id'], 'seed': 101,
                'arm': 'declared_off', 'boot': 't01-D', 'key': [task['suite'], task['id'], 101],
                'task': task, 'task_sha256': report.audit_records.object_sha(task),
                'body_template': body, 'wire': wire, 'wire_sha256': report.digest(wire.encode()),
                'execution_policy': {'TF_DSV41_COMPUTE_LANE': '0'}}
    ids = [1, 2, 3]
    frame = {'choices': [{'index': 0, 'delta': {'content': '4'}, 'finish_reason': 'stop'}],
             'usage': {'prompt_tokens': 2, 'completion_tokens': 3},
             'tensorfold': {'token_ids': ids, 'token_sha256': report.client3.old.sha_ids(ids)}}
    row = report.client3.old.parse_stream(['data: ' + json.dumps(frame)])
    row.update({k: template[k] for k in ('suite', 'id', 'seed', 'arm', 'boot', 'task_sha256', 'execution_policy')})
    row.update(status='ok', transport_status='ok', diagnostic=False, sse_done=True,
               answer_eligible=True, elapsed_seconds=1.5, request=body,
               actual_request_wire=wire, actual_request_wire_bytes=len(wire.encode()),
               actual_request_wire_sha256=report.digest(wire.encode()))
    prompt = {'key': template['key'], 'declared_off_ids': [6, 7]}
    binding = {'qualified': True, 'native_frame_count': 0, 'calls': [], 'delivered_successes': 0}
    row['initial_prompt_reconciliation'] = report.reconciliation.reconcile(row, dict(template, body=body), prompt)
    row['native_accounting'] = report.client3.accounting(row, 8192)
    row['attempts'] = report.client3.assemble_attempts(row['sse_frames'])
    row['native_binding'] = binding
    row['input_accounting'] = report.client3.input_accounting(row, binding)
    row['call_accounting'] = report.client3.call_accounting(row, None)
    safety = {'binding': binding, 'qualified': True, 'references': []}
    worker_sha = '0' * 64
    with patch.object(report.client3, 'authenticate_calls', return_value=binding):
        checked, audit = report.audit_row(row, safety, template, prompt, None, worker_sha)
        assert audit['qualified'], audit
        assert checked['safety']['qualified']
        assert checked['server_seconds'] is None and checked['tool_execution_seconds'] is None
        assert checked['elapsed_seconds'] == 1.5
        assert report.grade.grade(checked, task, 4)['grade'] == 'correct'
        assert report.analysis.returned_tokens(checked) == 3
        assert report.analysis.initial_tokens(checked) == 2
        missing, audit = report.audit_row(row, None, template, prompt, None, worker_sha)
        assert not audit['qualified'] and not missing['safety']['qualified']
        assert report.analysis.returned_tokens(missing) == 3  # own native evidence remains complete
        for field, value in [('actual_request_wire', wire + ' '), ('task_sha256', 'bad'),
                             ('content', '5'), ('native_binding', {'qualified': True, 'calls': ['counter-only']})]:
            mutant = copy.deepcopy(row); mutant[field] = value
            _, audit = report.audit_row(mutant, safety, template, prompt, None, worker_sha)
            assert not audit['qualified'], field
    call = {'id': 'fake-call', 'code': 'print(4)', 'output': '4', 'status': 'ok', 'inserted': True}
    replay = {'status': 'ok', 'normalized_body_base64': base64.b64encode(b'4').decode()}
    assert report.reference_matches(call, replay)
    assert not report.reference_matches(dict(call, output='5'), replay)
    assert not report.reference_matches(call, dict(replay, status='exit_error'))
    failed_call=dict(call,status='exit_error',output='retained error')
    failed_replay={'status':'exit_error','normalized_body_base64':base64.b64encode(b'retained error').decode(),
                   'payload_sha256':report.audit_records.object_sha({'mode':'reference','code':call['code']}),
                   'worker_sha256':worker_sha,
                   'image':{'id':sorted(report.audit_records.IMAGE_IDS)[0],'revision':report.audit_records.REVISION}}
    failed_safety={'qualified':True,'references':[{'call_id':call['id'],'matches':True,'result':failed_replay}]}
    assert report.reference_audit([failed_call],failed_safety,worker_sha)==[]
    assert report.reference_audit([dict(failed_call,output='wrong')],failed_safety,worker_sha)
    assert report.reference_audit([failed_call],dict(failed_safety,references=[]),worker_sha)
    bound={'qualified':True,'calls':[{'id':call['id'],'native_bound':True,'delivered':True,'status':'exit_error',
           'source_sha256':report.digest(call['code'].encode()),'output_sha256':report.digest(b'retained error')}]}
    failed_safety['binding']=bound
    failed_row={'id':'FAKE','seed':101,'arm':'lane','native_binding':bound,'safety':failed_safety,
                'tensorfold':{'compute':{'calls':[failed_call]}}}
    assert report.analysis._row_safe(failed_row)
    assert not report.analysis._verified_activation(failed_row)
    # An asserted matches bit cannot rescue wrong bytes or missing provenance.
    wrong_safety = {'references': [{'call_id': call['id'], 'matches': True, 'result': replay}]}
    assert report.audit_records.reference_audit([call], wrong_safety, report.reference_matches, worker_sha)
    calls = [{'id': 'fake-call', 'status': 'ok', 'inserted': True}]
    d = copy.deepcopy(row)
    d['sse_frames'][0]['tensorfold']['compute'] = {'calls': calls, 'inserted_tokens': 0}
    d['tensorfold']['compute'] = {'calls': calls, 'inserted_tokens': 0}
    with patch.object(report.client3, 'authenticate_calls', return_value=binding):
        _, audit = report.audit_row(d, safety, template, prompt, None, worker_sha)
    assert any('D EXECUTION' in gap for gap in audit['gaps'])
    # The renderer consumes actual computed values, including null costs and bounds.
    arm = {'solved': 0, 'scheduled': 240}
    mixture = {'arms': {'D': arm, 'L': arm}, 'solve_difference_L_minus_D': None,
               'solve_difference_ci95': None, 'difference_bounds': [-1, 1], 'sign_flip': None,
               'gains': [], 'losses': [], 'complete_both_uncensored_sensitivity': {},
               'costs': {'returned_ids': {'ratio_L_over_D': None, 'ratio_ci95': None}}}
    artifact = {'integrity': {'qualified': False, 'gaps': ['missing evidence']}, 'budget': None,
                'analysis': {'status': 'INCOMPLETE', 'observed': 0, 'missing_identities': [None] * 480,
                    'overall': mixture, 'families': {}, 'qualification': {'integrity': False},
                    'activation': {'distinct_tasks': 0, 'status': 'INSUFFICIENT EXECUTION EXPOSURE'},
                    'interpretation': 'FAKE_NOT_COLLECTION'}}
    markdown = report.render_results(artifact)
    for text in ('INCOMPLETE', '0 / 480', report.SPEC_SHA256, 'not directly observed',
                 'historical context only', 'unknown costs', 'null', '[-1, 1]'):
        assert text in markdown, text
    print(json.dumps({'passed': True, 'status': 'FAKE_NOT_COLLECTION',
                      'own_analysis_grader_paths': True, 'actual_wire_and_SSE_recomputed': True,
                      'missing_safety_wrong_delivery_D_execution_rejected': True,
                      'unknown_timing_preserved': True, 'real_network': False,
                      'real_tasks_generated': 0}))


if __name__ == '__main__':
    main()
