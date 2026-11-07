#!/usr/bin/env python3
"""A1 exact-wire transport and conservative prompt accounting; no implicit GO."""
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'phase2'))
import client as old
from reconciliation import wire

canonical = old.canonical
NativeDecoder = old.NativeDecoder


def assemble_attempts(frames):
    calls, malformed = {}, []
    for frame in frames:
        for choice in frame.get('choices', []):
            for delta in (choice.get('delta') or {}).get('tool_calls') or []:
                index = delta.get('index')
                if type(index) is not int or index < 0:
                    malformed.append(delta); continue
                key = (choice.get('index', 0), index)
                item = calls.setdefault(key, {'choice': key[0], 'index': index, 'id': '',
                                             'type': '', 'name': '', 'arguments': ''})
                for field in ('id', 'type'):
                    if field in delta:
                        if not isinstance(delta[field], str): malformed.append(delta)
                        else: item[field] += delta[field]
                for field in ('name', 'arguments'):
                    value = (delta.get('function') or {}).get(field)
                    if value is not None:
                        if not isinstance(value, str): malformed.append(delta)
                        else: item[field] += value
    return {'calls': [calls[k] for k in sorted(calls)], 'malformed': malformed}


def accounting(record, cap=8192):
    # A valid execution-off tool-call termination is scored incorrect, but its
    # observed native costs remain valid. Transport failure cannot be promoted.
    adjusted = dict(record, status=record.get('transport_status', record.get('status')))
    return old.accounting(adjusted, cap)


def generation_errors(record):
    errors = []
    for index, frame in enumerate(record.get('sse_frames') or []):
        if frame.get('error') is not None:
            errors.append({'frame': index, 'error': frame['error']})
        for choice in frame.get('choices') or []:
            if choice.get('error') is not None or (choice.get('delta') or {}).get('error') is not None:
                errors.append({'frame':index, 'choice':choice})
    if not record.get('sse_done'):
        errors.append({'framing':'missing DONE'})
    if record.get('finish_reason') is None or record.get('finish_reason') in ('error','failed','cancelled'):
        errors.append({'finish_reason':record.get('finish_reason')})
    return errors


def call_accounting(record, decoder):
    """Source-qualified call counts; missing/ambiguous fields stay unavailable."""
    fields = ('attempted','blocked_dispatches','successful_executions','failed_executions',
              'delivered_authenticated_results','reply_used_result')
    result = dict.fromkeys(fields)
    result.update(complete=False, coverage=dict.fromkeys(fields,False),
                  blocked_semantics='source-inferred prevented D call attempts, not a deployed dispatch-spy counter')
    binding = record['native_binding']
    if not (binding.get('qualified') and record['native_accounting']['complete']
            and record.get('sse_done') and record.get('transport_status')=='ok'):
        return result
    reports = ((record.get('tensorfold') or {}).get('compute') or {}).get('calls',[])
    attempts = record['attempts']
    if record['arm'] in ('plain','declared_off'):
        if reports or attempts['malformed']:
            return result
        n = len(attempts['calls'])
        result.update(attempted=n, blocked_dispatches=n if record['arm']=='declared_off' else 0,
                      successful_executions=0,failed_executions=0,
                      delivered_authenticated_results=0,reply_used_result=0)
    else:
        bound = binding.get('calls',[])
        all_bound = len(bound)==len(reports) and all(c.get('native_bound') for c in bound)
        if all_bound:
            result['successful_executions'] = sum(c.get('status')=='ok' for c in reports)
            failed_statuses = {'exit_error','timeout','output_limit'}
            if all(c.get('status') in failed_statuses|{'ok'} for c in reports):
                result['failed_executions'] = sum(c.get('status') in failed_statuses for c in reports)
            result['delivered_authenticated_results'] = sum(c.get('inserted') is True for c in reports)
            result['reply_used_result'] = int(result['delivered_authenticated_results']>0)
        result['blocked_dispatches'] = 0 if not reports or all(c.get('status')!='unavailable' for c in reports) else None
        # A terminal call can be withheld from SSE at the cap without a report.
        # Never silently classify that unreported attempt as zero.
        try:
            ids = record['tensorfold']['token_ids']
            start = binding.get('frames',[])[-1]['end_token'] if binding.get('frames') else 0
            tail = decoder.decode(ids[start:])
            final = tail.rsplit(b'</think>',1)[-1]
            native_call = '<｜DSML｜'.encode() in final
            if not native_call and all_bound:
                result['attempted'] = len(reports)
        except (KeyError, ValueError, TypeError, AttributeError):
            pass
    result['coverage'] = {field:type(result[field]) is int for field in fields}
    result['complete'] = all(result['coverage'].values())
    return result


def authenticate_calls(record, decoder):
    copy = dict(record)
    external = bool(record.get('external_tool_calls'))
    if external and record['arm'] != 'declared_off':
        return {'qualified': False, 'reason': 'unexpected client-visible call', 'delivered_successes': 0}
    if external:
        # The old authenticator's remaining checks prove absence of forced tool
        # result frames and reconciled native IDs; they do not bind D call source.
        copy['external_tool_calls'] = []
    result = old.authenticate_calls(copy, decoder)
    result['external_attempts_present'] = external
    result['external_attempt_source_native_bound'] = False if external else None
    return result


def input_accounting(record, binding):
    recon = record['initial_prompt_reconciliation']
    tf = record.get('tensorfold') or {}
    compute = tf.get('compute')
    known_zero = compute is None or (isinstance(compute, dict) and compute.get('calls') == []
                                     and compute.get('inserted_tokens') == 0)
    single = (binding.get('qualified') is True and known_zero
              and binding.get('native_frame_count') == 0
              and record.get('transport_status') == 'ok'
              and record.get('sse_done') is True)
    initial = recon['server_initial_prompt_tokens']
    returned = record['native_accounting']['returned_tokens']
    complete = bool(single and recon['qualified_under_proposed_count_contract']
                    and record['native_accounting']['complete'])
    return {'classification': 'SINGLE-ROUND' if single else 'MULTI-ROUND/INPUT-ACCOUNTING-UNKNOWN',
            'complete': complete, 'initial_prompt_tokens': initial,
            'qualified': complete, 'processed_input_tokens': initial if complete else None,
            'provenance': 'pinned3a chat.generate/compute_segments: authenticated zero forced frames and no input-bearing continuation' if complete else None,
            'prompt_tokens': initial if complete else None,
            'prompt_plus_returned_tokens': initial + returned if complete else None,
            'served_array_equality_proven': False,
            'reason': 'source-qualified no input-bearing continuation' if complete else
                      'full input ledger unavailable or evidence incomplete; no imputation'}


class Transport:
    def __init__(self, endpoint, go_file, spec_file, spec_sha, timeout=360):
        old.Transport(endpoint, go_file, spec_file, spec_sha, timeout)
        self.endpoint, self.go = endpoint, Path(go_file)
        self.spec, self.spec_sha, self.timeout = Path(spec_file), spec_sha, timeout

    def __call__(self, body):
        if self.go.read_text().strip() != self.spec_sha or hashlib.sha256(self.spec.read_bytes()).hexdigest() != self.spec_sha:
            raise ValueError('externally frozen A1 GO binding differs')
        if body.get('stream') is not True or body.get('return_token_ids') is not True:
            raise ValueError('native streaming required')
        if body.get('max_tokens') != 8192 or body.get('deadline_s') != 270:
            raise ValueError('cap/deadline differs')
        if signal.getitimer(signal.ITIMER_REAL)[0]:
            raise ValueError('unexpected active alarm')
        def expired(_sig, _frame): raise TimeoutError('whole-request client deadline')
        previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, self.timeout)
        start = time.monotonic()
        sent = wire(body)
        row = {'status': 'error', 'transport_status': 'error', 'raw_response': None,
               'sse_frames': [], 'sse_done': False, 'content': '', 'reasoning': '', 'usage': None,
               'tensorfold': {}, 'finish_reason': None, 'actual_request_wire': sent,
               'actual_request_wire_bytes': len(sent.encode()),
               'actual_request_wire_sha256': hashlib.sha256(sent.encode()).hexdigest()}
        try:
            request = urllib.request.Request(self.endpoint, data=sent.encode(),
                                             headers={'Content-Type':'application/json'}, method='POST')
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                row.update(old.parse_stream(response, start, row['sse_frames']))
                row.update(status='ok', transport_status='ok', http_status=response.status)
        except (TimeoutError, OSError, ValueError, urllib.error.HTTPError) as error:
            if row['sse_frames']:
                row.update(old.parse_stream(['data: '+canonical(f) for f in list(row['sse_frames'])]))
            row['status'] = 'timeout' if isinstance(error, TimeoutError) else 'error'
            row['transport_status'] = row['status']
            row['error'] = type(error).__name__ + ': ' + str(error)
        finally:
            row['elapsed_seconds'] = time.monotonic()-start
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        row['attempts'] = assemble_attempts(row['sse_frames'])
        row['generation_errors'] = generation_errors(row)
        if row['transport_status']=='ok' and row['generation_errors']:
            row['status']='error'
            row['error']='native SSE generation error or incomplete completion framing'
        row['native_accounting'] = accounting(row)
        return row
