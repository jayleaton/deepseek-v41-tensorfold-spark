#!/usr/bin/env python3
"""Frozen SSE transport and native whole-continuation accounting. No implicit endpoint."""
import argparse
import hashlib
import io
import json
import signal
import time
import urllib.error
import urllib.request
from pathlib import Path


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def sha_ids(ids):
    return hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()


def parse_stream(lines, started=None, frames=None):
    content, reasoning, calls = [], [], []
    frames = [] if frames is None else frames
    usage, runtime, finish, response_id, done = None, {}, None, None, False
    tool_calls = []
    for raw in lines:
        line = raw.decode('utf-8').strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith('data:'):
            continue
        text = line[5:].strip()
        if text == '[DONE]':
            done = True
            break
        frame = json.loads(text)
        frames.append(frame)
        response_id = frame.get('id', response_id)
        if frame.get('usage'):
            usage = frame['usage']
        if isinstance(frame.get('tensorfold'), dict):
            runtime.update(frame['tensorfold'])
        for choice in frame.get('choices', []):
            delta = choice.get('delta') or {}
            content.append(delta.get('content') or '')
            reasoning.append(delta.get('reasoning_content') or '')
            if delta.get('tool_calls'):
                tool_calls.extend(delta['tool_calls'])
            if isinstance(delta.get('compute'), dict):
                calls.append(delta['compute'])
            if choice.get('finish_reason'):
                finish = choice['finish_reason']
        if isinstance(frame.get('compute'), dict):
            calls.append(frame['compute'])
    # The server's final merged runtime is authoritative; incremental call deltas remain in frames.
    body = {'id': response_id, 'choices': [{'index': 0, 'message': {
        'role': 'assistant', 'content': ''.join(content), 'reasoning_content': ''.join(reasoning)},
        'finish_reason': finish}], 'usage': usage, 'tensorfold': runtime}
    if tool_calls:
        body['choices'][0]['message']['tool_calls'] = tool_calls
    return {'raw_response': canonical(body), 'sse_frames': frames, 'sse_done': done,
            'content': ''.join(content), 'reasoning': ''.join(reasoning), 'finish_reason': finish,
            'usage': usage, 'tensorfold': runtime, 'external_tool_calls': tool_calls,
            'live_compute_reports': calls}


def accounting(record, cap=8192):
    tf, usage = record.get('tensorfold') or {}, record.get('usage') or {}
    ids, digest = tf.get('token_ids'), tf.get('token_sha256')
    native = isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids)
    hash_ok = bool(native and isinstance(digest, str) and sha_ids(ids) == digest)
    compute = tf.get('compute') if 'compute' in tf else None
    shape_ok = 'compute' not in tf or (isinstance(compute, dict) and
        isinstance(compute.get('calls'), list) and type(compute.get('prefix_tokens')) is int and
        compute['prefix_tokens'] >= 0 and 'inserted_tokens' in compute)
    # Frozen engine omits compute entirely on zero calls/insertions; a present
    # but incomplete object is missing evidence, never a zero-cost observation.
    inserted = compute.get('inserted_tokens') if isinstance(compute, dict) else (0 if 'compute' not in tf else None)
    completion = usage.get('completion_tokens')
    valid_counts = shape_ok and type(completion) is int and completion >= 0 and type(inserted) is int and inserted >= 0
    reconcile = bool(native and valid_counts and len(ids) == completion + inserted)
    within_cap = bool(native and len(ids) <= cap)
    # Missing/error/deadline frames cannot manufacture zero-cost evidence.
    complete = bool(record.get('status') == 'ok' and record.get('sse_done') and hash_ok and reconcile and within_cap)
    return {'complete': complete, 'returned_tokens': len(ids) if native else None,
            'hash_matches': hash_ok, 'api_completion_tokens': completion,
            'inserted_tokens': inserted, 'ids_equal_completion_plus_inserted': reconcile,
            'within_aggregate_cap': within_cap, 'coverage_reason':
            'complete' if complete else 'missing/invalid native IDs, hash, usage, framing or aggregate-cap evidence'}


class Transport:
    def __init__(self, endpoint, go_file, spec_file, spec_sha, timeout=360):
        if endpoint != 'http://127.0.0.1:8000/v1/chat/completions':
            raise ValueError('only the explicitly approved rank0-host direct endpoint is permitted')
        self.endpoint, self.go = endpoint, Path(go_file)
        self.spec, self.spec_sha, self.timeout = Path(spec_file), spec_sha, timeout
        if timeout < 330:
            raise ValueError('spec requires client wait >=330 seconds')

    def __call__(self, body):
        if self.go.read_text().strip() != self.spec_sha:
            raise ValueError('GO file must hold the externally supplied frozen spec SHA')
        if hashlib.sha256(self.spec.read_bytes()).hexdigest() != self.spec_sha:
            raise ValueError('frozen spec changed')
        if body.get('stream') is not True or body.get('return_token_ids') is not True:
            raise ValueError('native-ID streaming contract required')
        if body.get('max_tokens') != 8192 or body.get('deadline_s') != 270:
            raise ValueError('aggregate cap/deadline differs from frozen spec')
        if signal.getitimer(signal.ITIMER_REAL)[0]:
            raise ValueError('unexpected active alarm')
        def expired(_sig, _frame):
            raise TimeoutError('whole-request client deadline')
        previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, self.timeout)
        start = time.monotonic()
        row = {'status': 'error', 'raw_response': None, 'sse_frames': [], 'sse_done': False,
               'content': '', 'reasoning': '', 'usage': None, 'tensorfold': {}, 'finish_reason': None}
        try:
            request = urllib.request.Request(self.endpoint, data=canonical(body).encode(),
                                             headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                row.update(parse_stream(response, start, row['sse_frames']))
                row['status'] = 'ok' if not row.get('external_tool_calls') else 'error'
                row['http_status'] = response.status
                if row.get('external_tool_calls'):
                    row['error'] = 'unexpected external client tool calls'
        except (TimeoutError, OSError, ValueError, urllib.error.HTTPError) as error:
            if row['sse_frames']:
                saved = list(row['sse_frames'])
                row.update(parse_stream(['data: ' + canonical(x) for x in saved]))
            row['status'] = 'timeout' if isinstance(error, TimeoutError) else 'error'
            row['error'] = type(error).__name__ + ': ' + str(error)
            # Frames parsed before a cancellation are not synthesized into a complete reply.
        finally:
            row['elapsed_seconds'] = time.monotonic() - start
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        row['native_accounting'] = accounting(row, body['max_tokens'])
        return row


class NativeDecoder:
    """Decode IDs directly from the pinned tokenizer vocabulary, without retokenizing output."""
    def __init__(self, path, expected_sha):
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_sha:
            raise ValueError('tokenizer hash drift')
        obj = json.loads(data)
        alphabet = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        chars, number = alphabet[:], 0
        for value in range(256):
            if value not in alphabet:
                alphabet.append(value)
                chars.append(256 + number)
                number += 1
        inverse = {chr(c): b for b, c in zip(alphabet, chars)}
        added = {x['id']: x['content'].encode() for x in obj.get('added_tokens', [])}
        self.pieces = {index: bytes(inverse[c] for c in token)
                       for token, index in obj['model']['vocab'].items() if index not in added}
        self.pieces.update(added)

    def decode(self, ids):
        return b''.join(self.pieces[i] for i in ids)


def authenticate_calls(record, decoder):
    from native_frames import authenticate
    return authenticate(record, decoder)


def selftest(out):
    ids = [11, 12, 13, 14, 15]
    tf = {'token_ids': ids, 'token_sha256': sha_ids(ids), 'compute': {'inserted_tokens': 2, 'prefix_tokens': 4, 'calls': []}}
    frames = [{'choices': [{'delta': {'content': '323'}}]},
              {'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 3}, 'tensorfold': tf}]
    lines = [b'data: ' + canonical(x).encode() + b'\n' for x in frames] + [b'data: [DONE]\n']
    row = parse_stream(lines)
    row['status'] = 'ok'
    assert accounting(row)['complete'] and accounting(row)['returned_tokens'] == 5
    bad = dict(row, tensorfold=dict(tf, token_sha256='bad'))
    assert not accounting(bad)['complete']
    assert not accounting(dict(row, usage=None))['complete']
    assert not accounting(row, 4)['complete']
    assert not accounting(dict(row, sse_done=False))['complete']
    assert not accounting(dict(row, tensorfold=dict(tf, compute={})))['complete']
    Path(out).write_text(json.dumps({'passed': True, 'network': False,
        'whole_native_count': 5, 'completion_only': 3, 'missing_hash_usage_cap_framing_rejected': True}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest', required=True)
    selftest(parser.parse_args().selftest)
