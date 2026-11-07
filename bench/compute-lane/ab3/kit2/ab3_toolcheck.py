#!/usr/bin/env python3
"""AB3 tool-result check (T5 spec, "Safety, deadlines and censoring"): every tool call in a set of reply receipts,
S3 (in-model runner) and sandbox (code tool, lane) alike, is located in the reply's native returned token stream,
its emitted code and its inserted result bytes authenticated there, and the result checked against an independent
re-run of the exact emitted code in the grader jail. This script never runs model code itself.

    python3 -I ab3_toolcheck.py jobs   TOKENIZER OUT_JOBS.jsonl RECEIPTS...             the calls to re-run
    python3 -I run_jobs2.py JOBS.jsonl RESULTS.jsonl                                     (on a node: the jail)
    python3 -I ab3_toolcheck.py verify TOKENIZER RESULTS.jsonl OUT_PREFIX RECEIPTS...   OUT_PREFIX.json and .md

RECEIPTS are reply JSON files, directories of them (arm.json and files without `extra` are skipped) or globs; each
receipt's directory is one arm/run in the summary. Verdict per call:
  ok           delivered bytes authenticated in the stream and == normalize(re-run)
  wrong        an ok call whose delivered result differs from the re-run, or an error not attributed as one
  error-ok     a failed call, attributed by its status line, whose result reproduces or asserts no number
  inconclusive not located/authenticated in the stream, re-run missing or failed in the jail, a mismatch on code
               that may be nondeterministic (time, random, ...), or missing fields
  undelivered  a call whose result the engine did not insert (no room); nothing reached the model
An arm PASSes only with zero wrong, zero inconclusive and no receipt-level gap; any wrong is FAIL, else INCONCLUSIVE.

Pinned transport (engine tf-imr-commit at 3bf584af1c5407006f154e0c70d05c6f495d0487):
  result text  zig/src/core/compute_runner.zig:35-59 `body` (S3: core/compute_circuit.zig:135-137, same rule),
               cut to max_result_tokens by server/compute_segments.zig:213-222 `capTokens`
  inserted ids server/compute_segments.zig:260-276 `suffix` = render(history+call turn+tool turns) past
               render(history+call turn); DeepSeek-V4.1 template families/deepseek_v41/serve/template.zig:441-446
               (tool turns merged into one user turn, "\\n\\n" between results) and :460-464 (generation prompt;
               tools offered, so thinking stays open: "<｜Assistant｜><think>"); typed image markers escaped (:19-20)
  stream       compute_segments.zig:284-321: token_ids = every segment's ids + each inserted suffix, token_sha256
               over them (server/tokens.zig:106-116, decimal ids joined by ","); S3 fields :186-198 (`executor`,
               `runner_abstain` = S3 declined, the sandbox ran it = fallback; `verify` = shadow run)."""
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

TOKENIZER_SHA = 'c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
EOS, USER, ASSISTANT, THINK = '<｜end▁of▁sentence｜>', '<｜User｜>', '<｜Assistant｜>', '<think>'
IMAGE, IMAGE_ESCAPED = '<｜deepseek_image｜>', '<​｜deepseek_image｜>'
D = '｜DSML｜'
CALLS_OPEN = f'<{D} calls>'.encode()
INVOKE = re.compile(rf'<{D}\s?invoke\s+name="([^"]*)"\s*>'.encode())
PARAM = re.compile(rf'<{D}\s?parameter\s+name="code"\s+string="(true|false)"\s*>(.*?)(?:</{D}\s?parameter>|$)'
                   .encode(), re.S)
STATUS_LINE = {'exit_error': re.compile(r'\[(exit status -?\d+|killed by signal \d+)\]$'),
               'timeout': re.compile(r'\[timed out after (\d+) ms\]$'),
               'output_limit': re.compile(r'\[output truncated at \d+ bytes\]$'),
               'cancelled': re.compile(r'\[cancelled\]$'), 'unavailable': re.compile(r'\[sandbox unavailable\]$')}
CAPPED = re.compile(r'(?:^|\n)\[output truncated at \d+ tokens\]$')
# Code whose output may legitimately differ between runs: a mismatch there is INCONCLUSIVE, never wrong (conservative;
# both preludes seed `random` with 0 and PYTHONHASHSEED=0, but S3 and other RNGs need not follow that).
NONDET = re.compile(r'\b(time|datetime|random|secrets|uuid|threading|multiprocessing|asyncio|concurrent|tracemalloc|'
                    r'resource|gc|psutil|platform|socket|subprocess)\b|urandom|SystemRandom|getpid|\bid\(|perf_counter')


def load_pieces(path):
    """S1's pinned byte decoder (s1-w9-waste/bounds/token_bytes.py): id -> bytes, never via replacement text."""
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == TOKENIZER_SHA, 'tokenizer sha256 differs from the pin'
    t = json.loads(raw)
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = visible.copy()
    for b in range(256):
        if b not in visible:
            visible.append(b)
            chars.append(256 + sum(c >= 256 for c in chars))
    inverse = {chr(c): b for b, c in zip(visible, chars)}
    added = {r['id']: r['content'].encode() for r in t['added_tokens']}
    out = {i: bytes(inverse[c] for c in s) for s, i in t['model']['vocab'].items() if i not in added}
    out.update(added)
    eos = [i for i, b in added.items() if b == EOS.encode()]
    return out, eos[0]


def body(status, stdout, stderr, tail=''):
    """compute_runner.zig:35-59: stdout, "[stderr]" block, the non-ok status line (`tail`, the delivered one)."""
    t = stdout
    if stderr:
        t += ('[stderr]\n' if not t else '\n[stderr]\n') + stderr
    if status != 'ok':
        if t and not t.endswith('\n'):
            t += '\n'
        t += tail
    if t.endswith('\n'):
        t = t[:-1]
    return t or '[no output]'


def inserted_bytes(outputs):
    """template.zig:441-446 + :460-464 with tools offered: the suffix after the model's held EOS."""
    blocks = '\n\n'.join('<tool_result>' + o.replace(IMAGE, IMAGE_ESCAPED) + '</tool_result>' for o in outputs)
    return (USER + blocks + ASSISTANT + THINK).encode()


def receipts(args):
    for a in args:
        p = Path(a)
        files = sorted(p.glob('*.json')) if p.is_dir() else [Path(x) for x in sorted(glob.glob(a))] or [p]
        for f in files:
            r = json.loads(f.read_text())
            r = r.get('raw', r)
            if f.name != 'arm.json' and isinstance(r, dict) and 'extra' in r:
                yield f, r


def locate(r, pieces, eos):
    """(problems, [(call, emitted code or None, authenticated bool, why)]) for one receipt, from its native stream."""
    tf = (r.get('extra') or {}).get('tensorfold') or {}
    comp, ids, bad = tf.get('compute') or {}, tf.get('token_ids'), []
    calls = comp.get('calls') or []
    if not isinstance(ids, list):
        return ['no token_ids'], [(c, None, False, 'no token_ids') for c in calls]
    if hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest() != tf.get('token_sha256'):
        bad.append('token_sha256 does not hash token_ids')
    if len(ids) != (r.get('usage') or {}).get('completion_tokens', -1) + comp.get('inserted_tokens', 0):
        bad.append('len(token_ids) != completion_tokens + inserted_tokens')
    if any(i not in pieces for i in ids):
        return bad + ['unknown token id'], [(c, None, False, 'unknown token id') for c in calls]
    starts, buf = [], bytearray()
    for i in ids:
        starts.append(len(buf))
        buf += pieces[i]
    starts.append(len(buf))
    buf = bytes(buf)
    # an inserted suffix is the only text that can follow the model's own EOS inside one reply
    holds = [k for k, i in enumerate(ids) if i == eos and buf.startswith(USER.encode(), starts[k + 1])]
    sent = [c for c in calls if c.get('inserted') is True]
    rest = [c for c in calls if c.get('inserted') is not True]
    out, n_ins, last_end, at = [], 0, 0, 0
    for k in holds:
        block = buf[buf.rfind(CALLS_OPEN, 0, starts[k]):starts[k]] if buf.rfind(CALLS_OPEN, 0, starts[k]) >= 0 else b''
        codes = []
        for m in INVOKE.finditer(block):
            nxt = INVOKE.search(block, m.end())
            p = PARAM.search(block[m.end():nxt.start() if nxt else len(block)])
            try:
                raw = p.group(2).decode() if p else None
            except UnicodeDecodeError:
                raw = None
            if raw is not None and p.group(1) == b'false':
                try:
                    raw = json.loads(raw)
                except ValueError:
                    raw = None
            codes.append((m.group(1).decode(), raw))
        group, at = sent[at:at + len(codes)], at + len(codes)
        if not codes or len(group) != len(codes):
            out += [(c, None, False, 'call block does not match the reports') for c in group]
            bad.append(f'hold at token {k}: {len(codes)} emitted calls, {len(group)} reports')
            continue
        want = inserted_bytes([c.get('output', '') for c in group])
        s = starts[k + 1]
        end = next((j for j in range(k + 1, len(ids) + 1) if starts[j] == s + len(want)), None)
        ok = buf[s:s + len(want)] == want and end is not None
        if ok:
            n_ins += end - (k + 1)
            last_end = end
        for c, (name, code) in zip(group, codes):
            why = '' if ok else 'inserted bytes != transport(output) at the hold'
            if ok and (name != 'python' or code != c.get('code')):
                why = 'emitted call != reported name/code'
            out.append((c, code, not why, why))
    out += [(c, None, False, 'reported inserted, not found in the stream') for c in sent[at:]]
    out += [(c, None, False, 'undelivered') for c in rest]
    if sent and (n_ins != comp.get('inserted_tokens') or last_end != comp.get('prefix_tokens')):
        bad.append('inserted_tokens/prefix_tokens disagree with the located suffixes')
    if bad:
        out = [(c, code, False, why or '; '.join(bad)) if why != 'undelivered' else (c, code, a, why)
               for c, code, a, why in out]
    return bad, out


def key_of(f, n):
    return f'{f.parent.parent.name}/{f.parent.name}/{f.stem}#{n}'


def timeout_of(c, default):
    m = STATUS_LINE['timeout'].search(c.get('output') or '') if c.get('status') == 'timeout' else None
    return int(m.group(1)) / 1000 if m else default


def verdict(c, code, res):
    """(verdict, why) for a located, authenticated, delivered call and its re-run row (or None)."""
    st, out = c.get('status'), c.get('output')
    if not isinstance(out, str) or st is None:
        return 'inconclusive', 'missing status/output'
    nondet = bool(NONDET.search(code))
    if st != 'ok':
        m = STATUS_LINE.get(st) and STATUS_LINE[st].search(out)
        if not m or (st == 'exit_error' and m.group(1).startswith('exit status')
                     and m.group(1) != f'exit status {c.get("exit_code")}'):
            return 'wrong', f'status {st} not attributed by its status line'
        if res and 'jail_error' not in res and res['status'] != 'ok' and \
                body(res['status'], res['stdout'], res['stderr'], m.group(0)) == out:
            return 'error-ok', 're-run reproduces the error'
        if not re.search(r'\d', out[:out.find('[stderr]')] if '[stderr]' in out else out[:m.start()]):
            return 'error-ok', 'attributed failure, no number asserted'
        return 'inconclusive', 'failed call with printed numbers that the re-run does not reproduce'
    if res is None:
        return 'inconclusive', 'no re-run result'
    if 'jail_error' in res or res['status'] == 'timeout':
        return 'inconclusive', 're-run failed in the jail: ' + (res.get('jail_error') or 'timeout')[:120]
    if res['status'] != 'ok' and re.search(r'MemoryError|RecursionError|^jail ', res['stderr'], re.M):
        return 'inconclusive', 're-run hit a jail resource limit (512 MiB there, RLIMIT_AS 1 GiB in the engine)'
    want = body(res['status'], res['stdout'], res['stderr'], '[exit status 1]')
    cap = CAPPED.search(out)
    if cap:
        head = out[:cap.start()]
        same = want.startswith(head) and len(want) > len(head)
    else:
        same = want == out
    if same:
        return 'ok', 'capped result is a prefix of the re-run' if cap else ''
    return ('inconclusive', 'mismatch on possibly nondeterministic code') if nondet else ('wrong', 'mismatch')


def main():
    mode, pieces_path = sys.argv[1], sys.argv[2]
    pieces, eos = load_pieces(pieces_path)
    if mode == 'jobs':
        out, n = Path(sys.argv[3]), 0
        with out.open('w') as w:
            for f, r in receipts(sys.argv[4:]):
                for n_call, (c, code, auth, why) in enumerate(locate(r, pieces, eos)[1]):
                    if code is not None and why != 'undelivered':
                        w.write(json.dumps({'key': key_of(f, n_call), 'code': code, 'timeout_s': timeout_of(c, 10.0),
                                            'code_sha256': hashlib.sha256(code.encode()).hexdigest()}) + '\n')
                        n += 1
        print(f'{n} jobs -> {out}')
        return
    assert mode == 'verify', __doc__
    lines = [json.loads(x) for x in Path(sys.argv[3]).read_text().splitlines() if x.strip()]
    probe = next((x['probe'] for x in lines if 'probe' in x), None)
    results = {x['key']: x for x in lines if 'key' in x}
    arms, rows = {}, []
    for f, r in receipts(sys.argv[5:]):
        a = arms.setdefault(str(f.parent), {'receipts': 0, 'calls': 0, 'verdicts': {}, 'executors': {},
                                            'fallbacks': [], 'receipt_gaps': []})
        a['receipts'] += 1
        bad, located = locate(r, pieces, eos)
        if bad:
            a['receipt_gaps'].append(f'{f.name}: ' + '; '.join(bad))
        for n_call, (c, code, auth, why) in enumerate(located):
            k = key_of(f, n_call)
            res = results.get(k)
            if res and res.get('code_sha256') != hashlib.sha256((code or '').encode()).hexdigest():
                res = {'jail_error': 're-run code differs from the emitted code'}
            v, why = ('undelivered', why) if why == 'undelivered' else verdict(c, code, res) if auth else \
                ('inconclusive', why)
            ex = c.get('executor') or 'sandbox (lane, no S3)'
            a['calls'] += 1
            a['verdicts'][v] = a['verdicts'].get(v, 0) + 1
            a['executors'].setdefault(ex, {}).setdefault(v, 0)
            a['executors'][ex][v] += 1
            if c.get('runner_abstain') or c.get('verify'):
                a['fallbacks'].append({'call': k, 'executor': ex, 'runner_abstain': c.get('runner_abstain'),
                                       'verify': c.get('verify')})
            rows.append({'call': k, 'verdict': v, 'why': why, 'executor': ex, 'status': c.get('status'),
                         'rerun': None if res is None else res.get('status', res.get('jail_error'))})
    for a in arms.values():
        v = a['verdicts']
        a['overall'] = 'FAIL' if v.get('wrong') else 'INCONCLUSIVE' if v.get('inconclusive') or a['receipt_gaps'] \
            or not (probe and probe.get('passed')) else 'PASS'
    rep = {'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'tokenizer_sha256': TOKENIZER_SHA,
           'results': sys.argv[3], 'rerun_jail': probe, 'arms': arms, 'calls': rows}
    pre = sys.argv[4]
    Path(pre + '.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False) + '\n')
    md = ['# AB3 tool-result check', '', f'checker {rep["checker_sha256"][:16]}, re-run jail: {probe}', '',
          '| arm/run | receipts | calls | ok | wrong | error-ok | inconclusive | undelivered | executors | fallbacks '
          '| overall |', '|' + '---|' * 11]
    for name, a in arms.items():
        v = a['verdicts']
        ex = ', '.join(f'{e}: {sum(d.values())}' for e, d in a['executors'].items()) or '-'
        md.append(f'| {Path(name).parent.name}/{Path(name).name} | {a["receipts"]} | {a["calls"]} | '
                  + ' | '.join(str(v.get(x, 0)) for x in ('ok', 'wrong', 'error-ok', 'inconclusive', 'undelivered'))
                  + f' | {ex} | {len(a["fallbacks"])} | {a["overall"]} |')
    md += [''] + [f'- {x["call"]}: {x["verdict"]} ({x["why"]})' for x in rows if x['verdict'] not in ('ok',)]
    md += [f'- gap {g}' for a in arms.values() for g in a['receipt_gaps']]
    Path(pre + '.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
