#!/usr/bin/env python3
"""The §14 report from a window's receipts. Runs after the window, off the quiet path (nice 19; grading runs model code
in bwrap).

    python3 -B analyze.py --run DIR --tasks tasks.jsonl [--runner-cpus 0-4,10-14] [--engine-cpus 19]

DIR holds one directory per boot (<boot>/<arm>/<task id>.json, <boot>/<arm>/arm.json, <boot>/rank0.log, replay files).
Arm labels are <boot>/<arm>, e.g. off/a, off/d, cr/r, inl/r; the base is the boot's arm a, whatever the boot is called. Writes DIR/grades.jsonl (cached), DIR/report.json and
DIR/report.md.

Per arm and set: solved rate; reasoning tokens (total and before/after the first delivered result); wall time; decode
tok/s with held time removed; tokens and seconds to the first cell; cells, owned results, abstains and pause rounds per
task. Runner arms also get: owned results that are right vs the model's own guess at the same position (replay), and
trigger-to-first-owned-token latency (the runner's held ms, and the client's gap at the owned token). Placement: every
sampled tf-runner thread must stay on --runner-cpus and off --engine-cpus."""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import seeds  # noqa: E402
import spans  # noqa: E402
import tasks as T  # noqa: E402

OWNED = re.compile(r'reply (\d+) cell (\d+) \((\S+) (\w+)\): owned (\d+) tokens, (\w+) after (\d+) steps, run (\d+) ms, '
                   r'held (\d+) ms')
TRACE_OWNED = re.compile(r'reply (\d+) cell (\d+) \(trace\): owned (\d+) tokens, (\d+) steps, run (\d+) ms, held (\d+) ms')
EXACT_LINE = re.compile(r'\[exact[^:\]\n]{0,40}: ([^\]\n]*?) = (-?\d+)(?: remainder (-?\d+))?\]')  # trace line, any wording
ABSTAIN = re.compile(r'reply (\d+) cell (\d+) \((\S+) (\w+)\): abstain \((\w+)\), held (\d+) ms')
SUMMARY = re.compile(r'reply (\d+): (\d+) cells closed, (\d+) owned, (\d+) abstained at a header, (\d+) never claimed')
FORMS = {'fence': 'fence', 'repl': 'repl', 'inline_expr': 'inline'}
THINK_CLOSE = '</think>'


def cpus(text):
    out = set()
    for part in filter(None, text.split(',')):
        lo, _, hi = part.partition('-')
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return None if not xs else xs[min(len(xs) - 1, int(p * len(xs)))]


def wilson(k, n, z=1.96):
    """95% Wilson interval for k of n."""
    if not n:
        return None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / (1 + z * z / n)
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else statistics.mean(xs)


def text_of(rec):
    return rec['reasoning'] + (THINK_CLOSE + rec['content'] if rec['content'] or rec.get('finish') else '')


def chunk_ends(rec):
    """[(client s, text_of position where the chunk ends)]: content chunks start after the think close."""
    out, n, closed = [], 0, False
    for t, r, c in rec['chunks']:
        n += r
        if c and not closed:
            n += len(THINK_CLOSE)
            closed = True
        n += c
        out.append((t, n))
    return out


def time_at(rec, pos):
    """Client time (s) when the stream had delivered `pos` characters of text_of(rec)."""
    ends = chunk_ends(rec)
    return next((t for t, n in ends if n >= pos), ends[-1][0] if ends else None)


def gap_at(rec, pos):
    """The client's wait at a result position: from the chunk that reached it to the next chunk."""
    ends = chunk_ends(rec)
    for (t, n), (t2, _) in zip(ends, ends[1:]):
        if n >= pos:
            return round((t2 - t) * 1000)
    return None


def log_replies(path, start=0, end=None):
    """runner.zig's lines in a byte range of the log (one arm's), per reply id: cells (index -> fields), summary."""
    out = {}
    if not path.exists():
        return out
    with open(path, 'rb') as f:
        f.seek(start)
        data = f.read() if end is None else f.read(max(0, end - start))
    for line in data.decode(errors='replace').splitlines():
        for rx, kind in ((OWNED, 'owned'), (TRACE_OWNED, 'trace'), (ABSTAIN, 'abstain'), (SUMMARY, 'summary')):
            m = rx.search(line)
            if not m:
                continue
            rid = int(m.group(1))
            r = out.setdefault(rid, {'cells': {}, 'summary': None})
            if kind == 'summary':
                r['summary'] = dict(zip(('closed', 'owned', 'abstained', 'never'), map(int, m.groups()[1:])))
            elif kind == 'trace':
                r['cells'][int(m.group(2))] = {'lang': '', 'form': 'trace', 'outcome': 'owned', 'tokens': int(m.group(3)),
                                               'status': 'ok', 'steps': int(m.group(4)), 'run_ms': int(m.group(5)),
                                               'held_ms': int(m.group(6))}
            elif kind == 'owned':
                r['cells'][int(m.group(2))] = {'lang': m.group(3), 'form': FORMS.get(m.group(4), m.group(4)),
                                               'outcome': 'owned', 'tokens': int(m.group(5)), 'status': m.group(6),
                                               'steps': int(m.group(7)), 'run_ms': int(m.group(8)),
                                               'held_ms': int(m.group(9))}
            else:
                r['cells'][int(m.group(2))] = {'lang': m.group(3), 'form': FORMS.get(m.group(4), m.group(4)),
                                               'outcome': 'abstain', 'why': m.group(5), 'held_ms': int(m.group(6))}
            break
    return out


def logged_forms(lr):
    """The forms of a log reply's claimed cells (owned or abstained) in index order, trace cells left out: what the
    client scanner can see. Trace cells take runner indexes the scanner has no cell for, so cells pair by form order."""
    return [c['form'] for _, c in sorted(lr['cells'].items()) if c['form'] != 'trace']


def claimed(cells):
    """The scanner cells the runner can have claimed: every inline/REPL cell and fences with a header after them."""
    return [c for c in cells if c['at'] is not None]


def visible_cells(text, forms):
    """The scanner's cells, less any inside an [exact: ...] line: that text is the runner's own (a trace cell)."""
    spans_ = [(m.start(), m.end()) for m in EXACT_LINE.finditer(text)]
    return [c for c in spans.cells(text, forms) if not any(a <= c['start'] < b or a < c['end'] <= b for a, b in spans_)]


def pair_cells(lr, cells):
    """Log cell index -> scanner cell. Without trace cells the runner's index is the scanner's (closed order, as in
    windows 1-2). With trace cells (they take runner indexes the scanner has no cell for), by order within each form:
    the k-th logged inline cell is the k-th claimed inline cell."""
    if not any(c['form'] == 'trace' for c in lr['cells'].values()):
        return {i: cells[i] for i in lr['cells'] if i < len(cells) and cells[i]['form'] == lr['cells'][i]['form']}
    by_form, out = {}, {}
    for c in claimed(cells):
        by_form.setdefault(c['form'], []).append(c)
    seen = {}
    for i, c in sorted(lr['cells'].items()):
        if c['form'] == 'trace':
            continue
        k = seen.get(c['form'], 0)
        seen[c['form']] = k + 1
        if k < len(by_form.get(c['form'], [])):
            out[i] = by_form[c['form']][k]
    return out


def trace_lines(text):
    return [{'form': 'trace', 'lang': '', 'code': m.group(1), 'value': m.group(2), 'remainder': m.group(3),
             'start': m.start(), 'end': m.end(), 'at': m.start()} for m in EXACT_LINE.finditer(text)]


def trace_right(t):
    """Whether a trace line's value is right (None: not plain arithmetic). A divmod line `a ÷ b = q remainder r` is
    right when a = q*b + r with r between 0 and b, as Python's divmod gives."""
    if t.get('remainder') is None:
        v = spans.exact(t['code'])
        return None if v is None else v == int(t['value'])
    d = 0
    for i, ch in enumerate(t['code']):  # the top-level ÷: compound operands come wrapped in parentheses
        d += (ch == '(') - (ch == ')')
        if ch == '÷' and d == 0:
            a, b = spans.exact(t['code'][:i]), spans.exact(t['code'][i + 1:])
            if a is None or b is None or b == 0 or a.denominator != 1 or b.denominator != 1:
                return None
            return divmod(int(a), int(b)) == (int(t['value']), int(t['remainder']))
    return None


def match_replies(logs, recs, forms, window):
    """Log reply id -> task id. Replies are numbered in admission order, so candidates are taken in first-token order
    within a window (the concurrency); the best signature (the claimed cells' forms in order) wins."""
    order = sorted((r for r in recs.values() if r['chunks']), key=lambda r: r['sent_unix'] + r['chunks'][0][0])
    every = {r['id']: visible_cells(text_of(r), forms) for r in order}
    scanned = {tid: [c['form'] for c in claimed(cl)] for tid, cl in every.items()}
    traced = {r['id']: len(trace_lines(text_of(r))) for r in order}
    free = [r['id'] for r in order if every[r['id']] or traced[r['id']]]
    out, exact = {}, 0

    def fits(tid, lr, ntrace):
        """(score, exact): without trace cells, windows 1-2's signature (closed count, forms at the logged indexes);
        with them, the claimed forms in order and the [exact:] line count."""
        if not ntrace:
            sc = every[tid]
            closed = (lr['summary'] or {}).get('closed', len(lr['cells']))
            at = [i < len(sc) and sc[i]['form'] == c['form'] for i, c in lr['cells'].items()]
            return 2 * (len(sc) == closed) + sum(at), len(sc) == closed and all(at)
        sc, want = scanned[tid], logged_forms(lr)
        prefix = next((i for i, (x, y) in enumerate(zip(sc, want)) if x != y), min(len(sc), len(want)))
        same = sc == want and traced[tid] == ntrace
        return 3 * (sc == want) + 2 * (traced[tid] == ntrace) + prefix, same

    pos = {r['id']: i for i, r in enumerate(order)}
    rid0 = min(logs, default=0)
    for rid in sorted(logs):
        lr = logs[rid]
        ntrace = sum(c['form'] == 'trace' for c in lr['cells'].values())
        best, score = None, None
        for tid in free[:window * 2]:
            s, _ = fits(tid, lr, ntrace)
            key = (s, -abs(pos[tid] - (rid - rid0)))  # ties: nearest its admission place (ids count admissions)
            if score is None or key > score:
                best, score = tid, key
        if best is None:
            continue
        free.remove(best)
        out[rid] = best
        exact += fits(best, lr, ntrace)[1]
    return out, exact


def grade_all(run, tasks, recs_by_arm):
    path = run / 'grades.jsonl'
    cache = {}
    if path.exists():
        for l in path.read_text().splitlines():
            g = json.loads(l)
            cache[(g['arm'], g['id'])] = g
    with path.open('a') as f:
        for arm, recs in recs_by_arm.items():
            for tid, rec in recs.items():
                if (arm, tid) in cache:
                    continue
                if rec.get('error') or rec.get('finish') not in ('stop', 'done', 'tool_calls', None):
                    g = {'ok': False, 'why': f'finish {rec.get("finish")} {rec.get("error", "")}'.strip()}
                else:
                    try:
                        g = T.check(tasks[tid], rec['content'])
                    except Exception as exc:
                        g = {'ok': False, 'why': f'checker error {exc!r}'[:300]}
                g.update(arm=arm, id=tid)
                f.write(json.dumps(g) + '\n')
                f.flush()
                cache[(arm, tid)] = g
    for (arm, tid), g in cache.items():  # cached before the sign-aware final: fill it in, the file stays as it was
        if 'ok_signed' not in g:
            rec = recs_by_arm.get(arm, {}).get(tid) or {}
            num = tasks[tid]['check']['kind'] == 'number' and 'got' in g
            g['ok_signed'] = T.same_number(T.final_number(rec.get('content')), tasks[tid]['check']['truth']) if num else g['ok']
    return cache


def placement(arm_meta, want, engine):
    seen_allowed, seen_last, engine_hits, n = set(), set(), 0, 0
    for s in arm_meta.get('placement', []):
        for tid, allowed, last, ticks in s['runner']:
            n += 1
            seen_allowed.add(allowed)
            seen_last.add(last)
            engine_hits += last in engine
    if n == 0:
        return {'samples': 0, 'verdict': 'no tf-runner thread seen'}
    ok = all(cpus(a) <= want for a in seen_allowed) and seen_last <= want and engine_hits == 0
    return {'samples': n, 'allowed': sorted(seen_allowed), 'last_cpus': sorted(seen_last), 'on_engine_cpus': engine_hits,
            'verdict': 'PASS' if ok else 'FAIL'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--runner-cpus', default='0-4,10-14')
    ap.add_argument('--engine-cpus', default='19')
    a = ap.parse_args()
    run = Path(a.run)
    tasks = {t['id']: t for t in map(json.loads, open(a.tasks))}
    recs_by_arm, metas = {}, {}
    for armdir in sorted(p.parent for p in run.glob('*/*/arm.json')):
        label = f'{armdir.parent.name}/{armdir.name}'
        metas[label] = json.loads((armdir / 'arm.json').read_text())
        recs_by_arm[label] = {p.stem: json.loads(p.read_text()) for p in armdir.glob('*.json') if p.name != 'arm.json'}
    grades = grade_all(run, tasks, recs_by_arm)
    report = {'arms': {}, 'sets': sorted({t['set'] for t in tasks.values()})}
    detail = []
    for label, recs in recs_by_arm.items():
        boot, arm = label.split('/')
        meta = metas[label]
        forms = tuple(meta.get('forms', 'fence,repl').split(','))
        logs = log_replies(run / boot / 'rank0.log', *meta.get('rank_log_bytes', [0, None])) if arm[0] == 'r' else {}
        mapping, exact = match_replies(logs, recs, forms, meta.get('concurrency', 4)) if logs else ({}, 0)
        by_task = {v: k for k, v in mapping.items()}
        guesses = {}
        for name in ('replay.jsonl', 'replay-validity.jsonl'):
            p = run / boot / arm / name
            if p.exists():
                for g in map(json.loads, p.read_text().splitlines()):
                    guesses[(g['id'], g['pos'])] = g
        rows = []
        for tid, rec in recs.items():
            u = rec.get('usage') or {}
            text = text_of(rec)
            cl = visible_cells(text, forms)
            reasoning_tok = (u.get('completion_tokens_details') or {}).get('reasoning_tokens') or rec.get('reasoning_tokens_counted')
            ctoks = u.get('completion_tokens')
            per_char = (ctoks / len(text)) if ctoks and text else None
            lr = logs.get(by_task.get(tid), {'cells': {}, 'summary': None})
            held = [c['held_ms'] for c in lr['cells'].values() if c['outcome'] == 'owned']
            first_cell = cl[0] if cl else None
            owned_rows = []
            paired, tl = pair_cells(lr, cl), trace_lines(text)
            tk = [i for i, c in sorted(lr['cells'].items()) if c['form'] == 'trace']
            for i, c in sorted(lr['cells'].items()):
                if c['outcome'] != 'owned':
                    continue
                if c['form'] == 'trace':  # judged from its own [exact: E = V] line; the model wrote no value there
                    k = tk.index(i)
                    if k >= len(tl):
                        continue
                    t = tl[k]
                    ok = trace_right(t)
                    owned_rows.append({'index': i, 'form': 'trace', 'owned': t['value'], 'guess': None,
                                       'owned_right': ok, 'guess_right': None, 'truth_known': ok is not None, 'held_ms': c['held_ms'],
                                       'client_gap_ms': gap_at(rec, t['end']), 'pos': t['start'], 'no_guess': True})
                    continue
                cell = paired.get(i)
                if cell is None or cell['at'] is None:  # unpaired, or a fence with no result in the text: not judged
                    continue
                truth = spans.truth(cell)
                got = spans.value_at(text, cell['at'], cell['form'])
                g = guesses.get((tid, cell['at']))
                guess = None if not g or g.get('guess') is None else spans.value_at(g['guess'], 0, cell['form'])
                owned_rows.append({'index': i, 'form': cell['form'], 'owned': got[:80], 'guess': None if guess is None else guess[:80],
                                   'owned_right': spans.right(truth, got), 'guess_right': spans.right(truth, guess),
                                   'truth_known': truth is not None, 'held_ms': c['held_ms'],
                                   'client_gap_ms': gap_at(rec, cell['at']), 'pos': cell['at']})
            first_owned = min((o['pos'] for o in owned_rows), default=None)
            first_code = next((c for c in cl if c['form'] in ('fence', 'repl')), None)
            forms_count = {}
            for c in lr['cells'].values():
                k = f"{c['form']} {c['outcome']}"
                forms_count[k] = forms_count.get(k, 0) + 1
            guide = ((rec.get('extra', {}).get('tensorfold') or {}).get('runner_guide') or {}).get('sha256')
            reason_len = len(rec['reasoning'])
            # fence/REPL cells by where the model wrote them and what the runner did: logged ones by their outcome,
            # visible fences the log never names as `unclaimed`
            code_cells = {}
            logged = set()
            for i, c in lr['cells'].items():
                if c['form'] in ('fence', 'repl'):
                    cell = paired.get(i)
                    where = '?' if cell is None else 'reasoning' if cell['start'] < reason_len else 'content'
                    k = f"{c['form']} {where} {c['outcome']}"
                    code_cells[k] = code_cells.get(k, 0) + 1
                    if cell is not None:
                        logged.add(cell['start'])
            if logs:
                for c in cl:
                    if c['form'] == 'fence' and c['start'] not in logged:
                        k = f"fence {'reasoning' if c['start'] < reason_len else 'content'} unclaimed"
                        code_cells[k] = code_cells.get(k, 0) + 1
            row = {'id': tid, 'set': rec['set'], 'solved': grades.get((label, tid), {}).get('ok', False),
                   'solved_signed': grades.get((label, tid), {}).get('ok_signed', False),
                   'finish': rec.get('finish'), 'error': rec.get('error'), 'wall_s': rec['elapsed_s'],
                   'completion_tokens': ctoks, 'reasoning_tokens': reasoning_tok,
                   'decode_tps': None, 'engine_tps': (rec.get('extra', {}).get('tensorfold') or {}).get('tokens_per_second'),
                   'first_cell_s': time_at(rec, first_cell['end']) if first_cell else None,
                   'first_cell_tokens': round(first_cell['end'] * per_char) if first_cell and per_char else None,
                   'cells': len(cl), 'owned': len(owned_rows), 'pause_rounds': len(held),
                   'abstained': sum(c['outcome'] == 'abstain' for c in lr['cells'].values()),
                   'reasoning_before_delivery': round(min(first_owned, reason_len) * per_char) if first_owned is not None and per_char else None,
                   'reasoning_after_delivery': round(max(0, reason_len - first_owned) * per_char) if first_owned is not None and per_char else None,
                   'owned_detail': owned_rows,
                   'first_code_cell_tokens': round(first_code['end'] * per_char) if first_code and per_char else None,
                   'content_tokens': (ctoks - reasoning_tok) if ctoks is not None and reasoning_tok is not None else None,
                   'forms': forms_count, 'code_cells': code_cells, 'guide_sha256': guide,
                   'lane_calls': len(((rec.get('extra', {}).get('tensorfold') or {}).get('compute') or {}).get('calls', []))}
            if rec['chunks'] and ctoks:
                span = rec['chunks'][-1][0] - rec['chunks'][0][0] - sum(held) / 1000
                row['decode_tps'] = round((ctoks - 1) / span, 2) if span > 0 else None
            rows.append(row)
            detail.append(dict(row, arm=label))
        validity = [g for k, g in guesses.items() if arm[0] in 'ad']
        agree = sum(1 for g in validity if g.get('guess') and g['reply'].startswith(g['guess'][:min(len(g['guess']), 40)]))
        cut_path = run / boot / arm / 'cut.jsonl'
        cut = [json.loads(l)['id'] for l in cut_path.read_text().splitlines()] if cut_path.exists() else []
        planned = meta.get('planned', list(recs))
        arm_rep = {'boot': boot, 'arm': arm, 'tasks': len(rows), 'planned': len(planned), 'cut_at_cap': len(cut),
                   'not_started': len([t for t in planned if t not in recs and t not in cut]), 'placement': placement(meta, cpus(a.runner_cpus), cpus(a.engine_cpus)),
                   'log_replies': len(logs), 'log_matched': len(mapping), 'log_matched_exact': exact,
                   'replay_validity': {'positions': len(validity), 'agree': agree} if validity else None,
                   'wall_s': round(meta['ended_unix'] - meta['started_unix']), 'sets': {}}
        for s in report['sets'] + ['ALL']:
            rs = [r for r in rows if s == 'ALL' or r['set'] == s]
            if not rs:
                continue
            owned = [o for r in rs for o in r['owned_detail']]
            judged = [o for o in owned if o['truth_known']]
            arm_rep['sets'][s] = {
                'n': len(rs), 'solved': sum(r['solved'] for r in rs), 'solved_signed': sum(r['solved_signed'] for r in rs),
                'solved_rate': round(sum(r['solved'] for r in rs) / len(rs), 3),
                'solved_ci95': wilson(sum(r['solved'] for r in rs), len(rs)),
                'reasoning_tokens_mean': mean([r['reasoning_tokens'] for r in rs]),
                'reasoning_tokens_total': sum(r['reasoning_tokens'] or 0 for r in rs),
                'completion_tokens_total': sum(r['completion_tokens'] or 0 for r in rs),
                'completion_tokens_mean': mean([r['completion_tokens'] for r in rs]),
                'content_tokens_mean': mean([r['content_tokens'] for r in rs]),
                'first_code_cell_tokens_median': q([r['first_code_cell_tokens'] for r in rs], 0.5),
                'forms': {k: sum(r['forms'].get(k, 0) for r in rs) for k in sorted({k for r in rs for k in r['forms']})},
                'code_cells': {k: sum(r['code_cells'].get(k, 0) for r in rs) for k in sorted({k for r in rs for k in r['code_cells']})},
                'guide_sha256': sorted({r['guide_sha256'] for r in rs if r['guide_sha256']}),
                'wall_s_mean': mean([r['wall_s'] for r in rs]), 'wall_s_p95': q([r['wall_s'] for r in rs], 0.95),
                'decode_tps_median': q([r['decode_tps'] for r in rs], 0.5),
                'engine_tps_median': q([r['engine_tps'] for r in rs], 0.5),
                'first_cell_s_median': q([r['first_cell_s'] for r in rs], 0.5),
                'first_cell_tokens_median': q([r['first_cell_tokens'] for r in rs], 0.5),
                'cells_per_task': mean([r['cells'] for r in rs]), 'owned_per_task': mean([r['owned'] for r in rs]),
                'abstained_per_task': mean([r['abstained'] for r in rs]),
                'pause_rounds_per_task': mean([r['pause_rounds'] for r in rs]),
                'reasoning_before_delivery_mean': mean([r['reasoning_before_delivery'] for r in rs]),
                'reasoning_after_delivery_mean': mean([r['reasoning_after_delivery'] for r in rs]),
                'owned': len(owned), 'owned_judged': len(judged),
                'owned_right': sum(o['owned_right'] is True for o in judged),
                'guess_right': sum(o['guess_right'] is True for o in judged),
                'guess_missing': sum(o['guess'] is None and not o.get('no_guess') for o in judged),
                'held_ms_p50': q([o['held_ms'] for o in owned], 0.5), 'held_ms_p95': q([o['held_ms'] for o in owned], 0.95),
                'client_gap_ms_p50': q([o['client_gap_ms'] for o in owned], 0.5),
                'lane_calls_per_task': mean([r['lane_calls'] for r in rs]),
            }
        report['arms'][label] = arm_rep
    def base_of(label):  # the runner-off arm (a) of the same concurrency or seed, whatever its boot is called
        seed = seeds.split(label)[1]
        want = 'a' + ('-c1' if label.endswith('-c1') else '') + ('' if seed is None else f'-s{seed}')
        return next((x for x in report['arms'] if x.split('/', 1)[-1] == want), None)

    saved = {}
    for label in report['arms']:
        base = base_of(label)
        if label == base or base is None:
            continue
        both = [r for r in detail if r['arm'] == label and r['solved']]
        ref = {r['id']: r for r in detail if r['arm'] == base and r['solved']}
        pairs = [(ref[r['id']], r) for r in both if r['id'] in ref]
        saved[label] = {'vs': base, 'solved_by_both': len(pairs),
                        'reasoning_tokens_saved': sum((x['reasoning_tokens'] or 0) - (y['reasoning_tokens'] or 0) for x, y in pairs),
                        'completion_tokens_saved': sum((x['completion_tokens'] or 0) - (y['completion_tokens'] or 0) for x, y in pairs),
                        'completion_tokens_base': sum(x['completion_tokens'] or 0 for x, _ in pairs),
                        'wall_s_saved': round(sum(x['wall_s'] - y['wall_s'] for x, y in pairs), 1),
                        'solved_rate_delta': round(report['arms'][label]['sets']['ALL']['solved_rate'] - report['arms'][base]['sets']['ALL']['solved_rate'], 3)}
    report['thinking_saved'] = saved
    # decode on/off per reply: each runner reply against the same task in the runner-off arm at its concurrency, split
    # by whether the runner owned a cell in it; held-adjusted (client span minus held ms) and the engine's own figure
    dec = {}
    for label in report['arms']:
        base = base_of(label)
        if not label.split('/', 1)[-1].startswith('r') or base is None:
            continue
        ref = {r['id']: r for r in detail if r['arm'] == base}
        for kind, keep in (('owned', lambda r: r['owned'] > 0), ('none', lambda r: r['owned'] == 0)):
            pairs = [(ref[r['id']], r) for r in detail if r['arm'] == label and keep(r) and r['id'] in ref]
            ratio = lambda k: q([y[k] / x[k] - 1 for x, y in pairs if x[k] and y[k]], 0.5)
            dec[f'{label} {kind}'] = {'vs': base, 'replies': len(pairs),
                                      'held_adjusted_change_median': ratio('decode_tps'),
                                      'engine_change_median': ratio('engine_tps'),
                                      'held_adjusted_change_range': [q([y['decode_tps'] / x['decode_tps'] - 1 for x, y in pairs if x['decode_tps'] and y['decode_tps']], p) for p in (0, 1)]}
    report['decode_vs_off'] = dec
    report['seeds'] = seeds.pooled(report, detail)
    (run / 'report.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    (run / 'detail.jsonl').write_text(''.join(json.dumps(r, default=str) + '\n' for r in detail))
    (run / 'report.md').write_text(markdown(report))
    print(markdown(report))


def fmtv(v, nd=1):
    if v is None:
        return '-'
    return f'{v:.{nd}f}' if isinstance(v, float) else str(v)


def markdown(rep):
    out = ['# §14 measurement report', '']
    cols = ['arm', 'set', 'n', 'solved', 'sign-aware', 'reason+content tok', 'reason tok', 'content tok', 'wall s', 'decode tok/s', '1st cell s', 'cells', 'owned',
            'owned right', 'guess right', 'held p50/p95 ms']
    out += ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for label, ar in rep['arms'].items():
        for s, v in ar['sets'].items():
            out.append('| ' + ' | '.join([label, s, str(v['n']), f'{v["solved"]} ({v["solved_rate"]:.0%}, {v["solved_ci95"][0]:.0%}-{v["solved_ci95"][1]:.0%})', str(v['solved_signed']),
                                          fmtv(v['completion_tokens_mean'], 0), fmtv(v['reasoning_tokens_mean'], 0),
                                          fmtv(v['content_tokens_mean'], 0), fmtv(v['wall_s_mean']),
                                          fmtv(v['decode_tps_median']), fmtv(v['first_cell_s_median']),
                                          fmtv(v['cells_per_task'], 2), str(v['owned']),
                                          f'{v["owned_right"]}/{v["owned_judged"]}', f'{v["guess_right"]}/{v["owned_judged"]}',
                                          f'{fmtv(v["held_ms_p50"])}/{fmtv(v["held_ms_p95"])}']) + ' |')
    out += ['', '| arm | wall s | planned / cut at cap / not started | placement | log replies matched (exact) | replay validity |',
            '|---|---|---|---|---|---|']
    for label, ar in rep['arms'].items():
        p = ar['placement']
        rv = ar['replay_validity']
        out.append(f'| {label} | {ar["wall_s"]} | {ar["planned"]} / {ar["cut_at_cap"]} / {ar["not_started"]} | '
                   f'{p["verdict"]} ({p["samples"]} samples) | '
                   f'{ar["log_matched"]}/{ar["log_replies"]} ({ar["log_matched_exact"]}) | '
                   f'{"-" if not rv else str(rv["agree"]) + "/" + str(rv["positions"])} |')
    out += ['', '| arm | runner cells by form and outcome (all tasks) | first fence/REPL cell, token offset (median) | runner_guide sha256 seen |',
            '|---|---|---|---|']
    for label, ar in rep['arms'].items():
        v = ar['sets']['ALL']
        fc = ', '.join(f'{k} {n}' for k, n in v['forms'].items()) or '-'
        out.append(f'| {label} | {fc} | {fmtv(v["first_code_cell_tokens_median"], 0)} | '
                   f'{", ".join(x[:12] for x in v["guide_sha256"]) or "-"} |')
    arith = [x for x in ('add30', 'mul30', 'mul_big', 'div_big', 'bignum')]
    out += ['', '| arm | fence/REPL cells by place and outcome: arithmetic sets | code sets (mbpp, algo, build) |', '|---|---|---|']
    for label, ar in rep['arms'].items():
        if not label.split('/', 1)[-1].startswith('r'):
            continue
        cols = []
        for keep in (lambda k: k in arith, lambda k: k not in arith and k != 'ALL'):
            tot = {}
            for st, v in ar['sets'].items():
                if keep(st):
                    for k, n in v.get('code_cells', {}).items():
                        tot[k] = tot.get(k, 0) + n
            cols.append(', '.join(f'{k} {n}' for k, n in sorted(tot.items())) or '-')
        out.append(f'| {label} | {cols[0]} | {cols[1]} |')
    out += ['', '| arm | vs | solved by both | reason+content tokens saved | reasoning tokens saved | wall s saved | solved-rate delta |',
            '|---|---|---|---|---|---|---|']
    for label, s in rep['thinking_saved'].items():
        share = s['completion_tokens_saved'] / s['completion_tokens_base'] if s['completion_tokens_base'] else 0
        out.append(f'| {label} | {s["vs"]} | {s["solved_by_both"]} | {s["completion_tokens_saved"]} ({share:+.1%}) | '
                   f'{s["reasoning_tokens_saved"]} | {s["wall_s_saved"]} | '
                   f'{s["solved_rate_delta"]:+.3f} |')
    out += ['', '| runner replies | vs | replies | decode change, held-adjusted (median, min..max) | engine tok/s change (median) |',
            '|---|---|---|---|---|']
    pct = lambda v: '-' if v is None else f'{v * 100:+.1f}%'
    for label, d in rep.get('decode_vs_off', {}).items():
        lo, hi = d['held_adjusted_change_range']
        out.append(f'| {label} | {d["vs"]} | {d["replies"]} | {pct(d["held_adjusted_change_median"])} ({pct(lo)}..{pct(hi)}) | '
                   f'{pct(d["engine_change_median"])} |')
    out += seeds.markdown(rep.get('seeds', {}))
    return '\n'.join(out) + '\n'


if __name__ == '__main__':
    main()
