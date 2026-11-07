#!/usr/bin/env python3
"""W9-shaped fixture for w9_report.py from real record-bearing measure receipts (stand-ins, never W9 observations).
  make_w9_fixture.py --out DIR --variant clean|cutrec|faults SOURCE_DIR...
Writes DIR/blocks/w9-block-NN-manifest.json (w9_kit.py's pairs: ten blocks x 45 tasks x seeds 4/5/6, G1 = seed 4 plus
the first 23 seed-6 tasks, G2 = seed 5 plus the other 22; odd blocks off/on/on/off, even on/off/off/on) and
DIR/run/run-NN/<label>/<id>.json. Each reply is a source receipt with a unique tag before its reasoning (records
shifted), a live set lagging its final records by one write (05aa3debe's newest set beside the final receipt), and
two server-deadline cuts per block in model text after the last record (finish `deadline`).
cutrec adds per block one on-arm deadline cut inside the reply's last engine write (a correct partial: r5's `cut`, r2's
wrong-value). faults adds: block 2 a live record its final contradicts, block 4 a wrong complete inline value, block 6
a deadline cut inside a write with a wrong landed digit, block 8 a client-closed live-only reply, block 9 an owned write
whose record is gone (its counter kept), block 10 one reply missing. Expected: clean PASS; faults FAIL in 2, 4, 6,
INCONCLUSIVE in 8, 9, 10, pooled FAIL."""
import argparse
import copy
import json
import os
from pathlib import Path

FIELDS = {'reasoning': 0, 'content': 1}


def runner(d):
    return d['extra']['tensorfold']['runner']


def sources(dirs):
    out = []
    for root in dirs:
        for dp, _, fs in os.walk(root):
            for f in sorted(fs):
                if not f.endswith('.json') or f == 'arm.json':
                    continue
                try:
                    d = json.load(open(os.path.join(dp, f)))
                    cells = runner(d).get('cells')
                except Exception:
                    continue
                if isinstance(d, dict) and d.get('finish') == 'stop' and cells and not runner(d).get('live') \
                        and isinstance(d.get('reasoning'), str):
                    out.append(os.path.join(dp, f))
    return sorted(out)


def last(cells):
    return max(cells, key=lambda c: (FIELDS.get(c.get('field'), 2), c.get('at') or 0))


def tag(d, text):
    """Prefix the reasoning with text and shift reasoning records (final and live) by its bytes."""
    n = len(text.encode())
    d['reasoning'] = text + d['reasoning']
    for c in runner(d)['cells'] + (d.get('live_cells') or []):
        if c.get('field') == 'reasoning' and c.get('at') is not None:
            c['at'] += n


def lagging(cells):
    """The stream's newest live set before the last write finished: the last record half landed."""
    live = copy.deepcopy(cells)
    c = last(live)
    v = str(c.get('value', ''))
    if len(v) > 1:
        c['value'] = v[:len(v) // 2]
        c['landed'] = max(0, (c.get('landed') or 1) - 1)
    return live


def cut_text(d, field, end):
    """End the reply at byte `end` of `field` (the content goes if the end is in the reasoning)."""
    d[field] = d[field].encode()[:end].decode('utf-8', 'ignore')
    if field == 'reasoning':
        d['content'] = ''
    d['finish'] = 'deadline'
    d['extra']['tensorfold']['finish'] = 'deadline'


def cut_after_last(d):
    """A server deadline in model text: after the last record, before the reply's own end."""
    c = last(runner(d)['cells'])
    f = c['field']
    end = (c.get('at') or 0) + len(str(c.get('value', '')).encode())
    rest = len(d[f].encode()) - end
    if rest < 8:
        return False
    cut_text(d, f, end + rest // 2)
    return True


def cut_inside_last(d, wrong=False):
    """A server deadline inside the last engine write: two landed characters of an inline or trace value."""
    cells = runner(d)['cells']
    c = last(cells)
    v = str(c.get('value', ''))
    if c.get('form') not in ('inline', 'trace') or (c.get('ids') or 0) < 3 or len(v.strip()) < 3:
        return False
    k = len(v) - len(v.lstrip()) + 2
    pre = v[:k]
    if wrong:
        if not pre[-1].isdigit():
            return False
        pre = pre[:-1] + str((int(pre[-1]) + 1) % 10)
        b = bytearray(d[c['field']].encode())
        b[(c['at'] or 0) + k - 1] = ord(pre[-1])
        d[c['field']] = b.decode()
    c['value'], c['landed'] = pre, 1
    cut_text(d, c['field'], (c.get('at') or 0) + len(pre.encode()))
    d['live_cells'] = copy.deepcopy([x for x in cells if x is not c]) + [dict(c, value=pre[:-1], landed=0)]
    return True


def wrong_value(d):
    """A wrong complete inline value: its last digit off by one, in the text and the record alike."""
    for c in runner(d)['cells']:
        v = str(c.get('value', ''))
        if c.get('form') == 'inline' and v.strip().isdigit() and len(v.strip()) >= 2:
            i = len(v.rstrip()) - 1
            nv = v[:i] + str((int(v[i]) + 1) % 10) + v[i + 1:]
            b = bytearray(d[c['field']].encode())
            b[(c['at'] or 0) + i] = ord(nv[i])
            d[c['field']] = b.decode()
            c['value'] = nv
            for x in d.get('live_cells') or []:
                if (x.get('cell'), x.get('at')) == (c.get('cell'), c.get('at')) and str(x.get('value')) == v:
                    x['value'] = nv
            return True
    return False


def contradict(d):
    """A live record its final record contradicts: the live value's first digit differs."""
    for x in d.get('live_cells') or []:
        v = str(x.get('value', ''))
        i = next((j for j, ch in enumerate(v) if ch.isdigit()), None)
        if i is not None:
            x['value'] = v[:i] + str((int(v[i]) + 1) % 10) + v[i + 1:]
            return True
    return False


def unrecorded(d):
    """An owned write without its record: the runner counter still counts it, the final and live records drop it."""
    cells = runner(d)['cells']
    if len(cells) < 2:
        return False
    gone = cells.pop(0)
    d['live_cells'] = [x for x in d.get('live_cells') or [] if (x.get('cell'), x.get('at')) != (gone.get('cell'), gone.get('at'))]
    return True


def live_only(d):
    """A client close: no final receipt, only the stream's newest live set (complete records), no usage."""
    run = runner(d)
    run['live'] = True
    run['cells'] = copy.deepcopy(run['cells'])
    d.pop('live_cells', None)
    d.pop('usage', None)
    d['finish'], d['closed_by'] = 'timeout', 'client'
    d['extra']['tensorfold'].pop('finish', None)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--variant', default='clean', choices=('clean', 'cutrec', 'faults'))
    ap.add_argument('sources', nargs='+')
    a = ap.parse_args()
    src = sources(a.sources)
    assert len(src) >= 100, f'only {len(src)} source receipts with records'
    out = Path(a.out)
    k, made, faults = 0, 0, []
    for n in range(1, 11):
        fam = [os.path.basename(src[(n * 45 + t) * 7919 % len(src)]).rsplit('-', 1)[0] for t in range(45)]
        ids = [f'fx-{fam[t]}-{n:02d}{t:02d}' for t in range(45)]
        order = ['off', 'on', 'on', 'off'] if n % 2 else ['on', 'off', 'off', 'on']
        g = {('G1', side): f'w9-b{n:02d}-{order.index(side) + 1}-{side}-G1' for side in ('off', 'on')}
        g.update({('G2', side): f'w9-b{n:02d}-{order.index(side, 2) + 1}-{side}-G2' for side in ('off', 'on')})
        pairs = []
        for t, tid in enumerate(ids):
            for seed in (4, 5, 6):
                grp = 'G1' if seed == 4 or (seed == 6 and t < 23) else 'G2'
                lab = {side: f"{g[(grp, side)]}/{'rf' if side == 'off' else 'rt'}-s{seed}" for side in ('off', 'on')}
                pairs.append({'id': tid, 'seed': seed, 'group': grp, 'block': n, 'pair_id': f'{tid}:s{seed}', **lab})
        (out / 'blocks').mkdir(parents=True, exist_ok=True)
        (out / 'blocks' / f'w9-block-{n:02d}-manifest.json').write_text(
            json.dumps({'block': n, 'fixture': 'make_w9_fixture.py stand-in, not W9 tasks', 'pairs': pairs}))
        todo = {'cut-after': 2, 'cutrec': 1 if a.variant == 'cutrec' else 0}
        if a.variant == 'faults':
            todo.update({2: {'contradict': 1}, 4: {'wrong': 1}, 6: {'cutwrong': 1}, 8: {'liveonly': 1}, 9: {'unrecorded': 1},
                         10: {'missing': 1}}
                        .get(n, {}))
        for p in pairs:
            for side in ('off', 'on'):
                d = json.load(open(src[k * 7919 % len(src)]))
                k += 1
                d['id'], d['arm'] = p['id'], p[side].split('/')[1].replace('-s', ':s')
                d['live_cells'] = lagging(runner(d)['cells'])
                tag(d, f"w9fix b{n:02d} {p['id']} s{p['seed']} {side}. ")
                what = None
                if side == 'on' and todo.get('missing'):
                    todo['missing'] -= 1
                    faults.append(f"block {n}: missing {p[side]}/{p['id']}")
                    continue
                for key, fn in (('contradict', contradict), ('wrong', wrong_value), ('liveonly', live_only), ('unrecorded', unrecorded),
                                ('cutwrong', lambda x: cut_inside_last(x, wrong=True)), ('cutrec', cut_inside_last)):
                    if side == 'on' and todo.get(key) and fn(d):
                        todo[key] -= 1
                        what = key
                        break
                if what is None and todo['cut-after'] and (p['seed'] == 4) and cut_after_last(d):
                    todo['cut-after'] -= 1
                    what = 'cut-after'
                if what:
                    faults.append(f"block {n}: {what} {p[side]}/{p['id']}")
                f = out / 'run' / f'run-{n:02d}' / p[side] / f"{p['id']}.json"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(d))
                made += 1
        left = {kk: v for kk, v in todo.items() if v}
        assert not left, f'block {n}: could not place {left}'
    (out / 'FIXTURE.txt').write_text('\n'.join([f'variant {a.variant}; {made} replies from {len(src)} sources'] + faults) + '\n')
    print(f'variant {a.variant}: {made} replies, {len(src)} sources; ' + '; '.join(faults))


if __name__ == '__main__':
    main()
