#!/usr/bin/env python3
"""Per-turn AB3 safety summary (T5-AB3-SPEC.md: no outcome peeking). Per boot of one turn's run copy: receipts present
vs the frozen plan, finish counts (deadline/timeout/length/error), id-hash integrity, boot pass and measure rc, and the
run's r7 gate. Deliberately prints no token, solve or tool-call figures.
Usage: python3 -I ab3_turnsafety.py LAYOUT TURN RUN_COPY"""
import collections
import json
import sys
from pathlib import Path

layout, turn, run = json.loads(Path(sys.argv[1]).read_text()), int(sys.argv[2]), Path(sys.argv[3])
t = next(x for x in layout['turns'] if x['turn'] == turn)
bl = next(x for x in layout['blocks'] if x['block'] == t['block'])
letter = {'plain': 'a', 'lane': 'd', 'full': 'rn', 'only': 'ro'}
status = json.loads((run / 'status.json').read_text()) if (run / 'status.json').exists() else {'attempts': []}
att = {a['name']: a for a in status['attempts']}
for name in t['boots']:
    arm = next(b['arm'] for b in bl['boots'] if b['name'] == name)
    fin, bad, n = collections.Counter(), [], 0
    for s in layout['seeds']:
        for u in bl['units']:
            f = run / name / f'{letter[arm]}-s{s}' / f'{u}.json'
            if not f.exists():
                bad.append(f'missing {u}:s{s}')
                continue
            r = json.loads(f.read_text())
            n += 1
            fin[r.get('finish') if not r.get('error') else 'error'] += 1
            if r.get('ids_check') != 'match':
                bad.append(f'ids {r.get("ids_check")} {u}:s{s}')
    a = att.get(name, {})
    print(f'{name} ({arm}): receipts {n}/60, boot pass {a.get("pass")} measure_rc {a.get("measure_rc")}, '
          f'finish {dict(fin)}, integrity problems {len(bad)} {bad[:3]}')
wr = run / 'window-report' / 'window-report.json'
g = json.loads(wr.read_text())['gate'] if wr.exists() else {}
print('r7:', {k: g.get(k) for k in ('verdict', 'records', 'ok', 'wrong-value', 'wrong-span', 'not-in-text', 'unparsed',
                                     'wrong-check', 'cut', 'inconclusive', 'short', 'unknown')} if g else 'MISSING')
