#!/usr/bin/env python3
"""Known-answer tests for ab3_analysis.py on synthetic full-shape AB3 runs built from the frozen layout (no AB3
outcome exists). Each scenario writes runs under WORK/<name>, runs the analysis and checks verdict and gates.
Usage: python3 -I test_ab3_analysis.py LAYOUT UNITS WORK"""

import sys as _archive_sys
import json
import random
import subprocess
import sys
from pathlib import Path

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

HERE = Path(__file__).resolve().parent
layout, units_path, work = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]), Path(sys.argv[3])
units = [json.loads(x) for x in units_path.read_text().splitlines() if x]
ARMS = {'plain': 'a', 'lane': 'd', 'full': 'rn', 'only': 'ro'}
R7 = '35f2bdf0243df7917b2b61098a1ff4a91bf1044854634912ab4800495db853fc'
IMAGES = ['sha256:26c0b4107869671656476006743be5288dc48c8ee1f5c5b3879bd606fba59535',
          'sha256:1cfb03ebae4f58de4ef61be0bd8527e7d88dd4c046bfce98844bc53305d37292']
PROMPT = {'plain': 100, 'lane': 406, 'only': 406, 'full': 518}


def build(name, T, solved=lambda a, t, s: True, finish=lambda a, t, s: 'stop', drop=(), ledger_sha=R7):
    root = work / name
    runs, ledger = [], {}
    for turn in layout['turns']:
        run = root / f'run-t{turn["turn"]:02d}'
        boots = {b['name']: b for bl in layout['blocks'] for b in bl['boots']}
        atts, grades = [], []
        for bname in turn['boots']:
            b = boots[bname]
            arm = b['arm']
            atts.append({'name': bname, 'pass': True, 'measure_rc': 0, 'extra': layout['arms'][arm]['extra']})
            (run / bname).mkdir(parents=True, exist_ok=True)
            (run / bname / 'arm-metadata.json').write_text(json.dumps({'ranks': [{'image_id': i} for i in IMAGES]}))
            bl = next(x for x in layout['blocks'] if x['block'] == turn['block'])
            for s in layout['seeds']:
                d = run / bname / f'{ARMS[arm]}-s{s}'
                d.mkdir(exist_ok=True)
                (d / 'arm.json').write_text(json.dumps({'templates_sha256': b['templates_sha256']}))
                for t in bl['units']:
                    if (arm, t, s) in drop:
                        continue
                    n = T(arm, t, s)
                    ins = 20 if arm != 'plain' else 0
                    rec = {'id': t, 'finish': finish(arm, t, s), 'elapsed_s': 10.0, 'ids_check': 'match',
                           'usage': {'completion_tokens': n - ins, 'prompt_tokens': PROMPT[arm],
                                     'prompt_tokens_details': {'cached_tokens': 0}},
                           'extra': {'tensorfold': {'token_ids': [1] * n,
                                                    'compute': {'inserted_tokens': ins, 'calls': []}}}}
                    (d / f'{t}.json').write_text(json.dumps(rec))
                    grades.append({'arm': f'{bname}/{ARMS[arm]}-s{s}', 'id': t, 'ok': solved(arm, t, s)})
        (run / 'status.json').write_text(json.dumps({'engine_sha': '3bf584af1c5407006f154e0c70d05c6f495d0487',
                                                     'attempts': atts}))
        (run / 'grades.jsonl').write_text(''.join(json.dumps(g) + '\n' for g in grades))
        (run / 'window-report').mkdir(exist_ok=True)
        (run / 'window-report' / 'window-report.json').write_text(json.dumps(
            {'gate': {'verdict': 'PASS', 'records': 10, 'ok': 10, 'wrong-value': 0, 'wrong-span': 0}, 'arms': []}))
        ledger[run.name] = {'oracle_sha_before': R7, 'oracle_sha_after': ledger_sha, 'fence': 'on'}
        runs.append(run)
    (root / 'tool.json').write_text(json.dumps({'overall': 'PASS', 'arms': {}}))
    (root / 'ledger.json').write_text(json.dumps(ledger))
    p = subprocess.run(['python3', '-I', str(HERE / 'ab3_analysis.py'), sys.argv[1], str(units_path),
                        str(root / 'tool.json'), str(root / 'ledger.json'), str(root / 'out'), *map(str, runs)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads((root / 'out.json').read_text())


def check(name, rep, verdict, **gates):
    bad = [f'verdict {rep["verdict"]} != {verdict}'] if rep['verdict'] != verdict else []
    bad += [f'gate {g} {rep["gates"].get(g)} != {v}' for g, v in gates.items() if rep['gates'].get(g) != v]
    print(('PASS ' if not bad else 'FAIL ') + name, bad or '', rep.get('primary', {}).get('change_pct'),
          rep.get('primary', {}).get('ci95'))
    return not bad


ok = True
base = {'plain': 1500, 'lane': 1000, 'only': 950, 'full': 800}
rep = build('meets', lambda a, t, s: base[a])
ok &= check('exact -20%, all gates pass', rep, 'MEETS AB3', token=True, solves=True, censoring=True, safety=True,
            provenance=True) and all(abs(x + 20) < 1e-9 for x in [rep['primary']['change_pct'], *rep['primary']['ci95']])
ok &= check('-5% fails the token gate', build('small', lambda a, t, s: {**base, 'full': 950}[a]), 'DOES NOT MEET',
            token=False)
ids = sorted(u['id'] for u in units)
four = {(ids[i], 101) for i in range(4)}
ok &= check('4 isolated losses fail solves', build('loss4', lambda a, t, s: base[a],
            solved=lambda a, t, s: not (a == 'full' and (t, s) in four)), 'DOES NOT MEET', solves=False, token=True)
ok &= check('one task lost in 2 seeds fails solves', build('repeat', lambda a, t, s: base[a],
            solved=lambda a, t, s: not (a == 'full' and t == ids[0] and s in (101, 102))), 'DOES NOT MEET',
            solves=False)
three = {(ids[i], 100 + 1 + i) for i in range(3)}
gains = {(ids[10 + i], 101) for i in range(3)}
ok &= check('3 isolated losses, 3 gains pass solves', build('loss3', lambda a, t, s: base[a],
            solved=lambda a, t, s: not (a == 'full' and (t, s) in three or a == 'lane' and (t, s) in gains)),
            'MEETS AB3', solves=True)
ok &= check('3 losses, 0 gains fail (net decrease)', build('loss3net', lambda a, t, s: base[a],
            solved=lambda a, t, s: not (a == 'full' and (t, s) in three)), 'DOES NOT MEET', solves=False)
ok &= check('exactly -10% passes the point rule', build('ten', lambda a, t, s: {**base, 'full': 900}[a]),
            'MEETS AB3', token=True)
lanecen = {(t, s) for t in ids[:7] for s in (101, 102, 103)}  # 21 lane deadlines
ok &= check('21 lane deadlines fail censoring', build('cen', lambda a, t, s: base[a],
            finish=lambda a, t, s: 'deadline' if a == 'lane' and (t, s) in lanecen else 'stop'), 'DOES NOT MEET',
            censoring=False)
ok &= check('missing receipt is INCONCLUSIVE', build('missing', lambda a, t, s: base[a], drop={('full', ids[5], 102)}),
            'INCONCLUSIVE')
ok &= check('oracle sha drift is INCONCLUSIVE', build('drift', lambda a, t, s: base[a], ledger_sha='0' * 64),
            'INCONCLUSIVE')
rng = random.Random(7)
noise = {(a, t, s): rng.randint(300, 3000) for a in base for t in ids for s in (101, 102, 103)}
rep = build('noisy', lambda a, t, s: noise[(a, t, s)] if a != 'full' else int(noise[('lane', t, s)] * 0.8))
lo, hi = rep['primary']['ci95']
ok &= check('noisy lane, full = 0.8 lane', rep, 'MEETS AB3') and lo <= rep['primary']['change_pct'] <= hi
print('ALL PASS' if ok else 'SOME FAILED')
sys.exit(0 if ok else 1)
