#!/usr/bin/env python3
"""Re-grades windows' number tasks with the sign-aware final beside their kept grades.jsonl; code grades carry over.
Usage: python3 -B regrade.py TASKS.jsonl RUN [RUN...] > regrade.md  (RUN: grades.jsonl, <boot>/<arm>/*.json)"""
import json
import sys
from pathlib import Path

import tasks as T


def regrade(run, tasks):
    """Per arm: (raw solved, sign-aware solved, n), and every reply whose grade changes."""
    grades = {(g['arm'], g['id']): g for g in map(json.loads, (run / 'grades.jsonl').read_text().splitlines())}
    arms, changed = {}, []
    for armdir in sorted(p.parent for p in run.glob('**/arm.json')):
        label = f'{armdir.parent.name}/{armdir.name}'
        raw = signed = n = 0
        for f in sorted(armdir.glob('*.json')):
            if f.name == 'arm.json':
                continue
            rec = json.loads(f.read_text())
            g, c = grades[(label, rec['id'])], tasks[rec['id']]['check']
            num = c['kind'] == 'number' and 'got' in g  # a finished number reply (cut or failed ones stay failed)
            got = T.final_number(rec.get('content')) if num else None
            ok = T.same_number(got, c['truth']) if num else g['ok']
            n, raw, signed = n + 1, raw + g['ok'], signed + ok
            if ok != g['ok']:
                changed.append((label, rec['id'], g.get('got'), got, g['ok'], ok))
        arms[label] = (raw, signed, n)
    return arms, changed


def main():
    tasks = {t['id']: t for t in map(json.loads, open(sys.argv[1]))}
    for arg in sys.argv[2:]:
        run = Path(arg)
        arms, changed = regrade(run, tasks)
        print(f'## {run.name}\n\n| arm | solved (raw, kept) | solved (sign-aware) | n |\n|---|---|---|---|')
        for label, (raw, signed, n) in arms.items():
            print(f'| {label} | {raw} | {signed} | {n} |')
        print(f'\n{len(changed)} grade changes' + (':' if changed else '.') + '\n')
        if changed:
            print('| arm | task | raw final | sign-aware final | raw | sign-aware |\n|---|---|---|---|---|---|')
            for label, tid, got, new, a, b in changed:
                print(f'| {label} | {tid} | {got} | {new} | {"ok" if a else "fail"} | {"ok" if b else "fail"} |')
        print()


if __name__ == '__main__':
    main()
