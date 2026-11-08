#!/usr/bin/env python3
"""W8 kit (T5-W8-TRACE-SPEC.md): the W6 45 tasks, trace off (rf) vs on (rt), seeds 1-3 as four ABBA boots
(off/G1, on/G1, on/G2, off/G2) plus T2's profiled boot (off/G1's settings and seed 1 on 12 tasks, TF_DSV41_PROFILE).
G1 = seed 1 + a family-stratified 23 of seed 3; G2 = seed 2 + the other 22. Writes CONFIGS and MANIFEST (frozen).
Usage: python3 -B w8_kit.py TASKS.jsonl CONFIGS.json MANIFEST.json"""
import hashlib
import json
import sys

import tasks as T

SETS = 'add30,mul30,mul_big,div_big,bignum,mbpp,algo'
SAMPLE = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20}  # W7's sampling (DSV4.1 serving defaults)
GUIDED = dict(SAMPLE, runner_guide='@trust')
BANK = '/opt/tensorfold/share/runner/graph-v7-powmod/bank/bank.json'  # TE's default from 4b39258f3 on (26862f0a)
PINS = ['TF_DSV41_RUNNER_STATED=0', 'TF_DSV41_RUNNER_INLINE_CHAIN=0', 'TF_DSV41_RUNNER_INLINE_MARK=0',
        'TF_DSV41_RUNNER_SYNC_GRAPH=0', 'TF_DSV41_RUNNER_COMPILE_STEPS=10522', 'TF_DSV41_RUNNER_CELLS=1']
DEADLINE_S = 300
PROFILED = 12


def env(trace):
    forms = 'fence,repl,inline,trace' if trace else 'fence,repl,inline'
    return ['TF_DSV41_COMPUTE_LANE=1', 'TF_DSV41_RUNNER=1',
            'TF_DSV41_RUNNER_CIRCUIT=/opt/tensorfold/share/runner/calc.tfpc', f'TF_DSV41_RUNNER_FORMS={forms}',
            'TF_DSV41_RUNNER_BACKEND=graph', f'TF_DSV41_RUNNER_GRAPH_BANK={BANK}', *PINS]


def boot(name, arm, seeds, ids3, extra, ids=None, start_by_min=None, needs=()):
    arms = [f'{arm}:s{s}' for s in seeds]
    arm_ids = {f'{arm}:s3': ids3} if 3 in seeds else {}
    if ids is not None:
        arm_ids = {a: ids for a in arms}
    guard = {'needs': list(needs)} if needs else {}
    if start_by_min:
        guard['start_by_s'] = start_by_min * 60
    return {'name': name, 'pack': 'k6', 'extra': extra, 'loose': True, **guard,
            'measure': {'arms': ','.join(arms), 'forms': 'fence,repl,inline', 'place': 'on', 'decode_check': False,
                        'sample': 'first', 'cap_s': 0, 'request_deadline_s': DEADLINE_S, 'max_tokens': 32768,
                        'arm_opts': {arm: GUIDED}, 'arm_ids': arm_ids, 'sets': SETS}}


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    keep = set(SETS.split(','))
    chosen, _ = T.sample([t for t in rows if t['set'] in keep], 'first')
    ids = sorted(t['id'] for t in chosen)
    assert len(ids) == 45, len(ids)
    # family-stratified: alternate over the ids sorted by family then id, G1 first (23/22 whatever the family sizes)
    by_family = sorted(ids, key=lambda i: (i.rsplit('-', 1)[0], i))
    g1, g2 = sorted(by_family[0::2]), sorted(by_family[1::2])
    assert (len(g1), len(g2)) == (23, 22), (len(g1), len(g2))
    by_fam = {}
    for tid in ids:
        by_fam.setdefault(tid.rsplit('-', 1)[0], []).append(tid)
    prof = sorted(x for f in sorted(by_fam) for x in by_fam[f][:2])[:PROFILED]  # two ids per family, first 12
    # ~90 min of GPU: a late boot is skipped, never half a pair (a skipped block reruns whole on the same image)
    configs = [boot('t4w8offG1', 'rf', (1, 3), g1, env(False)),
               boot('t4w8onG1', 'rt', (1, 3), g1, env(True), needs=['t4w8offG1']),
               boot('t4w8onG2', 'rt', (2, 3), g2, env(True), start_by_min=40),
               boot('t4w8offG2', 'rf', (2, 3), g2, env(False), start_by_min=65, needs=['t4w8onG2']),
               boot('t4w8prof', 'rf', (1,), [], env(False) + ['TF_DSV41_PROFILE=/state/profile.json'], ids=prof,
                    start_by_min=80, needs=['t4w8offG1'])]
    pairs = [{'id': t, 'seed': s, 'group': g, 'off': f't4w8off{g}/rf-s{s}', 'on': f't4w8on{g}/rt-s{s}'}
             for s, g, group in ((1, 'G1', ids), (2, 'G2', ids), (3, 'G1', g1), (3, 'G2', g2)) for t in group]
    manifest = {'tasks': ids, 'ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                'tasks_sha256': hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest(),  # w8_gate checks the file
                'groups': {'G1': g1, 'G2': g2}, 'profiled': {'boot': 't4w8prof', 'arm': 'rf-s1', 'tasks': prof,
                                                             'control': 't4w8offG1/rf-s1'},
                'settings': dict(SAMPLE, request_deadline_s=DEADLINE_S, max_tokens=32768, bank=BANK, pins=PINS),
                'order': [c['name'] for c in configs], 'pairs': pairs}
    with open(sys.argv[2], 'w') as f:
        f.write('[\n' + ',\n'.join(' ' + json.dumps(c) for c in configs) + '\n]\n')
    with open(sys.argv[3], 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f'{len(pairs)} pairs, G1 {len(g1)} G2 {len(g2)}, profiled {prof}')


if __name__ == '__main__':
    main()
