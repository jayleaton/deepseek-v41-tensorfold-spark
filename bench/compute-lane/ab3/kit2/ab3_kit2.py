#!/usr/bin/env python3
"""AB3 kit to T5-AB3-SPEC.md (sha256 81b34c1a…): frozen units, requests, order and configs before any request.
Units: per family (add30 11, algo 5, bignum 3, div_big 8, mbpp 20, mul30 3, mul_big 10) the first IDs of T5's W9 rows
(manifest sha256 checked) by (sha256("AB3-v1:" + id), id), the 11 pilot IDs excluded. Blocks: three of 20 tasks,
family-balanced (families in lexicographic order, each family's ranked IDs dealt round-robin with one global cursor).
Seeds 101/102/103. Within a block the order is random.Random(20261104).shuffle of the block's sorted (task, seed)
tuples (a fresh RNG per block), shared by all four arms. Every arm is its own cold boot; boot orders per block are the
spec's: plain/lane/full/only, lane/full/only/plain, full/only/plain/lane. Two boots per GPU turn (six turns, each
under 45 min), keeping that order. Requests are frozen body templates sent by measure.py's templates mode.
Writes into OUT: ab3-units.jsonl (T5's row lines, byte for byte), ab3-bN-units.jsonl, ab3-<boot>.json (templates),
ab3-tNN-configs.json and ab3-layout.json (every hash and setting).
Usage: python3 -I ab3_kit2.py --rows ROWS --manifest MANIFEST --manifest-sha SHA256 --pilot PILOT_UNITS
       --guides guides.json --pilot-configs AB_B_CONFIGS --out DIR"""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

SALT = 'AB3-v1:'
K = {'add30': 11, 'algo': 5, 'bignum': 3, 'div_big': 8, 'mbpp': 20, 'mul30': 3, 'mul_big': 10}
BLOCKS, SEEDS, ORDER_SEED = 3, (101, 102, 103), 20261104
ORDERS = (('plain', 'lane', 'full', 'only'), ('lane', 'full', 'only', 'plain'), ('full', 'only', 'plain', 'lane'))
SAMPLING = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20}
TOOLS = [{'type': 'code_interpreter'}]
S3 = 'TF_DSV41_COMPUTE_LANE_RUNNER'
LETTER = {'plain': 'a', 'lane': 'd', 'full': 'rn', 'only': 'ro'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def body(prompt, seed, tools, guide):
    b = {'model': '__QUALIFIED_MODEL__', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 32768,
         'stream': True, 'stream_options': {'include_usage': True}, 'chat_template_kwargs': {'enable_thinking': True},
         'seed': seed, **SAMPLING}
    if tools:
        b['tools'] = copy.deepcopy(TOOLS)
    if guide:
        b['runner_guide'] = guide
    return b


def template_sha(b):  # measure.template_sha: sorted keys, no spaces, UTF-8 as written
    return sha(json.dumps(b, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())


def main():
    ap = argparse.ArgumentParser()
    for k in ('rows', 'manifest', 'manifest_sha', 'pilot', 'guides', 'pilot_configs', 'out'):
        ap.add_argument('--' + k.replace('_', '-'), required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mbytes, rbytes = Path(a.manifest).read_bytes(), Path(a.rows).read_bytes()
    assert sha(mbytes) == a.manifest_sha, 'manifest.json is not the posted one'
    assert json.loads(mbytes)['files_sha256']['tasks-private.jsonl'] == sha(rbytes), 'rows are not the manifest\'s'
    lines = {json.loads(x)['id']: x for x in rbytes.decode().splitlines() if x.strip()}
    rows = {i: json.loads(x) for i, x in lines.items()}
    pilot = {json.loads(x)['id'] for x in Path(a.pilot).read_text().splitlines() if x.strip()}
    assert len(pilot) == 11 and pilot <= set(rows)
    trust = json.loads(Path(a.guides).read_text())['trust']
    runner_env = json.loads(Path(a.pilot_configs).read_text())[0]['extra']  # the pilot's R+N boot, commit 0
    for knob in ('COMMIT', 'STATED', 'INLINE_CHAIN', 'INLINE_MARK', 'SYNC_GRAPH'):
        assert f'TF_DSV41_RUNNER_{knob}=0' in runner_env, knob
    assert 'TF_DSV41_RUNNER_COMPILE_STEPS=10522' in runner_env and any(x.startswith(S3 + '=') for x in runner_env)
    lane_env = ['TF_DSV41_COMPUTE_LANE=1', 'TF_DSV41_RUNNER_COMMIT=0']
    arms = {'plain': (lane_env, 'fence,repl,inline', 'off', False, None),
            'lane': (lane_env, 'fence,repl,inline', 'off', True, None),
            'full': (runner_env, 'fence,repl,inline,trace', 'on', True, trust),
            'only': ([x for x in runner_env if not x.startswith(S3)], 'fence,repl,inline,trace', 'on', True, None)}

    picked = {f: sorted((i for i, r in rows.items() if r['set'] == f and i not in pilot),
                        key=lambda i: (sha((SALT + i).encode()), i))[:k] for f, k in sorted(K.items())}
    assert all(len(picked[f]) == k for f, k in K.items())
    blocks, cursor = [[] for _ in range(BLOCKS)], 0
    for f in sorted(K):
        for i in picked[f]:
            blocks[cursor % BLOCKS].append(i)
            cursor += 1
    assert all(len(b) == 20 for b in blocks)
    units = out / 'ab3-units.jsonl'
    units.write_text(''.join(lines[i] + '\n' for b in blocks for i in b))
    layout = {'spec': 'T5-AB3-SPEC.md',
              'spec_sha256': '81b34c1a79a754efca01ebee85f47c74490a029f23e4e0533799205b60dd524e',
              'salt': SALT, 'k': K, 'excluded_pilot_units': sorted(pilot), 'manifest_sha256': a.manifest_sha,
              'rows_sha256': sha(rbytes), 'guide_trust_sha256': sha(trust.encode()), 'units': units.name,
              'units_sha256': sha(units.read_bytes()), 'seeds': SEEDS, 'sampling': SAMPLING, 'max_tokens': 32768,
              'request_deadline_s': 300, 'client_wait_s': 360, 'server_deadline_field': 'deadline_s',
              'concurrency': 4, 'order': 'random.Random(20261104).shuffle(sorted((task, seed))) per block, all arms',
              'arms': {n: {'letter': LETTER[n], 'extra': e, 'forms': f, 'place': p, 'tools': t, 'guide': bool(g)}
                       for n, (e, f, p, t, g) in arms.items()}, 'blocks': [], 'turns': []}
    configs_by_boot = {}
    for n, ids in enumerate(blocks, 1):
        bfile = out / f'ab3-b{n}-units.jsonl'
        bfile.write_text(''.join(lines[i] + '\n' for i in ids))
        order = sorted((i, s) for i in ids for s in SEEDS)
        random.Random(ORDER_SEED).shuffle(order)
        boots = []
        for slot, arm in enumerate(ORDERS[n - 1], 1):
            extra, forms, place, tools, guide = arms[arm]
            name = f'ab3-b{n}-{slot}-{arm}'
            reqs = []
            for i, s in order:
                bt = body(rows[i]['prompt'], s, tools, guide)
                reqs.append({'within_boot': len(reqs) + 1, 'pair_id': f'{i}:s{s}', 'task_id': i, 'seed': s,
                             'body_template': bt, 'body_template_sha256': template_sha(bt)})
            tfile = out / f'{name}.json'
            tfile.write_text(json.dumps({'boot': name, 'arm': LETTER[arm], 'block': n, 'slot': slot,
                                         'requests': reqs}) + '\n')
            configs_by_boot[name] = {
                'name': name, 'pack': 'k6', 'extra': extra, 'loose': True,
                'measure': {'arms': LETTER[arm], 'forms': forms, 'place': place, 'decode_check': False,
                            'tasks': bfile.name, 'templates': tfile.name, 'concurrency': 4, 'max_tokens': 32768,
                            'request_deadline_s': 300, 'token_ids': True, 'no_count_fallback': True,
                            'server_deadline_field': 'deadline_s', 'deadline_grace_s': 60}}
            boots.append({'name': name, 'arm': arm, 'slot': slot, 'requests': len(reqs), 'templates': tfile.name,
                          'templates_sha256': sha(tfile.read_bytes())})
        layout['blocks'].append({'block': n, 'units': ids, 'tasks': bfile.name, 'tasks_sha256': sha(bfile.read_bytes()),
                                 'order': [f'{i}:s{s}' for i, s in order],
                                 'order_sha256': sha(json.dumps(order).encode()), 'boots': boots})
        for half in (0, 1):
            t = 2 * (n - 1) + half + 1
            names = [b['name'] for b in boots[2 * half:2 * half + 2]]
            cfile = out / f'ab3-t{t:02d}-configs.json'
            cfile.write_text('[\n' + ',\n'.join(' ' + json.dumps(configs_by_boot[x]) for x in names) + '\n]\n')
            layout['turns'].append({'turn': t, 'block': n, 'boots': names, 'tasks': bfile.name,
                                    'configs': cfile.name, 'configs_sha256': sha(cfile.read_bytes())})
    (out / 'ab3-layout.json').write_text(json.dumps(layout, indent=1) + '\n')
    print(f'{len(layout["turns"])} turns, {sum(len(b) for b in blocks)} units, '
          f'{sum(x["requests"] for b in layout["blocks"] for x in b["boots"])} requests; units sha256 '
          f'{layout["units_sha256"]}; layout sha256 {sha((out / "ab3-layout.json").read_bytes())}')
    for b in layout['blocks']:
        print(b['block'], [x['name'] for x in b['boots']], {f: sum(rows[i]['set'] == f for i in b['units']) for f in K})


if __name__ == '__main__':
    main()
