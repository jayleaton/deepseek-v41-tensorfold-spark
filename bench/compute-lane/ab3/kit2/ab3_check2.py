#!/usr/bin/env python3
"""Checks an AB3 kit dir against T5-AB3-SPEC.md before freeze: units (counts, salt order, no pilot ID, T5's row
bytes), seeds, the shared within-block order (recomputed), boot orders, each arm's env and request bodies (tools,
guide bytes, sampling, prompt), the client wait (300 + 60 s), pairing (each task x seed once per arm) and hashes.
Usage: python3 -I ab3_check2.py KIT_DIR PILOT_CONFIGS ROWS"""

import sys as _archive_sys
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

kit, pilot_env = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())[0]['extra']
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
L = json.loads((kit / 'ab3-layout.json').read_text())
K = {'add30': 11, 'algo': 5, 'bignum': 3, 'div_big': 8, 'mbpp': 20, 'mul30': 3, 'mul_big': 10}
assert L['k'] == K and L['seeds'] == [101, 102, 103] and L['salt'] == 'AB3-v1:'
trust = json.loads((kit / 'guides.json').read_text())['trust']
assert sha(trust.encode()) == L['guide_trust_sha256'] and L['guide_trust_sha256'].startswith('53ed328f')
t5 = {json.loads(x)['id']: x for x in Path(sys.argv[3]).read_text().splitlines() if x.strip()}
ulines = (kit / L['units']).read_text().splitlines()
rows = {json.loads(x)['id']: json.loads(x) for x in ulines}
assert all(t5[json.loads(x)['id']] == x for x in ulines) and len(rows) == 60
assert not set(rows) & set(L['excluded_pilot_units']) and len(L['excluded_pilot_units']) == 11
for f, k in K.items():  # recompute the selection from T5's rows
    want = sorted((i for i, x in t5.items() if json.loads(x)['set'] == f and i not in L['excluded_pilot_units']),
                  key=lambda i: (sha(('AB3-v1:' + i).encode()), i))[:k]
    assert sorted(want) == sorted(i for i, r in rows.items() if r['set'] == f), f
orders = [('plain', 'lane', 'full', 'only'), ('lane', 'full', 'only', 'plain'), ('full', 'only', 'plain', 'lane')]
S3 = 'TF_DSV41_COMPUTE_LANE_RUNNER'
env = {'plain': ['TF_DSV41_COMPUTE_LANE=1', 'TF_DSV41_RUNNER_COMMIT=0'],
       'lane': ['TF_DSV41_COMPUTE_LANE=1', 'TF_DSV41_RUNNER_COMMIT=0'], 'full': pilot_env,
       'only': [x for x in pilot_env if not x.startswith(S3)]}
seen, cfgs = collections.Counter(), {}
for t in L['turns']:
    assert sha((kit / t['configs']).read_bytes()) == t['configs_sha256']
    cfgs.update({c['name']: c for c in json.loads((kit / t['configs']).read_text())})
assert len(cfgs) == 12 and [t['boots'] for t in L['turns']] == [[b['name'] for b in bl['boots'][h:h + 2]]
                                                                for bl in L['blocks'] for h in (0, 2)]
for bl in L['blocks']:
    assert sha((kit / bl['tasks']).read_bytes()) == bl['tasks_sha256'] and len(bl['units']) == 20
    order = sorted((i, s) for i in bl['units'] for s in (101, 102, 103))
    random.Random(20261104).shuffle(order)
    assert bl['order'] == [f'{i}:s{s}' for i, s in order]
    assert [b['arm'] for b in bl['boots']] == list(orders[bl['block'] - 1])
    for b in bl['boots']:
        assert sha((kit / b['templates']).read_bytes()) == b['templates_sha256']
        arm, c = b['arm'], cfgs[b['name']]
        m = c['measure']
        assert c['extra'] == env[arm] and m['request_deadline_s'] + m['deadline_grace_s'] >= 360
        assert (m['concurrency'], m['max_tokens'], m['server_deadline_field']) == (4, 32768, 'deadline_s')
        spec = json.loads((kit / b['templates']).read_text())
        assert [f"{r['task_id']}:s{r['seed']}" for r in spec['requests']] == bl['order']  # same order, every arm
        for r in spec['requests']:
            bt = r['body_template']
            assert bt['messages'] == [{'role': 'user', 'content': rows[r['task_id']]['prompt']}]
            assert (bt['temperature'], bt['top_p'], bt['top_k'], bt['max_tokens'], bt['seed']) == \
                (1.0, 0.95, 20, 32768, r['seed']) and bt['chat_template_kwargs'] == {'enable_thinking': True}
            assert ('tools' in bt) == (arm != 'plain') and bt.get('tools', [{'type': 'code_interpreter'}]) == \
                [{'type': 'code_interpreter'}]
            assert ('runner_guide' in bt) == (arm == 'full') and bt.get('runner_guide', trust) == trust
            assert sha(json.dumps(bt, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()) == \
                r['body_template_sha256']
            seen[(arm, r['task_id'], r['seed'])] += 1
assert all(seen[(a, i, s)] == 1 for a in env for i in rows for s in (101, 102, 103)) and sum(seen.values()) == 720
print('AB3 kit matches the spec:', len(rows), 'units, 720 requests, 180 pairs per arm; units', L['units_sha256'][:16])
