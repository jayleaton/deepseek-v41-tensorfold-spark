#!/usr/bin/env python3
"""Bounded P-only fake checks; parent runs on Spark, never on dev-host.

check(root=...) checks pinned real public inputs and in-memory negative controls.
check_stage(path, root=...) additionally reads the actual staged files.
No model requests, subprocesses, SSH, GPU, answers, or original-file writes.
"""
import copy
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import p_stage as p


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(index, payloads, root=p.ROOT):
    blobs = p.sources(root)
    require(all(index.get(k) == v and type(index.get(k)) is type(v)
                for k, v in p.STATUS.items()), 'approval/status drift')
    p.check_prefix(index['prefix'])
    require(index['base_spec_sha256'] == p.BASE_SPEC, 'base identity drift')
    require(index['source_pins'] == p.PINS, 'source pins drift')
    require(index['candidate_boot'] == 'p-all' and index['requests'] == 240
            and index['candidate_order'] == ['t01-L:120', 't02-L:120'], 'candidate layout drift')
    require(set(payloads) == set(index['files']) | {'P-stage-index.json'}, 'file inventory drift')
    require(p.encoded(index) == payloads['P-stage-index.json'], 'index bytes drift')
    for name, expected in index['files'].items():
        require(p.digest(payloads[name]) == expected, 'staged hash drift: ' + name)
    document = json.loads(payloads['requests-P.json'])
    require(all(document.get(k) == v for k, v in p.STATUS.items()), 'request status drift')
    rows = document['requests']
    require(len(rows) == 240, '240 P rows required')
    tasks = [json.loads(line) for line in blobs['dataset-v1/tasks-public.jsonl'].decode().splitlines()]
    public = {t['id']: t for t in tasks}
    expected_keys = {('compute_dl', t['id'], s) for t in tasks for s in (101, 102)}
    require(len({tuple(r['key']) for r in rows}) == 240
            and {tuple(r['key']) for r in rows} == expected_keys, 'task/seed coverage drift')
    original_rows = []
    for turn in ('t01', 't02'):
        original_rows.extend(json.loads(blobs[f'{p.OLD_PREFIX}/frozen-v1/boots/{turn}-L.json'])['requests'])
    for i, (row, old) in enumerate(zip(rows, original_rows)):
        task = public[row['id']]
        require(set(task) == {'id', 'family', 'suite', 'slot', 'attempt', 'generator_label', 'inputs', 'prompt'},
                'nonpublic task fields')
        require(row['task'] == task and row['key'] == old['key']
                and row['seed'] == old['seed'] and row['task_sha256'] == old['task_sha256'],
                'public mapping/order drift')
        require(set(row) == {'arm', 'suite', 'id', 'seed', 'key', 'task', 'task_sha256',
                            'body_template', 'wire', 'body_sha256', 'wire_sha256',
                            'execution_policy', 'matched_L'}, 'extra request metadata')
        expected = copy.deepcopy(old['body_template'])
        del expected['tools']
        body = row['body_template']
        require(body == expected and list(body) == list(expected), 'P must only remove tools')
        require('tools' not in body and body['seed'] == row['seed']
                and body['messages'] == [{'content': task['prompt'], 'role': 'user'}], 'wire prompt/tools drift')
        require({k: v for k, v in body.items() if k not in ('messages', 'seed')} == p.SETTINGS,
                'sampling settings drift')
        require(row['wire'] == p.wire(expected)
                and row['wire_sha256'] == row['body_sha256'] == p.digest(row['wire'].encode()),
                'wire bytes/hash drift')
        require(row['arm'] == 'plain' and row['execution_policy'] == {'TF_DSV41_COMPUTE_LANE': '0'},
                'plain pool policy drift')
        require(row['matched_L'] == {'boot': ('t01-L' if i < 120 else 't02-L'),
                                   'position': i % 120, 'wire_sha256': old['wire_sha256']},
                'L provenance drift')
    helper, _ = p.helpers(Path(root).resolve())
    for rank in (0, 1):
        record = json.loads(payloads[f'rank{rank}/effective-env.json'])
        require(record == index['ranks'][rank], 'rank manifest/index drift')
        env = record['effective_env']
        off = p.off_values(helper)
        require(record['expected_off'] == off and all(env.get(k) == v for k, v in off.items()),
                'lane/optional flag enabled')
        require(not any(any(word in k for word in ('TRACE', 'GUIDE', 'S3')) for k in env),
                'inherited optional feature')
        require(not any(k.startswith(('GUARD', 'SPARK')) for k in env), 'copied thermal overrides')
        require(env['TF_DSV41_RUNNER_FORMS'] == 'fence,repl,inline', 'runner forms drift')
        require(env['ZIG_COMMIT'] == p.REVISION and record['image_id'] == p.IMAGE_IDS[rank]
                and env['IMAGE'] == helper.IMAGE, 'rank image/source drift')
        target = p.node_root(rank, index['prefix'])
        require(record['rank'] == rank and record['lane'] == 0 and record['arm'] == 'P', 'rank policy drift')
        require(record['root'] == str(target)
                and env['STATE_DIR'] == str(target / 'state/p-all')
                and env['CACHE_DIR'] == str(target / 'cache/p-all')
                and env['CONTAINER_PREFIX'] == index['prefix'] + '-p-all'
                and record['config'] == str(target / f'configs/p-all-r{rank}.env'), 'fresh paths drift')
        require(payloads[record['stage_config']] == helper.env_text(env).encode(), 'effective env/config drift')
        require(all(record.get(k) == v for k, v in p.STATUS.items()), 'rank approval drift')
    # Full deterministic comparison also catches changes to inherited calibration,
    # transport, paths, provenance, or off assertions not enumerated above.
    expected_index, expected_payloads = p.plan(root, index['prefix'])
    require(index == expected_index and payloads == expected_payloads, 'deterministic plan drift')


def check_stage(path, root=p.ROOT):
    """Read actual files; reject added files, symlinks and modified bytes."""
    path = Path(path)
    require(not path.is_symlink(), 'symlink stage')
    files = list(path.rglob('*'))
    require(not any(f.is_symlink() for f in files), 'symlink stage member')
    payloads = {str(f.relative_to(path)): f.read_bytes() for f in files if f.is_file()}
    index = json.loads(payloads['P-stage-index.json'])
    validate(index, payloads, root)
    return {'staging_only': True, 'collection_authorized': False, 'checked_requests': 240}


def check(root=p.ROOT):
    """Real pinned public inputs, FAKE_NOT_COLLECTION mutations; no writes."""
    index, payloads = p.plan(root)
    validate(index, payloads, root)
    passed = []

    def rejected(name, action):
        try:
            action()
        except (ValueError, KeyError, TypeError):
            passed.append(name)
        else:
            raise AssertionError('negative control accepted: ' + name)

    def mutated_document(name, mutate):
        changed = copy.deepcopy(index)
        data = dict(payloads)
        doc = json.loads(data['requests-P.json'])
        mutate(doc)
        data['requests-P.json'] = p.encoded(doc)
        changed['files']['requests-P.json'] = p.digest(data['requests-P.json'])
        data['P-stage-index.json'] = p.encoded(changed)
        rejected(name, lambda: validate(changed, data, root))

    mutated_document('missing pair', lambda d: d['requests'].pop())
    mutated_document('duplicate pair', lambda d: d['requests'].__setitem__(1, copy.deepcopy(d['requests'][0])))
    mutated_document('order changed', lambda d: d['requests'].reverse())
    mutated_document('tools empty still forbidden', lambda d: d['requests'][0]['body_template'].update(tools=[]))
    mutated_document('sampling changed', lambda d: d['requests'][0]['body_template'].update(top_p=0.9))
    mutated_document('hidden wire metadata', lambda d: d['requests'][0]['body_template'].update(arm='P'))
    mutated_document('private answer', lambda d: d['requests'][0]['task'].update(answer='FAKE'))
    mutated_document('prompt changed', lambda d: d['requests'][0]['body_template']['messages'][0].update(content='FAKE'))
    mutated_document('wire corruption', lambda d: d['requests'][0].update(wire='{}'))
    mutated_document('P pool enabled', lambda d: d['requests'][0]['execution_policy'].update(TF_DSV41_COMPUTE_LANE='1'))
    helper, _ = p.helpers(Path(root).resolve())
    for rank in (0, 1):
        for flag in p.off_values(helper):
            changed, data = copy.deepcopy(index), dict(payloads)
            record = changed['ranks'][rank]
            record['effective_env'][flag] = '1'
            name = f'rank{rank}/effective-env.json'
            data[name] = p.encoded(record)
            changed['files'][name] = p.digest(data[name])
            data['P-stage-index.json'] = p.encoded(changed)
            rejected(f'rank{rank} {flag}', lambda: validate(changed, data, root))
        for field in ('STATE_DIR', 'CACHE_DIR', 'CONTAINER_PREFIX'):
            changed, data = copy.deepcopy(index), dict(payloads)
            record = changed['ranks'][rank]
            record['effective_env'][field] = p.OLD_PREFIX
            name = f'rank{rank}/effective-env.json'
            data[name] = p.encoded(record)
            changed['files'][name] = p.digest(data[name])
            config = record['stage_config']
            data[config] = helper.env_text(record['effective_env']).encode()
            changed['files'][config] = p.digest(data[config])
            data['P-stage-index.json'] = p.encoded(changed)
            rejected(f'rank{rank} stale {field}', lambda: validate(changed, data, root))
    rejected('old namespace', lambda: p.check_prefix(p.OLD_PREFIX))
    rejected('path escape namespace', lambda: p.check_prefix(p.PREFIX + '/../old'))
    changed = copy.deepcopy(index)
    changed['collection_authorized'] = True
    rejected('approval promotion', lambda: validate(changed, payloads, root))
    blobs = p.sources(root)
    for name in blobs:
        changed_blobs = dict(blobs)
        changed_blobs[name] += b'\nFAKE_DRIFT'
        rejected('source/data hash ' + name, lambda: p.verify_blobs(changed_blobs))
    return {'status': 'FAKE_NOT_COLLECTION', 'collection_authorized': False,
            'positive': ['240 exact matched-L tools removals', '120 public tasks x 2 seeds',
                         'both rank pinned fresh configs; lane and optional features off'],
            'rejected': passed}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
