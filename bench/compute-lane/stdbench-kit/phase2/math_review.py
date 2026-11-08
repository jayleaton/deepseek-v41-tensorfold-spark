#!/usr/bin/env python3
"""Seal all MATH replies for independent blinded review; merge sealed labels offline."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import stat
import tempfile

from analysis import ARMS, SEEDS, canonical, digest, identity, manifest_tasks, read_jsonl, unresolved_bounds, write_json

LABELS = ('correct', 'incorrect', 'unresolved')


def math_task_map(tasks, *, _trusted_selftest=False):
    math_tasks = [t for t in tasks if t.get('kind') == 'math_manual']
    if len(math_tasks) != (2 if _trusted_selftest else 20):
        raise ValueError('seal requires all 20 normalized MATH tasks (80 replies)')
    if len({t['suite'] for t in math_tasks}) != 1:
        raise ValueError('MATH tasks must belong to one suite')
    task_map = {}
    for task in math_tasks:
        key = task['suite'], task['id']
        if key in task_map or not all(isinstance(x, str) and x for x in key):
            raise ValueError('duplicate/invalid MATH task')
        if not isinstance(task.get('prompt'), str) or not isinstance(task.get('answer'), str):
            raise ValueError('normalized problem and official answer strings required')
        task_map[key] = task
    return task_map


def validate_math_inputs(tasks, rows, *, _trusted_selftest=False):
    task_map = math_task_map(tasks, _trusted_selftest=_trusted_selftest)
    math_tasks = list(task_map.values())
    suite = math_tasks[0]['suite']
    expected = {(s, i, seed, arm) for s, i in task_map for seed in SEEDS for arm in ARMS}
    selected = {}
    for row in rows:
        if row.get('suite') != suite:
            continue
        key = identity(row)
        if type(row['seed']) is not int:
            raise ValueError('integer generation seed required')
        if key not in expected or key in selected:
            raise ValueError('unexpected/duplicate MATH response')
        if not isinstance(row.get('status'), str) or not row['status']:
            raise ValueError('response status required')
        content = row.get('content')
        if content is None and row['status'] != 'ok':
            content = ''
        if not isinstance(content, str):
            raise ValueError('actual delivered final content must be a string')
        selected[key] = (row, content)
    if set(selected) != expected:
        raise ValueError('all scheduled MATH replies required; no outcome-selected packet')
    return task_map, selected


def build_preseal(tasks, *, _trusted_selftest=False):
    task_map = math_task_map(tasks, _trusted_selftest=_trusted_selftest)
    rng = random.Random(20261105)
    keys = sorted((s, i, seed, arm) for s, i in task_map for seed in SEEDS for arm in ARMS)
    rng.shuffle(keys)
    mapping, order = {}, []
    for key in keys:
        opaque = 'C-' + format(rng.getrandbits(128), '032x')
        if opaque in mapping:
            raise ValueError('opaque ID collision')
        mapping[opaque] = dict(zip(('suite', 'id', 'seed', 'arm'), key))
        order.append(opaque)
    sorted_tasks = sorted(task_map.values(), key=lambda t: (t['suite'], t['id']))
    return {'version': 1, 'phase': 'before_collection', 'rng': 20261105,
            'tasks': sorted_tasks, 'tasks_sha256': digest(sorted_tasks),
            'mapping': mapping, 'order': order, 'trusted_selftest_fixture': _trusted_selftest}


def validate_preseal(presealed, tasks, expected_sha256, *, _trusted_selftest=False):
    if not expected_sha256 or digest(presealed) != expected_sha256:
        raise ValueError('precollection mapping differs from pinned preseal hash')
    if (presealed.get('version') != 1 or presealed.get('rng') != 20261105
            or presealed.get('phase') != 'before_collection'):
        raise ValueError('invalid precollection mapping format')
    if presealed.get('trusted_selftest_fixture') and not _trusted_selftest:
        raise ValueError('synthetic preseal refused by production')
    task_map = math_task_map(tasks, _trusted_selftest=_trusted_selftest)
    expected_tasks = sorted(task_map.values(), key=lambda t: (t['suite'], t['id']))
    if (digest(expected_tasks) != presealed['tasks_sha256']
            or presealed['tasks'] != expected_tasks):
        raise ValueError('MATH task cohort differs from precollection seal')
    order, mapping = presealed['order'], presealed['mapping']
    if len(order) != len(set(order)) or set(order) != set(mapping):
        raise ValueError('duplicate/missing presealed opaque IDs')
    expected = {(s, i, seed, arm) for s, i in task_map for seed in SEEDS for arm in ARMS}
    actual = [identity(value) for value in mapping.values()]
    if (len(actual) != len(expected) or set(actual) != expected
            or any(type(v['seed']) is not int for v in mapping.values())):
        raise ValueError('presealed mapping must cover both arms/seeds exactly once')
    return mapping


def preseal(tasks, key_path):
    private = build_preseal(tasks)
    write_private(key_path, private)
    return {'candidates': len(private['mapping']), 'preseal_sha256': digest(private),
            'instruction': 'Pin this private hash in the precollection freeze; do not distribute mapping.'}


def build_seal(tasks, rows, *, preseal_key=None, preseal_sha256=None, _trusted_selftest=False):
    task_map, selected = validate_math_inputs(tasks, rows, _trusted_selftest=_trusted_selftest)
    if preseal_key is None:
        if not _trusted_selftest:
            raise ValueError('production seal requires pinned precollection mapping')
        preseal_key = build_preseal(tasks, _trusted_selftest=True)
        preseal_sha256 = digest(preseal_key)
    mapping = validate_preseal(preseal_key, tasks, preseal_sha256, _trusted_selftest=_trusted_selftest)
    packet, entries = [], {}
    for opaque in preseal_key['order']:
        key = identity(mapping[opaque])
        row, content = selected[key]
        task = task_map[key[:2]]
        # No source IDs, arm, seed, grade, timing, status, ownership or paired outcomes.
        packet.append({'candidate_id': opaque, 'problem': task['prompt'],
                       'official_answer': task['answer'], 'candidate_final_content': content})
        entries[opaque] = {'row': row, 'row_sha256': digest(row)}
    private = {'version': 1, 'rng': 20261105, 'packet_sha256': digest(packet),
               'tasks_sha256': digest(sorted(task_map.values(), key=lambda t: (t['suite'], t['id']))),
               'responses_sha256': digest([selected[k][0] for k in sorted(selected)]),
               'preseal': preseal_key, 'preseal_sha256': preseal_sha256,
               'entries': entries, 'trusted_selftest_fixture': _trusted_selftest}
    return packet, private


def write_private(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        stream.write(canonical(value) + '\n')
    if stat.S_IMODE(os.stat(path).st_mode) != 0o600:
        raise ValueError('private key must have mode 0600')


def seal(tasks, rows, packet_path, key_path, *, preseal_path, preseal_sha256):
    if Path(packet_path).resolve() == Path(key_path).resolve():
        raise ValueError('public and private paths must differ')
    if Path(packet_path).exists() or Path(key_path).exists():
        raise ValueError('refuse to overwrite sealed artifacts')
    if stat.S_IMODE(Path(preseal_path).stat().st_mode) != 0o600:
        raise ValueError('precollection mapping must have mode 0600')
    presealed = json.loads(Path(preseal_path).read_text(encoding='utf-8'))
    packet, private = build_seal(tasks, rows, preseal_key=presealed, preseal_sha256=preseal_sha256)
    write_private(key_path, private)
    write_json(packet_path, packet)
    return {'candidates': len(packet), 'packet_sha256': digest(packet),
            'private_key_sha256': digest(private), 'rng': 20261105,
            'instruction': 'Distribute packet only; seal two independent complete reviews before merge.'}


def load_labels(path):
    if str(path).lower().endswith('.csv'):
        with open(path, newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream))
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('review JSON must be an array')
    return data


def validate_labels(labels, reviewer_id, allowed, *, require_all):
    result = {}
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise ValueError('nonempty reviewer identity required')
    for row in labels:
        candidate = row.get('candidate_id')
        if candidate not in allowed or candidate in result:
            raise ValueError('unexpected/duplicate review candidate')
        if row.get('reviewer_id') != reviewer_id:
            raise ValueError('reviewer identity mismatch')
        if row.get('label') not in LABELS or not isinstance(row.get('reason'), str) or not row['reason'].strip():
            raise ValueError('every label requires correct/incorrect/unresolved and a reason')
        result[candidate] = dict(row)
    if require_all and set(result) != set(allowed):
        raise ValueError('each initial reviewer must independently label every candidate')
    return result


def validate_seal(packet, private, key_sha256, *, _trusted_selftest=False):
    if digest(private) != key_sha256:
        raise ValueError('private key differs from presealed key hash')
    if private.get('version') != 1 or private.get('rng') != 20261105:
        raise ValueError('unsupported seal')
    if private.get('trusted_selftest_fixture') and not _trusted_selftest:
        raise ValueError('trusted synthetic seal refused by real merge')
    if len(packet) != (8 if _trusted_selftest else 80):
        raise ValueError('full 80-answer review packet required')
    if digest(packet) != private['packet_sha256']:
        raise ValueError('public packet hash mismatch')
    mapping = validate_preseal(private['preseal'], private['preseal']['tasks'],
                               private['preseal_sha256'], _trusted_selftest=_trusted_selftest)
    if private['tasks_sha256'] != private['preseal']['tasks_sha256']:
        raise ValueError('postcollection task hash differs from precollection seal')
    ids = [item['candidate_id'] for item in packet]
    if ids != private['preseal']['order']:
        raise ValueError('packet order differs from precollection mapping')
    if len(set(ids)) != len(ids) or set(ids) != set(private['entries']):
        raise ValueError('duplicate/missing packet candidate identity')
    rows = []
    allowed_fields = {'candidate_id', 'problem', 'official_answer', 'candidate_final_content'}
    for item in packet:
        if set(item) != allowed_fields or not all(isinstance(v, str) for v in item.values()):
            raise ValueError('packet contains prohibited metadata or non-string content')
        entry = private['entries'][item['candidate_id']]
        if identity(entry['row']) != identity(mapping[item['candidate_id']]):
            raise ValueError('response rekeyed from precollection candidate identity')
        if digest(entry['row']) != entry['row_sha256']:
            raise ValueError('sealed response hash mismatch')
        rows.append(entry['row'])
    if len({identity(r) for r in rows}) != len(rows):
        raise ValueError('duplicate private response mapping')
    if digest(sorted(rows, key=identity)) != private['responses_sha256']:
        raise ValueError('response cohort hash mismatch')
    # Do not trust a self-consistent but incomplete mapping: require its full cross product.
    task_ids = {(r['suite'], r['id']) for r in rows}
    if len(task_ids) != (2 if _trusted_selftest else 20) or len({s for s, _ in task_ids}) != 1:
        raise ValueError('invalid private MATH task cohort')
    expected = {(s, i, seed, arm) for s, i in task_ids for seed in SEEDS for arm in ARMS}
    if {identity(r) for r in rows} != expected:
        raise ValueError('incomplete private MATH cross product')
    return set(ids)


def merge(packet, private, first, second, first_id, second_id, *, key_sha256,
          third=None, third_id=None, _trusted_selftest=False):
    candidates = validate_seal(packet, private, key_sha256, _trusted_selftest=_trusted_selftest)
    if first_id == second_id or third_id in (first_id, second_id):
        raise ValueError('two independent reviewers and a distinct third reviewer required')
    a = validate_labels(first, first_id, candidates, require_all=True)
    b = validate_labels(second, second_id, candidates, require_all=True)
    disputes = {c for c in candidates if a[c]['label'] != b[c]['label']}
    if third is not None:
        c = validate_labels(third, third_id, disputes, require_all=False)
    else:
        if third_id is not None:
            raise ValueError('third reviewer identity supplied without labels')
        c = {}
    resolved, audit = [], []
    for candidate in sorted(candidates):
        votes = [a[candidate], b[candidate]]
        if candidate in c:
            votes.append(c[candidate])
        labels = Counter(v['label'] for v in votes)
        label = next((label for label in ('correct', 'incorrect') if labels[label] >= 2), 'unresolved')
        row = dict(private['entries'][candidate]['row'])
        # Generation failures are incorrect even if a reviewer labels their empty text correct.
        row['grade'] = ('incorrect' if row['status'] != 'ok' else
                        'manual_review' if label == 'unresolved' else label)
        row['math_review'] = {'candidate_id': candidate, 'label': label,
                              'packet_sha256': private['packet_sha256'], 'key_sha256': key_sha256,
                              'reviewers': [v['reviewer_id'] for v in votes], 'sealed': True}
        resolved.append(row)
        audit.append({'candidate_id': candidate, 'labels': votes, 'decision': label,
                      'effective_grade': row['grade']})
    resolved.sort(key=identity)
    clusters = []
    for task in sorted({(r['suite'], r['id']) for r in resolved}):
        indexed = {identity(r): r for r in resolved}
        clusters.append([(indexed[task + (seed, 'plain')], indexed[task + (seed, 'lane')]) for seed in SEEDS])
    summary = {'packet_sha256': private['packet_sha256'], 'private_key_sha256': key_sha256,
               'first_review_sha256': digest(first), 'second_review_sha256': digest(second),
               'third_review_sha256': digest(third) if third is not None else None,
               'reviewer_ids': [first_id, second_id] + ([third_id] if third is not None else []),
               'disputed_candidates': sorted(disputes), 'third_reviewed': len(c),
               'bounds': unresolved_bounds(clusters), 'labels': audit,
               'instruction': 'Keep aggregate output private until all reviewer labels are sealed.'}
    return resolved, summary


def selftest():
    tasks = [{'suite': 'fixture_math', 'id': str(i), 'kind': 'math_manual',
              'prompt': 'What is 1+1?', 'answer': '2'} for i in range(2)]
    rows = [{'suite': t['suite'], 'id': t['id'], 'seed': seed, 'arm': arm,
             'content': '2', 'status': 'ok', 'grade': 'correct'}
            for t in tasks for seed in SEEDS for arm in ARMS]
    packet, private = build_seal(tasks, rows, _trusted_selftest=True)
    presealed = build_preseal(tasks, _trusted_selftest=True)
    pinned = digest(presealed)
    assert private['preseal_sha256'] == pinned
    mutated = json.loads(canonical(presealed))
    mutated['mapping'][mutated['order'][0]]['arm'] = 'changed'
    try:
        build_seal(tasks, rows, preseal_key=mutated, preseal_sha256=pinned, _trusted_selftest=True)
    except ValueError:
        pass
    else:
        raise AssertionError('mutated precollection key accepted')
    assert (packet, private) == build_seal(tasks, list(reversed(rows)), _trusted_selftest=True)
    assert all(set(item) == {'candidate_id', 'problem', 'official_answer', 'candidate_final_content'}
               for item in packet)
    for bad in (rows[:-1], rows + [rows[0]]):
        try:
            build_seal(tasks, bad, _trusted_selftest=True)
        except ValueError:
            pass
        else:
            raise AssertionError('partial/duplicate packet accepted')
    first = [{'candidate_id': p['candidate_id'], 'reviewer_id': 'independent-a',
              'label': 'correct', 'reason': 'exact integer'} for p in packet]
    second = [dict(v, reviewer_id='independent-b') for v in first]
    plain_id = next(k for k, v in private['entries'].items() if v['row']['arm'] == 'plain')
    lane_id = next(k for k, v in private['entries'].items() if v['row']['arm'] == 'lane')
    for row in second:
        if row['candidate_id'] in (plain_id, lane_id):
            row.update(label='incorrect', reason='independent disagreement')
    merged, audit = merge(packet, private, first, second, 'independent-a', 'independent-b',
                          key_sha256=digest(private), _trusted_selftest=True)
    assert sum(r['grade'] == 'manual_review' for r in merged) == 2
    assert audit['bounds']['difference_adverse'] == -.25
    assert audit['bounds']['difference_favorable'] == .25
    third = [{'candidate_id': plain_id, 'reviewer_id': 'independent-c',
              'label': 'correct', 'reason': 'adjudicated exact integer'}]
    merged, _ = merge(packet, private, first, second, 'independent-a', 'independent-b',
                       key_sha256=digest(private), third=third, third_id='independent-c',
                       _trusted_selftest=True)
    assert sum(r['grade'] == 'manual_review' for r in merged) == 1
    try:
        merge(packet, private, first, second, 'independent-a', 'independent-a',
              key_sha256=digest(private), _trusted_selftest=True)
    except ValueError:
        pass
    else:
        raise AssertionError('same reviewer accepted')
    with tempfile.TemporaryDirectory(prefix='stdbench-math-selftest-') as directory:
        path = Path(directory) / 'key.json'
        write_private(path, private)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    return {'selftest': 'passed', 'fixture_candidates': 8}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('preseal', help='before collection: private candidate mapping from tasks only')
    p.add_argument('--manifest', required=True, help='frozen analysis manifest with normalized MATH tasks')
    p.add_argument('--private-key', required=True)
    p = sub.add_parser('seal')
    p.add_argument('--tasks', required=True, help='kit normalized tasks JSONL')
    p.add_argument('--responses', nargs='+', required=True, help='bench response JSONL for both arms')
    p.add_argument('--packet', required=True)
    p.add_argument('--private-key', required=True)
    p.add_argument('--preseal', required=True, help='mode0600 precollection private mapping')
    p.add_argument('--preseal-sha256', required=True, help='precollection canonical mapping hash')
    p = sub.add_parser('merge')
    p.add_argument('--packet', required=True)
    p.add_argument('--private-key', required=True)
    p.add_argument('--key-sha256', required=True, help='canonical private key hash recorded at seal')
    p.add_argument('--first', required=True)
    p.add_argument('--first-id', required=True)
    p.add_argument('--second', required=True)
    p.add_argument('--second-id', required=True)
    p.add_argument('--third')
    p.add_argument('--third-id')
    p.add_argument('--out', required=True, help='private canonical graded MATH JSONL')
    p.add_argument('--audit', required=True, help='private review evidence and bounds JSON')
    sub.add_parser('selftest')
    args = parser.parse_args()
    if args.command == 'selftest':
        print(canonical(selftest()))
    elif args.command == 'preseal':
        manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
        if manifest['seeds'] != list(SEEDS):
            raise ValueError('precollection manifest seeds must be [101,102]')
        tasks = manifest_tasks(manifest)
        if 'tasks_sha256' in manifest and digest(tasks) != manifest['tasks_sha256']:
            raise ValueError('manifest tasks hash mismatch')
        print(canonical(preseal(tasks, args.private_key)))
    elif args.command == 'seal':
        print(canonical(seal(read_jsonl(args.tasks),
                             [r for path in args.responses for r in read_jsonl(path)],
                             args.packet, args.private_key,
                             preseal_path=args.preseal, preseal_sha256=args.preseal_sha256)))
    else:
        key_path = Path(args.private_key)
        if stat.S_IMODE(key_path.stat().st_mode) != 0o600:
            raise ValueError('private key mode must be 0600')
        if any(Path(p).exists() for p in (args.out, args.audit)):
            raise ValueError('refuse to overwrite sealed merge outputs')
        if Path(args.out).resolve() == Path(args.audit).resolve():
            raise ValueError('graded output and audit paths must differ')
        rows, audit = merge(json.loads(Path(args.packet).read_text(encoding='utf-8')),
                            json.loads(key_path.read_text(encoding='utf-8')),
                            load_labels(args.first), load_labels(args.second), args.first_id, args.second_id,
                            key_sha256=args.key_sha256,
                            third=load_labels(args.third) if args.third else None, third_id=args.third_id)
        # Audit and merged rows contain private arm/seed mappings; restrict them too.
        write_private(args.audit, audit)
        descriptor = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(canonical(row) + '\n')
        print(canonical({'merged_candidates': len(rows), 'audit_sha256': digest(audit)}))


if __name__ == '__main__':
    main()
