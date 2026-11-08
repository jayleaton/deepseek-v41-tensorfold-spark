"""Read pinned RANK0-HOST exclusions. No remote access, discovery, or empty fallback.

load_exclusions(config=None) -> (records, audit). Defaults are authoritative RANK0-HOST
paths, never development mirrors. config may override *_root for a parent-staged
CPU fixture and *_manifest_sha256/freeze_sha256 only with prospectively trusted
pins. The original pilot source is mandatory: pilot_path + pilot_sha256 identify
an original JSON list/JSONL of eleven {id,prompt} rows, NOT reconstructed W9 rows.
For original request receipts instead supply pilot_receipts, a list of eleven
{path,sha256,id,request_pointer}; request_pointer is a JSON Pointer to the actual
request body. Each body must contain exactly one original user string message.
The pinned layout + W9 rows independently validate those original pilot prompts.
The original a-s4 receipt directory located by the parent is
<rank0-work>/dsv41-bench/out/imr-measure-20261104T074954Z/ab-a-plain-lane/a-s4.
Supply exact receipt files/pins/pointers after read-only inspection; a directory
name alone cannot establish which JSON field contains the original request.

Historical answers/tests/output are never returned. Stdbench analysis-lane tasks
are original task specs, not inference replies. The dataset lock and selected
normalized originals are independently checked against their frozen manifest.
"""
import hashlib
import json
from pathlib import Path

from disjointness import DisjointnessError, normalize_prompt, source_review_inventory

W9_SHA = '092d6a83593adaa148344e886ccd393ef19e78fa94c98774c1711a0ac0213cb9'
W9_MANIFEST_SHA = '1aad3a3e1f73c08e925b362659ebe6da8029a1b25d5b480b30527ae1f630b4ae'
AB3_SHA = '6053d86cecd00bc91cf78b2606307766fb97f4076fae2dbb6833be99930d0174'
AB3_FREEZE_SHA = '93417d27c61d8c52069ec1b33cb83d79f26e6cd3fdc0b8d33127e784b6315e7a'
STDBENCH_MANIFEST_SHA = '29d8b4531e098f42960cf9f8afbe113d7905316d9f98aec30034b1ba6ed30cd3'
PILOT_IDS = frozenset('w9-' + x for x in (
    'add30-051', 'add30-062', 'algo-013', 'bignum-010', 'div_big-035',
    'mbpp-773', 'mul30-010', 'mul30-014', 'mul30-018', 'mul_big-043', 'mul_big-055'))


def _json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DisjointnessError('duplicate source JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique)


def _read(path, expected, audit):
    path = Path(path)
    if not isinstance(expected, str) or len(expected) != 64:
        raise DisjointnessError('BLOCK: missing trusted SHA256 for ' + str(path))
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DisjointnessError('BLOCK: missing exclusion source ' + str(path)) from exc
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise DisjointnessError('BLOCK: exclusion hash mismatch: ' + str(path))
    audit.append({'path': str(path.resolve()), 'sha256': actual, 'bytes': len(data)})
    return data


def _relative(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise DisjointnessError('freeze path escapes source root')
    return path


def _pinned(root, manifest, relative, audit, field='files_sha256'):
    expected = manifest.get(field, {}).get(relative)
    return _read(_relative(root, relative), expected, audit)


def _rows(data):
    text = data.decode('utf-8').strip()
    if not text:
        raise DisjointnessError('BLOCK: empty exclusion source')
    # JSON arrays and JSONL are both accepted, irrespective of filename.
    if text.startswith('['):
        rows = _json(text)
    else:
        rows = [_json(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise DisjointnessError('exclusion source must contain objects')
    return rows


def _public(row, source):
    if not isinstance(row.get('id'), str) or not row['id']:
        raise DisjointnessError('original task ID missing: ' + source)
    normalize_prompt(row.get('prompt'))
    result = {key: row[key] for key in ('id', 'prompt', 'set', 'spec', 'family', 'inputs', 'suite') if key in row}
    result['source'] = source
    return result


def _unique(rows, count, label):
    if len(rows) != count or len({row.get('id') for row in rows}) != count:
        raise DisjointnessError(f'BLOCK: {label} requires exactly {count} unique original tasks')


def _pointer(value, pointer):
    if not isinstance(pointer, str) or (pointer and not pointer.startswith('/')):
        raise DisjointnessError('invalid receipt JSON pointer')
    for part in pointer.split('/')[1:]:
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def _pilot(config, audit):
    if config.get('pilot_path'):
        return _rows(_read(config['pilot_path'], config.get('pilot_sha256'), audit))
    receipts = config.get('pilot_receipts')
    if not isinstance(receipts, list) or len(receipts) != 11:
        raise DisjointnessError('BLOCK: original eleven pilot units path/hash or eleven pinned request receipts required; relative pilot tasks=units.jsonl is not an absolute source')
    rows = []
    for receipt in receipts:
        data = _json(_read(receipt['path'], receipt.get('sha256'), audit))
        try:
            request = _pointer(data, receipt['request_pointer'])
            users = [message['content'] for message in request['messages'] if message['role'] == 'user']
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise DisjointnessError('invalid original pilot request location') from exc
        if len(users) != 1 or not isinstance(users[0], str):
            raise DisjointnessError('ambiguous original pilot user request')
        rows.append({'id': receipt['id'], 'prompt': users[0]})
    return rows


def load_exclusions(config=None):
    config = dict(config or {})
    audit = []
    wroot = Path(config.get('w9_root', '<rank0-work>/imr-t5-w9/frozen'))
    aroot = Path(config.get('ab3_root', '<rank0-work>/ab3-20261104'))
    sroot = Path(config.get('stdbench_root', '<rank0-work>/stdbench/phase2/frozen-v3'))
    wm = _json(_read(wroot / 'manifest.json', config.get('w9_manifest_sha256', W9_MANIFEST_SHA), audit))
    wbytes = _pinned(wroot, wm, 'tasks-public.json', audit)
    if hashlib.sha256(wbytes).hexdigest() != W9_SHA:
        raise DisjointnessError('W9 taskset does not match frozen 450-task pin')
    w9 = _rows(wbytes)
    audit[-1]['count'] = len(w9)
    _unique(w9, 450, 'W9')
    families = {}
    for row in w9:
        families[row.get('set')] = families.get(row.get('set'), 0) + 1
    if families != {'add30': 80, 'mul30': 20, 'mul_big': 80, 'div_big': 60, 'bignum': 20, 'algo': 40, 'mbpp': 150}:
        raise DisjointnessError('W9 family census mismatch')
    originals = {row['id']: _public(row, 'W9') for row in w9}
    for row in originals.values():
        row.update(source_path=str(wroot / 'tasks-public.json'), source_sha256=W9_SHA)

    af = _json(_read(aroot / 'AB3-FREEZE.json', config.get('ab3_freeze_sha256', AB3_FREEZE_SHA), audit))
    kit = aroot / 'kit2'
    abytes = _pinned(kit, af, 'ab3-units.jsonl', audit, 'kit_files_sha256')
    if hashlib.sha256(abytes).hexdigest() != AB3_SHA or af.get('units_sha256') != AB3_SHA:
        raise DisjointnessError('AB3 units pin mismatch')
    ab3 = _rows(abytes)
    audit[-1]['count'] = len(ab3)
    _unique(ab3, 60, 'AB3')
    if set(af.get('unit_ids', [])) != {row['id'] for row in ab3}:
        raise DisjointnessError('AB3 freeze ID mismatch')
    layout = _json(_pinned(kit, af, 'ab3-layout.json', audit, 'kit_files_sha256'))
    if (layout.get('manifest_sha256') != config.get('w9_manifest_sha256', W9_MANIFEST_SHA)
            or layout.get('rows_sha256') != wm['files_sha256'].get('tasks-private.jsonl')
            or layout.get('units_sha256') != AB3_SHA):
        raise DisjointnessError('AB3 layout lineage disagrees with W9/AB3 freezes')
    if len(layout.get('excluded_pilot_units', [])) != 11 or set(layout['excluded_pilot_units']) != PILOT_IDS:
        raise DisjointnessError('AB3 pilot layout must pin the exact eleven original IDs')
    pilots = _pilot(config, audit)
    _unique(pilots, 11, 'AB3 original pilot')
    if {row['id'] for row in pilots} != PILOT_IDS:
        raise DisjointnessError('original pilot ID mismatch')
    dedup = []
    for label, rows in (('AB3', ab3), ('AB3-pilot', pilots)):
        for row in rows:
            original = originals.get(row['id'])
            if original is None or row.get('prompt') != original['prompt']:
                raise DisjointnessError('BLOCK: W9 descendant original prompt differs: ' + row['id'])
            if 'spec' in row and row['spec'] != original.get('spec'):
                raise DisjointnessError('W9 descendant structured operands differ: ' + row['id'])
            dedup.append({'source': label, 'id': row['id'], 'deduplicated_against': 'W9',
                          'original_prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest()})
    if {row['id'] for row in ab3} & PILOT_IDS:
        raise DisjointnessError('pilot units unexpectedly selected into AB3')

    sm = _json(_read(sroot / 'manifest.json', config.get('stdbench_manifest_sha256', STDBENCH_MANIFEST_SHA), audit))
    analysis = _json(_pinned(sroot, sm, 'analysis-lane.json', audit))
    lock = _json(_pinned(sroot, sm, 'sources/datasets.lock.json', audit))
    # The freeze stores selected normalized originals, not the raw remote datasets.
    # The raw-dataset digest/revision must match each selected row's provenance.
    datasets = {}
    for dataset in lock['datasets']:
        suite = dataset['suite']
        data = _pinned(sroot, sm, 'sources/' + suite + '.normalized.jsonl', audit)
        selected = _rows(data)
        audit[-1]['count'] = len(selected)
        datasets[suite] = {row['id']: row for row in selected}
        if len(datasets[suite]) != len(selected):
            raise DisjointnessError('duplicate normalized source task ID')
        for row in selected:
            if row.get('source_sha256') != dataset['sha256'] or row.get('source_revision') != dataset['revision']:
                raise DisjointnessError('stdbench original provenance disagrees with dataset lock')
    std = []
    for suite, block in analysis['suites'].items():
        for row in block['tasks']:
            original = datasets.get(suite, {}).get(row['id'])
            if original is None or not row['prompt'].startswith(original['prompt']):
                raise DisjointnessError('stdbench task lacks its pinned original problem')
            public = _public(row, 'stdbench/' + suite)
            public['suite'] = suite
            public['original_prompt'] = original['prompt']
            public['source_path'] = str(sroot / 'analysis-lane.json')
            public['source_sha256'] = sm['files_sha256']['analysis-lane.json']
            std.append(public)
    _unique(std, 92, 'stdbench')
    records = list(originals.values()) + std
    return records, {'status': 'source-pins-verified-not-semantic-seal', 'files': audit,
                     'counts': {'W9': 450, 'AB3': 60, 'AB3-pilot': 11, 'stdbench': 92,
                                'deduplicated_W9_descendants': 71, 'unique_records': len(records)},
                     'deduplications': dedup,
                     'semantic_review_inventory': source_review_inventory(records)}
