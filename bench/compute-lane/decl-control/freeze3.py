#!/usr/bin/env python3
"""Freeze fresh A1 three-arm request/layout manifest; CPU-only, no dispatch."""
import hashlib
import json
from pathlib import Path
import sys
from reconciliation import digest, wire, compare_declared_templates

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'phase2/frozen-v3'
BASE_SHA = 'adb05167fff21469ee7251555cb243f1f109c4d8e3de237fdcaa59ada8eb55e6'
A1_SHA = 'e7308621c645aef4402924a1b47c1a52e05baabe00239f915011f8431bbd0e89'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())
def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(obj, f, indent=2, ensure_ascii=False); f.write('\n')


def freeze(out):
    assert sha(HERE/'T5-DECL-CONTROL-SPEC.md') == BASE_SHA
    assert sha(HERE/'T5-DECL-CONTROL-SPEC-A1.md') == A1_SHA
    old = load(OLD/'manifest.json')
    originals = {}
    for entry in old['boots']:
        if entry['arm'] != 'lane': continue
        boot = load(OLD/entry['file'])
        diagnostics = boot['diagnostics']
        for row in boot['requests']:
            key = (row['suite'],row['id'],row['seed'])
            assert key not in originals
            originals[key] = row
    prompts = {tuple(r['key']):r for r in map(json.loads,(HERE/'prompt-arrays.jsonl').read_text().splitlines())}
    draft = load(HERE/'DRAFT-requests-NOT-QUALIFIED.json')
    assert len(originals) == 184 and len(draft['requests']) == 552
    assert len(prompts) == 186
    assert not out.exists()
    out.mkdir()
    entries, turns, all_requests = [], [], []
    for layout in draft['layouts']:
        turn = f't{layout["turn"]:02}'
        names = []
        for arm in layout['arms']:
            boot_name = f'{turn}-{arm}'
            rows = []
            for item in draft['requests']:
                if item['turn'] != layout['turn'] or item['arm'] != arm: continue
                key = tuple(item['key'])
                original = originals[key]
                assert item['body']['messages'] == original['body_template']['messages']
                cpu = prompts[key]
                assert cpu['declared_off_ids'] == cpu['declared_lane_ids'] == cpu['original_builtin_lane_ids']
                rows.append(dict(original, body_template=item['body'],
                                 body_sha256=digest(item['wire']), wire=item['wire'],
                                 wire_sha256=item['wire_sha256'], key=item['key'], arm=arm,
                                 execution_policy=item['execution_policy']))
            assert [r['key'] for r in rows] == layout['pair_order']
            diags = []
            for original in diagnostics:
                body = dict(original['body_template'])
                body.pop('tools',None)
                if arm != 'plain': body['tools'] = load(HERE/'canonical-python-tools.json')
                raw = wire(body)
                key = [original['suite'],original['id'],original['seed']]
                diags.append(dict(original, body_template=body, wire=raw, wire_sha256=digest(raw),
                                  body_sha256=digest(raw), key=key, arm=arm,
                                  execution_policy={'TF_DSV41_COMPUTE_LANE':str(int(arm=='lane'))}))
                assert tuple(key) in prompts
            name = f'boots/{boot_name}.json'
            save(out/name, {'boot':boot_name,'arm':arm,'diagnostics':diags,'requests':rows})
            entries.append({'boot':boot_name,'arm':arm,'file':name,'sha256':sha(out/name)})
            names.append(boot_name); all_requests.extend(rows)
        turns.append({'name':turn,'boots':names})
    assert len(all_requests) == 552 and len(entries) == 9
    pairs = {}
    for r in all_requests:
        pairs.setdefault(tuple(r['key']),{})[r['arm']] = dict(r,body=r['body_template'])
    for key, arms in pairs.items():
        assert compare_declared_templates(arms['declared_off'],arms['lane'],prompts[key])
    previous = load(OLD/'analysis-lane.json')
    tasks = [t for suite in previous['suites'].values() for t in suite['tasks']]
    manifest = {'tasks':tasks,'seeds':[101,102],'arms':['plain','declared_off','lane'],
                'base_spec_sha256':BASE_SHA,'spec_sha256':A1_SHA}
    save(out/'analysis-manifest.json',manifest)
    save(out/'manifest.json', {'base_spec_sha256':BASE_SHA,'spec_sha256':A1_SHA,
                              'boots':entries,'turns':turns,'fresh_replies':552,
                              'reserve':'only unmeasured fixed units; no automatic replacement',
                              'cpu_prompt_arrays_sha256':sha(HERE/'prompt-arrays.jsonl'),
                              'draft_sha256':sha(HERE/'DRAFT-requests-NOT-QUALIFIED.json'),
                              'analysis_manifest_sha256':sha(out/'analysis-manifest.json')})
    print(json.dumps({'freeze':'PASS','fresh_replies':552,'boots':9,'turns':3,
                      'manifest_sha256':sha(out/'manifest.json'),'GPU':False}))


if __name__ == '__main__': freeze(Path(sys.argv[1]))
