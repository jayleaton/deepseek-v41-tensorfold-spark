#!/usr/bin/env python3
"""CPU-only artifact checks; never qualify collection from reconstructed IDs."""
import hashlib
import json
from pathlib import Path
import py_compile
import tempfile

root = Path(__file__).resolve().parent
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(root / 'T5-DECL-CONTROL-SPEC.md') == 'adb05167fff21469ee7251555cb243f1f109c4d8e3de237fdcaa59ada8eb55e6'
proof = json.loads((root / 'cpu-proof.json').read_text())
assert proof['cpu_prompt_arrays_equal']
assert proof['collection_qualified'] is False
assert proof['dispatch_negative_control_qualified'] is False
assert proof['actual_served_prompt_array_capture_available'] is False
rows = [json.loads(s) for s in (root / 'prompt-arrays.jsonl').read_text().splitlines()]
assert len(rows) == 186
for row in rows:
    assert row['declared_off_ids'] == row['declared_lane_ids'] == row['original_builtin_lane_ids']
    assert row['held_off'] is False and row['held_lane'] is True
assert digest(root / 'prompt-arrays.jsonl') == proof['prompt_arrays_sha256']
draft = json.loads((root / 'DRAFT-requests-NOT-QUALIFIED.json').read_text())
assert draft['status'] == 'CPU_FEASIBILITY_ONLY_NOT_COLLECTABLE'
assert len(draft['requests']) == 552
by_pair = {}
for row in draft['requests']:
    by_pair.setdefault(tuple(row['key']), {})[row['arm']] = row
assert len(by_pair) == 184
for arms in by_pair.values():
    assert set(arms) == {'plain', 'declared_off', 'lane'}
    assert arms['declared_off']['wire'] == arms['lane']['wire']
    assert arms['declared_off']['execution_policy'] == {'TF_DSV41_COMPUTE_LANE': '0'}
    assert arms['lane']['execution_policy'] == {'TF_DSV41_COMPUTE_LANE': '1'}
    assert 'tools' not in arms['plain']['body']
with tempfile.TemporaryDirectory(prefix='decl-control-syntax-') as scratch:
    for p in sorted(root.glob('*.py')):
        py_compile.compile(str(p), cfile=str(Path(scratch) / (p.name + 'c')), doraise=True)
result = {'artifact_checks': 'PASS', 'full_array_pairs': len(rows), 'draft_fresh_requests': 552,
          'collection': 'BLOCKED', 'source_sha256': {p.name: digest(p) for p in sorted(root.glob('*.py'))}}
(root / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
