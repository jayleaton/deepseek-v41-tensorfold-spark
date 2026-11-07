#!/usr/bin/env python3
"""AB3 freeze manifest (T5-AB3-SPEC.md registration): sha256 of every file in the frozen kit and every turn copy (each
must equal the kit), the layout's units/order/config/template hashes, the pinned image IDs, engine, r7 oracle, guide,
rendered tool schema and engine runner prelude, written before the first AB3 request.
Usage: python3 -I ab3_freeze.py KIT_DIR OUT_JSON FROZEN_AT_UTC TURN_DIR..."""

import sys as _archive_sys
import hashlib
import json
import sys
from pathlib import Path

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

kit, out, at, turns = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], [Path(p) for p in sys.argv[4:]]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
files = {str(p.relative_to(kit)): sha(p) for p in sorted(kit.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
for t in turns:
    mine = {str(p.relative_to(t)): sha(p) for p in sorted(t.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    assert mine == files, f'{t} differs from the kit'
L = json.loads((kit / 'ab3-layout.json').read_text())
schema = json.loads((kit / 'rendered-tool-schema.json').read_text())
oracle = Path('<rank0-work>/owned-oracle/tool/oracle.py')
freeze = {
    'frozen_at_utc': at, 'spec': 'T5-AB3-SPEC.md',
    'spec_sha256': '81b34c1a79a754efca01ebee85f47c74490a029f23e4e0533799205b60dd524e',
    'engine_sha': '3bf584af1c5407006f154e0c70d05c6f495d0487', 'image_tag': 'tensorfold-dsv41-imr:r-3bf584af1c54',
    'image_ids': {'rank0-host': 'sha256:26c0b4107869671656476006743be5288dc48c8ee1f5c5b3879bd606fba59535',
                  'rank1-host': 'sha256:1cfb03ebae4f58de4ef61be0bd8527e7d88dd4c046bfce98844bc53305d37292'},
    'arms_base': 'imr-3bf584af1c54', 'r7_oracle_sha256': sha(oracle),
    'r7_pinned': '35f2bdf0243df7917b2b61098a1ff4a91bf1044854634912ab4800495db853fc',
    'guide_trust_sha256': L['guide_trust_sha256'], 'tool_definition_sha256': schema['definition_sha256'],
    'tool_rendered_probes': {k: (v['count'], v['tokens_sha256']) for k, v in schema['probes'].items()},
    'engine_runner_prelude_sha256': sha(kit / 'engine_runner.txt'), 'units_sha256': L['units_sha256'],
    'unit_ids': [json.loads(x)['id'] for x in (kit / L['units']).read_text().splitlines() if x],
    'blocks': [{'block': b['block'], 'units': b['units'], 'order_sha256': b['order_sha256'],
                'boots': {x['name']: x['templates_sha256'] for x in b['boots']}} for b in L['blocks']],
    'turns': [{'turn': t['turn'], 'boots': t['boots'], 'configs_sha256': t['configs_sha256']} for t in L['turns']],
    'turn_dirs': [str(t) for t in turns], 'kit': str(kit), 'kit_files_sha256': files}
assert freeze['r7_oracle_sha256'] == freeze['r7_pinned'], 'the live r7 oracle is not the pinned one'
out.write_text(json.dumps(freeze, indent=1) + '\n')
print(out, sha(out), len(files), 'files')
