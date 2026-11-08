#!/usr/bin/env python3
"""Parent-run CPU check only: env/boot/wait gates; no engines or subprocesses."""
import hashlib
import json
from pathlib import Path
import tempfile

import stage_turns as stage
import staged_gate as gate


def rejects(call):
    try:
        call()
    except (ValueError, FileExistsError):
        return
    raise AssertionError('expected closed gate')


def main():
    helper = stage.helpers()
    # Fake hostile inheritance must never reach a generated env.
    fake = {'IMAGE': 'ignored', 'ZIG_COMMIT': 'ignored',
            'TF_DSV41_RUNNER': '1', 'TF_DSV41_COMPUTE_LANE': '1',
            'TF_DSV41_TRACE': '1', 'TF_DSV41_GUIDE': '1', 'TF_DSV41_S3': '1',
            'GUARD_PAUSE_C': '999', 'UNRELATED_PRIVATE_VALUE': 'never-export',
            'TF_DSV41_DRAFT_CHAIN': '1', 'TF_TP_MASTER': 'transport-fixture'}
    env = stage.clean_env(helper, fake, Path('/fixture'), 'staged_decl_fake', 't01-D', 'D')
    assert env['TF_DSV41_COMPUTE_LANE'] == '0'
    assert env['TF_DSV41_DRAFT_CHAIN'] == '1' and env['TF_TP_MASTER'] == 'transport-fixture'
    assert all(env[f'TF_DSV41_{key}'] == '0' for key in helper.ZERO_FLAGS)
    assert not any(k in env for k in ('TF_DSV41_TRACE', 'TF_DSV41_GUIDE', 'TF_DSV41_S3',
                                    'GUARD_PAUSE_C', 'UNRELATED_PRIVATE_VALUE'))
    # Scratch is inside the only permitted artifact subtree and always cleaned.
    with tempfile.TemporaryDirectory(prefix='staged_selfcheck_', dir=stage.HERE) as scratch:
        out = Path(scratch) / 'staged_decl_selfcheck'
        stage.generate(stage.PHASE2 / 'references', out, out.name)
        index = json.loads((out / 'manifest.json').read_text())
        gate.validate_boots(index)
        assert len(index['boots']) == 9 and len(index['files']) == 27
        assert len(list(out.glob('rank*/configs/*.env'))) == 18
        for relative, expected in index['files'].items():
            assert hashlib.sha256((out / relative).read_bytes()).hexdigest() == expected
        seen = set()
        for boot in index['boots']:
            manifest = json.loads((out / boot['manifest']).read_text())
            meta = manifest['decl_control']
            assert 'phase2' not in manifest and meta['arm'] == boot['arm']
            assert meta['lane'] == int(boot['arm'] == 'L')
            assert meta['declaration'] == (boot['arm'] != 'P')
            for rank, item in enumerate(meta['ranks']):
                configured = item['effective_env']
                assert configured['ZIG_COMMIT'] == helper.REVISION
                assert manifest['arms'][boot['arm']]['ranks'][rank]['image_id'] == helper.IMAGE_IDS[rank]
                assert configured['TF_DSV41_COMPUTE_LANE'] == str(int(boot['arm'] == 'L'))
                assert all(configured[f'TF_DSV41_{key}'] == '0' for key in helper.ZERO_FLAGS)
                for key in ('state_dir', 'cache_dir', 'container_name', 'config'):
                    assert item[key] not in seen
                    seen.add(item[key])
        rejects(lambda: stage.generate(stage.PHASE2 / 'references', out, out.name))
        broken = json.loads(json.dumps(index))
        broken['boots'][0]['arm'] = 'L'
        rejects(lambda: gate.validate_boots(broken))
    stage.validate_waits(stage.WAITS)
    for key, value in (('client_wait_s', 329), ('outer_timeout_s', 7199),
                       ('server_deadline_s', 269), ('aggregate_returned_ids', 8191), ('concurrency', 2)):
        rejects(lambda: stage.validate_waits(dict(stage.WAITS, **{key: value})))
    assert gate.AMENDED_SPEC_SHA256 == stage.A1_SHA and gate.BINDING_SHA256 is None
    for turn in (*stage.ORDERS, 'reserve'):
        rejects(lambda: gate.validate_binding(turn))
    print('PASS: staging env/boot/wait gates only; no campaign qualification')


if __name__ == '__main__':
    main()
