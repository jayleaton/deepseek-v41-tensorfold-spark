#!/usr/bin/env python3
"""Parent-run offline CPU env/context checks. No Docker, SSH, engine or queue."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import generate_envs3 as gen
import rank_ops3 as ops


HERE = Path(__file__).resolve().parent


def rejects(call):
    try:
        call()
    except (ValueError, FileExistsError):
        return
    raise AssertionError('expected refusal')


def function_tree(path, name):
    tree = ast.parse(path.read_text())
    return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))


def check_context(path, rank, meta):
    # Fake only original config loading and thermal source. No rank command runs.
    configured = dict(meta['effective_env'], GUARD_PAUSE_C='fake-node-policy')

    def thermal(argv, **kwargs):
        assert argv == ['bash', '-c', 'set -a; source /etc/spark/thermal.env; env -0']
        return SimpleNamespace(stdout=b'GUARD_PAUSE_C=fake-node-policy' + bytes([0]))

    original = SimpleNamespace(config_env=lambda cfg: dict(configured))
    with patch.object(ops, 'load_module', return_value=original), patch.object(ops.subprocess, 'run', side_effect=thermal):
        result = ops.context(path, rank)
    assert result[3] == configured and result[-1] == meta['container_name']


def main():
    references = HERE.parent / 'phase2/references'
    with tempfile.TemporaryDirectory(prefix='staged_rank_check3_', dir=HERE) as scratch:
        root = Path(scratch)
        out = root / gen.PREFIX
        index = gen.generate(references, out)
        assert [(b['boot'], b['arm']) for b in index['boots']] == [
            (f'{turn}-{arm}', arm) for turn, arms in gen.ORDERS for arm in arms]
        assert len(index['boots']) == 9 and len(index['files']) == 27
        assert len(list(out.glob('rank*/configs/*.env'))) == 18
        assert all(not b['boot'].startswith('pilot') for b in index['boots'])
        for relative, digest in index['files'].items():
            assert hashlib.sha256((out / relative).read_bytes()).hexdigest() == digest
        all_paths = set()
        for turn, arms in gen.ORDERS:
            configs = {}
            for arm in arms:
                manifest_path = out / 'boots' / f'{turn}-{arm}' / 'manifest.json'
                manifest = json.loads(manifest_path.read_text())
                assert manifest['active_arms'] == ['D'] and set(manifest['arms']) == {'D'}
                assert manifest['phase2']['schema'] == 1
                assert manifest['phase2']['arm'] == arm
                assert manifest['decl_control_a1']['a1_spec_sha256'] == gen.A1_SPEC_SHA256
                for rank, meta in enumerate(manifest['phase2']['ranks']):
                    env = meta['effective_env']
                    configs[arm, rank] = env
                    record = manifest['arms']['D']['ranks'][rank]
                    assert record['image_id'] == gen.IMAGE_IDS[rank]
                    assert env['IMAGE'] == gen.IMAGE and env['ZIG_COMMIT'] == gen.REVISION
                    assert env['TF_DSV41_COMPUTE_LANE'] == str(int(arm == 'lane'))
                    assert all(env[f'TF_DSV41_{flag}'] == '0' for flag in gen.ZERO_FLAGS)
                    assert all(not k.startswith(('GUARD', 'SPARK')) for k in env)
                    controls = {f'TF_DSV41_{flag}' for flag in gen.ZERO_FLAGS}
                    controls.update(('TF_DSV41_COMPUTE_LANE', 'TF_DSV41_RUNNER_FORMS'))
                    assert not any(any(word in k for word in gen.STRIP) and k not in controls for k in env)
                    text = (out / meta['stage_config']).read_text()
                    assert 'source /etc/spark/thermal.env || return 1' in text
                    assert not any(line.startswith(('export GUARD', 'export SPARK')) for line in text.splitlines())
                    for key in ('config', 'state_dir', 'cache_dir', 'container_name'):
                        assert meta[key] not in all_paths
                        all_paths.add(meta[key])
                    assert meta['root'] == f'<rank{rank}-work>/stdbench/decl-control/{gen.PREFIX}'
                    check_context(manifest_path, rank, meta)
            for rank in (0, 1):
                def normalized(arm):
                    return {k: v for k, v in configs[arm, rank].items()
                            if k not in ('STATE_DIR', 'CACHE_DIR', 'CONTAINER_PREFIX')}
                assert normalized('plain') == normalized('declared_off')
                plain, lane = normalized('plain'), normalized('lane')
                assert {k for k in plain.keys() | lane.keys() if plain.get(k) != lane.get(k)} == {'TF_DSV41_COMPUTE_LANE'}
        rejects(lambda: gen.generate(references, out))
        first = json.loads((out / 'boots/t01-plain/manifest.json').read_text())
        bad_path = root / 'staged_bad_manifest.json'
        for boot in ('pilot-plain', 't04-plain', 't01-unknown'):
            first['phase2']['boot'] = boot
            bad_path.write_text(json.dumps(first))
            rejects(lambda: ops.context(bad_path, 0))
    # Verify inherited actual native snapshot implementation was preserved exactly;
    # this is source equality, not a live-container snapshot qualification.
    old = HERE.parent / 'phase2/rank_ops.py'
    for name in ('engine_environment', 'docker_command', 'inspect_container', 'operate'):
        assert function_tree(old, name) == function_tree(HERE / 'rank_ops3.py', name)
    print('PASS: 9 boots/18 envs, fake rank context, policy and snapshot-source preservation; CPU qualification pending')


if __name__ == '__main__':
    main()
