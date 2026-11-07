#!/usr/bin/env python3
"""CPU-only staging. No subprocess, network, queue, lock or engine operations."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE2 = HERE.parent.parent / 'phase2'
FROZEN_SHA = 'adb05167fff21469ee7251555cb243f1f109c4d8e3de237fdcaa59ada8eb55e6'
A1_SHA = 'e7308621c645aef4402924a1b47c1a52e05baabe00239f915011f8431bbd0e89'
ORDERS = {'t01': ('P', 'D', 'L'), 't02': ('D', 'L', 'P'), 't03': ('L', 'P', 'D')}
WAITS = {'aggregate_returned_ids': 8192, 'server_deadline_s': 270,
         'client_wait_s': 330, 'outer_timeout_s': 7200, 'concurrency': 1}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def helpers():
    # Import definitions only; generate_envs.main is guarded. Never import rank_ops.
    spec = importlib.util.spec_from_file_location('staged_env_helpers', PHASE2 / 'generate_envs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_waits(waits):
    if set(waits) != set(WAITS) or any(type(v) is not int for v in waits.values()):
        raise ValueError('invalid wait contract')
    if any(waits[k] != WAITS[k] for k in ('aggregate_returned_ids', 'server_deadline_s', 'concurrency')):
        raise ValueError('inference contract changed')
    if waits['client_wait_s'] < 330 or waits['outer_timeout_s'] < 7200:
        raise ValueError('wait below required floor')


def clean_env(helper, reference, root, prefix, boot, arm):
    # Preserve node-specific transport/assets; exclude inherited optional features.
    env = {k: v for k, v in reference.items()
           if helper.allowed(k) and not k.startswith(('GUARD', 'SPARK'))
           and not any(word in k for word in (*helper.STRIP, 'S3', 'COMMIT', 'STATED', 'CHAIN', 'MARK', 'SYNC_GRAPH'))}
    # ZIG_COMMIT and native DRAFT_CHAIN are engine identity/tuning, not runner features.
    for key in ('TF_DSV41_DRAFT_CHAIN',):
        if key in reference:
            env[key] = reference[key]
    env.update(IMAGE=helper.IMAGE, ZIG_COMMIT=helper.REVISION,
               CONTAINER_PREFIX=f'{prefix}-{boot}',
               STATE_DIR=str(root / 'state' / boot), CACHE_DIR=str(root / 'cache' / boot))
    env.update({f'TF_DSV41_{key}': '0' for key in helper.ZERO_FLAGS})
    env['TF_DSV41_COMPUTE_LANE'] = str(int(arm == 'L'))
    env['TF_DSV41_RUNNER_FORMS'] = 'fence,repl,inline'
    env.setdefault('NCCL_CUMEM_ENABLE', '0')
    env.setdefault('NCCL_CUMEM_HOST_ENABLE', '0')
    helper.reject_banned(env)
    return env


def generate(references, out, prefix):
    if not re.fullmatch(r'staged_decl_[A-Za-z0-9][A-Za-z0-9_.-]*', prefix):
        raise ValueError('fresh staged_decl_ namespace required')
    if out.exists() or out.is_symlink():
        raise FileExistsError('output already exists')
    if out.name != prefix:
        raise ValueError('output basename must equal fresh prefix')
    if not out.resolve().is_relative_to(HERE) or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError('output must stay inside stage-turns without symlinks')
    helper = helpers()
    spec = HERE.parent / 'T5-DECL-CONTROL-SPEC.md'
    if digest(spec) != FROZEN_SHA:
        raise ValueError('original frozen spec changed; explicit staging review required')
    base = json.loads((references / 'base-arms.json').read_text())['arms']['D']
    refs = [helper.read_reference(references / f'base-r{rank}.env') for rank in (0, 1)]
    for rank, env in enumerate(refs):
        if env['IMAGE'] not in (helper.IMAGE, helper.IMAGE_IDS[rank]) or env['ZIG_COMMIT'] != helper.REVISION:
            raise ValueError('reference engine/image mismatch')
        if base['ranks'][rank]['image_id'] != helper.IMAGE_IDS[rank]:
            raise ValueError('rank image ID mismatch')
    files, boots = {}, []
    for turn, arms in ORDERS.items():
        for arm in arms:
            boot = f'{turn}-{arm}'
            record = copy.deepcopy(base)
            record.pop('alias_of', None)
            record.update(engine_sha=helper.REVISION, description=f'STAGED ONLY declaration-control {boot}')
            ranks, settings = [], []
            for rank in (0, 1):
                root = Path(f'<rank{rank}-work>/stdbench/decl-control') / prefix
                env = clean_env(helper, refs[rank], root, prefix, boot, arm)
                relative = f'rank{rank}/configs/{boot}-r{rank}.env'
                # Keep existing thermal sourcing and safe assignment serialization.
                files[relative] = helper.env_text(env).replace('Generated phase-2 config', 'STAGED ONLY declaration-control config')
                config = str(root / 'configs' / f'{boot}-r{rank}.env')
                flags = {k: v for k, v in env.items() if k.startswith(('TF_', 'GLM53_TF'))}
                settings.append(flags)
                record['ranks'][rank].update(config=config, image_id=helper.IMAGE_IDS[rank], settings=flags)
                ranks.append({'rank': rank, 'root': str(root), 'config': config,
                              'stage_config': relative, 'container_name': f'{prefix}-{boot}-r{rank}',
                              'state_dir': env['STATE_DIR'], 'cache_dir': env['CACHE_DIR'],
                              'table_source': str(Path(record['ranks'][rank]['recipe_dir']) / 'config/pfdense-table.json'),
                              'effective_env': helper.filtered(env), 'guard_source': '/etc/spark/thermal.env'})
            record['settings'] = {k: v for k, v in settings[0].items() if settings[1].get(k) == v}
            record['native_settings'] = {}
            manifest_path = f'boots/{boot}/manifest.json'
            manifest = {'schema': 'decl-control-staging-v1', 'status': 'STAGED_NOT_QUALIFIED',
                        'active_arms': [arm], 'arms': {arm: record},
                        'decl_control': {'prefix': prefix, 'boot': boot, 'turn': turn, 'arm': arm,
                                         'lane': int(arm == 'L'), 'declaration': arm != 'P', 'ranks': ranks},
                        'waits': WAITS}
            files[manifest_path] = json.dumps(manifest, indent=2, sort_keys=True) + '\n'
            boots.append({'boot': boot, 'turn': turn, 'arm': arm, 'manifest': manifest_path})
    index = {'schema': 'decl-control-staging-v1', 'status': 'STAGED_NOT_QUALIFIED',
             'prefix': prefix, 'original_spec_sha256': FROZEN_SHA,
             'amended_spec_sha256': A1_SHA, 'controller_sha256': None,
             'image': helper.IMAGE, 'revision': helper.REVISION, 'image_ids': helper.IMAGE_IDS,
             'boots': boots, 'orders': ORDERS, 'waits': WAITS,
             'reserve': {'turn': 'reserve', 'new_boots': False, 'unmeasured_fixed_units_only': True},
             'files': {p: hashlib.sha256(v.encode()).hexdigest() for p, v in files.items()},
             'references': {p.name: digest(p) for p in references.iterdir() if p.name in
                            ('base-arms.json', 'base-r0.env', 'base-r1.env')},
             'helper_sha256': digest(PHASE2 / 'generate_envs.py'),
             'limitations': ['No request/layout/CPU qualification seal', 'A1 served-count/wire/conditional-input gates pending',
                             'No forced-call dispatch qualification', 'Old rank_ops/controller reject this schema',
                             'Future sealed three-arm controller and rank adapter required; CPU qualification pending']}
    # Materialize only after all staging validation; exclusive files, never overwrite.
    out.mkdir(parents=True, exist_ok=False)
    for relative, content in files.items():
        helper.dump_new(out / relative, content)
    helper.json_new(out / 'manifest.json', index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, default=PHASE2 / 'references')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    index = generate(args.references, args.out, args.prefix)
    print(json.dumps({'status': index['status'], 'boots': len(index['boots']), 'envs': 18}))


if __name__ == '__main__':
    main()
