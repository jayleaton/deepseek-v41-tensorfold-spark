#!/usr/bin/env python3
"""Stage new phase-2 configurations; never install, connect to nodes, or launch."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shlex

PREFIX = 'stdbench-20261105-v1'
REVISION = '3a429d6f122d9eb2900c43e622933d311e4be44f'
IMAGE = 'tensorfold-dsv41-imr:r-3a429d6f122d'
IMAGE_IDS = (
    'sha256:5be7d7d40f267590b9e7b41410f48d40048af24e004203629a347d3d7c16a8cd',
    'sha256:80beaba732d1ed9439e012b007d8dbe19ec2f8962f50fbac29a18c8cd6fda306',
)
BANNED = ('KV_SPLIT_DEDUP', 'KV_SPLIT_EARLY', 'TF_TP_ROCE_CHUNKED_KB')
STRIP = ('RUNNER', 'TRACE', 'GUIDE', 'COMPUTE_LANE')
ZERO_FLAGS = ('RUNNER', 'COMPUTE_LANE_RUNNER', 'RUNNER_COMMIT', 'RUNNER_STATED',
              'RUNNER_INLINE_CHAIN', 'RUNNER_INLINE_MARK', 'RUNNER_SYNC_GRAPH', 'PREPARED_WRITE')
PATH_KEYS = {'MODEL_DIR', 'ENGRAM_DIR', 'ASSETS_DIR', 'PREPARED_DIR', 'STATE_DIR',
             'CACHE_DIR', 'CONFIG', 'HOST', 'PORT', 'IMAGE', 'ZIG_COMMIT', 'CONTAINER_PREFIX'}


def allowed(key):
    return (key.startswith(('TF_', 'GLM53_TF', 'NCCL', 'GUARD', 'SPARK'))
            or key in PATH_KEYS)


def filtered(env):
    return {k: v for k, v in sorted(env.items()) if allowed(k)}


def reject_banned(value):
    if any(word in str(value) for word in BANNED):
        raise ValueError('forbidden experiment flag')


def read_reference(path):
    """Read the sealed literal assignment format without executing reference code."""
    result = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        parts = shlex.split(line, comments=True)
        if not parts:
            continue
        if parts[0] == 'export':
            parts.pop(0)
        if len(parts) != 1 or '=' not in parts[0]:
            raise ValueError(f'{path}:{number}: expected literal assignment')
        key, value = parts[0].split('=', 1)
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            raise ValueError('invalid env key')
        if any(c in value for c in ('$','`','\n','\r')):
            raise ValueError('reference is not literal')
        result[key] = value
    reject_banned(result)
    return result


def dump_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(value)


def json_new(path, value):
    dump_new(path, json.dumps(value, indent=2, sort_keys=True) + '\n')


def env_text(env):
    # Guard values must come from the node, never from a copied numeric override.
    lines = ['# Generated phase-2 config; source only the node thermal policy.',
             'set -a', 'source /etc/spark/thermal.env || return 1']
    lines += [f'export {key}={shlex.quote(value)}' for key, value in sorted(env.items())]
    return '\n'.join(lines) + '\n'


def generate(references, out, prefix=PREFIX):
    if not re.fullmatch(r'stdbench-[A-Za-z0-9_.-]+', prefix):
        raise ValueError('invalid owned prefix')
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    base = json.loads((references / 'base-arms.json').read_text())
    refs = [read_reference(references / f'base-r{rank}.env') for rank in (0, 1)]
    for rank, env in enumerate(refs):
        if env['IMAGE'] not in (IMAGE, IMAGE_IDS[rank]) or env['ZIG_COMMIT'] != REVISION:
            raise ValueError('reference image/revision differs from pin')
        if base['arms']['D']['ranks'][rank]['image_id'] != IMAGE_IDS[rank]:
            raise ValueError('manifest reference image differs from pin')
    # Two pilot boots, then two fresh boots for each of six turns.
    orders = [('pilot', ('plain', 'lane'))] + [
        (f't{turn:02d}', ('plain', 'lane') if turn % 2 else ('lane', 'plain'))
        for turn in range(1, 7)]
    staged = []
    for turn, arms in orders:
        for arm in arms:
            boot = f'{turn}-{arm}'
            manifest = copy.deepcopy(base)
            record = copy.deepcopy(base['arms']['D'])
            record['engine_sha'] = REVISION
            record['description'] = f'phase2 {boot}: normal G, explicit compute lane {int(arm == "lane")}'
            record.pop('alias_of', None)
            manifest.update(active_arms=['D'], arms={'D': record})
            rank_meta = []
            rank_settings = []
            for rank in (0, 1):
                root = Path(f'<rank{rank}-work>/stdbench/phase2') / prefix
                env = {k: v for k, v in refs[rank].items()
                       if not any(word in k for word in STRIP)
                       and not k.startswith(('GUARD', 'SPARK'))}
                env.update(IMAGE=IMAGE, ZIG_COMMIT=REVISION,
                           CONTAINER_PREFIX=f'{prefix}-{boot}',
                           STATE_DIR=str(root / 'state' / boot),
                           CACHE_DIR=str(root / 'cache' / boot))
                env.update({f'TF_DSV41_{key}': '0' for key in ZERO_FLAGS})
                env['TF_DSV41_COMPUTE_LANE'] = str(int(arm == 'lane'))
                env['TF_DSV41_RUNNER_FORMS'] = 'fence,repl,inline'
                # Make the frozen native adapter's two default exports explicit.
                env.setdefault('NCCL_CUMEM_ENABLE', '0')
                env.setdefault('NCCL_CUMEM_HOST_ENABLE', '0')
                reject_banned(env)
                relative = Path('configs') / f'{boot}-r{rank}.env'
                cfg = root / relative
                record['ranks'][rank].update(config=str(cfg), image_id=IMAGE_IDS[rank])
                settings = {k: v for k, v in env.items()
                            if k.startswith(('TF_', 'GLM53_TF'))}
                rank_settings.append(settings)
                record['ranks'][rank]['settings'] = settings
                stage = Path(f'rank{rank}') / relative
                staged.append((stage, env_text(env)))
                rank_meta.append({
                    'rank': rank, 'root': str(root), 'config': str(cfg),
                    'stage_config': str(stage), 'container_name': f'{prefix}-{boot}-r{rank}',
                    'state_dir': env['STATE_DIR'], 'cache_dir': env['CACHE_DIR'],
                    'table_source': str(Path(record['ranks'][rank]['recipe_dir']) / 'config/pfdense-table.json'),
                    'effective_env': filtered(env), 'guard_source': '/etc/spark/thermal.env',
                })
            critical = {f'TF_DSV41_{key}' for key in ZERO_FLAGS}
            critical.update(('TF_DSV41_COMPUTE_LANE', 'TF_DSV41_RUNNER_FORMS'))
            if any(rank_settings[0].get(key) != rank_settings[1].get(key) for key in critical):
                raise ValueError('critical feature settings differ across ranks')
            # Original rank.py checks shared settings. The wrapper checks each
            # rank's full sealed settings, allowing legitimate transport differences.
            record['settings'] = {key: value for key, value in rank_settings[0].items()
                                  if rank_settings[1].get(key) == value}
            record['native_settings'] = {}
            manifest['phase2'] = {'schema': 1, 'prefix': prefix, 'boot': boot,
                                  'arm': arm, 'lane': int(arm == 'lane'), 'ranks': rank_meta}
            reject_banned(manifest['arms'])
            staged.append((Path('boots') / boot / 'manifest.json', json.dumps(manifest, indent=2, sort_keys=True) + '\n'))
    out.mkdir(parents=True, exist_ok=False)
    for relative, content in staged:
        dump_new(out / relative, content)
    inventory = {str(path): hashlib.sha256(content.encode()).hexdigest() for path, content in staged}
    boots = [{'boot': f'{turn}-{arm}', 'arm': arm, 'lane': int(arm == 'lane'),
              'manifest': f'boots/{turn}-{arm}/manifest.json'}
             for turn, arms in orders for arm in arms]
    index = {'schema': 1, 'prefix': prefix, 'image': IMAGE, 'revision': REVISION,
             'image_ids': IMAGE_IDS, 'boots': boots, 'files': inventory,
             'guard_source': '/etc/spark/thermal.env',
             'references': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [references / 'base-arms.json', references / 'base-r0.env', references / 'base-r1.env']}}
    json_new(out / 'manifest.json', index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, default=Path(__file__).parent / 'references')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prefix', default=PREFIX)
    args = parser.parse_args()
    result = generate(args.references, args.out, args.prefix)
    print(json.dumps({'output': str(args.out), 'boots': len(result['boots']), 'prefix': result['prefix']}))


if __name__ == '__main__':
    main()
