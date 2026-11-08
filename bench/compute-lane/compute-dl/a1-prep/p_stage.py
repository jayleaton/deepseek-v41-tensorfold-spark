#!/usr/bin/env python3
"""CPU-only P preparation. No execution, transport, controller or approval API.

Parent runs this module only in bounded Spark CPU scopes. Import is read-only;
stage() writes exclusively to a new a1-prep descendant after all validation.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREFIX = 'stdbench-compute-a1-prep-20261105-v1'
OLD_PREFIX = 'stdbench-compute-dl-20261105-v1'
BASE_SPEC = 'c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141'
REVISION = '3a429d6f122d9eb2900c43e622933d311e4be44f'
IMAGE_IDS = (
    'sha256:5be7d7d40f267590b9e7b41410f48d40048af24e004203629a347d3d7c16a8cd',
    'sha256:80beaba732d1ed9439e012b007d8dbe19ec2f8962f50fbac29a18c8cd6fda306',
)
SETTINGS = {'chat_template_kwargs': {'enable_thinking': True}, 'deadline_s': 270,
            'max_tokens': 8192, 'model': 'DeepSeek-V4.1-Flash-TF',
            'reasoning_effort': 'high', 'return_token_ids': True, 'stream': True,
            'stream_options': {'include_usage': True}, 'temperature': 1.0,
            'top_k': 20, 'top_p': 0.95}
PINS = {
    'T5-COMPUTE-DL-SPEC.md': BASE_SPEC,
    'dataset-v1/tasks-public.jsonl': '3e1e07d7a573eb194c0568afac2ef2112011b2b2cada9012fbbce4e6f47f7548',
    'runtime_stage.py': 'fcb48012cae6f0ca16ca14581894527872af8d493a1c64243891727fac744942',
    'protocol.py': '0b8bb2e7db728a593f8450843711008dd39925897a4f4a4b7fb8a3ec5e50dd1f',
    'remaining-collection.json': '6fdf99b3dbfd5d77e9ae3c04c3a23c85c18e78a0762933561aa34c1fbd57da43',
    f'{OLD_PREFIX}/frozen-v1/manifest.json': '8aee124a6c4fcb68208d8bd710bd8f2a9413c153359240c43bc99ec670db8f2f',
    f'{OLD_PREFIX}/frozen-v1/boots/t01-L.json': 'f044822f55fb1e80c6509514c3d271fa7c5936ecec8933b60ad621939e4e98e7',
    f'{OLD_PREFIX}/frozen-v1/boots/t02-L.json': '1259822c1ffca99073bf2e26fce9f5ad86206eb7fc412dc3f08157c1df8264a2',
    '../decl-control/generate_envs3.py': '01ba062c5eccee41033dab42697eb5e5ff06392c2bf2c9a462ea0db2e6ec0076',
    '../decl-control/canonical-python-tools.json': '1d112d68e2df0a69c1a6222daedef99f8afb8323e872d5c0849563a83812361f',
    '../phase2/references/base-arms.json': '9bbacf49fd8bf7917321409c23c8c7f68836d4d28fe456181b7957daac0f5d4f',
    '../phase2/references/base-r0.env': 'c0301508e99561c4e457d5c40e26eb23667a8243b2602210fa23018afa96c61b',
    '../phase2/references/base-r1.env': '96cbacd85464af61073619b4e96c83180bb902112b1ef8062e543cf6c0085edc',
}
STATUS = {'staging_only': True, 'collection_authorized': False,
          'A1sha': None, 'specpending': True, 'status': 'SPEC_PENDING',
          'order_status': 'ORDER_NOT_FROZEN'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def verify_blobs(blobs):
    """Exact pin set; also used by fake drift controls without writing sources."""
    if set(blobs) != set(PINS):
        raise ValueError('source pin coverage differs')
    for name, expected in PINS.items():
        if digest(blobs[name]) != expected:
            raise ValueError('source/data drift: ' + name)


def sources(root=ROOT):
    blobs = {name: (Path(root) / name).read_bytes() for name in PINS}
    verify_blobs(blobs)
    return blobs


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helpers(root):
    # Verify bytes before importing. Refuse a conflicting preloaded helper.
    path = (root / '../decl-control/generate_envs3.py').resolve()
    existing = sys.modules.get('generate_envs3')
    if existing is not None and Path(existing.__file__).resolve() != path:
        raise ValueError('foreign generate_envs3 module')
    helper = load_module('generate_envs3', path)
    sys.modules['generate_envs3'] = helper
    runtime = load_module('_p_original_runtime_stage', root / 'runtime_stage.py')
    if helper.REVISION != REVISION or tuple(helper.IMAGE_IDS) != IMAGE_IDS:
        raise ValueError('helper image/source identity differs')
    return helper, runtime


def node_root(rank, prefix):
    return Path(f'<rank{rank}-work>/stdbench/compute-dl') / 'a1-prep' / prefix


def check_prefix(prefix):
    if not re.fullmatch(r'stdbench-compute-a1-prep-[A-Za-z0-9][A-Za-z0-9_.-]*', prefix):
        raise ValueError('fresh P namespace required')


def off_values(helper):
    values = {f'TF_DSV41_{key}': '0' for key in helper.ZERO_FLAGS}
    values['TF_DSV41_COMPUTE_LANE'] = '0'
    return values


def templates(tasks, originals, tools, runtime):
    public = {t['id']: t for t in tasks}
    rows = []
    for turn in ('t01', 't02'):
        old = originals[turn]
        if old['boot'] != turn + '-L' or old['arm'] != 'L' or len(old['requests']) != 120:
            raise ValueError('original L boot identity/coverage differs')
        expected_order = [(t['id'], seed) for t, seed in runtime.ordered_pairs(tasks, turn)]
        if [(r['id'], r['seed']) for r in old['requests']] != expected_order:
            raise ValueError('original pair order differs')
        for original in old['requests']:
            task = public[original['id']]
            seed = original['seed']
            key = ['compute_dl', task['id'], seed]
            body = original['body_template']
            if (original['task'] != task or original['key'] != key
                    or original['task_sha256'] != digest(runtime.canonical(task).encode())
                    or original['arm'] != 'lane'
                    or original['execution_policy'] != {'TF_DSV41_COMPUTE_LANE': '1'}):
                raise ValueError('matched public L identity differs')
            if (body.get('tools') != tools or body.get('seed') != seed
                    or body.get('messages') != [{'content': task['prompt'], 'role': 'user'}]
                    or {k: v for k, v in body.items() if k not in ('tools', 'seed', 'messages')} != SETTINGS
                    or wire(body) != original['wire']
                    or digest(wire(body).encode()) != original['wire_sha256']
                    or original['body_sha256'] != original['wire_sha256']):
                raise ValueError('matched L wire/settings differ')
            plain = copy.deepcopy(body)
            del plain['tools']
            sent = wire(plain)
            rows.append({'arm': 'plain', 'suite': 'compute_dl', 'id': task['id'],
                         'seed': seed, 'key': key, 'task': copy.deepcopy(task),
                         'task_sha256': original['task_sha256'], 'body_template': plain,
                         'wire': sent, 'body_sha256': digest(sent.encode()),
                         'wire_sha256': digest(sent.encode()),
                         'execution_policy': {'TF_DSV41_COMPUTE_LANE': '0'},
                         'matched_L': {'boot': turn + '-L', 'position': len(rows) % 120,
                                       'wire_sha256': original['wire_sha256']}})
    if len(rows) != 240 or {tuple(r['key']) for r in rows} != {
            ('compute_dl', t['id'], s) for t in tasks for s in (101, 102)}:
        raise ValueError('P coverage differs')
    return rows


def plan(root=ROOT, prefix=PREFIX):
    """Return (index, relative-path -> bytes), with no writes or launches."""
    root = Path(root).resolve()
    check_prefix(prefix)
    blobs = sources(root)
    helper, runtime = helpers(root)
    tasks = runtime.read_tasks(root / 'dataset-v1/tasks-public.jsonl')
    originals = {turn: runtime.load_text(blobs[f'{OLD_PREFIX}/frozen-v1/boots/{turn}-L.json'].decode())
                 for turn in ('t01', 't02')}
    tools = runtime.load_text(blobs['../decl-control/canonical-python-tools.json'].decode())
    rows = templates(tasks, originals, tools, runtime)
    remaining = runtime.load_text(blobs['remaining-collection.json'].decode())
    if (remaining['collection_authorized'] is not False or remaining['count'] != 138
            or remaining['spec_sha256'] != BASE_SPEC):
        raise ValueError('original remaining inventory differs')
    payloads = {'requests-P.json': encoded(dict(STATUS, boot='p-all', arm='P', requests=rows))}
    base = runtime.load_text(blobs['../phase2/references/base-arms.json'].decode())
    ranks = []
    off = off_values(helper)
    for rank in (0, 1):
        reference = root / f'../phase2/references/base-r{rank}.env'
        ref = helper.read_reference(reference)
        if (ref['IMAGE'] not in (helper.IMAGE, IMAGE_IDS[rank]) or ref['ZIG_COMMIT'] != REVISION
                or base['arms']['D']['ranks'][rank]['image_id'] != IMAGE_IDS[rank]):
            raise ValueError('reference image/source pin differs')
        env = {k: v for k, v in ref.items() if not any(word in k for word in helper.STRIP)
               and not k.startswith(('GUARD', 'SPARK'))}
        target = node_root(rank, prefix)
        env.update(IMAGE=helper.IMAGE, ZIG_COMMIT=REVISION, CONTAINER_PREFIX=prefix + '-p-all',
                   STATE_DIR=str(target / 'state/p-all'), CACHE_DIR=str(target / 'cache/p-all'))
        env.update(off)
        env['TF_DSV41_RUNNER_FORMS'] = 'fence,repl,inline'
        env.setdefault('NCCL_CUMEM_ENABLE', '0')
        env.setdefault('NCCL_CUMEM_HOST_ENABLE', '0')
        helper.reject_banned(env)
        if 'IMR_PUBLISH' in str(env):
            raise ValueError('publication flag forbidden')
        relative = f'rank{rank}/configs/p-all-r{rank}.env'
        payloads[relative] = helper.env_text(env).encode()
        record = dict(STATUS, rank=rank, boot='p-all', arm='P', lane=0,
                      root=str(target), config=str(target / f'configs/p-all-r{rank}.env'),
                      stage_config=relative, image=helper.IMAGE, image_id=IMAGE_IDS[rank],
                      engine_sha=REVISION, effective_env=env, expected_off=off,
                      guard_source='/etc/spark/thermal.env',
                      container_name=prefix + f'-p-all-r{rank}',
                      table_source=str(Path(base['arms']['D']['ranks'][rank]['recipe_dir']) / 'config/pfdense-table.json'),
                      source_pins=PINS.copy())
        payloads[f'rank{rank}/effective-env.json'] = encoded(record)
        ranks.append(record)
    index = dict(STATUS, schema=1, prefix=prefix, candidate_boot='p-all',
                 base_spec_sha256=BASE_SPEC, base_prefix=OLD_PREFIX, requests=240,
                 tasks=120, seeds=[101, 102], candidate_order=['t01-L:120', 't02-L:120'],
                 image_ids=list(IMAGE_IDS), engine_sha=REVISION, ranks=ranks,
                 dispatch_reuse={'proof_sha256': '77e43ebc13afdc16a8304836869953f8ceadfc8c6775b8f551aceb3b15af0436',
                                 'status': 'PENDING_PARENT_SOURCE_IMAGE_IDENTITY_RECHECK',
                                 'new_dispatch_proof_claimed': False},
                 source_pins=PINS.copy(),
                 pending=['JJ A1 SHA steer', 'T5 order overrides', 'parent native prompt/clipping gates',
                          'parent two-rank effective config and source/dispatch checks',
                          'parent bounded CPU/client/turn-preflight checks'],
                 files={name: digest(data) for name, data in payloads.items()})
    payloads['P-stage-index.json'] = encoded(index)
    return index, payloads


def stage(out=None, *, root=ROOT, prefix=PREFIX):
    """Create a fresh a1-prep descendant only. Returns staging index, never GO."""
    parent = (Path(root).resolve() / 'a1-prep').resolve()
    destination = Path(out).absolute() if out is not None else parent / 'staged-P'
    if ('..' in destination.parts or destination == parent
            or parent not in destination.parents
            or any(p.is_symlink() for p in (destination, *destination.parents))):
        raise ValueError('output must be a fresh non-symlink a1-prep descendant')
    index, payloads = plan(root, prefix)
    # Check pins again immediately before writing; never overwrite partial output.
    sources(root)
    destination.mkdir(exist_ok=False)
    for name, data in payloads.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
    return index
