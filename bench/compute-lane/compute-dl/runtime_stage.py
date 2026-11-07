#!/usr/bin/env python3
"""CPU-only request/config staging from an EXISTING public dataset.

stage(rootout=None, tasks_file=None) creates a new portable staging directory.
It never generates tasks, invokes programs, renders tokens, creates GO, initializes
the budget ledger, or authorizes collection. Run only in the parent's CPU guard.
materialize(stage_root, rank, destination=None) copies checked stage artifacts to
a NEW rank-local prefix directory, without SSH or changing source files.

Layout: frozen-v1/ (controller request manifest), rank{0,1}/configs/,
rank{0,1}/boots/ (rank-local campaign bindings), prompt-inputs.json (242 records),
stage-campaign.json, source-hashes.json, materialization.json and stage-manifest.json.
rank0-host retains BOTH rank manifest trees: the controller reads the rank0-host shadow
of rank1-host's manifest when checking its hash. Each rank's configs are flattened
from rankN/configs into its actual prefix/configs. Campaign paths already name
the eventual Spark paths; the parent must finish/seal campaign-v1.json separately.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / 'decl-control'
sys.path.append(str(LEGACY))
from generate_envs3 import (IMAGE, IMAGE_IDS, REVISION, STRIP, ZERO_FLAGS,
                            env_text, filtered, read_reference, reject_banned)

PREFIX = 'stdbench-compute-dl-20261105-v1'
SPEC_SHA256 = 'c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141'
TOKENIZER_SHA256 = 'c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
FAMILIES = ('integer', 'modular', 'division', 'dp', 'graph', 'aggregation')
SEEDS = (101, 102)
ORDERS = (('t01', ('D', 'L')), ('t02', ('L', 'D')))
CLIENT_ARM = {'D': 'declared_off', 'L': 'lane'}
PUBLIC_KEYS = {'id', 'family', 'suite', 'slot', 'attempt', 'generator_label', 'inputs', 'prompt'}
INPUT_KEYS = {'integer': {'a', 'b', 'c'}, 'modular': {'a', 'e', 'm'},
              'division': {'a', 'b'}, 'dp': {'rows', 'cols', 'matrix'},
              'graph': {'n', 'edges'}, 'aggregation': {'count', 'records'}}
SETTINGS = {'chat_template_kwargs': {'enable_thinking': True}, 'deadline_s': 270,
            'max_tokens': 8192, 'model': 'DeepSeek-V4.1-Flash-TF',
            'reasoning_effort': 'high', 'return_token_ids': True, 'stream': True,
            'stream_options': {'include_usage': True}, 'temperature': 1.0,
            'top_k': 20, 'top_p': 0.95}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def wire(body):
    return json.dumps(body, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def json_text(value):
    # Never sort request/tool keys when writing a document that will be reloaded.
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError('nonfinite JSON constant: ' + value)


def load_text(text):
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid_constant)


def spark_root(rank):
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError('rank must be 0 or 1')
    return Path(f'<rank{rank}-work>/stdbench/compute-dl')


def prefix_root(rank):
    return spark_root(rank) / PREFIX


def new_directory(path):
    path = Path(path).absolute()
    if '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('unsafe output directory')
    path.mkdir(exist_ok=False)  # Existing/partial namespaces must never be reused.
    return path


def write_new(root, relative, data):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('unsafe stage-relative path')
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as output:
        output.write(data)


def read_tasks(tasks_file):
    tasks = [load_text(line) for line in Path(tasks_file).read_text().splitlines() if line.strip()]
    if len(tasks) != 120:
        raise ValueError('exactly 120 existing public tasks required')
    seen = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != PUBLIC_KEYS:
            raise ValueError('public task schema differs; private answers/extra fields forbidden')
        family, slot = task['family'], task['slot']
        if family not in FAMILIES or type(slot) is not int or not 0 <= slot < 20:
            raise ValueError('invalid family/slot')
        if (family, slot) in seen:
            raise ValueError('duplicate family/slot')
        seen.add((family, slot))
        if task['id'] != f'compute-dl-v1-{family}-{slot + 1:03d}' or task['suite'] != 'compute_dl':
            raise ValueError('task ID/suite differs from frozen protocol')
        attempt = task['attempt']
        if type(attempt) is not int or not 0 <= attempt < 1000:
            raise ValueError('invalid collision attempt')
        if task['generator_label'] != f'compute-dl-v1|20261105|{family}|{slot}|{attempt}':
            raise ValueError('generator label differs')
        if not isinstance(task['inputs'], dict) or set(task['inputs']) != INPUT_KEYS[family]:
            raise ValueError('public typed-input schema differs')
        if not isinstance(task['prompt'], str) or not task['prompt'].strip():
            raise ValueError('public prompt missing')
    if seen != {(family, slot) for family in FAMILIES for slot in range(20)}:
        raise ValueError('incomplete family/slot grid')
    return tasks


def ordered_pairs(tasks, turn):
    if turn not in ('t01', 't02'):
        raise ValueError('unknown turn')
    low = 0 if turn == 't01' else 10
    result = []
    for family in FAMILIES:
        units = [(t, seed) for t in tasks if t['family'] == family and low <= t['slot'] < low + 10
                 for seed in SEEDS]
        units.sort(key=lambda pair: (pair[0]['id'], pair[1]))
        random.Random(20261105).shuffle(units)  # Fresh stream for EACH family/turn.
        result.extend(units)
    if len(result) != 120:
        raise ValueError('turn must have 120 task/seed pairs')
    return result


def request_template(task, seed, arm, original_body, tools):
    body = copy.deepcopy(original_body)
    message = copy.deepcopy(body['messages'][0])
    message['content'] = task['prompt']
    message['role'] = 'user'
    body['messages'] = [message]
    body['seed'] = seed
    body['tools'] = copy.deepcopy(tools)
    sent = wire(body)
    return {'suite': 'compute_dl', 'id': task['id'], 'seed': seed,
            'key': ['compute_dl', task['id'], seed], 'arm': CLIENT_ARM[arm],
            'task': copy.deepcopy(task), 'task_sha256': digest(canonical(task).encode()),
            'body_template': body, 'wire': sent, 'body_sha256': digest(sent.encode()),
            'wire_sha256': digest(sent.encode()),
            'execution_policy': {'TF_DSV41_COMPUTE_LANE': str(int(arm == 'L'))}}


def diagnostic_templates(originals, arm):
    result = copy.deepcopy(originals)
    for row in result:
        row['arm'] = CLIENT_ARM[arm]
        row['execution_policy'] = {'TF_DSV41_COMPUTE_LANE': str(int(arm == 'L'))}
        # The fixed diagnostic answers are public, but keep task objects public-only.
        row['task'].pop('answer', None)
        row['task_sha256'] = digest(canonical(row['task']).encode())
    return result


def rank_artifacts(base, references, boot, arm):
    manifest = copy.deepcopy(base)
    record = copy.deepcopy(base['arms']['D'])
    record.update(engine_sha=REVISION, description=f'compute D/L {boot}; lane={int(arm == "L")}')
    record.pop('alias_of', None)
    manifest.update(active_arms=['D'], arms={'D': record})
    manifest.pop('decl_control_a1', None)
    metas, settings, configs = [], [], {}
    off = {f'TF_DSV41_{key}': '0' for key in ZERO_FLAGS}
    for rank in (0, 1):
        root = prefix_root(rank)
        env = {k: v for k, v in references[rank].items()
               if not any(word in k for word in STRIP)
               and not k.startswith(('GUARD', 'SPARK'))}
        env.update(IMAGE=IMAGE, ZIG_COMMIT=REVISION, CONTAINER_PREFIX=f'{PREFIX}-{boot}',
                   STATE_DIR=str(root / 'state' / boot), CACHE_DIR=str(root / 'cache' / boot))
        env.update(off)
        env.update(TF_DSV41_COMPUTE_LANE=str(int(arm == 'L')), TF_DSV41_RUNNER_FORMS='fence,repl,inline')
        env.setdefault('NCCL_CUMEM_ENABLE', '0')
        env.setdefault('NCCL_CUMEM_HOST_ENABLE', '0')
        reject_banned(env)
        if 'IMR_PUBLISH' in str(env):
            raise ValueError('publication flag forbidden')
        cfg = root / 'configs' / f'{boot}-r{rank}.env'
        stage_cfg = f'rank{rank}/configs/{boot}-r{rank}.env'
        configs[stage_cfg] = env_text(env).encode()
        values = {k: v for k, v in env.items() if k.startswith(('TF_', 'GLM53_TF'))}
        settings.append(values)
        record['ranks'][rank].update(config=str(cfg), image_id=IMAGE_IDS[rank], settings=values)
        metas.append({'rank': rank, 'root': str(root), 'config': str(cfg), 'stage_config': stage_cfg,
                      'container_name': f'{PREFIX}-{boot}-r{rank}', 'state_dir': env['STATE_DIR'],
                      'cache_dir': env['CACHE_DIR'], 'effective_env': filtered(env),
                      'table_source': str(Path(record['ranks'][rank]['recipe_dir']) / 'config/pfdense-table.json'),
                      'expected_off': dict(off), 'guard_source': '/etc/spark/thermal.env'})
    record['settings'] = {k: v for k, v in settings[0].items() if settings[1].get(k) == v}
    record['native_settings'] = {}
    manifest['phase2'] = {'schema': 1, 'prefix': PREFIX, 'boot': boot, 'arm': arm,
                          'lane': int(arm == 'L'), 'runner_forms': 'fence,repl,inline',
                          'guard_source': '/etc/spark/thermal.env', 'ranks': metas}
    for rank in (0, 1):
        local = copy.deepcopy(manifest)
        local['compute_dl'] = {'status': 'STAGED_CPU_GATES_PENDING',
                               'campaign': str(prefix_root(rank) / 'campaign-v1.json'),
                               'spec_sha256': SPEC_SHA256, 'boot_min_gib': 114}
        configs[f'rank{rank}/boots/{boot}/manifest.json'] = json_text(local).encode()
    return configs


def stage(rootout=None, tasks_file=None, *, references=None, spec_file=None, legacy=None):
    """Write a NEW CPU stage from 120 already generated public records; return index.

    No gate is promoted and no missing seal is manufactured. The parent owns
    evidence qualification and campaign-v1.json/GO creation after these artifacts.
    """
    legacy = Path(legacy) if legacy is not None else LEGACY
    references = Path(references) if references is not None else HERE.parent / 'phase2/references'
    spec_file = Path(spec_file) if spec_file is not None else HERE / 'T5-COMPUTE-DL-SPEC.md'
    tasks_file = Path(tasks_file) if tasks_file is not None else HERE / 'dataset-v1/tasks-public.jsonl'
    rootout = Path(rootout) if rootout is not None else HERE / 'runtime-stage-v1'
    if sha(spec_file) != SPEC_SHA256:
        raise ValueError('actual frozen spec SHA differs')
    tasks = read_tasks(tasks_file)
    tokenizer = references / 'tokenizer.json'
    if sha(tokenizer) != TOKENIZER_SHA256:
        raise ValueError('A1 tokenizer pin differs')
    frozen = legacy / 'frozen-a1-v1'
    old_index = load_text((frozen / 'manifest.json').read_text())
    entries = [e for e in old_index['boots'] if e['boot'] == 't01-lane' and e['arm'] == 'lane']
    if len(entries) != 1:
        raise ValueError('exact A1 lane diagnostic source missing')
    old_path = frozen / entries[0]['file']
    old_path.resolve().relative_to(frozen.resolve())
    if sha(old_path) != entries[0]['sha256']:
        raise ValueError('frozen A1 boot drift')
    old = load_text(old_path.read_text())
    tools_path = legacy / 'canonical-python-tools.json'
    tools = load_text(tools_path.read_text())
    original_body = old['requests'][0]['body_template']
    if {k: v for k, v in original_body.items() if k not in ('messages', 'seed', 'tools')} != SETTINGS:
        raise ValueError('A1 request settings differ from compute D/L spec')
    if original_body['tools'] != tools or list(original_body['messages'][0]) != ['content', 'role']:
        raise ValueError('A1 declaration/message ordering differs')
    diags = old['diagnostics']
    if [(r['key'], str(r['reference'])) for r in diags] != [
            (['diagnostic', 'python-17x19', 101], '323'),
            (['diagnostic', 'python-sum11', 101], '55')]:
        raise ValueError('frozen 323/55 diagnostic identity differs')
    for row in diags:
        body = row['body_template']
        if body['tools'] != tools or wire(body) != row['wire'] or digest(row['wire'].encode()) != row['wire_sha256']:
            raise ValueError('A1 diagnostic body/declaration drift')
        if {k: v for k, v in body.items() if k not in ('messages', 'seed', 'tools')} != SETTINGS:
            raise ValueError('A1 diagnostic settings differ')
    base = load_text((references / 'base-arms.json').read_text())
    refs = [read_reference(references / f'base-r{rank}.env') for rank in (0, 1)]
    for rank, env in enumerate(refs):
        if env['IMAGE'] not in (IMAGE, IMAGE_IDS[rank]) or env['ZIG_COMMIT'] != REVISION:
            raise ValueError('base rank image/source pin differs')
        if base['arms']['D']['ranks'][rank]['image_id'] != IMAGE_IDS[rank]:
            raise ValueError('base manifest image pin differs')
    payloads, boots, turns, prompt_inputs = {}, [], [], {}
    for turn, arms in ORDERS:
        pairs = ordered_pairs(tasks, turn)
        names = []
        by_arm = {}
        for arm in arms:
            name = f'{turn}-{arm}'
            rows = [request_template(t, seed, arm, original_body, tools) for t, seed in pairs]
            by_arm[arm] = rows
            diagnostics = diagnostic_templates(diags, arm)
            relative = f'boots/{name}.json'
            data = json_text({'boot': name, 'arm': arm, 'requests': rows, 'diagnostics': diagnostics}).encode()
            payloads['frozen-v1/' + relative] = data
            boots.append({'boot': name, 'arm': arm, 'file': relative, 'sha256': digest(data)})
            names.append(name)
            payloads.update(rank_artifacts(base, refs, name, arm))
            for row in rows + diagnostics:
                record = {'key': row['key'], 'messages': row['body_template']['messages']}
                key = tuple(row['key'])
                if key in prompt_inputs and prompt_inputs[key] != record:
                    raise ValueError('D/L renderer messages differ')
                prompt_inputs[key] = record
        if any(d['key'] != l['key'] or d['wire'] != l['wire'] for d, l in zip(by_arm['D'], by_arm['L'])):
            raise ValueError('D/L wire/order mismatch')
        turns.append({'name': turn, 'boots': names})
    if len(prompt_inputs) != 242:
        raise ValueError('renderer requires exactly 240 task pairs plus two diagnostics')
    index = {'schema': 1, 'prefix': PREFIX, 'spec_sha256': SPEC_SHA256,
             'status': 'STAGED_CPU_GATES_PENDING', 'collection_qualified': False,
             'turns': turns, 'boots': boots, 'fresh_replies': 480, 'diagnostic_replies': 8,
             'tasks_public_sha256': sha(tasks_file), 'seeds': list(SEEDS), 'families': list(FAMILIES)}
    payloads['frozen-v1/manifest.json'] = json_text(index).encode()
    renderer_records = [r for r in prompt_inputs.values() if r['key'][0] != 'diagnostic']
    renderer_records += [r for r in prompt_inputs.values() if r['key'][0] == 'diagnostic']
    payloads['prompt-inputs.json'] = json_text(renderer_records).encode()
    root = prefix_root(0)
    campaign = {'prefix': PREFIX, 'spec': str(spark_root(0) / 'T5-COMPUTE-DL-SPEC.md'),
                'spec_sha256': SPEC_SHA256, 'arms': ['D', 'L'], 'model_alias': SETTINGS['model'],
                'boot_min_gib': 114, 'freeze': str(root / 'frozen-v1'),
                'freeze_sha256': digest(payloads['frozen-v1/manifest.json']),
                'layout': {'turns': turns, 'boots': boots},
                'rank_manifests': {b['boot']: {str(r): str(prefix_root(r) / f'rank{r}/boots' / b['boot'] / 'manifest.json')
                                               for r in (0, 1)} for b in boots},
                'rank_manifest_sha256': {b['boot']: {str(r): digest(payloads[f'rank{r}/boots/{b["boot"]}/manifest.json'])
                                                    for r in (0, 1)} for b in boots},
                'cpu_receipt': str(root / 'cpu-seal/verification.json'), 'cpu_receipt_sha256': None,
                'cpu_prompt_arrays': str(root / 'prompt-arrays.jsonl'), 'cpu_prompt_arrays_sha256': None,
                'code_sha256': {}, 'go_file': str(root / 'GO'),
                'tokenizer': str(spark_root(0).parent / 'phase2/references/tokenizer.json'),
                'tokenizer_sha256': TOKENIZER_SHA256, 'image_ids': list(IMAGE_IDS),
                'timeouts': {'client_wait_s': 360, 'gpu_parent_s': 43200, 'remote_control_s': 43200, 'turn_s': 43200},
                'transport_contract': {'max_tokens': 8192, 'deadline_s': 270},
                'diagnostic_tensorfold': {'enable_thinking': True, 'reasoning_effort': 'high'},
                'g_restore_step': {'format': 'g-restore-v1', 'events': ['SERVING', 'MODELS200', 'POPPED']},
                'budget': {'limit_seconds': 18000, 'admission_seconds': 15900},
                'budget_ledger': str(root / 'budget.json'),
                'status': 'STAGED_CPU_GATES_PENDING', 'collection_qualified': False,
                'pending': ['native 242-record prompt render and equality proof', 'independent references/disjointness seals',
                            'source/image/rank CPU evidence and source-identical safety proof',
                            'final campaign CPU receipt/source seals and external GO']}
    from protocol import require_staging_ready
    require_staging_ready(campaign)  # Read-only parent gate, never collection approval.
    payloads['stage-campaign.json'] = json_text(campaign).encode()
    source_files = {'spec': spec_file, 'tasks_public': tasks_file, 'a1/frozen-manifest': frozen / 'manifest.json',
                    'a1/lane-boot': old_path, 'a1/canonical-python-tools': tools_path,
                    'a1/generate_envs3.py': LEGACY / 'generate_envs3.py', 'runtime_stage.py': Path(__file__),
                    'references/base-arms.json': references / 'base-arms.json',
                    'references/base-r0.env': references / 'base-r0.env',
                    'references/base-r1.env': references / 'base-r1.env', 'references/tokenizer.json': tokenizer,
                    'a1/prompt-gate': legacy / 'prompt-gate'}
    sources = {name: {'path': str(path.resolve()), 'sha256': sha(path)} for name, path in source_files.items()}
    payloads['source-hashes.json'] = json_text({'schema': 1, 'sources': sources,
        'source_paths_are_provenance_only': True, 'runtime_paths': 'stage-campaign.json',
        'task_sha256_contract': 'SHA256 of UTF-8 canonical public task JSON; sorted keys, compact separators, ensure_ascii=False',
        'body_sha256_contract': 'SHA256 of exact UTF-8 wire bytes; preserve insertion order'}).encode()
    plan = {str(rank): materialization_plan(payloads, rank) for rank in (0, 1)}
    payloads['materialization.json'] = json_text({'schema': 1, 'prefix': PREFIX, 'ranks': plan,
        'campaign_note': 'stage-campaign.json is preliminary; parent binds rank-local campaign-v1.json after CPU seals'}).encode()
    inventory = {name: digest(data) for name, data in payloads.items()}
    result = dict(index, files=inventory, materialization='materialization.json',
                  source_hashes='source-hashes.json', renderer_input='prompt-inputs.json',
                  preliminary_campaign='stage-campaign.json')
    rootout = new_directory(rootout)
    for name, data in payloads.items():
        write_new(rootout, name, data)
    write_new(rootout, 'stage-manifest.json', json_text(result).encode())
    return result


def materialization_plan(files, rank):
    """Portable stage-relative input -> rank-prefix-relative output mapping."""
    spark_root(rank)
    result = {}
    for name in files:
        if name.startswith(f'rank{rank}/configs/'):
            result[name] = name.split('/', 1)[1]
        elif name.startswith('rank') and '/boots/' in name:
            if rank == 0 or name.startswith(f'rank{rank}/'):
                result[name] = name  # rank0-host also needs rank1-host's manifest shadow.
        elif name.startswith('frozen-v1/') or name in ('prompt-inputs.json', 'stage-campaign.json', 'source-hashes.json'):
            result[name] = name
    return result


def materialize(stage_root, rank, destination=None):
    """Local-only exclusive materialization; never writes campaign-v1.json or GO.

    An explicit destination allows a parent CPU fake to use a temporary directory.
    Production default is precisely the rank's Spark prefix. Stage inventory hashes
    are checked before any output directory is created; no overwrite/recovery mode.
    """
    stage_root = Path(stage_root)
    manifest = load_text((stage_root / 'stage-manifest.json').read_text())
    if manifest['prefix'] != PREFIX or manifest['spec_sha256'] != SPEC_SHA256:
        raise ValueError('stage identity differs')
    plan = materialization_plan(manifest['files'], rank)
    contents = {}
    for source, target in plan.items():
        path = stage_root / source
        path.resolve().relative_to(stage_root.resolve())
        data = path.read_bytes()
        if digest(data) != manifest['files'][source]:
            raise ValueError('stage artifact drift: ' + source)
        contents[target] = data
    destination = new_directory(destination if destination is not None else prefix_root(rank))
    for name, data in contents.items():
        write_new(destination, name, data)
    receipt = {'schema': 1, 'rank': rank, 'prefix': PREFIX, 'spec_sha256': SPEC_SHA256,
               'stage_manifest_sha256': sha(stage_root / 'stage-manifest.json'),
               'files': {name: digest(data) for name, data in contents.items()},
               'collection_qualified': False}
    write_new(destination, 'materialized.json', json_text(receipt).encode())
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('stage', help='stage EXISTING public tasks; no task generation or execution')
    build.add_argument('--out', type=Path)
    build.add_argument('--tasks-file', type=Path)
    build.add_argument('--references', type=Path)
    build.add_argument('--spec-file', type=Path)
    build.add_argument('--legacy', type=Path)
    deploy = commands.add_parser('materialize', help='copy checked artifacts locally into a NEW prefix only')
    deploy.add_argument('stage_root', type=Path)
    deploy.add_argument('rank', type=int, choices=(0, 1))
    deploy.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.command == 'stage':
        result = stage(args.out, args.tasks_file, references=args.references, spec_file=args.spec_file, legacy=args.legacy)
    else:
        result = materialize(args.stage_root, args.rank, args.destination)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
