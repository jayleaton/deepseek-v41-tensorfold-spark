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
LEGACY = HERE.parent.parent / 'decl-control'
sys.path.append(str(LEGACY))
from generate_envs3 import (IMAGE, IMAGE_IDS, REVISION, STRIP, ZERO_FLAGS,
                            env_text, filtered, read_reference, reject_banned)

PREFIX = 'stdbench-compute-a2-20261106-v1'
SPEC_SHA256 = "8eaa9575592f519b504d08022d400602e4b552874c8424b02c59c34f4af174e2"
TOKENIZER_SHA256 = 'c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
FAMILIES = ('integer', 'modular', 'division', 'dp', 'graph', 'aggregation')
SEEDS = (101, 102)
ORDERS = (('t01', ('D', 'L')), ('t02', ('L', 'D')))
CLIENT_ARM = {'D': 'declared_off', 'L': 'lane', 'P': 'plain'}
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
        # These are already set by the pinned native launch adapter. Make its
        # rank identity/kernel defaults explicit in the complete effective seal.
        env.update(TF_TP_WORLD='2',TF_TP_RANK=str(rank),TF_TP_DEVICE='0',
                   TF_TP_KERNEL_IMAGE='/opt/tensorfold/share/tp_mailbox.fatbin')
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
        elif name.startswith('frozen-v1/') or name in ('prompt-inputs.json', 'stage-campaign.json', 'source-hashes.json', 'retention.json', 'adoption.json'):
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



# A1 retains reusable task/config serializers above; collection layout below is new.
from protocol import BASE_SHA, A1_TURNS
OLD_PREFIX='stdbench-compute-dl-20261105-v1'
PUBLIC_SHA='3e1e07d7a573eb194c0568afac2ef2112011b2b2cada9012fbbce4e6f47f7548'
REMAINING_SHA='6fdf99b3dbfd5d77e9ae3c04c3a23c85c18e78a0762933561aa34c1fbd57da43'
OLD_FREEZE_SHA='8aee124a6c4fcb68208d8bd710bd8f2a9413c153359240c43bc99ec670db8f2f'

def plain(row):
    row=copy.deepcopy(row)
    del row['body_template']['tools']
    row['arm']='plain'; row['execution_policy']={'TF_DSV41_COMPUTE_LANE':'0'}
    row['wire']=wire(row['body_template'])
    row['wire_sha256']=row['body_sha256']=digest(row['wire'].encode())
    return row

def old_boots(root):
    frozen=root/OLD_PREFIX/'frozen-v1'
    if sha(frozen/'manifest.json')!=OLD_FREEZE_SHA: raise ValueError('original request manifest drift')
    index=load_text((frozen/'manifest.json').read_text()); boots={}
    for entry in index['boots']:
        path=frozen/entry['file']; path.resolve().relative_to(frozen.resolve())
        if sha(path)!=entry['sha256']: raise ValueError('original boot drift')
        boots[entry['boot']]=load_text(path.read_text())
    if set(boots)!={'t01-D','t01-L','t02-L','t02-D'}: raise ValueError('original grid differs')
    return boots

def retention(root,boots,observed_sources=None):
    """Seal all durable raw/response sources, not merely report rows."""
    if sha(root/'remaining-collection.json')!=REMAINING_SHA: raise ValueError('remaining mask drift')
    remaining=load_text((root/'remaining-collection.json').read_text())
    if remaining['count']!=138 or remaining['counts']!={'D':120,'L':18}: raise ValueError('remaining counts differ')
    grid={(b['arm'],*r['key']):r for b in boots.values() for r in b['requests']}
    if len(grid)!=480: raise ValueError('original grid is not unique480')
    missing=[]
    for item in remaining['remaining_original_requests']:
        key=(item['arm'],*item['key']); row=grid[key]
        for field in ('wire_sha256','task_sha256'):
            if row[field]!=item[field]: raise ValueError('remaining original wire/task differs')
        if sha(root/OLD_PREFIX/'frozen-v1/boots'/Path(item['original_boot_file']).name)!=item['original_boot_sha256']:
            raise ValueError('remaining boot source differs')
        missing.append(key)
    if len(set(missing))!=138: raise ValueError('duplicate missing key')
    sources=list(observed_sources) if observed_sources is not None else sorted(
        p for name in ('raw-observations.jsonl','responses.jsonl') for p in (root/'runs').glob(OLD_PREFIX+'-*/**/'+name))
    seen_raw=set(); seen_response=set(); hashes={}; records=[]
    for source in sources:
        source=Path(source)
        source_name=str(source).replace('<rank1-home>/','<rank0-home>/',1)
        hashes[source_name]=sha(source)
        boot=source.parent.name
        if boot not in boots: raise ValueError('foreign observed boot')
        for number,line in enumerate(source.read_text().splitlines(),1):
            row=load_text(line)
            if source.name=='raw-observations.jsonl':
                if row.get('diagnostic') is True: continue
                key=(boots[boot]['arm'],*row['template_key']); target=seen_raw
            elif source.name=='responses.jsonl':
                if row.get('diagnostic') is True: continue
                key=(boots[boot]['arm'],row['suite'],row['id'],row['seed']); target=seen_response
                if row['actual_request_wire_sha256']!=grid[key]['wire_sha256']: raise ValueError('original response wire differs')
            else: raise ValueError('unsupported raw source')
            if key not in grid or key in target: raise ValueError('foreign/duplicate observed request')
            target.add(key); records.append({'key':list(key),'source':source_name,'line':number,'line_sha256':digest(line.encode())})
    expected=set(grid)-set(missing)
    if seen_raw!=expected or seen_response!=expected or len(expected)!=342: raise ValueError('durable observed mask differs; reconcile before resume')
    if ('L','compute_dl','compute-dl-v1-aggregation-015',101) not in expected: raise ValueError('stopped observation missing')
    return {'observed':342,'remaining':138,'counts':{'D':120,'L':222},
            'raw_source_sha256':hashes,'records':records,
            'keys':[{'key':list(k),'observed':k in expected} for k in grid]}, set(missing)

def stage(rootout=None, *, root=HERE.parent, observed_sources=None, references=None, spec_file=None, prior_path=None, prior_sha=None, audit_path=None, audit_sha=None):
    """Parent CPU scope only; source staging, no execution or GO creation."""
    root=Path(root).resolve(); references=Path(references or root.parent/'phase2/references')
    spec_file=Path(spec_file or HERE/'T5-COMPUTE-DL-SPEC-A2.md')
    if sha(spec_file)!=SPEC_SHA256 or sha(root/'T5-COMPUTE-DL-SPEC.md')!=BASE_SHA: raise ValueError('protocol drift')
    if sha(root/'dataset-v1/tasks-public.jsonl')!=PUBLIC_SHA: raise ValueError('public dataset drift')
    tasks=read_tasks(root/'dataset-v1/tasks-public.jsonl'); boots=old_boots(root)
    retained,missing=retention(root,boots,observed_sources)
    base=load_text((references/'base-arms.json').read_text())
    refs=[read_reference(references/f'base-r{r}.env') for r in (0,1)]
    for r,env in enumerate(refs):
        if env['IMAGE'] not in (IMAGE,IMAGE_IDS[r]) or env['ZIG_COMMIT']!=REVISION or base['arms']['D']['ranks'][r]['image_id']!=IMAGE_IDS[r]: raise ValueError('rank image pin differs')
    from adoption import validate, suffix, A1_PREFIX
    prior_source=Path(prior_path or root/A1_PREFIX/"budget.json")
    audit_source=Path(audit_path or root/"a1-continuation/final-audit-v3/audit.json")
    prior_sha=prior_sha or sha(prior_source);audit_sha=audit_sha or sha(audit_source)
    prior,a,rows,audit_files=validate(root,prior_source,prior_sha,audit_source,audit_sha)
    remaining,observed,full_p=suffix(boots,rows)
    new=[("A2-plain-P","P",remaining)]
    payloads={'retention.json':json_text(retained).encode()}; entries=[]; prompts={}; schedule=[]
    for b in boots.values():
        schedule.extend({'key':[b['arm'],*r['key']],'observed':True,
                         'wire_sha256':r['wire_sha256'],'task_sha256':r['task_sha256']} for r in b['requests'])
    for name,arm,rows in new:
        diags=copy.deepcopy(boots['t01-'+('L' if arm=='L' else 'D')]['diagnostics'])
        if arm=='P': diags=[plain(r) for r in diags]
        if [(r['key'][1],str(r['reference'])) for r in diags]!=[('python-17x19','323'),('python-sum11','55')]: raise ValueError('diagnostics differ')
        for row in rows+diags:
            if wire(row['body_template'])!=row['wire'] or digest(row['wire'].encode())!=row['wire_sha256']: raise ValueError('exact wire differs')
            if row['arm']!=CLIENT_ARM[arm] or row['execution_policy']!={'TF_DSV41_COMPUTE_LANE':str(int(arm=='L'))}: raise ValueError('arm/lane differs')
            prompt={'key':row['key'], 'messages':row['body_template']['messages']}
            if tuple(row['key']) in prompts and prompts[tuple(row['key'])]!=prompt: raise ValueError('same key prompt drift')
            prompts[tuple(row['key'])]=prompt
        if arm=='P': schedule.extend({"key":["P",*r["key"]],"observed":("P",*r["key"]) in observed,"wire_sha256":r["wire_sha256"],"task_sha256":r["task_sha256"]} for r in full_p)
        data=json_text({'boot':name,'arm':arm,'requests':rows,'diagnostics':diags}).encode()
        payloads['frozen-v1/boots/'+name+'.json']=data
        entries.append({'boot':name,'arm':arm,'file':'boots/'+name+'.json','sha256':digest(data)})
        payloads.update(rank_artifacts(base,refs,name,arm))
    if len(schedule)!=720 or len({tuple(r['key']) for r in schedule})!=720: raise ValueError('A1 schedule coverage differs')
    index={'schema':1,'prefix':PREFIX,'spec_sha256':SPEC_SHA256,'parent_spec_sha256':BASE_SHA,
           'turns':A1_TURNS,'boots':entries,'fresh_replies':len(remaining),'scheduled_replies':720,
           'retention_sha256':digest(payloads['retention.json']),'schedule':schedule,
           'tasks_public_sha256':PUBLIC_SHA,'collection_qualified':False}
    payloads['frozen-v1/manifest.json']=json_text(index).encode()
    prompt_source=root/"stdbench-compute-a1-20261105-v2/prompt-inputs.json"
    payloads["prompt-inputs.json"]=prompt_source.read_bytes()
    qualified_inputs=load_text(payloads["prompt-inputs.json"].decode())
    qualified_by_key={tuple(r["key"]):r for r in qualified_inputs}
    if len(qualified_inputs)!=len(qualified_by_key): raise ValueError("qualified input coverage differs")
    if len(qualified_inputs)!=242 or any(qualified_by_key.get(key)!=row for key,row in prompts.items()): raise ValueError("P suffix prompt bytes changed")
    target=prefix_root(0)
    old_campaign=load_text((root/OLD_PREFIX/'campaign-v1.json').read_text())
    campaign=copy.deepcopy(old_campaign)
    campaign.update(prefix=PREFIX,spec=str(spark_root(0)/'a2-continuation/T5-COMPUTE-DL-SPEC-A2.md'),spec_sha256=SPEC_SHA256,
        base_spec=str(spark_root(0)/'T5-COMPUTE-DL-SPEC.md'),base_spec_sha256=BASE_SHA,arms=['D','L','P'],
        freeze=str(target/'frozen-v1'),freeze_sha256=digest(payloads['frozen-v1/manifest.json']),layout={'turns':A1_TURNS,'boots':entries},
        cpu_receipt=str(target/'cpu-seal/verification.json'),cpu_receipt_sha256=None,
        cpu_prompt_arrays=str(target/'prompt-arrays.jsonl'),cpu_prompt_arrays_sha256=None,
        code_sha256={},go_file=str(target/'GO'),budget_ledger=str(target/'budget.json'),
        original_budget=str(spark_root(0)/OLD_PREFIX/'budget.json'),original_budget_sha256=sha(root/OLD_PREFIX/'budget.json'),
        prior_budget=str(spark_root(0)/A1_PREFIX/"budget.json"),prior_budget_sha256=prior_sha,
        a1_audit=str(spark_root(0)/"a1-continuation/final-audit-v3/audit.json"),a1_audit_sha256=audit_sha,
        a1_audit_files_sha256=audit_files,a1_spec_sha256="0dca1fa52838388ea1e7afbc3a4c5e9747775f64e871640df7a66496369e5453",a1_spec=str(spark_root(0)/"a2-continuation/T5-COMPUTE-DL-SPEC-A1.md"),
        adoption_receipt=str(target/"adoption.json"),adoption_receipt_sha256=None,
        revalidation_receipt=str(spark_root(0)/'a1-prep/revalidation-v2/qualification.json'),revalidation_receipt_sha256=None,
        cap_policy={'binary':str(spark_root(0)/'a1-prep/zig-out/bin/native-cap'),
          'tokenizer':str(spark_root(0).parent/'phase2/references/tokenizer.json'),
          'source':str(spark_root(0).parent/'decl-control/dispatch-high'),
          'receipt':str(spark_root(0)/'a1-prep/native-build-receipt.json')},
        status='STAGED_CPU_GATES_PENDING',collection_qualified=False)
    campaign['rank_manifests']={b['boot']:{str(r):str(prefix_root(r)/f'rank{r}/boots'/b['boot']/'manifest.json') for r in (0,1)} for b in entries}
    campaign['rank_manifest_sha256']={b['boot']:{str(r):digest(payloads[f'rank{r}/boots/{b["boot"]}/manifest.json']) for r in (0,1)} for b in entries}
    campaign["budget"]={"limit_seconds":36000,"admission_seconds":33900}
    campaign['timeouts']['client_wait_s']=360
    import datetime
    freeze=root/"a2-report/freeze.json";event=load_text(freeze.read_text())
    if event["spec_sha256"]!=SPEC_SHA256 or event["event"]!="A2_FROZEN_BEFORE_A1_CUTOFF": raise ValueError("A2 freeze missing")
    receipt=dict(spec_sha256=SPEC_SHA256,adopted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        old_limits={"limit_seconds":18000,"admission_seconds":15900},new_limits=campaign["budget"],
        prior_ledger_sha256=prior_sha,unchanged_held_seconds=prior["held_seconds"],unchanged_turns=prior["turns"],
        audit_files_sha256=audit_files,observed_mask=[r["key"] for r in schedule if r["observed"]],
        observed_count=len(observed),remaining_count=len(remaining),optional_G_restore_breaks=1,
        outcome_informed=True,single_new_turn="A2-plain",freeze_path=str(freeze),freeze_sha256=sha(freeze))
    payloads["adoption.json"]=json_text(receipt).encode()
    campaign["adoption_receipt_sha256"]=digest(payloads["adoption.json"])
    from protocol import require_staging_ready
    require_staging_ready(campaign)
    from budget import validate_original
    validate_original(root/OLD_PREFIX/'budget.json',campaign['original_budget_sha256'])
    payloads['stage-campaign.json']=json_text(campaign).encode()
    sources={str(p):sha(p) for p in [spec_file,root/'T5-COMPUTE-DL-SPEC.md',root/'remaining-collection.json',root/'LEDGER.md',*HERE.glob('*.py'),*HERE.glob('*.sh')]}
    payloads['source-hashes.json']=json_text({'sources':sources}).encode()
    result=dict(index,files={n:digest(b) for n,b in payloads.items()},renderer_input='prompt-inputs.json')
    destination=new_directory(rootout or HERE/'runtime-stage-v1')
    for name,data in payloads.items(): write_new(destination,name,data)
    write_new(destination,'stage-manifest.json',json_text(result).encode())
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(); commands=parser.add_subparsers(dest='command',required=True)
    make=commands.add_parser('stage'); make.add_argument('--out',type=Path); make.add_argument('--root',type=Path,default=HERE.parent)
    make.add_argument('--prior-budget',type=Path); make.add_argument('--prior-budget-sha256')
    make.add_argument('--a1-audit',type=Path); make.add_argument('--a1-audit-sha256')
    install=commands.add_parser('materialize'); install.add_argument('stage_root',type=Path); install.add_argument('rank',type=int,choices=(0,1)); install.add_argument('--destination',type=Path)
    args=parser.parse_args()
    print(json_text(stage(args.out,root=args.root,prior_path=args.prior_budget,prior_sha=args.prior_budget_sha256,audit_path=args.a1_audit,audit_sha=args.a1_audit_sha256) if args.command=='stage' else materialize(args.stage_root,args.rank,args.destination)))
