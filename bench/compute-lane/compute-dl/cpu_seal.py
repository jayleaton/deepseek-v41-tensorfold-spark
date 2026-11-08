#!/usr/bin/env python3
"""Offline final CPU qualification; no subprocess, generation, SSH or GPU work.

Run ONLY in the parent's bounded Spark CPU scope. All paths in code_sha256 are
canonical rank0-host paths; on rank1-host reads are mapped to its synchronized mirror.

API / CLI:
  provisional(stage): validate stage and materialized files, return a campaign
    with collection gate absent. CLI: provisional --stage DIR --out NEW_FILE.
    This is for rank_ops verify --dry-run ONLY. It never writes GO.
  seal(stage, checks, *, dataset=None, pin=None, exclusions=None): validate actual
    evidence, exclusively write cpu-seal/verification.json, campaign-sealed.json,
    then GO (last). CLI: seal --stage DIR --checks checks.json [--dataset DIR]
    [--pin FILE] [--exclusions FILE]. No overwrites, including partial seals.

checks.json maps the eight check names generation,outcome,disjoint,lifecycle,
runtime,budget,controller,report to actual cpu_checks.py receipts. Select the latest
passing controller receipt explicitly. Each receipt must bind every source the
corresponding check exercises (see CHECK_SOURCES); a PASS flag alone is rejected.
Other inputs use the existing producer filenames: PREFIX/prompt-proof.json,
prompt-arrays.jsonl, cpu-rank-{0,1}.json, cpu-preflight.json; baseline-r{0,1}.json;
dataset generation/reference/exclusion/candidate receipts and generation-pin-v1.
Rank1 proof paths are resolved in PREFIX/rank1-cpu-proof/{boot}/{basename}; their
original proof bytes/hash and rank0-host canonical source paths are preserved.

Output campaign-sealed.json deliberately differs from campaign-v1.json: rank
qualification may have put a provisional file at campaign-v1.json already. The
parent must install identical final campaign bytes on BOTH ranks as campaign-v1
before collection. No campaign or CPU receipt hashes itself. GO is not included
in code_sha256. No budget ledger is initialized. Missing/drifting evidence raises
ValueError/FileNotFoundError before final output. Copy ALL sealed canonical files
into the rank1-host mirror before running protocol.require_collection_ready there.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime_stage as runtime
from protocol import QUALIFIED_SPEC_SHA256, require_staging_ready

SPEC = QUALIFIED_SPEC_SHA256
PUBLIC_SHA = '3e1e07d7a573eb194c0568afac2ef2112011b2b2cada9012fbbce4e6f47f7548'
ROOT = runtime.prefix_root(0)
LEGACY = HERE.parent / 'decl-control'
CHECK_SOURCES = {
    'generation': ('generation_check.py', 'generate_tasks.py', 'reference_a.py', 'reference_b.py'),
    'outcome': ('outcome_check.py', 'grade.py', 'analysis.py'),
    'disjoint': ('disjoint_check.py', 'disjointness.py', 'exclusion_sources.py'),
    'lifecycle': ('lifecycle_check.py', 'controller.py', 'protocol.py', 'rank_ops.py'),
    'runtime': ('runtime_check.py', 'runtime_stage.py'),
    'budget': ('budget_check.py', 'budget.py'),
    'controller': ('controller_safety_check.py', 'controller.py', 'budget.py', 'protocol.py', 'rank_ops.py'),
    'report': ('report_check.py', 'collect_report.py', 'analysis.py', 'grade.py'),
}
GATES = {'generated_tasks': 120, 'scheduled_replies': 480, 'disjointness': True,
         'independent_references': True, 'prompt_pairs': 242, 'prior_dispatch': True,
         'rank_checks': 8, 'baseline_rank_checks': 2, 'preflight': True,
         'fake_checks': True, 'grader_analysis_pinned': True, 'pre_generation_pin': True}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical_path(path):
    return str(Path(path)).replace('<rank1-home>/', '<rank0-home>/', 1)


def remap_tree(value, mapper):
    if isinstance(value, dict):
        return {key: remap_tree(item, mapper) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [remap_tree(item, mapper) for item in value]
    if isinstance(value, str) and value.startswith(('<rank0-home>/', '<rank1-home>/')):
        return str(mapper(value))
    return value


def local(path):
    path = Path(canonical_path(path))
    if str(HERE).startswith('<rank1-home>/'):
        return Path(str(path).replace('<rank0-home>/', '<rank1-home>/', 1))
    return path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode()


def checked_relative(root, name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, 'unsafe relative evidence path')
    target = Path(root) / path
    require(target.resolve().is_relative_to(Path(root).resolve()), 'evidence escapes root')
    return target


class Inputs:
    """Hash the actual consumed bytes, and detect changes before publication."""
    def __init__(self):
        self.hashes = {}
        self.actual = {}

    def read(self, path, expected=None):
        path = local(path)
        require(path.is_absolute() and '..' not in path.parts, 'absolute evidence path required')
        require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink evidence forbidden')
        raw = path.read_bytes()
        sha = digest(raw)
        if expected is not None:
            require(isinstance(expected, str) and re.fullmatch('[a-f0-9]{64}', expected)
                    and sha == expected, 'evidence hash drift: ' + str(path))
        name = canonical_path(path)
        require(name not in self.hashes or self.hashes[name] == sha, 'input changed during validation')
        self.hashes[name] = sha
        self.actual[name] = path
        return raw

    def sha(self, path, expected=None):
        return digest(self.read(path, expected))

    def obj(self, path, expected=None):
        return runtime.load_text(self.read(path, expected).decode('utf-8'))

    def rows(self, path, expected=None):
        return [runtime.load_text(line) for line in self.read(path, expected).decode().splitlines() if line.strip()]

    def stable(self):
        for name, path in self.actual.items():
            require(digest(path.read_bytes()) == self.hashes[name], 'evidence changed: ' + name)


def stage_inputs(inputs, stage):
    stage = local(Path(stage).absolute())
    index = inputs.obj(stage / 'stage-manifest.json')
    require(index['prefix'] == runtime.PREFIX and index['spec_sha256'] == SPEC, 'wrong stage identity')
    for name, sha in index['files'].items():
        inputs.read(checked_relative(stage, name), sha)
    cfg = inputs.obj(stage / 'stage-campaign.json')
    require(cfg.get('collection_qualified') is False and cfg.get('cpu_receipt_sha256') is None,
            'stage campaign is already promoted')
    require_staging_ready(cfg)
    require(cfg['image_ids'] == list(runtime.IMAGE_IDS) and cfg['boot_min_gib'] == 114,
            'stage image/admission differs')
    require(inputs.sha(cfg['spec']) == SPEC, 'spec drift')
    require(index['turns'] == cfg['layout']['turns'] and index['boots'] == cfg['layout']['boots'],
            'stage layout drift')
    sources = inputs.obj(stage / 'source-hashes.json')['sources']
    for source in sources.values():
        inputs.read(source['path'], source['sha256'])
    require(sources['tasks_public']['sha256'] == PUBLIC_SHA, 'stage did not use adopted actual tasks')
    for rank in (0, 1):
        plan = runtime.materialization_plan(index['files'], rank)
        # Rank1 artifacts are validated from rank0-host's shadow; materialized.json
        # itself must be copied from rank1 as materialized-r1.json on rank0-host.
        receipt_path = ROOT / ('materialized.json' if rank == 0 else 'materialized-r1.json')
        receipt = inputs.obj(receipt_path)
        require(receipt['rank'] == rank and receipt['prefix'] == runtime.PREFIX
                and receipt['spec_sha256'] == SPEC
                and receipt['stage_manifest_sha256'] == inputs.sha(stage / 'stage-manifest.json'),
                'materialization provenance differs')
        expected_files = {target: index['files'][source] for source, target in plan.items()}
        require(receipt['files'] == expected_files, 'materialized inventory differs')
        for target, sha in expected_files.items():
            inputs.read(ROOT / target, sha)
    frozen = inputs.obj(Path(cfg['freeze']) / 'manifest.json', cfg['freeze_sha256'])
    require(frozen['tasks_public_sha256'] == PUBLIC_SHA and frozen['fresh_replies'] == 480
            and frozen['seeds'] == [101, 102] and frozen['families'] == list(runtime.FAMILIES),
            'frozen cohort differs')
    for boot in cfg['layout']['boots']:
        for rank in (0, 1):
            relative = f'rank{rank}/boots/{boot["boot"]}/manifest.json'
            path = ROOT / relative
            inputs.read(path, cfg['rank_manifest_sha256'][boot['boot']][str(rank)])
            # Stage native rank1 spelling must normalize to the rank0-host shadow.
            require(canonical_path(cfg['rank_manifests'][boot['boot']][str(rank)]) == str(path),
                    'rank manifest path differs')
    return cfg, sources


def pin_sources(inputs):
    # Seal all direct compute sources, including this module and added report
    # code, plus legacy launch/transport/native-frame/replay dependencies.
    for folder in (HERE, LEGACY):
        for pattern in ('*.py', '*.sh'):
            for path in sorted(folder.glob(pattern)):
                inputs.read(path)
    phase = HERE.parent / 'phase2'
    for name in ('client.py', 'native_frames.py', 'analysis.py', 'grade_all.py', 'audit_records.py',
                 'jail.py', 'jail_worker.py', 'jail.md', 'cpu-rank0-host.sh', 'cpu-rootrank1_host.sh'):
        inputs.read(phase / name)
    for name in ('bench.py', 'grader_worker.py', 'datasets.lock.json'):
        inputs.read(HERE.parent / 'kit' / name)


def provisional(stage):
    """Return a verified provisional campaign; never sufficient for collection."""
    inputs = Inputs()
    cfg, _ = stage_inputs(inputs, stage)
    pin_sources(inputs)
    cfg = copy.deepcopy(cfg)
    cfg['code_sha256'] = dict(sorted(inputs.hashes.items()))
    # No CPU receipt, GO, or qualification flag is invented for dry-run access.
    require_staging_ready(cfg)
    inputs.stable()
    return cfg


def dataset_evidence(inputs, dataset, pin_path, exclusions):
    import grade
    from disjointness import CollisionIndex
    dataset = local(Path(dataset).absolute())
    generation = inputs.obj(dataset / 'generation.json')
    require(generation['passed'] is True and generation['spec_sha256'] == SPEC
            and generation['task_count'] == 120 and generation['model_requests'] == 0
            and generation['public_sha256'] == PUBLIC_SHA, 'generation receipt differs')
    pin = inputs.obj(pin_path, generation['pre_generation_pin_sha256'])
    require(pin['spec_sha256'] == SPEC and pin['study_candidates_generated_before_this_pin'] == 0,
            'prospective pin absent')
    require(set(pin['source_sha256']) == {'generate_tasks.py', 'reference_a.py', 'reference_b.py',
                                        'disjointness.py', 'exclusion_sources.py'}, 'pin source set differs')
    for name, sha in pin['source_sha256'].items():
        inputs.read(HERE / name, sha)
    require(generation['generation_source_sha256'] == pin['source_sha256']['generate_tasks.py']
            and generation['python_version'] == pin['python_version'], 'generation pin differs')
    exclusion_config = inputs.obj(exclusions, generation['exclusion_config_sha256'])
    public = dataset / 'tasks-public.jsonl'
    tasks = runtime.read_tasks(local(public))
    inputs.read(public, PUBLIC_SHA)
    by_id = {task['id']: task for task in tasks}
    refs = inputs.obj(dataset / 'reference-verification.json')
    require(refs['passed'] is True and refs['tasks'] == 120 and refs['independent_processes'] is True
            and refs['spec_sha256'] == SPEC and refs['public_sha256'] == PUBLIC_SHA
            and refs['source_sha256'] == pin['source_sha256']
            and refs['python_version'] == pin['python_version'], 'reference verification differs')
    answers = []
    for letter in ('a', 'b'):
        path = dataset / f'answers-{letter}.json'
        receipt = inputs.obj(Path(str(path) + '.receipt.json'))
        values = inputs.obj(path, refs[f'reference_{letter}_sha256'])
        require(receipt['status'] == 'PASS' and receipt['count'] == 120
                and receipt['reference'] == letter.upper() and receipt['spec_sha256'] == SPEC
                and receipt['input_sha256'] == PUBLIC_SHA
                and receipt['source_sha256'] == pin['source_sha256'][f'reference_{letter}.py']
                and receipt['output_sha256'] == inputs.sha(path)
                and receipt['python_version'] == pin['python_version']
                and canonical_path(receipt['input_path']) == canonical_path(public)
                and canonical_path(receipt['output_path']) == canonical_path(path), 'reference process binding differs')
        require(set(values) == set(by_id), 'reference ID coverage differs')
        for ident, value in values.items():
            require(grade.valid_value(value, by_id[ident]['family']), 'malformed typed reference')
        answers.append(values)
    require(answers[0] == answers[1], 'independent reference disagreement')
    manual = inputs.obj(dataset / 'manual-resolutions.json')
    original_manual = inputs.obj(HERE / 'manual-resolutions-v1.json', generation['manual_resolutions_sha256'])
    require(manual == original_manual, 'reserialized manual resolution content differs')
    inventory = inputs.obj(HERE / 'exclusion-inventory-v1/records.json')
    audit = inputs.obj(dataset / 'exclusion-audit.json')
    require(audit == inputs.obj(HERE / 'exclusion-inventory-v1/audit.json'), 'exclusion inventory drift')
    # Reparse the authoritative original inputs. Cached record/audit agreement
    # alone cannot establish that those records came from the pinned sources.
    from exclusion_sources import load_exclusions
    actual_records, actual_audit = load_exclusions(remap_tree(exclusion_config, local))
    require(remap_tree(actual_records, canonical_path) == remap_tree(inventory, canonical_path)
            and remap_tree(actual_audit, canonical_path) == remap_tree(audit, canonical_path),
            'exclusions differ from authoritative source inputs')
    require(audit['counts'] == {'W9': 450, 'AB3': 60, 'AB3-pilot': 11, 'stdbench': 92,
                              'deduplicated_W9_descendants': 71, 'unique_records': 542}, 'exclusion census differs')
    for item in audit['files']:
        inputs.read(item['path'], item['sha256'])
    prior = {row['id']: row for row in inventory}
    require(len(inventory) == len(prior) == 542, 'prior coverage differs')
    unresolved = inputs.obj(HERE / 'exclusion-inventory-v1/unresolved.json')
    require(set(manual) == {row['id'] for row in unresolved} and len(manual) == 68,
            'manual resolution coverage differs')
    for ident, resolution in manual.items():
        original = prior[ident]
        evidence = resolution['original_evidence']
        require(evidence['path'] == original['source_path'] and evidence['sha256'] == original['source_sha256']
                and ident in evidence['location'] and resolution['reason']
                and resolution['prompt_sha256'] == digest(original['prompt'].encode())
                and resolution['decision'] == 'nonoverlap' and resolution['signature'] is None,
                'manual resolution binding differs: ' + ident)
        inputs.read(evidence['path'], evidence['sha256'])
    collision_index = CollisionIndex(inventory, manual)
    candidates = inputs.rows(dataset / 'candidates.jsonl')
    require(120 <= len(candidates) <= 120000, 'candidate ledger size differs')
    accepted, collisions = [], []
    next_attempt = {(f, s): 0 for f in runtime.FAMILIES for s in range(20)}
    completed = set()
    for record in candidates:
        candidate = record['candidate']
        key = (candidate['family'], candidate['slot'])
        require(key in next_attempt and key not in completed
                and candidate['attempt'] == next_attempt[key], 'undocumented attempt replacement')
        actual = collision_index.check(candidate)
        require(actual == record['collision'], 'recorded collision differs from actual exclusion inputs')
        if actual is None:
            collision_index.add(candidate)
            accepted.append(candidate)
            completed.add(key)
        else:
            collisions.append(actual)
            next_attempt[key] += 1
    require(accepted == tasks and len(completed) == 120, 'candidate ledger does not yield actual cohort')
    require(collisions == inputs.obj(dataset / 'collisions.json')
            and len(collisions) == generation['collision_count'], 'collision evidence differs')
    return tasks


def requests_and_prompts(inputs, cfg, tasks):
    tools = inputs.obj(LEGACY / 'canonical-python-tools.json')
    old_index = inputs.obj(LEGACY / 'frozen-a1-v1/manifest.json')
    old_entry = [e for e in old_index['boots'] if e['boot'] == 't01-lane' and e['arm'] == 'lane']
    require(len(old_entry) == 1, 'original diagnostic source missing')
    old_boot = inputs.obj(checked_relative(LEGACY / 'frozen-a1-v1', old_entry[0]['file']),
                          old_entry[0]['sha256'])
    prompt_inputs = inputs.obj(ROOT / 'prompt-inputs.json')
    require(len(prompt_inputs) == 242 and len({tuple(r['key']) for r in prompt_inputs}) == 242,
            'renderer input coverage differs')
    messages = {tuple(r['key']): r['messages'] for r in prompt_inputs}
    boot_data = {}
    for entry in cfg['layout']['boots']:
        boot = inputs.obj(Path(cfg['freeze']) / entry['file'], entry['sha256'])
        require(boot['boot'] == entry['boot'] and boot['arm'] == entry['arm']
                and len(boot['requests']) == 120 and len(boot['diagnostics']) == 2, 'boot coverage differs')
        turn = entry['boot'][:3]
        pairs = runtime.ordered_pairs(tasks, turn)
        require([row['key'] for row in boot['requests']] == [['compute_dl', t['id'], seed] for t, seed in pairs],
                'request ordering differs')
        for row, (task, seed) in zip(boot['requests'], pairs):
            require(row['task'] == task and row['seed'] == seed
                    and row['task_sha256'] == digest(runtime.canonical(task).encode()), 'request task drift')
            require(row['body_template']['messages'] == [{'content': task['prompt'], 'role': 'user'}],
                    'task prompt differs from public task')
        require([(r['key'], str(r['reference'])) for r in boot['diagnostics']] == [
            (['diagnostic', 'python-17x19', 101], '323'), (['diagnostic', 'python-sum11', 101], '55')],
            'diagnostic identity differs')
        require([r['wire'] for r in boot['diagnostics']] == [r['wire'] for r in old_boot['diagnostics']],
                'diagnostic request bytes differ from prior frozen source')
        for row in boot['requests'] + boot['diagnostics']:
            body = row['body_template']
            require({k: v for k, v in body.items() if k not in ('messages', 'seed', 'tools')} == runtime.SETTINGS
                    and body['tools'] == tools and body['seed'] == row['key'][2], 'request sampling/tools drift')
            require(row['wire'] == runtime.wire(body) and row['wire_sha256'] == digest(row['wire'].encode()),
                    'request wire drift')
            require(messages[tuple(row['key'])] == body['messages'], 'renderer input/request mismatch')
            require(row['arm'] == runtime.CLIENT_ARM[entry['arm']]
                    and row['execution_policy'] == {'TF_DSV41_COMPUTE_LANE': str(int(entry['arm'] == 'L'))},
                    'request policy drift')
        boot_data[entry['boot']] = boot
    for turn in ('t01', 't02'):
        d, l = (boot_data[turn + '-' + arm] for arm in ('D', 'L'))
        for section in ('requests', 'diagnostics'):
            require([r['wire'] for r in d[section]] == [r['wire'] for r in l[section]], 'D/L bytes differ')
    proof = inputs.obj(ROOT / 'prompt-proof.json')
    require(proof['passed'] is True and proof['spec_sha256'] == SPEC and proof['pairs'] == 242
            and proof['dataset_pairs'] == 240 and proof['diagnostic_pairs'] == 2
            and proof['actual_served_array_equality_observed'] is False, 'prompt proof scope differs')
    inputs.read(HERE / 'render_prompts.py', proof['source_sha256'])
    inputs.read(ROOT / 'prompt-inputs.json', proof['input_sha256'])
    old = inputs.obj(LEGACY / 'cpu-proof.json')
    inputs.read(LEGACY / 'prompt-gate', old['binary_sha256'])
    require(proof['binary_sha256'] == old['binary_sha256'], 'renderer binary pin differs')
    inputs.read(cfg['tokenizer'], runtime.TOKENIZER_SHA256)
    require(proof['tokenizer_sha256'] == old['tokenizer_sha256'] == cfg['tokenizer_sha256'], 'tokenizer differs')
    for name, sha in old['native_source_sha256'].items():
        inputs.read(LEGACY / 'native' / name, sha)
    for name, sha in old['harness_sha256'].items():
        inputs.read(LEGACY / name, sha)
    rows = inputs.rows(ROOT / 'prompt-arrays.jsonl', proof['arrays_sha256'])
    require([r['key'] for r in rows] == [r['key'] for r in prompt_inputs], 'full CPU array coverage differs')
    for row in rows:
        ids = row['declared_off_ids']
        require(isinstance(ids, list) and ids and all(type(i) is int and 0 <= i < 2**32 for i in ids)
                and ids == row['declared_lane_ids'] == row['original_builtin_lane_ids']
                and row['held_off'] is False and row['held_lane'] is True
                and row['explicit_tools'] == tools, 'CPU array/arming/declaration mismatch')
    return proof['arrays_sha256']


def prior_evidence(inputs):
    old = inputs.obj(LEGACY / 'cpu-final-a1-v2/verification.json')
    require(old['passed'] is True and old['fresh_replies'] == 552, 'prior CPU seal missing')
    for path, sha in old['code_sha256'].items():
        inputs.read(path, sha)
    dispatch_root = LEGACY / 'dispatch-high'
    proof = inputs.obj(dispatch_root / 'dispatch-proof.json', old['dispatch_proof_sha256'])
    require(proof['verdict'] == 'PASS' and proof['observed_fixtures'] == 8
            and proof['guard_mutant_compiled'] is True and proof['guard_mutant_behaviorally_caught'] is True
            and proof['whole_stream_noncall_and_wrong_arm_controls'] is True
            and proof['deployed_binary_instrumentation'] is False, 'prior dispatch proof differs')
    for name, sha in proof['sha256'].items():
        inputs.read(checked_relative(dispatch_root, name), sha)
    sources = inputs.obj(dispatch_root / 'unmodified-source-sha256.json')
    require(len(sources) == proof['production_sources_unchanged'] == 3436, 'prior production census differs')
    for name, sha in sources.items():
        inputs.read(checked_relative(dispatch_root, name), sha)
    # Replay the read-only receipt audit, never run the fixture or old prompts.
    from dispatch_audit3 import audit
    audit(dispatch_root)
    inputs.read(dispatch_root / 'fixture-receipts.jsonl')
    for rank in (0, 1):
        receipt = inputs.obj(HERE / f'baseline-r{rank}.json')
        require(receipt['passed'] is True and type(receipt['rank']) is int and receipt['rank'] == rank
                and receipt['status'] == 'REUSED_A1_EVIDENCE_ONLY'
                and receipt['old_sealed_files_unchanged'] == len(old['code_sha256'])
                and receipt['production_source_hashes_rechecked'] == 3436
                and receipt['image_id'] == runtime.IMAGE_IDS[rank]
                and receipt['dispatch_proof_sha256'] == inputs.sha(dispatch_root / 'dispatch-proof.json')
                and receipt['dispatch_fixture_freshly_rerun'] is False
                and receipt['collection_authorized'] is False, 'baseline rank evidence differs')
        inputs.read(HERE / 'baseline_check.py', receipt['source_sha256'])


def rank_evidence(inputs, cfg):
    for rank in (0, 1):
        receipt = inputs.obj(ROOT / f'cpu-rank-{rank}.json')
        require(receipt['passed'] is True and type(receipt['rank']) is int and receipt['rank'] == rank
                and receipt['spec_sha256'] == SPEC, 'rank receipt differs')
        inputs.read(HERE / 'rank_ops.py', receipt['source_sha256'])
        inputs.read(HERE / 'qualify_ranks.py', receipt['runner_sha256'])
        require([r['boot'] for r in receipt['checks']] == ['t01-D', 't01-L', 't02-L', 't02-D'],
                'four distinct rank checks required')
        for check in receipt['checks']:
            boot = check['boot']
            manifest_path = ROOT / f'rank{rank}/boots/{boot}/manifest.json'
            manifest = inputs.obj(manifest_path, check['manifest_sha256'])
            original = Path(check['proof'])
            expected_parent = runtime.prefix_root(rank) / 'proof' / boot
            require(original.parent == expected_parent and original.name.startswith('verify-')
                    and original.suffix == '.json', 'unexpected rank proof path')
            path = original if rank == 0 else ROOT / 'rank1-cpu-proof' / boot / original.name
            proof = inputs.obj(path, check['sha256'])
            log = ROOT / f'cpu-verify-{boot}-r{rank}.log'
            text = inputs.read(log, check['log_sha256']).decode()
            event = runtime.load_text(text.strip().splitlines()[-1])
            require(event['proof'] == str(original) and event['command'] == 'verify'
                    and event['dry_run'] is True and event['boot'] == boot and event['rank'] == rank,
                    'rank stdout/proof binding differs')
            require(proof['command'] == 'verify' and proof['dry_run'] is True and proof['rank'] == rank
                    and proof['boot'] == boot and proof['manifest_sha256'] == check['manifest_sha256']
                    == cfg['rank_manifest_sha256'][boot][str(rank)]
                    and proof['image_id'] == runtime.IMAGE_IDS[rank], 'rank dry-run proof differs')
            meta = manifest['phase2']['ranks'][rank]
            expected = meta['effective_env']
            effective = {k: v for k, v in proof['env'].items() if not k.startswith(('GUARD', 'SPARK'))}
            require(proof['container_name'] == meta['container_name'] and effective == expected,
                    'rank effective environment differs')
            args = proof['argv']
            require(runtime.IMAGE_IDS[rank] in args,'dry-run pinned image absent')
            docker_args=args[:args.index(runtime.IMAGE_IDS[rank])]
            launch_checks={'docker':args[:2] == ['docker','run'],
                           'roundtrip':shlex.split(proof['launch_command'])==args,
                           'name_count':docker_args.count('--name')==1,
                           'name':docker_args[docker_args.index('--name')+1]==meta['container_name'],
                           'image':runtime.IMAGE_IDS[rank] in args}
            require(all(launch_checks.values()),'dry-run launch identity differs: '+boot+' r'+str(rank)+' '+
                    ','.join(k for k,v in launch_checks.items() if not v))
            container_env = proof['container_env']
            for key, value in expected.items():
                if key.startswith(('TF_', 'GLM53_TF', 'NCCL')):
                    require(container_env.get(key) == value, 'launcher dropped effective control: ' + key)
            require(any(args[i] in ('-v', '--volume') and args[i+1].endswith(':/prepared:ro')
                        for i in range(len(args)-1)), 'prepared mount not read-only')


def fake_and_preflight(inputs, checks):
    require(set(checks) == set(CHECK_SOURCES), 'explicit eight CPU check receipts required')
    for name, sources in CHECK_SOURCES.items():
        receipt = inputs.obj(checks[name])
        require(receipt['name'] == name and type(receipt['returncode']) is int and receipt['returncode'] == 0
                and receipt['GPU_authorized'] is False and isinstance(receipt['stdout'], str)
                and isinstance(receipt['stderr'], str), 'CPU fake check failed: ' + name)
        # Do not invalidate independent checks when unrelated modules were added.
        for source in sources:
            inputs.read(HERE / source, receipt['source_sha256'][source])
    receipt = inputs.obj(ROOT / 'cpu-preflight.json')
    require(receipt['passed'] is True and receipt['spec_sha256'] == SPEC, 'preflight failed')
    inputs.read(HERE / 'preflight_cpu.py', receipt['source_sha256'])
    require([Path(c['script']).name for c in receipt['checks']] == ['launch.sh', 'turn.sh', 'fence.sh'],
            'three shell preflights required')
    for check in receipt['checks']:
        require(check['returncode'] == 0 and isinstance(check['stdout'], str) and isinstance(check['stderr'], str)
                and canonical_path(check['script']) == canonical_path(HERE / Path(check['script']).name),
                'preflight command result differs')
        inputs.read(check['script'], check['sha256'])
    # The producer does not retain dependency hashes: bind current dependency
    # bytes too, and require parents to rerun preflight after any such drift.
    queue = Path('<rank0-work>/queue/turn-preflight.sh')
    inputs.read(queue)


def exclusive(path, data):
    path = local(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'unsafe output path')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def seal(stage, checks, *, dataset=None, pin=None, exclusions=None):
    """Validate existing artifacts, then publish an exclusive deterministic seal."""
    dataset = dataset or HERE / 'dataset-v1'
    pin = pin or HERE / 'generation-pin-v1.json'
    exclusions = exclusions or HERE / 'exclusions-config.json'
    targets = (ROOT / 'cpu-seal/verification.json', ROOT / 'campaign-sealed.json', ROOT / 'GO')
    require(not any(local(p).exists() or local(p).is_symlink() for p in targets), 'seal outputs already exist')
    inputs = Inputs()
    cfg, _ = stage_inputs(inputs, stage)
    pin_sources(inputs)
    checks_obj = inputs.obj(Path(checks).absolute()) if not isinstance(checks, dict) else checks
    tasks = dataset_evidence(inputs, dataset, pin, exclusions)
    arrays_sha = requests_and_prompts(inputs, cfg, tasks)
    prior_evidence(inputs)
    rank_evidence(inputs, cfg)
    fake_and_preflight(inputs, checks_obj)
    # Both ranks mirror runtime artifacts under stdbench. Historical exclusion
    # originals remain read-only at their authoritative rank0-host locations.
    # Preserve and recheck those pins separately; never install files into an
    # existing W9/AB3/pilot namespace on rank1-host.
    runtime_base='<rank0-work>/stdbench/'
    code = dict(sorted((p,h) for p,h in inputs.hashes.items() if p.startswith(runtime_base)))
    preparation = dict(sorted((p,h) for p,h in inputs.hashes.items() if not p.startswith(runtime_base)))
    require(len(code) >= 25 and all(name.startswith('<rank0-work>/') for name in code),
            'final source map must contain canonical rank0-host runtime paths only')
    forbidden = {canonical_path(p) for p in targets} | {str(ROOT / 'campaign-v1.json')}
    require(not forbidden.intersection(code), 'cyclic campaign/receipt/GO seal')
    receipt = {'passed': True, 'spec_sha256': SPEC, 'gates': GATES, 'code_sha256': code,
               'preparation_file_sha256':preparation,
               'freeze_sha256': cfg['freeze_sha256'], 'cpu_prompt_arrays_sha256': arrays_sha,
               'GPU': False, 'prompt_claim': 'CPU-matched/count-checked; served initial arrays unobserved',
               'multi_round_input_totals': 'WITHHELD',
               'campaign_installation': 'Install campaign-sealed.json bytes as campaign-v1.json on both ranks before collection'}
    cfg = copy.deepcopy(cfg)
    cfg.update(code_sha256=code, cpu_receipt_sha256=digest(encode(receipt)),
               cpu_prompt_arrays_sha256=arrays_sha, status='CPU_QUALIFIED', collection_qualified=True, pending=[])
    require(canonical_path(cfg['cpu_receipt']) == str(targets[0])
            and canonical_path(cfg['cpu_prompt_arrays']) == str(ROOT / 'prompt-arrays.jsonl')
            and canonical_path(cfg['go_file']) == str(targets[2]), 'unexpected seal destination bindings')
    require_staging_ready(cfg)
    inputs.stable()
    exclusive(targets[0], encode(receipt))
    exclusive(targets[1], encode(cfg))
    # Re-read exact produced bytes before authorizing. A partial publication has
    # no GO and cannot collect; existing partial output is never overwritten.
    require(local(targets[0]).read_bytes() == encode(receipt)
            and local(targets[1]).read_bytes() == encode(cfg), 'seal write/readback differs')
    inputs.stable()
    exclusive(targets[2], (SPEC + '\n').encode())
    return {'passed': True, 'cpu_receipt': str(targets[0]), 'cpu_receipt_sha256': cfg['cpu_receipt_sha256'],
            'campaign': str(targets[1]), 'campaign_sha256': digest(encode(cfg)),
            'go_file': str(targets[2]), 'sealed_files': len(code), 'gates': GATES}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('provisional')
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = commands.add_parser('seal')
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--checks', type=Path, required=True)
    p.add_argument('--dataset', type=Path)
    p.add_argument('--pin', type=Path)
    p.add_argument('--exclusions', type=Path)
    args = parser.parse_args()
    if args.command == 'provisional':
        cfg = provisional(args.stage)
        exclusive(args.out.absolute(), encode(cfg))
        result = {'collection_qualified': False, 'campaign': str(args.out)}
    else:
        result = seal(args.stage, args.checks, dataset=args.dataset, pin=args.pin, exclusions=args.exclusions)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
