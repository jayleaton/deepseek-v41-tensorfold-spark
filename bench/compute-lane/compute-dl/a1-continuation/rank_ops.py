#!/usr/bin/env python3
"""CPU-staged compute D/L rank operations; owned copy of A1 cleanup."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid
import time
import math

sys.dont_write_bytecode = True
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'decl-control'))
from protocol import require_collection_ready, require_staging_ready

from generate_envs3 import IMAGE, IMAGE_IDS, REVISION, STRIP, ZERO_FLAGS, filtered, reject_banned as a1_reject_banned


def reject_banned(value):
    a1_reject_banned(value)
    if 'IMR_PUBLISH' in str(value):
        raise ValueError('publication flag forbidden')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def minimal_env():
    return {k: v for k, v in os.environ.items()
            if k in ('PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'TMPDIR')}


def exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def owned(path, root):
    path = Path(path)
    root = Path(root)
    if not path.is_absolute() or '..' in path.parts or path == root:
        raise ValueError('invalid owned path')
    path.relative_to(root)
    # Refuse redirects, including symlink ancestors outside the owned root.
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f'symlink in owned path: {parent}')
    if path.resolve() != path:
        raise ValueError('owned path resolves elsewhere')
    return path


def context(manifest_path, rank, *, qualification_only=False):
    if rank not in (0, 1):
        raise ValueError('invalid rank')
    manifest = json.loads(Path(manifest_path).read_text())
    binding = manifest['compute_dl']
    campaign = (require_staging_ready if qualification_only else require_collection_ready)(binding['campaign'])
    if binding['spec_sha256'] != campaign['spec_sha256']:
        raise ValueError('rank belongs to another protocol')
    if binding['boot_min_gib'] != campaign['boot_min_gib']:
        raise ValueError('rank memory admission differs from campaign')
    for path, digest in campaign['code_sha256'].items():
        local = Path(path.replace('<rank0-home>/', f'<rank{rank}-home>/', 1))
        if hashlib.sha256(local.read_bytes()).hexdigest() != digest:
            raise ValueError('rank source seal drift: ' + str(local))
    expected_manifest = campaign['rank_manifest_sha256'][manifest['phase2']['boot']][str(rank)]
    if hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest() != expected_manifest:
        raise ValueError('rank manifest differs from campaign seal')
    record = manifest['arms']['D']
    phase = manifest['phase2']
    meta = phase['ranks'][rank]
    item = record['ranks'][rank]
    prefix, boot = phase['prefix'], phase['boot']
    if not re.fullmatch(r'stdbench-compute-a1-[A-Za-z0-9_.-]+', prefix):
        raise ValueError('invalid owned prefix')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', boot):
        raise ValueError('invalid boot')
    entries = [b for b in campaign['layout']['boots'] if b['boot'] == boot]
    if len(entries) != 1 or campaign['prefix'] != prefix:
        raise ValueError('boot absent from frozen campaign')
    arm = entries[0]['arm']
    if arm not in ('D', 'L', 'P') or phase['arm'] != arm or phase['lane'] != int(arm == 'L'):
        raise ValueError('boot/arm/lane mismatch')
    root = Path(f'<rank{rank}-work>/stdbench/compute-dl') / prefix
    name = f'{prefix}-{boot}-r{rank}'
    if meta['root'] != str(root) or meta['container_name'] != name or meta['rank'] != rank:
        raise ValueError('rank namespace differs')
    if record['engine_sha'] != REVISION or item['image_id'] != IMAGE_IDS[rank]:
        raise ValueError('manifest pin differs')
    if record.get('image_revision_label') != 'org.tensorfold.revision':
        raise ValueError('unexpected image label')
    cfg = owned(item['config'], root)
    if cfg != root / 'configs' / f'{boot}-r{rank}.env' or meta['config'] != str(cfg):
        raise ValueError('configuration path differs')
    bench = Path(f'<rank{rank}-work>/dsv41-bench')
    original_path = bench / 'ops/rank.py'
    # config_env otherwise inherits unrelated host flags and launch variables.
    saved = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(minimal_from(saved))
        os.environ.update(DSV41ROOT=str(bench), DSV41_BENCH_ROOT=str(bench))
        original = load_module('original_rank', original_path)
        env = original.config_env(cfg)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    reject_banned(env)
    expected = meta['effective_env']
    actual = {k: v for k, v in filtered(env).items() if not k.startswith(('GUARD', 'SPARK'))}
    exact_optional(expected, env)
    if actual != expected:
        raise ValueError('effective configuration differs from generated manifest')
    rank_settings = {k: v for k, v in actual.items() if k.startswith(('TF_', 'GLM53_TF'))}
    if item['settings'] != rank_settings:
        raise ValueError('per-rank manifest settings differ')
    controls = {f'TF_DSV41_{key}': '0' for key in ZERO_FLAGS}
    controls.update(TF_DSV41_COMPUTE_LANE=str(phase['lane']),
                    TF_DSV41_RUNNER_FORMS=phase['runner_forms'])
    if phase['guard_source'] != '/etc/spark/thermal.env':
        raise ValueError('thermal policy source differs')
    if meta['expected_off'] != {f'TF_DSV41_{key}': '0' for key in ZERO_FLAGS}:
        raise ValueError('missing explicit expected-off assertions')
    if any(env.get(key) != value for key, value in controls.items()):
        raise ValueError('explicit declaration-control controls differ')
    if any(any(word in key for word in STRIP) and key not in controls for key in env):
        raise ValueError('unexpected inherited experiment flag')
    # Refuse numeric guard overrides added to the generated config after staging.
    thermal = subprocess.run(['bash', '-c', 'set -a; source /etc/spark/thermal.env; env -0'],
                             env=minimal_env(), capture_output=True, check=True)
    thermal_env = dict(x.decode().split('=', 1) for x in thermal.stdout.split(b'\0') if b'=' in x)
    for key, value in env.items():
        if key.startswith(('GUARD', 'SPARK')) and thermal_env.get(key) != value:
            raise ValueError('guard differs from node thermal policy')
    if env.get('IMAGE') != IMAGE or env.get('ZIG_COMMIT') != REVISION:
        raise ValueError('config image tag/revision differs')
    for key, folder in (('STATE_DIR', 'state'), ('CACHE_DIR', 'cache')):
        target = owned(env[key], root)
        if target != root / folder / boot or meta[key.lower()] != str(target):
            raise ValueError('runtime directory differs')
    if env['CONTAINER_PREFIX'] != f'{prefix}-{boot}':
        raise ValueError('container prefix differs')
    return manifest, record, meta, env, root, bench, name


def minimal_from(source):
    return {k: v for k, v in source.items()
            if k in ('PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'TMPDIR')}


def prepare(record, meta, env, rank, root):
    """Only fresh owned state/cache may be created; shared prepared is never touched."""
    if not Path(env['PREPARED_DIR']).is_dir():
        raise ValueError('existing prepared directory is absent')
    source = Path(record['ranks'][rank]['recipe_dir']) / 'config/pfdense-table.json'
    if str(source) != meta['table_source'] or source.is_symlink():
        raise ValueError('tuning source differs')
    frozen = json.loads((source.parent.parent / 'BENCH-SOURCE.json').read_text())
    content = source.read_bytes()
    digest = frozen['files']['config/pfdense-table.json']
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError('frozen tuning table changed')
    cache = owned(env['CACHE_DIR'], root)
    table = cache / 'pfdense-table.json'
    if not cache.exists():
        cache.mkdir(parents=True, exist_ok=False)
        with table.open('xb') as stream:
            stream.write(content)
    # Repeated verify is safe; never replace existing data or accept cache extras.
    if table.is_symlink() or set(p.name for p in cache.iterdir()) != {'pfdense-table.json'}:
        raise ValueError('cache must contain only the frozen tuning table')
    if hashlib.sha256(table.read_bytes()).hexdigest() != digest:
        raise ValueError('owned tuning table differs')
    state = owned(env['STATE_DIR'], root)
    if not state.exists():
        state.mkdir(parents=True, exist_ok=False)
    (state / 'sessions').mkdir(exist_ok=True)
    owned(state / 'sessions', root)


def verify_original(manifest_path, rank, bench):
    env = dict(minimal_env(), DSV41ROOT=str(bench), DSV41_BENCH_ROOT=str(bench),
               DSV41_MANIFEST=str(Path(manifest_path).resolve()), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, str(bench / 'ops/rank.py'), 'verify', 'D', str(rank)],
                            env=env, text=True, capture_output=True, check=True)
    # rank.py verifies asset sizes, image/revision, frozen recipe/adapter and table.
    return result.stdout


def docker_command(output, env, image_id, name):
    candidates = []
    for line in output.splitlines():
        if line.strip().startswith('docker run '):
            candidates.append(shlex.split(line))
    if len(candidates) != 1:
        raise ValueError('expected exactly one original dry-run Docker command')
    args = candidates[0]
    if args[:2] != ['docker', 'run']:
        raise ValueError('unexpected command')
    # Preserve the frozen launch command, changing identity/mount/image only,
    # then explicitly forward sealed flags omitted by an older launcher.
    arity = {'--name', '--gpus', '--network', '--device', '--ulimit', '--cap-add',
             '-v', '--volume', '-e', '--env', '--ipc'}
    switches = {'--rm', '--init', '--ipc=host'}
    i, names, prepared, image_index = 2, 0, 0, None
    docker_env = {}
    while i < len(args):
        option = args[i]
        if option in switches:
            i += 1
            continue
        if option in arity:
            if i + 1 >= len(args):
                raise ValueError('truncated Docker option')
            value = args[i + 1]
            if option == '--name':
                names += 1
                args[i + 1] = name
            elif option in ('-v', '--volume'):
                pieces = value.split(':')
                if len(pieces) in (2, 3) and pieces[1] == '/prepared':
                    if pieces[0] != env['PREPARED_DIR']:
                        raise ValueError('prepared mount differs')
                    prepared += 1
                    args[i + 1] = f'{pieces[0]}:/prepared:ro'
            elif option in ('-e', '--env'):
                key, sep, val = value.partition('=')
                if not (key in filtered({key: ''}) or key == 'TENSORFOLD_REQUEST_LOG'):
                    raise ValueError('non-whitelisted Docker environment')
                if not sep and key not in env:
                    raise ValueError('unresolved launch environment: ' + key)
                docker_env[key] = val if sep else env[key]
            i += 2
            continue
        if option.startswith('-'):
            raise ValueError('unsupported Docker option: ' + option)
        image_index = i
        break
    if names != 1 or prepared != 1 or image_index is None:
        raise ValueError('missing or duplicate launch identity/mount')
    if args[image_index] != env['IMAGE']:
        raise ValueError('original launch image differs')
    args[image_index] = image_id
    additions = []
    for key, value in sorted(env.items()):
        if key.startswith(('TF_', 'GLM53_TF', 'NCCL')):
            if key not in docker_env:
                additions.extend(('-e', f'{key}={value}'))
                docker_env[key] = value
            elif docker_env[key] != value:
                raise ValueError('launcher changed sealed effective flag: ' + key)
    args[image_index:image_index] = additions
    reject_banned(args)
    reject_banned(docker_env)
    # Explicit controls must actually reach the container, not just host config.
    for key, value in env.items():
        if key.startswith(('TF_', 'GLM53_TF', 'NCCL')) and docker_env.get(key) != value:
            raise ValueError('launcher dropped/changed effective flag: ' + key)
    exact_optional(env, docker_env)
    return args, filtered(docker_env)


def inspect_container(name, image):
    result = subprocess.run(['docker', 'container', 'inspect', name], text=True, capture_output=True)
    if result.returncode:
        # A daemon/permission error must never be mistaken for an absent container.
        if any(text in result.stderr.lower() for text in ('no such container:', 'no such object:')):
            return None
        raise RuntimeError('container inspect failed: ' + result.stderr.strip())
    items = json.loads(result.stdout)
    if len(items) != 1 or items[0]['Name'] != '/' + name or items[0]['Image'] != image:
        raise ValueError('container name/image differs; refusing operation')
    return items[0]


def proof_path(root, boot, command):
    directory = owned(root / 'proof' / boot, root)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f'{command}-{uuid.uuid4().hex}.json'


def record_proof(root, proof):
    path = proof_path(root, proof['boot'], proof['command'])
    exclusive_json(path, proof)
    # Keep environment/launch details in the proof artifact, out of terminal JSON.
    print(json.dumps({'command': proof['command'], 'rank': proof['rank'],
                      'boot': proof['boot'], 'dry_run': proof['dry_run'],
                      'proof': str(path)}), flush=True)


def engine_environment(name, rank):
    probe = '''import json,os
out=[]
for entry in os.listdir('/proc'):
 if not entry.isdigit(): continue
 try:
  args=open('/proc/'+entry+'/cmdline','rb').read().split(b'\\0')
  args=[x.decode() for x in args if x]
  if not args or os.path.basename(args[0]) not in ('tensorfold-dsv41','tf-dsv41','tf-dsv41-m1'): continue
  if ('serve' if int(os.environ['STD_RANK'])==0 else 'follow') not in args: continue
  env=dict(x.decode().split('=',1) for x in open('/proc/'+entry+'/environ','rb').read().split(b'\\0') if b'=' in x)
  out.append(dict(pid_namespace=int(entry),argv=args,env=env))
 except (FileNotFoundError,ProcessLookupError): pass
if len(out)!=1: raise RuntimeError('expected exactly one native rank process')
print(json.dumps(out[0]))'''
    result = subprocess.run(['docker', 'exec', '-e', f'STD_RANK={rank}', name,
                             '/usr/bin/python3', '-I', '-S', '-c', probe],
                            capture_output=True, text=True, timeout=30, check=True)
    return json.loads(result.stdout)


def cleanup_identity(manifest_path, rank):
    """Derive only the exact owned target; never consult mutable GO/config/code."""
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError('invalid cleanup rank')
    base = Path(f'<rank{rank}-work>/stdbench/compute-dl')
    path = Path(manifest_path)
    parts = path.relative_to(base).parts
    if len(parts) != 5:
        raise ValueError('unexpected owned rank manifest layout')
    prefix, rank_dir, folder, boot, filename = parts
    if not re.fullmatch(r'stdbench-compute-a1-[A-Za-z0-9_.-]+', prefix):
        raise ValueError('invalid cleanup prefix')
    if boot not in ('A1-resume-L', 'A1-resume-D', 'A1-plain-P'):
        raise ValueError('invalid frozen cleanup boot')
    if (rank_dir, folder, filename) != (f'rank{rank}', 'boots', 'manifest.json'):
        raise ValueError('cleanup manifest rank/path differs')
    root = base / prefix
    owned(path, root)  # Includes all ancestors, even if manifest was revoked/removed.
    return root, prefix, boot, f'{prefix}-{boot}-r{rank}'


def launch_ownership(manifest_path, rank, root, manifest, name):
    """Retain immutable launch identity before Docker; no approval is inferred."""
    expected_root, prefix, boot, expected_name = cleanup_identity(manifest_path, rank)
    if root != expected_root or name != expected_name or manifest['phase2']['boot'] != boot:
        raise ValueError('qualified launch identity differs from retained ownership')
    path = owned(root / 'ownership' / f'{boot}-r{rank}.json', root)
    value = {'schema': 1, 'prefix': prefix, 'boot': boot, 'rank': rank,
             'root': str(root), 'container_name': name, 'image_id': IMAGE_IDS[rank],
             'engine_sha': REVISION, 'manifest': str(manifest_path),
             'manifest_sha256': hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
             'campaign': str(root / 'campaign-v1.json')}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return path


def stop_owned(manifest_path, rank, dry_run=False):
    """Safety-only shutdown survives revoked collection approval and source drift.

    A pending launcher can be cancelled before it has a launch receipt. A present
    container can only be stopped with the retained launch identity plus exact
    Docker name/image verification. Never source a config or load engine code.
    """
    root, prefix, boot, name = cleanup_identity(manifest_path, rank)
    receipt_path = owned(root / 'ownership' / f'{boot}-r{rank}.json', root)
    marker = owned(root / 'started' / f'{boot}-r{rank}', root)
    retained = None
    if receipt_path.exists():
        retained = json.loads(receipt_path.read_text())
        if type(retained.get('rank')) is not int or type(retained.get('schema')) is not int:
            raise ValueError('invalid retained launch identity types')
        expected = {'schema': 1, 'prefix': prefix, 'boot': boot, 'rank': rank,
                    'root': str(root), 'container_name': name, 'image_id': IMAGE_IDS[rank],
                    'engine_sha': REVISION, 'manifest': str(manifest_path),
                    'campaign': str(root / 'campaign-v1.json')}
        if any(retained.get(key) != value for key, value in expected.items()):
            raise ValueError('retained launch ownership/pins differ; refusing shutdown')
        if not re.fullmatch(r'[0-9a-f]{64}', retained.get('manifest_sha256', '')):
            raise ValueError('invalid retained launch manifest hash')
        if marker.read_text() != name + '\n':
            raise ValueError('retained launch marker differs')
    proof = {'command': 'stop', 'rank': rank, 'boot': boot, 'container_name': name,
             'image_id': IMAGE_IDS[rank], 'dry_run': dry_run,
             'ownership_receipt': str(receipt_path) if retained else None,
             'retained_launch': retained}
    # Current qualification is evidence only. A refusal never vetoes safety drain.
    try:
        require_collection_ready(str(root / 'campaign-v1.json'))
        current_sha = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
        if retained is not None and current_sha != retained['manifest_sha256']:
            raise ValueError('manifest drift since retained launch')
        proof['collection_qualification'] = {'passed': True}
    except Exception as error:
        proof['collection_qualification'] = {'passed': False, 'error': type(error).__name__ + ': ' + str(error)}
    if dry_run:
        proof['planned_argv'] = ['docker', 'stop', '--time', '120', name]
        proof['verdict'] = 'DRY_RUN'
        record_proof(root, proof)
        return 0
    try:
        cancel = owned(root / 'cancelled' / f'{boot}-r{rank}', root)
        cancel.parent.mkdir(parents=True, exist_ok=True)
        cancel.touch(exist_ok=True)  # Also cancels an admitted but not-yet-launched rank.
        container = inspect_container(name, IMAGE_IDS[rank])
        proof['present'] = container is not None
        if container is not None:
            if retained is None:
                raise ValueError('present container has no retained launch ownership')
            subprocess.run(['docker', 'stop', '--time', '120', name], check=True)
            remaining = inspect_container(name, IMAGE_IDS[rank])
            if remaining is not None:
                waited = subprocess.run(['docker', 'wait', name], capture_output=True,
                                        text=True, timeout=150)
                remaining = inspect_container(name, IMAGE_IDS[rank])
                if waited.returncode and remaining is not None:
                    raise RuntimeError('container wait failed: ' + waited.stderr.strip())
                if remaining is not None and remaining['State']['Running']:
                    raise ValueError('container has not drained')
        proof['verdict'] = 'DRAINED'
    except BaseException as error:
        proof.update(verdict='FAIL', error=repr(error))
        raise
    finally:
        record_proof(root, proof)
    return 0


def operate(command, manifest_path, rank, dry_run=False, admission_deadline=None):
    if command == 'stop':
        return stop_owned(manifest_path, rank, dry_run)
    manifest, record, meta, env, root, bench, name = context(manifest_path, rank,
        qualification_only=command=='verify' and dry_run)
    image = IMAGE_IDS[rank]
    proof = {'command': command, 'rank': rank, 'boot': manifest['phase2']['boot'],
             'container_name': name, 'image_id': image, 'dry_run': dry_run,
             'env': filtered(env), 'manifest_sha256': hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()}
    if command == 'start':
        if not dry_run and (type(admission_deadline) not in (int,float) or not math.isfinite(admission_deadline)):
            raise ValueError('operator-held admission deadline required for boot')
        if not dry_run and not 0<admission_deadline-time.time()<=15903:
            raise ValueError('boot admission deadline expired or exceeds frozen study interval')
        minimum = manifest['compute_dl']['boot_min_gib']
        if type(minimum) is not int or minimum <= 0:
            raise ValueError('explicit memory admission required')
        env['DSV41_BOOT_MIN_GIB'] = str(minimum)
        proof['boot_min_gib'] = minimum
        proof['admission_deadline_epoch']=admission_deadline
    cancel = owned(root / 'cancelled' / f'{manifest["phase2"]["boot"]}-r{rank}', root)
    if command == 'snapshot':
        if dry_run:
            proof['planned_argv'] = ['docker', 'container', 'inspect', name]
        else:
            container = inspect_container(name, image)
            proof['present'] = container is not None
            if command == 'snapshot' and (container is None or not container['State'].get('Running')):
                raise RuntimeError('snapshot requires the owned container to be present and running')
            if container is not None and command == 'snapshot':
                effective = dict(value.split('=', 1) for value in container['Config']['Env'] if '=' in value)
                proof['container_env'] = effective
                reject_banned(effective)
                native = engine_environment(name, rank)
                proof['native_process'] = native
                reject_banned(native['env'])
                exact_optional(meta['effective_env'], native['env'])
                for key, value in meta['expected_off'].items():
                    if effective.get(key) != value or native['env'].get(key) != value:
                        raise ValueError('expected-off assertion failed: ' + key)
                for key, value in meta['effective_env'].items():
                    if key.startswith(('TF_', 'GLM53_TF', 'NCCL')) and native['env'].get(key) != value:
                        raise ValueError('running container effective flag differs: ' + key)
    else:
        prepare(record, meta, env, rank, root)
        output = verify_original(manifest_path, rank, bench)
        # The D manifest selects its frozen native adapter, rank.py's original
        # equivalent of run-zig.sh --dry-run. Never synthesize a serving command.
        args, effective = docker_command(output, env, image, name)
        proof.update(argv=args, launch_command=shlex.join(args), container_env=effective)
        if command == 'start' and not dry_run:
            if cancel.exists():
                raise RuntimeError('pending boot was cancelled before start')
            if inspect_container(name, image) is not None:
                raise ValueError('owned container already exists')
            state = Path(env['STATE_DIR'])
            if set(p.name for p in state.iterdir()) != {'sessions'} or any((state / 'sessions').iterdir()):
                raise ValueError('state is not fresh for this boot')
            marker = owned(root / 'started' / f'{manifest["phase2"]["boot"]}-r{rank}', root)
            marker.parent.mkdir(parents=True, exist_ok=True)
            if marker.exists() or marker.is_symlink():
                raise ValueError('boot was already attempted; generate a new namespace')
            # Fail closed if the node's production guard is absent/incompatible.
            sys.path.insert(0, str(bench / 'ops'))
            saved = dict(os.environ)
            try:
                os.environ.clear()
                os.environ.update(env)
                guard = load_module('mem_guard', bench / 'ops/mem_guard.py')
                guard.before_boot(str(rank),lambda:cancel.exists() or time.time()>=admission_deadline)
            finally:
                os.environ.clear()
                os.environ.update(saved)
            with marker.open('x') as stream:
                stream.write(name + '\n')
            if cancel.exists() or time.time()>=admission_deadline:
                raise RuntimeError('pending boot was cancelled after admission')
            proof['ownership_receipt'] = str(launch_ownership(manifest_path, rank, root, manifest, name))
            record_proof(root, proof)
            if cancel.exists() or time.time()>=admission_deadline:
                raise RuntimeError('cancelled or admission deadline reached before Docker start')
            # Foreground only. Parent tracks this rank process; no background job.
            return subprocess.run(args, env=env, check=False).returncode
    record_proof(root, proof)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify', 'start', 'stop', 'snapshot'))
    parser.add_argument('manifest', type=Path)
    parser.add_argument('rank', type=int, choices=(0, 1))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--admission-deadline-epoch',type=float)
    args = parser.parse_args()
    raise SystemExit(operate(args.command, args.manifest, args.rank, args.dry_run,args.admission_deadline_epoch))



def exact_optional(expected,actual):
    select=lambda env:{k:v for k,v in env.items() if k.startswith(('TF_','GLM53_TF','NCCL'))}
    if select(expected)!=select(actual): raise ValueError('unexpected/missing/changed optional engine flag')

if __name__=='__main__': main()
