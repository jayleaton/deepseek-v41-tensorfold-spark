#!/usr/bin/env python3
"""CPU-STAGED owned copy of controller3; no current collection authorization.

Campaign contract for the parent's eventual spec adoption:
  prefix, spec, spec_sha256, arms (D/L, optional P), model_alias, boot_min_gib;
  cpu_receipt{,_sha256}, cpu_prompt_arrays{,_sha256}, code_sha256;
  freeze, freeze_sha256, layout={turns,boots}; rank_manifests and
  rank_manifest_sha256 indexed by boot then string rank;
  go_file (external only), tokenizer{,_sha256}, image_ids;
  timeouts={client_wait_s,gpu_parent_s,remote_control_s,turn_s};
  transport_contract={max_tokens,deadline_s}, diagnostic_tensorfold;
  g_restore_step={format:'g-restore-v1',events:['SERVING','MODELS200','POPPED']};
  budget={limit_seconds:18000,admission_seconds:15900}, budget_ledger on rank0-host.

Frozen manifest: schema=1, prefix, spec_sha256, turns=[{name,boots}],
boots=[{boot,arm,file,sha256}]. These arrays exactly equal campaign.layout.
Each boot file: boot, arm (D/L/P), requests, diagnostics. Templates are built
by request_adapter, with suite/id/seed/task_sha256 provenance added; diagnostic
templates also carry the spec's reference. CPU arrays retain A1 reconciliation
field names. No task counts, seeds, generation or grading rules are selected.

Rank manifests retain native arms.D and phase2 for reviewed drain compatibility.
phase2 adds runner_forms and guard_source; ranks[].expected_off explicitly maps
every generation-module ZERO_FLAGS control to '0'. compute_dl contains campaign
(rank-local path), spec_sha256, boot_min_gib. Each rank's gate returns its actual
campaign dict. Rank paths use <rank{rank}-work>/stdbench/compute-dl.

Transport compatibility and actual G step schema still need explicit acceptance
against the future spec. The parent protocol remains the sole approval gate.
"""
import argparse
import base64
import datetime
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
sys.dont_write_bytecode = True
import time
import threading
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from protocol import require_collection_ready, A1_TURNS
from budget import HeldBudget, BudgetExhausted
sys.path.append(str(HERE.parent.parent / 'decl-control'))
sys.path.insert(0, str(HERE.parent.parent / 'kit'))
import bench
import client3 as client
from reconciliation import reconcile, wire

QUEUE = Path('<rank0-work>/queue/ORDER')
QUIET = QUEUE.parent / 'QUIET'
BENCH = Path('<rank0-work>/dsv41-bench')
SERVE = Path('<rank0-work>/r5/cand-e/EFAST2-RUN')
RESTORE = Path('<rank0-work>/r5/ecal/g-restore.sh')
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3', 'root@<rank1-link-ip>']
# No spec SHA is invented here. The parent protocol is the authority.
ARM_NAMES = {'D': 'declared_off', 'L': 'lane', 'P': 'plain'}
STAGED_DEPENDENCIES = (
    'externally supplied spec SHA and all parent protocol gates',
    'frozen turns/boots/request files and rank manifests with hashes',
    'CPU prompt arrays, tokenizer, source seals and transport contract',
    'spec-bound diagnostic expectations, thermal memory admission and timeouts',
    'request_adapter-built frozen templates with provenance/diagnostic metadata',
)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def container_absent(stderr):
    return any(text in stderr.lower() for text in ('no such object:', 'no such container:'))


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x') as output:
        output.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def remote(args):
    return SSH + [shlex.join(list(map(str, args)))]


def remote_rank(args, seconds=7200):
    if seconds < 7200:
        raise ValueError('remote control timeout below required floor')
    command=['systemd-run','--user','--scope',
             '--unit=stdbench-compute-a1-rctl-'+str(time.time_ns()),
             '-p','MemoryMax=2G','-p','MemorySwapMax=0',
             'env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MAKEFLAGS=-j1',
             'nice','-n','19','ionice','-c3','timeout',str(seconds),*args]
    return remote(command)


def run(args, log=None, timeout=7200, **kwargs):
    if log:
        with Path(log).open('a') as output:
            return subprocess.run(list(map(str, args)), stdout=output, stderr=subprocess.STDOUT,
                                  check=True, timeout=timeout, **kwargs)
    return subprocess.run(list(map(str, args)), check=True, timeout=timeout, **kwargs)


def head():
    lines = QUEUE.read_text().splitlines()
    return lines[0] if lines else ''


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def reference_matches(call, result):
    try:
        reference_bytes = base64.b64decode(result['normalized_body_base64'], validate=True)
    except (KeyError, ValueError):
        return False
    return call.get('status') == result.get('status') and reference_bytes == call.get('output', '').encode()


class Turn:
    def __init__(self, campaign, name):
        config = require_collection_ready(campaign)
        self.path, self.cfg, self.turn = Path(campaign).resolve(), config, name
        self.spec_sha = self.cfg['spec_sha256']
        self.prefix = self.cfg['prefix']
        if name == 'A1-plain':
            prior = load(HERE / 'runs' / (self.prefix + '-A1-resume') / 'verdict.json')
            if prior.get('verdict') != 'QUALIFIED' or prior.get('spec_sha256') != self.spec_sha:
                raise ValueError('A1-resume must finish and restore G before A1-plain')
        if not re.fullmatch(r'stdbench-compute-a1-[a-zA-Z0-9_.-]+', self.prefix):
            raise ValueError('invalid prefix')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
            raise ValueError('invalid turn name')
        self.me = self.prefix + '-' + name
        self.out = HERE / 'runs' / self.me
        self.lease = self.lock = self.thermal = None
        self.ranks = []
        self.stopped_g, self.fenced = False, False
        self.error = None
        self.cleaning = self.lease_expected = self.lease_lost = False
        self.lease_watcher = None

    def health(self):
        if self.lease_expected and (self.lease_lost or self.lease.poll() is not None):
            raise RuntimeError('rank1-host GPU lease lost')

    def watch_lease(self):
        while self.lease_expected:
            if self.lease.poll() is not None:
                self.lease_lost = True
                if not self.cleaning:
                    os.kill(os.getpid(), signal.SIGUSR1)
                return
            time.sleep(0.5)

    def event(self, step, verdict, **detail):
        record = {'at': now(), 'step': step, 'verdict': verdict, **detail}
        with (self.out / 'steps.jsonl').open('a') as log:
            log.write(json.dumps(record, sort_keys=True) + '\n')
        print(json.dumps(record, sort_keys=True), flush=True)

    def validate(self):
        if require_collection_ready(self.path) != self.cfg:
            raise ValueError('campaign changed since approval gate')
        if not re.fullmatch(r'[0-9a-f]{64}', self.spec_sha) or sha(self.cfg['spec']) != self.spec_sha:
            raise ValueError('externally supplied spec SHA differs')
        receipt = load(self.cfg['cpu_receipt'])
        if sha(self.cfg['cpu_receipt']) != self.cfg['cpu_receipt_sha256'] or not receipt.get('passed'):
            raise ValueError('CPU prerequisites not passed')
        if receipt['spec_sha256'] != self.spec_sha:
            raise ValueError('CPU proof belongs to another protocol')
        if self.cfg.get('completed_boots'):
            raise ValueError('fresh campaign does not accept retained-boot substitutions')
        expected_code = {p.replace('<rank1-home>/','<rank0-home>/',1): h
                         for p,h in receipt['code_sha256'].items()}
        if expected_code != self.cfg['code_sha256']:
            raise ValueError('CPU code seal differs from campaign')
        self.prompts = {tuple(r['key']): r for r in map(json.loads, Path(self.cfg['cpu_prompt_arrays']).read_text().splitlines())}
        if sha(self.cfg['cpu_prompt_arrays']) != self.cfg['cpu_prompt_arrays_sha256']:
            raise ValueError('CPU prompt arrays drift')
        for path, digest in self.cfg['code_sha256'].items():
            if sha(path) != digest:
                raise ValueError('frozen code drift: ' + path)
        frozen = Path(self.cfg['freeze'])
        index = load(frozen / 'manifest.json')
        if sha(frozen / 'manifest.json') != self.cfg['freeze_sha256']:
            raise ValueError('cohort/request freeze drift')
        if receipt['freeze_sha256'] != self.cfg['freeze_sha256']:
            raise ValueError('CPU request freeze differs from campaign')
        if receipt['cpu_prompt_arrays_sha256'] != self.cfg['cpu_prompt_arrays_sha256']:
            raise ValueError('CPU arrays differ from campaign')
        if index['schema'] != 1 or index['prefix'] != self.prefix or index['spec_sha256'] != self.spec_sha:
            raise ValueError('frozen layout identity differs')
        if index['turns'] != self.cfg['layout']['turns'] or index['boots'] != self.cfg['layout']['boots']:
            raise ValueError('campaign layout differs from frozen layout')
        if len(index['turns']) < 2:
            raise ValueError('at least two frozen turns required')
        turn_names = [r['name'] for r in index['turns']]
        boot_names = [r['boot'] for r in index['boots']]
        if len(set(turn_names)) != len(turn_names) or len(set(boot_names)) != len(boot_names):
            raise ValueError('duplicate layout identity')
        assigned = [b for t in index['turns'] for b in t['boots']]
        if sorted(assigned) != sorted(boot_names):
            raise ValueError('each boot must belong to exactly one turn')
        arms = set(self.cfg['arms'])
        if arms != {'D', 'L', 'P'} or index['turns'] != A1_TURNS or len(boot_names)!=3:
            raise ValueError('frozen two-turn D/L-only layout required')
        for turn in index['turns']:
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', turn['name']):
                raise ValueError('unsafe turn name')
            if [b['arm'] for name in turn['boots'] for b in index['boots'] if b['boot'] == name] != (['L','D'] if turn['name'] == 'A1-resume' else ['P']):
                raise ValueError('turn does not cover configured arms')
        for key, floor in (('client_wait_s', 360), ('gpu_parent_s', 7200), ('remote_control_s', 7200)):
            if type(self.cfg['timeouts'][key]) is not int or self.cfg['timeouts'][key] < floor:
                raise ValueError('timeout below required floor: ' + key)
        if type(self.cfg['timeouts']['turn_s']) is not int or self.cfg['timeouts']['turn_s'] < self.cfg['timeouts']['gpu_parent_s']:
            raise ValueError('turn timeout must cover GPU parent timeout')
        from generate_envs3 import IMAGE_IDS
        if self.cfg['image_ids'] != list(IMAGE_IDS):
            raise ValueError('campaign image pins differ from A1')
        if type(self.cfg['boot_min_gib']) is not int or self.cfg['boot_min_gib'] <= 0:
            raise ValueError('spec-bound boot memory admission required')
        if not self.cfg['diagnostic_tensorfold']:
            raise ValueError('explicit diagnostic extension expectations required')
        if self.cfg['g_restore_step'] != {'format':'g-restore-v1','events':['SERVING','MODELS200','POPPED']}:
            raise ValueError('actual G restoration log contract differs')
        # Unchanged A1 Transport refuses any other contract. This is a source
        # compatibility gate, never permission to choose these values for a study.
        if self.cfg['transport_contract'] != {'max_tokens': 8192, 'deadline_s': 270}:
            raise ValueError('spec contract unsupported by unchanged A1 Transport')
        turns = [row for row in index['turns'] if row['name'] == self.turn]
        if len(turns) != 1:
            raise ValueError('unknown turn')
        self.boots = []
        boot_map = {row['boot']: row for row in index['boots']}
        for name in turns[0]['boots']:
            entry = boot_map[name]
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
                raise ValueError('unsafe boot name')
            path = frozen / entry['file']
            path.resolve().relative_to(frozen.resolve())
            if sha(path) != entry['sha256']:
                raise ValueError('request templates drift')
            boot = load(path)
            if boot['boot'] != name or boot['arm'] != entry['arm'] or boot['arm'] not in arms:
                raise ValueError('boot identity/arm differs')
            boot['arm'] = ARM_NAMES[boot['arm']]
            for template in boot['requests'] + boot['diagnostics']:
                if template['arm'] != boot['arm']:
                    raise ValueError('request arm differs from boot')
                if wire(template['body_template']) != template['wire'] or hashlib.sha256(template['wire'].encode()).hexdigest() != template['wire_sha256']:
                    raise ValueError('frozen body/wire drift')
            boot['rank_manifest'] = self.cfg['rank_manifests'][name]
            for rank in (0, 1):
                local_manifest = boot['rank_manifest'][str(rank)].replace('<rank1-home>/', '<rank0-home>/', 1)
                if sha(local_manifest) != self.cfg['rank_manifest_sha256'][name][str(rank)]:
                    raise ValueError('rank manifest drift')
            self.boots.append(boot)
        return self.boots

    def preflight(self):
        self.validate()
        # Each rank's verify checks its sealed config and pinned native source.
        run(['bash', str(QUEUE.parent / 'turn-preflight.sh'), str(HERE / 'turn.sh'), str(HERE / 'fence.sh')], self.out / 'preflight.log')
        for boot in self.boots:
            paths = boot['rank_manifest']
            for rank in (0, 1):
                path = paths[str(rank)]
                args = ['python3', '-B', str(HERE / 'rank_ops.py') if rank == 0 else '<rank1-work>/stdbench/compute-dl/a1-runtime/rank_ops.py',
                        'verify', path, str(rank), '--dry-run']
                run(args if rank == 0 else remote_rank(args, self.cfg['timeouts']['remote_control_s']), self.out / f'preflight-{boot["boot"]}-r{rank}.log', timeout=self.cfg['timeouts']['gpu_parent_s'])
        self.event('preflight', 'PASS', spec_sha256=self.spec_sha, frozen_requests=self.cfg['freeze_sha256'])

    def queue(self):
        lines = QUEUE.read_text().splitlines()
        if any(line.startswith(self.prefix + '-') for line in lines):
            raise ValueError('turn already queued; never duplicate active work')
        with QUEUE.open('a') as output:
            output.write(self.me + '\n')
        self.event('queue', 'APPENDED_AT_TAIL', name=self.me)
        until = time.monotonic() + 43200
        while head() != self.me:
            if time.monotonic() > until:
                raise TimeoutError('not at queue head within 12 hours; retained queued entry requires explicit withdrawal')
            time.sleep(5)
        self.event('queue', 'AT_HEAD')
        self.preflight()

    def stop_g(self):
        current = SERVE.read_text().strip()
        if not re.fullmatch(r'efast2-[A-Za-z0-9_.-]+', current):
            raise ValueError('unrecognized G serving run')
        status = load(BENCH / 'out' / current / 'status.json')
        if not status.get('serving') or status.get('engine_sha') != '58bece1d3184269059fd3f7f755c52f4f295113d':
            raise ValueError('only the expected G serving controller may be stopped')
        # The current G controller owns its own rank teardown and lock release.
        (BENCH / 'out' / current / 'STOP').touch(exist_ok=False)
        self.stopped_g = True
        self.event('G-stop', 'REQUESTED', serving_run=current)

    def acquire(self):
        lease_cmd = 'exec 9</tmp/rank1-host-gpu.lock || exit 3; flock 9 || exit 3; echo LOCKED; cat >/dev/null'
        self.lease = subprocess.Popen(SSH + ['bash -c ' + shlex.quote(lease_cmd)],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=(self.out / 'rank1-host-lease.log').open('a'), text=True)
        # readline is bounded by the enclosing turn timeout and SSH keepalive. Lock acquisition is after G's STOP.
        if self.lease.stdout.readline().strip() != 'LOCKED':
            raise RuntimeError('rank1-host root read-only lease failed')
        self.lease_expected = True
        self.lease_watcher = threading.Thread(target=self.watch_lease, daemon=True)
        self.lease_watcher.start()
        self.lock = open('/tmp/rank1-host-gpu.lock', 'a')
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        self.event('gpu-locks', 'HELD', order='rank1-host-readonly then rank0-host')
        run(['bash', str(HERE / 'fence.sh'), 'on'], self.out / 'fence.log')
        self.fenced = True
        self.event('traffic-fence', 'ON')

    def admission_remaining(self):
        self.budget.ensure_can_start()
        return max(0.001,15900-self.budget.elapsed())

    def admission_output(self,command,**kwargs):
        timeout=min(self.cfg['timeouts']['gpu_parent_s'],self.admission_remaining())
        try:
            return subprocess.check_output(command,timeout=timeout,**kwargs)
        except subprocess.TimeoutExpired:
            self.budget.ensure_can_start()
            raise

    def cool(self):
        limit = time.monotonic() + self.cfg['timeouts']['gpu_parent_s']
        while time.monotonic() < limit:
            self.budget.ensure_can_start()
            rows = []
            for rank in (0, 1):
                command = ['bash', str(BENCH / 'ops/node.sh'), 'sensor'] if rank == 0 else remote_rank(['bash', '<rank1-work>/dsv41-bench/ops/node.sh', 'sensor'], self.cfg['timeouts']['remote_control_s'])
                result = self.admission_output(command,text=True).strip()
                fields = result.split(',')
                hot, state, available, pause, resume, kill, hold = fields[-7:]
                rows.append(state.startswith('normal') and int(hot) <= int(resume) and int(available) >= self.cfg['boot_min_gib'])
            if all(rows):
                return
            time.sleep(min(10,self.admission_remaining()))
        raise TimeoutError('thermal/memory boot admission refused')

    def rank_action(self, boot, rank, action, **kwargs):
        script = str(HERE / 'rank_ops.py') if rank == 0 else '<rank1-work>/stdbench/compute-dl/a1-runtime/rank_ops.py'
        command = ['python3', '-B', script, action, boot['rank_manifest'][str(rank)], str(rank)]
        kwargs.setdefault('timeout', self.cfg['timeouts']['gpu_parent_s'])
        return run(command if rank == 0 else remote_rank(command, self.cfg['timeouts']['remote_control_s']), self.out / boot['boot'] / f'{action}-r{rank}.log', **kwargs)

    def boot(self, boot):
        self.budget.ensure_can_start()
        directory = self.out / boot['boot']
        directory.mkdir(exist_ok=False)
        self.cool()
        self.budget.ensure_can_start()
        before=time.time()
        remote_clock=float(self.admission_output(remote_rank(['date','+%s.%N'],self.cfg['timeouts']['remote_control_s']),
                                                  text=True).strip())
        after=time.time()
        skew_bounds=[remote_clock-after,remote_clock-before]
        if max(abs(v) for v in skew_bounds)>3:
            raise RuntimeError('rank1-host clock uncertainty exceeds conservative boot-admission margin')
        self.event('clock-admission','QUALIFIED',offset_bounds_seconds=skew_bounds)
        self.budget.ensure_can_start()
        # Both ranks check this absolute admission boundary after memory waiting.
        # Five seconds are conservative allowance for the CPU-verified clock skew.
        self.boot_admission_deadline=time.time()+max(0,15900-self.budget.elapsed()-5)
        self.thermal = subprocess.Popen(['python3', str(BENCH / 'ops/guard.py'), str(directory), str(os.getpid())],
                                        stdout=(directory / 'thermal.log').open('a'), stderr=subprocess.STDOUT)
        for rank in (1, 0):
            self.budget.ensure_can_start()
            script = str(HERE / 'rank_ops.py') if rank == 0 else '<rank1-work>/stdbench/compute-dl/a1-runtime/rank_ops.py'
            args = ['python3', '-B', script, 'start', boot['rank_manifest'][str(rank)], str(rank),
                    '--admission-deadline-epoch',str(self.boot_admission_deadline)]
            process = subprocess.Popen(args if rank == 0 else remote_rank(args, self.cfg['timeouts']['remote_control_s']),
                                       stdout=(directory / f'rank{rank}.log').open('a'), stderr=subprocess.STDOUT)
            self.ranks.append((rank, process))
        until = time.monotonic() + self.cfg['timeouts']['gpu_parent_s']
        while time.monotonic() < until:
            self.budget.ensure_can_start()
            self.health()
            if any(p.poll() is not None for _, p in self.ranks) or self.thermal.poll() is not None:
                if time.time()>=self.boot_admission_deadline:
                    raise BudgetExhausted('rank admission cutoff reached during memory wait')
                raise RuntimeError('rank/thermal process ended before readiness')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=min(3,self.admission_remaining())) as reply:
                    model = json.load(reply)
                if model['data'][0]['id'] != self.cfg['model_alias']:
                    raise ValueError('unexpected model alias')
                break
            except OSError:
                time.sleep(min(2,self.admission_remaining()))
        else:
            raise TimeoutError('boot readiness deadline')
        for rank in (0, 1):
            self.rank_action(boot, rank, 'snapshot')
        self.event('boot', 'READY_EFFECTIVE_ENV_VERIFIED', boot=boot['boot'])

    def clear_quiet(self):
        if QUIET.exists() and QUIET.read_text().startswith(self.me + ' '):
            QUIET.unlink()

    def measure_one(self, boot, template, diagnostic=False):
        self.budget.ensure_can_start()
        self.health()
        body = template['body_template']
        if wire(body) != template['wire'] or hashlib.sha256(template['wire'].encode()).hexdigest() != template['wire_sha256']:
            raise ValueError('body SHA differs from frozen request')
        if QUIET.exists():
            raise ValueError('another owner has QUIET')
        with QUIET.open('x') as file:
            file.write(self.me + ' ' + (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=self.cfg['timeouts']['client_wait_s'] + 30)).isoformat() + '\n')
        try:
            if {k: body.get(k) for k in self.cfg['transport_contract']} != self.cfg['transport_contract']:
                raise ValueError('request differs from spec-bound transport contract')
            transport = client.Transport('http://127.0.0.1:8000/v1/chat/completions', self.cfg['go_file'], self.cfg['spec'], self.spec_sha, self.cfg['timeouts']['client_wait_s'])
            self.budget.ensure_can_start()
            if not diagnostic:
                attempt_dir = HERE / 'attempted' / self.prefix
                attempt_dir.mkdir(parents=True, exist_ok=True)
                attempt_key = hashlib.sha256(client.canonical([boot['arm'], *template['key']]).encode()).hexdigest()
                save(attempt_dir / (attempt_key + '.json'), {'boot':boot['boot'], 'key':template['key'], 'arm':boot['arm'], 'wire_sha256':template['wire_sha256'], 'at':now()})
            row = transport(body)
        finally:
            self.clear_quiet()
        # Raw observations survive adapter/accounting refusals as well.
        with (self.out / boot['boot'] / 'raw-observations.jsonl').open('a') as output:
            output.write(client.canonical({'template_key': template['key'], 'diagnostic': diagnostic, 'response': row}) + '\n')
        row.update(suite=template['suite'], id=template['id'], seed=template['seed'], arm=boot['arm'],
                   task_sha256=template['task_sha256'], request=body, boot=boot['boot'], diagnostic=diagnostic)
        row['execution_policy'] = template['execution_policy']
        decoder = client.NativeDecoder(self.cfg['tokenizer'], self.cfg['tokenizer_sha256'])
        from request_adapter import classify_observation
        row = classify_observation(row, template, self.prompts[tuple(template['key'])], decoder)
        binding = row['native_binding']
        row['tokens'] = bench.usage_counts(row.get('usage'))
        directory = self.out / boot['boot']
        target = directory / ('diagnostics.jsonl' if diagnostic else 'responses.jsonl')
        with target.open('a') as file:
            file.write(client.canonical(row) + '\n')
        if not row['initial_prompt_reconciliation']['qualified'] or not row['native_accounting']['complete']:
            raise RuntimeError('prompt/returned-ID reconciliation failed; observation retained')
        if boot['arm'] in ('plain', 'declared_off') and ((row.get('tensorfold') or {}).get('compute') or {}).get('calls'):
            raise RuntimeError('plain made native calls; observation retained, qualification stopped')
        # The reply is durable before any safety/reference checker is invoked.
        from jail import Jail
        delivered=[c for c in ((row.get('tensorfold') or {}).get('compute') or {}).get('calls',[]) if c.get('inserted')]
        must_authenticate=(row['status'] in ('ok','policy_terminated') or delivered
                           or binding.get('native_frame_count',0)>0)
        if must_authenticate and not binding['qualified']:
            failed={'binding':binding,'references':[],'qualified':False,
                    'reason':'native provenance/accounting failed before any independent replay'}
            with (directory/'safety.jsonl').open('a') as output:
                output.write(client.canonical(dict(suite=row['suite'],id=row['id'],seed=row['seed'],
                                                   arm=row['arm'],boot=boot['boot'],**failed))+'\n')
            raise RuntimeError('native provenance/accounting gate; response retained')
        references = []
        for call in ((row.get('tensorfold') or {}).get('compute') or {}).get('calls', []):
            if not call.get('inserted'):
                continue
            try:
                result = Jail(self.cfg['image_ids'][0]).run_reference(call['code'])
                from clipping import compare
                reference_ok, presentation = compare(self.cfg, call, result)
            except BaseException as error:
                references.append({'call_id':call.get('id'), 'matches':False,
                                   'qualification_error':repr(error)})
                with (directory / 'safety.jsonl').open('a') as output:
                    output.write(client.canonical({'suite':row['suite'], 'id':row['id'],
                        'seed':row['seed'], 'arm':row['arm'], 'boot':boot['boot'],
                        'binding':binding, 'references':references, 'qualified':False})+'\n')
                raise
            references.append({'call_id': call.get('id'), 'result': result, 'matches': reference_ok, 'presentation': presentation})
        safety = {'binding': binding, 'references': references,
                  'qualified': binding['qualified'] and all(x['matches'] for x in references)}
        with (directory / 'safety.jsonl').open('a') as file:
            file.write(client.canonical({'suite': row['suite'], 'id': row['id'], 'seed': row['seed'],
                                         'arm': row['arm'], 'boot': boot['boot'], **safety}) + '\n')
        if not safety['qualified'] and must_authenticate:
            raise RuntimeError('tool-result safety/reference gate')
        return row, safety

    def diagnostics(self, boot):
        calls = 0
        for template in boot['diagnostics']:
            row, safety = self.measure_one(boot, template, True)
            tf = row.get('tensorfold') or {}
            if (row['status'] not in ('ok','policy_terminated') or row.get('transport_status') != 'ok'
                    or not row['native_accounting']['complete']
                    or not row['initial_prompt_reconciliation']['qualified_under_proposed_count_contract']):
                raise RuntimeError('diagnostic protocol/accounting failure')
            if any(tf.get(k) != v for k, v in self.cfg['diagnostic_tensorfold'].items()):
                raise RuntimeError('server diagnostic extension differs from spec')
            calls += safety['binding'].get('delivered_successes', 0)
            if boot['arm'] == 'lane':
                successful = [call for call in (tf.get('compute') or {}).get('calls', [])
                              if call.get('inserted') and call.get('status') == 'ok']
                if not successful or not any(call.get('output') == str(template['reference']) for call in successful):
                    raise RuntimeError('diagnostic delivered output differs from fixed reference')
            if boot['arm'] in ('plain', 'declared_off') and ((tf.get('compute') or {}).get('calls')):
                raise RuntimeError('plain made lane calls')
        okay = calls > 0 if boot['arm'] == 'lane' else calls == 0
        self.event('diagnostics', 'QUALIFIED' if okay else 'FAIL', boot=boot['boot'], delivered_successes=calls)
        if not okay:
            raise RuntimeError('native activation gate failed')

    def stop_boot(self, boot):
        self.clear_quiet()
        errors = []
        if self.thermal:
            try:
                self.thermal.terminate()
                self.thermal.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.thermal.kill()  # Only the exact owned thermal helper PID.
                self.thermal.wait()
            except ProcessLookupError:
                pass
            except BaseException as error:
                errors.append('thermal helper cleanup: ' + repr(error))
            self.thermal = None
        for rank in (0, 1):
            try:
                self.rank_action(boot, rank, 'stop')
            except BaseException as error:
                errors.append(str(error))
        unsettled = []
        for rank, process in self.ranks:
            try:
                process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                errors.append(f'rank {rank} launcher did not settle')
                unsettled.append((rank, process))
        self.ranks = unsettled
        if errors:
            raise RuntimeError('; '.join(errors))
        self.event('boot-stop', 'DRAINED', boot=boot['boot'])

    def require_drained(self, boot):
        self.ranks = [(rank, process) for rank, process in self.ranks if process.poll() is None]
        if self.ranks:
            raise RuntimeError('pending rank launchers have not settled; retain ownership')
        meta = load(boot['rank_manifest']['0'])['phase2']['ranks']
        for rank in (0, 1):
            command = ['docker', 'inspect', '--format', '{{.State.Running}}', meta[rank]['container_name']]
            result = subprocess.run(command if rank == 0 else remote_rank(command, self.cfg['timeouts']['remote_control_s']), capture_output=True, text=True, timeout=self.cfg['timeouts']['gpu_parent_s'])
            if result.returncode == 0 and result.stdout.strip() == 'true':
                raise RuntimeError(f'owned rank {rank} remains running; retain ownership for recovery')
            if result.returncode and not container_absent(result.stderr):
                raise RuntimeError(f'cannot verify owned rank {rank} drain: ' + result.stderr.strip())
        self.event('owned-containers', 'DRAIN_VERIFIED', boot=boot['boot'])

    def release(self):
        self.clear_quiet()
        errors = []
        if self.fenced:
            try:
                run(['bash', str(HERE / 'fence.sh'), 'off'], self.out / 'fence.log')
                self.fenced = False
            except BaseException as error:
                errors.append(repr(error))
        if self.lock:
            try:
                fcntl.flock(self.lock, fcntl.LOCK_UN)
            except OSError as error:
                errors.append('rank0-host unlock: ' + repr(error))
            finally:
                self.lock.close()
            self.lock = None
        if self.lease:
            if self.lease_lost or self.lease.poll() is not None:
                errors.append('rank1-host lease was lost before intentional release')
            self.lease_expected = False
            try:
                self.lease.stdin.close()
            except BrokenPipeError:
                pass
            try:
                self.lease.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.lease.terminate()
                self.lease.wait(timeout=30)
            if self.lease_watcher:
                self.lease_watcher.join(timeout=2)
            self.lease = None
        self.event('gpu-locks', 'RELEASED')
        return errors

    def restore(self):
        if not self.stopped_g:
            return
        if head() != self.me:
            raise RuntimeError('lost own queue head; refusing to mutate another turn')
        if 'engine-Gr' in QUEUE.read_text().splitlines():
            raise RuntimeError('another G restore entry already exists; retain own head')
        run(['sed', '-i', '1a engine-Gr', str(QUEUE)])
        run(['sed', '-i', '1d', str(QUEUE)])
        run(['bash', str(RESTORE)], self.out / 'G-restore.log', timeout=self.cfg['timeouts']['gpu_parent_s'])
        text = (self.out / 'G-restore.log').read_text()
        current = SERVE.read_text().strip()
        state = load(BENCH / 'out' / current / 'status.json')
        if 'POPPED' not in text or not state.get('serving') or state.get('engine_sha') != '58bece1d3184269059fd3f7f755c52f4f295113d':
            raise RuntimeError('G step/status did not demonstrate restoration')
        step_path=self.out/'G-restore.log'
        actual_steps=[]
        for line in text.splitlines():
            if re.match(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ SERVING ',line):
                if current not in line or 'EF_SHA=58bece1d3184269059fd3f7f755c52f4f295113d' not in line:
                    raise RuntimeError('G serving log identifies another run/source')
                actual_steps.append({'step':'SERVING','line':line})
            elif re.match(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ /v1/models 200$',line):
                actual_steps.append({'step':'MODELS200','line':line})
            elif re.match(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ POPPED; ORDER:',line):
                actual_steps.append({'step':'POPPED','line':line})
        if [s['step'] for s in actual_steps]!=self.cfg['g_restore_step']['events']:
            raise RuntimeError('actual G restoration serving/models/pop steps missing or out of order')
        expected = load(BENCH / 'arms-gx.json')['arms']['D']['ranks']
        for rank in (0, 1):
            command = ['docker', 'inspect', '--format', '{{.Image}} {{.State.Running}}', f'tensorfold-zig-r{rank}']
            actual = subprocess.check_output(command if rank == 0 else remote_rank(command, self.cfg['timeouts']['remote_control_s']), text=True, timeout=self.cfg['timeouts']['gpu_parent_s']).strip()
            if actual != expected[rank]['image_id'] + ' true':
                raise RuntimeError('restored G container image/running state differs')
        self.event('G-restore', 'SERVING_VERIFIED', serving_run=current,
                   actual_step=actual_steps[-1], actual_step_path=str(step_path), actual_step_sha256=sha(step_path))
        self.stopped_g = False

    def execute(self, preflight_only=False):
        require_collection_ready(self.path)
        self.validate()  # Before even a preflight output directory is created.
        if preflight_only:
            self.out = HERE / 'preflight-runs' / (self.me + '-' + str(time.time_ns()))
        self.out.mkdir(parents=True, exist_ok=False)
        self.preflight()
        if preflight_only:
            return
        self.budget=HeldBudget(self.cfg['budget_ledger'],self.spec_sha,self.turn, original_path=self.cfg['original_budget'], original_sha=self.cfg['original_budget_sha256'])
        self.incomplete=False
        try:
            self.budget.ensure_can_start()
        except BudgetExhausted as error:
            self.event('collection','INCOMPLETE',reason=str(error))
            save(self.out/'verdict.json',{'verdict':'INCOMPLETE','error':None,'spec_sha256':self.spec_sha,'reason':str(error)})
            self.event('turn','INCOMPLETE')
            return
        try:
            self.queue()
        except BaseException:
            if head() == self.me:
                run(['sed', '-i', '1d', str(QUEUE)])
            else:
                run(['sed', '-i', '/^' + re.escape(self.me) + '$/d', str(QUEUE)])
            self.event('queue', 'OWN_ENTRY_WITHDRAWN_G_UNCHANGED')
            raise
        active = None
        try:
            self.budget.begin()
            self.stop_g()
            self.acquire()
            for boot in self.boots:
                active = boot
                self.boot(boot)
                self.diagnostics(boot)
                for request in boot['requests']:
                    row, safety = self.measure_one(boot, request)
                    self.event('response', 'RECORDED', boot=boot['boot'], suite=row['suite'], id=row['id'], seed=row['seed'],
                               status=row['status'], elapsed_seconds=round(row['elapsed_seconds'], 3),
                               native_accounting_complete=row['native_accounting']['complete'])
                self.stop_boot(boot)
                self.require_drained(boot)
                active = None
            self.event('collection', 'COMPLETE')
        except BudgetExhausted as error:
            self.incomplete=True
            self.event('collection','INCOMPLETE',reason=str(error),held_seconds=self.budget.elapsed())
        except BaseException as error:
            self.error = repr(error)
            self.event('collection', 'STOPPED', reason=self.error)
        finally:
            self.cleaning = True
            if active:
                try:
                    self.stop_boot(active)
                except BaseException as error:
                    self.error = str(error)
                    self.event('boot-stop', 'FAIL', reason=str(error))
                # Recovery is foreground and bounded; never hand a live owned rank to the next queue entry.
                for attempt in range(120):
                    try:
                        self.require_drained(active)
                        break
                    except BaseException as error:
                        self.event('recovery-drain', 'RETRY', attempt=attempt, reason=str(error))
                        for rank in (0, 1):
                            try:
                                self.rank_action(active, rank, 'stop')
                            except BaseException:
                                pass
                        time.sleep(5)
                else:
                    raise RuntimeError('owned ranks could not be drained; queue head retained, recovery required')
            try:
                release_errors = self.release()
                if release_errors:
                    self.error = '; '.join(release_errors)
            finally:
                if self.stopped_g:
                    self.restore()
                else:
                    # No STOP was requested: G is unchanged, so withdraw only
                    # this controller's exact entry rather than stranding head.
                    run(['sed','-i','/^'+re.escape(self.me)+'$/d',str(QUEUE)])
                    self.event('queue','OWN_ENTRY_WITHDRAWN_G_UNCHANGED')
                if not self.stopped_g:
                    self.budget.finish_after_verified_restore()
                    self.event('budget','RECORDED',held_seconds=self.budget.data['held_seconds'],
                               overrun_seconds=self.budget.data.get('overrun_seconds',0))
        verdict = 'STOPPED' if self.error is not None else 'INCOMPLETE' if self.incomplete else 'QUALIFIED'
        save(self.out / 'verdict.json', {'verdict': verdict, 'error': self.error, 'spec_sha256': self.spec_sha})
        self.event('turn', verdict)
        if self.error:
            raise RuntimeError(self.error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('campaign')
    parser.add_argument('turn')
    parser.add_argument('--protocol-check', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    turn = Turn(args.campaign, args.turn)
    if args.protocol_check:
        turn.validate()
        print(turn.me, turn.cfg['timeouts']['turn_s'])
        return
    def terminate(_signal, _frame):
        if not turn.cleaning:
            raise RuntimeError('thermal/external/lease termination; restore G')
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGUSR1, terminate)
    turn.execute(args.preflight_only)


if __name__ == '__main__':
    main()
