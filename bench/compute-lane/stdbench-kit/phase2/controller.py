#!/usr/bin/env python3
"""One foreground, spec-bound queued turn. No build or publication; restores G before settling."""
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
import time
import threading
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'kit'))
import bench
import client

QUEUE = Path('<rank0-work>/queue/ORDER')
QUIET = QUEUE.parent / 'QUIET'
BENCH = Path('<rank0-work>/dsv41-bench')
SERVE = Path('<rank0-work>/r5/cand-e/EFAST2-RUN')
RESTORE = Path('<rank0-work>/r5/ecal/g-restore.sh')
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3', 'root@<rank1-link-ip>']
SPEC_SHA = '9c7327007cfd15825a608f2f6b06f2b39e33b88548c5bb7deb1b209b7395d240'


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def container_absent(stderr):
    return any(text in stderr.lower() for text in ('no such object:', 'no such container:'))


def completed_boot(boot, receipt):
    """Resume only a complete frozen boot; never retry a recorded bad sample."""
    directory = Path(receipt['directory'])
    for name, digest in receipt['sha256'].items():
        if sha(directory / name) != digest:
            raise ValueError('retained boot evidence drift: ' + name)
    rows = lambda name: [json.loads(line) for line in (directory/name).read_text().splitlines()]
    observations = rows('responses.jsonl')
    expected = {(r['suite'],r['id'],r['seed']):r for r in boot['requests']}
    actual = {(r['suite'],r['id'],r['seed']):r for r in observations}
    if len(observations) != len(expected) or set(actual) != set(expected):
        raise ValueError('retained boot is incomplete/duplicate')
    diagnostics = rows('diagnostics.jsonl')
    if len(diagnostics) != len(boot['diagnostics']):
        raise ValueError('retained boot diagnostics incomplete')
    for row in observations + diagnostics:
        if row['arm'] != boot['arm'] or row['boot'] != boot['boot']:
            raise ValueError('retained boot identity differs')
        template = next((r for r in boot['requests']+boot['diagnostics'] if
                         (r['suite'],r['id'],r['seed']) == (row['suite'],row['id'],row['seed'])), None)
        if template is None or hashlib.sha256(client.canonical(row['request']).encode()).hexdigest() != template['body_sha256']:
            raise ValueError('retained request differs from freeze')
        if not client.accounting(row)['complete']:
            raise ValueError('retained boot lacks complete native accounting')
        if boot['arm'] == 'plain' and ((row.get('tensorfold') or {}).get('compute') or {}).get('calls'):
            raise ValueError('retained plain boot made calls')
    safety = rows('safety.jsonl')
    if len(safety) != len(observations)+len(diagnostics) or not all(r.get('qualified') is True for r in safety):
        raise ValueError('retained boot lacks complete safety qualification')
    return True


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x') as output:
        output.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def remote(args):
    return SSH + [shlex.join(list(map(str, args)))]


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
        self.path, self.cfg, self.turn = Path(campaign).resolve(), load(campaign), name
        self.prefix = self.cfg['prefix']
        if not re.fullmatch(r'stdbench-[a-zA-Z0-9_.-]+', self.prefix):
            raise ValueError('invalid prefix')
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
        if self.cfg['spec_sha256'] != SPEC_SHA or sha(self.cfg['spec']) != SPEC_SHA:
            raise ValueError('externally supplied spec SHA differs')
        if not load(self.cfg['cpu_receipt']).get('passed'):
            raise ValueError('CPU prerequisites not passed')
        for path, digest in self.cfg['code_sha256'].items():
            if sha(path) != digest:
                raise ValueError('frozen code drift: ' + path)
        frozen = Path(self.cfg['freeze'])
        index = load(frozen / 'manifest.json')
        if sha(frozen / 'manifest.json') != self.cfg['freeze_sha256']:
            raise ValueError('cohort/request freeze drift')
        turns = [row for row in index['turns'] if row['name'] == self.turn]
        if len(turns) != 1:
            raise ValueError('unknown turn')
        self.boots = []
        boot_map = {row['boot']: row for row in index['boots']}
        for name in turns[0]['boots']:
            entry = boot_map[name]
            path = frozen / entry['file']
            if sha(path) != entry['sha256']:
                raise ValueError('request templates drift')
            boot = load(path)
            boot['rank_manifest'] = self.cfg['rank_manifests'][name]
            retained = self.cfg.get('completed_boots', {}).get(name)
            if retained and completed_boot(boot, retained):
                self.event('resume', 'RETAINED_COMPLETE_BOOT', boot=name, directory=retained['directory'])
                continue
            self.boots.append(boot)
        if self.turn != 'pilot':
            previous = load(self.cfg['pilot_verdict'])
            if previous.get('verdict') != 'QUALIFIED':
                raise ValueError('activation pilot is not qualified')
        return self.boots

    def preflight(self):
        self.validate()
        run(['bash', str(QUEUE.parent / 'turn-preflight.sh'), str(HERE / 'turn.sh'), str(HERE / 'fence.sh')], self.out / 'preflight.log')
        for boot in self.boots:
            paths = boot['rank_manifest']
            for rank in (0, 1):
                path = paths[str(rank)]
                args = ['python3', '-B', str(HERE / 'rank_ops.py') if rank == 0 else '<rank1-work>/stdbench/phase2/rank_ops.py',
                        'verify', path, str(rank), '--dry-run']
                run(args if rank == 0 else remote(args), self.out / f'preflight-{boot["boot"]}-r{rank}.log')
        self.event('preflight', 'PASS', spec_sha256=SPEC_SHA, frozen_requests=self.cfg['freeze_sha256'])

    def queue(self):
        lines = QUEUE.read_text().splitlines()
        if self.me in lines:
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

    def cool(self):
        limit = time.monotonic() + 7200
        while time.monotonic() < limit:
            rows = []
            for rank in (0, 1):
                command = ['bash', str(BENCH / 'ops/node.sh'), 'sensor'] if rank == 0 else remote(['bash', '<rank1-work>/dsv41-bench/ops/node.sh', 'sensor'])
                result = subprocess.check_output(command, text=True, timeout=30).strip()
                fields = result.split(',')
                hot, state, available, pause, resume, kill, hold = fields[-7:]
                rows.append(state.startswith('normal') and int(hot) <= int(resume) and int(available) >= 114)
            if all(rows):
                return
            time.sleep(10)
        raise TimeoutError('thermal/memory boot admission refused')

    def rank_action(self, boot, rank, action, **kwargs):
        script = str(HERE / 'rank_ops.py') if rank == 0 else '<rank1-work>/stdbench/phase2/rank_ops.py'
        command = ['python3', '-B', script, action, boot['rank_manifest'][str(rank)], str(rank)]
        return run(command if rank == 0 else remote(command), self.out / boot['boot'] / f'{action}-r{rank}.log', **kwargs)

    def boot(self, boot):
        directory = self.out / boot['boot']
        directory.mkdir(exist_ok=False)
        self.cool()
        self.thermal = subprocess.Popen(['python3', str(BENCH / 'ops/guard.py'), str(directory), str(os.getpid())],
                                        stdout=(directory / 'thermal.log').open('a'), stderr=subprocess.STDOUT)
        for rank in (1, 0):
            script = str(HERE / 'rank_ops.py') if rank == 0 else '<rank1-work>/stdbench/phase2/rank_ops.py'
            args = ['python3', '-B', script, 'start', boot['rank_manifest'][str(rank)], str(rank)]
            process = subprocess.Popen(args if rank == 0 else remote(args),
                                       stdout=(directory / f'rank{rank}.log').open('a'), stderr=subprocess.STDOUT)
            self.ranks.append((rank, process))
        until = time.monotonic() + 7200
        while time.monotonic() < until:
            self.health()
            if any(p.poll() is not None for _, p in self.ranks) or self.thermal.poll() is not None:
                raise RuntimeError('rank/thermal process ended before readiness')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=3) as reply:
                    model = json.load(reply)
                if model['data'][0]['id'] != 'DeepSeek-V4.1-Flash-TF':
                    raise ValueError('unexpected model alias')
                break
            except OSError:
                time.sleep(2)
        else:
            raise TimeoutError('boot readiness deadline')
        for rank in (0, 1):
            self.rank_action(boot, rank, 'snapshot')
        self.event('boot', 'READY_EFFECTIVE_ENV_VERIFIED', boot=boot['boot'])

    def clear_quiet(self):
        if QUIET.exists() and QUIET.read_text().startswith(self.me + ' '):
            QUIET.unlink()

    def measure_one(self, boot, template, diagnostic=False):
        self.health()
        body = template['body_template']
        if hashlib.sha256(client.canonical(body).encode()).hexdigest() != template['body_sha256']:
            raise ValueError('body SHA differs from frozen request')
        if QUIET.exists():
            raise ValueError('another owner has QUIET')
        with QUIET.open('x') as file:
            file.write(self.me + ' ' + (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=390)).isoformat() + '\n')
        try:
            transport = client.Transport('http://127.0.0.1:8000/v1/chat/completions', self.cfg['go_file'], self.cfg['spec'], SPEC_SHA, 360)
            row = transport(body)
        finally:
            self.clear_quiet()
        row.update(suite=template['suite'], id=template['id'], seed=template['seed'], arm=boot['arm'],
                   task_sha256=template['task_sha256'], request=body, boot=boot['boot'], diagnostic=diagnostic)
        row['tokens'] = bench.usage_counts(row.get('usage'))
        directory = self.out / boot['boot']
        target = directory / ('diagnostics.jsonl' if diagnostic else 'responses.jsonl')
        with target.open('a') as file:
            file.write(client.canonical(row) + '\n')
        if boot['arm'] == 'plain' and ((row.get('tensorfold') or {}).get('compute') or {}).get('calls'):
            raise RuntimeError('plain made native calls; observation retained, qualification stopped')
        # The reply is durable before any safety/reference checker is invoked.
        from jail import Jail
        decoder = client.NativeDecoder(self.cfg['tokenizer'], self.cfg['tokenizer_sha256'])
        binding = client.authenticate_calls(row, decoder)
        if row['status'] == 'ok' and not binding['qualified']:
            raise RuntimeError('native accounting/provenance gate: ' + binding['reason'])
        references = []
        for call in ((row.get('tensorfold') or {}).get('compute') or {}).get('calls', []):
            if not call.get('inserted'):
                continue
            result = Jail(self.cfg['image_ids'][0]).run_reference(call['code'])
            reference_ok = reference_matches(call, result)
            references.append({'call_id': call.get('id'), 'result': result, 'matches': reference_ok})
        safety = {'binding': binding, 'references': references,
                  'qualified': binding['qualified'] and all(x['matches'] for x in references)}
        with (directory / 'safety.jsonl').open('a') as file:
            file.write(client.canonical({'suite': row['suite'], 'id': row['id'], 'seed': row['seed'],
                                         'arm': row['arm'], 'boot': boot['boot'], **safety}) + '\n')
        if not safety['qualified'] and row['status'] == 'ok':
            raise RuntimeError('tool-result safety/reference gate')
        return row, safety

    def diagnostics(self, boot):
        calls = 0
        for template in boot['diagnostics']:
            row, safety = self.measure_one(boot, template, True)
            tf = row.get('tensorfold') or {}
            if row['status'] != 'ok' or not row['native_accounting']['complete']:
                raise RuntimeError('diagnostic protocol/accounting failure')
            if tf.get('enable_thinking') is not True or tf.get('reasoning_effort') != 'high':
                raise RuntimeError('server did not accept thinking/high extension')
            calls += safety['binding'].get('delivered_successes', 0)
            if boot['arm'] == 'lane':
                successful = [call for call in (tf.get('compute') or {}).get('calls', [])
                              if call.get('inserted') and call.get('status') == 'ok']
                if successful and not any(call.get('output') == str(template['reference']) for call in successful):
                    raise RuntimeError('diagnostic delivered output differs from fixed reference')
            if boot['arm'] == 'plain' and ((tf.get('compute') or {}).get('calls')):
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
            result = subprocess.run(command if rank == 0 else remote(command), capture_output=True, text=True, timeout=30)
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
        run(['bash', str(RESTORE)], self.out / 'G-restore.log', timeout=7200)
        text = (self.out / 'G-restore.log').read_text()
        current = SERVE.read_text().strip()
        state = load(BENCH / 'out' / current / 'status.json')
        if 'POPPED' not in text or not state.get('serving') or state.get('engine_sha') != '58bece1d3184269059fd3f7f755c52f4f295113d':
            raise RuntimeError('G step/status did not demonstrate restoration')
        expected = load(BENCH / 'arms-gx.json')['arms']['D']['ranks']
        for rank in (0, 1):
            command = ['docker', 'inspect', '--format', '{{.Image}} {{.State.Running}}', f'tensorfold-zig-r{rank}']
            actual = subprocess.check_output(command if rank == 0 else remote(command), text=True, timeout=30).strip()
            if actual != expected[rank]['image_id'] + ' true':
                raise RuntimeError('restored G container image/running state differs')
        self.event('G-restore', 'SERVING_VERIFIED', serving_run=current)
        self.stopped_g = False

    def execute(self, preflight_only=False):
        if preflight_only:
            self.out = HERE / 'preflight-runs' / (self.me + '-' + str(time.time_ns()))
        self.out.mkdir(parents=True, exist_ok=False)
        self.preflight()
        if preflight_only:
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
                self.restore()
        verdict = 'QUALIFIED' if self.error is None else 'STOPPED'
        save(self.out / 'verdict.json', {'verdict': verdict, 'error': self.error, 'spec_sha256': SPEC_SHA})
        self.event('turn', verdict)
        if self.turn == 'pilot' and verdict == 'QUALIFIED':
            save(self.cfg['pilot_verdict'], {'verdict': 'QUALIFIED', 'steps': str(self.out / 'steps.jsonl'), 'spec_sha256': SPEC_SHA})
        if self.error:
            raise RuntimeError(self.error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('campaign')
    parser.add_argument('turn', choices=['pilot'] + [f't{i:02}' for i in range(1, 7)])
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    turn = Turn(args.campaign, args.turn)
    def terminate(_signal, _frame):
        if not turn.cleaning:
            raise RuntimeError('thermal/external/lease termination; restore G')
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGUSR1, terminate)
    turn.execute(args.preflight_only)


if __name__ == '__main__':
    main()
