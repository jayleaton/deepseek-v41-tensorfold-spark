#!/usr/bin/env python3
"""Native pair lifecycle; CONFIG must already be sourced and exported by serve.sh."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


class OpsError(Exception):
    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


def durable_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with temporary.open('w') as stream:
            json.dump(value, stream)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def http_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


class Lifecycle:
    def __init__(self, env=None, runner=subprocess.run, clock=time.monotonic,
                 sleep=time.sleep, http=http_json, root=ROOT, wall=time.time):
        self.env = dict(os.environ if env is None else env)
        self.runner, self.clock, self.sleep, self.http, self.wall = runner, clock, sleep, http, wall
        self.root = Path(root)
        self.worker = self.env.get('WORKER_SSH', '')
        if not self.worker or self.worker.startswith('-') or any(c.isspace() for c in self.worker):
            raise OpsError('WORKER_SSH must be a single SSH target, not an option', 2)
        for key in ('WORKER_REPO', 'WORKER_CONFIG', 'STATE_DIR'):
            if not Path(self.env.get(key, '')).is_absolute():
                raise OpsError(f'{key} must be absolute', 2)
        self.prefix = self.env.get('CONTAINER_PREFIX', 'tensorfold-zig')
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', self.prefix):
            raise OpsError('invalid CONTAINER_PREFIX', 2)
        self.revision = (self.root / 'config/tensorfold.rev').read_text().strip()
        if not re.fullmatch(r'[0-9a-f]{40}', self.revision):
            raise OpsError('invalid engine revision', 2)
        state = Path(self.env['STATE_DIR']) / 'native-ops'
        self.stop_file = Path(self.env.get('STOP_FILE', str(state / 'STOP')))
        self.lock_file = Path(self.env.get('LIFECYCLE_LOCK', str(state / 'lifecycle.lock')))
        self.watch_file = state / 'watch.json'
        lease = self.env.get('MAINTENANCE_LEASE', '')
        self.lease = Path(lease) if lease else None
        self.ready_timeout = self.number('READY_TIMEOUT', 900)
        self.command_timeout = self.number('OPS_COMMAND_TIMEOUT', 30)
        drain_budget = self.number('TF_DSV41_DRAIN_S', 20) + 60
        self.stop_timeout = math.ceil(max(80, drain_budget, self.number('STOP_TIMEOUT', 80)))
        self.http_timeout = self.number('HTTP_TIMEOUT', 5)
        self.interval = self.number('WATCH_INTERVAL', 10)
        self.fails = self.number('WATCH_FAILS', 3)
        self.cooldown = self.number('WATCH_MIN_HEAL', 300)
        self.deadline = None
        self.base_url = self.env.get('HEALTH_URL', f"http://127.0.0.1:{self.env.get('PORT', '8000')}").rstrip('/')

    def number(self, key, default):
        value = float(self.env.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise OpsError(f'{key} must be finite and positive', 2)
        return value

    def command(self, rank, argv, timeout=None):
        if rank == 1:
            argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', self.worker, shlex.join(argv)]
        limit = timeout or self.command_timeout
        if self.deadline is not None:
            limit = min(limit, self.deadline - self.clock())
            if limit <= 0:
                raise OpsError('native pair readiness timed out')
        try:
            result = self.runner(argv, cwd=self.root, env=self.env, capture_output=True,
                                 text=True, timeout=limit)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise OpsError('command unavailable or timed out') from error
        if result.returncode:
            raise OpsError(f'command failed (rank {rank}, exit {result.returncode})', result.returncode)
        return result.stdout.strip()

    def launch_command(self, rank, mode):
        argv = ['env', f"CONFIG={self.env.get('CONFIG', 'config/prod-zig.env')}",
                f'CONTAINER_PREFIX={self.prefix}', 'bash', 'scripts/run-zig.sh', str(rank), mode]
        if rank == 1:
            argv[1] = f"CONFIG={self.env['WORKER_CONFIG']}"
            argv = ['bash', '-c', f"cd {shlex.quote(self.env['WORKER_REPO'])} && {shlex.join(argv)}"]
        return self.command(rank, argv)

    @contextmanager
    def locked(self):
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_file.open('a') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise OpsError('lifecycle operation already in progress', 75) from error
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def policy(self, check_stop=True, operation="start"):
        if check_stop and self.stop_file.exists():
            raise OpsError('durable STOP is set; explicit start required', 78)
        if self.lease and self.lease.exists():
            raise OpsError('maintenance lease is present', 75)
        guard = self.env.get('GUARD_CONFIG')
        if guard:
            self.command(0, [sys.executable, 'scripts/guard/lifecycle.py', '--config', guard,
                             'check', '--operation', operation])

    def containers(self):
        pair = []
        for rank in (0, 1):
            # Listing must succeed: an unavailable daemon is never treated as an absent rank.
            names = self.command(rank, ['docker', 'container', 'ls', '--all', '--format', '{{.Names}}']).splitlines()
            name = f'{self.prefix}-r{rank}'
            if name not in names:
                pair.append(None)
                continue
            pair.append(self.inspect_owned(rank))
        return pair

    def inspect_owned(self, rank):
        name = f'{self.prefix}-r{rank}'
        try:
            records = json.loads(self.command(rank, ['docker', 'container', 'inspect', name]))
            record, = records
            labels = record['Config']['Labels'] or {}
            if (record['Name'] != '/' + name or labels.get('org.tensorfold.rank') != str(rank)
                    or labels.get('org.tensorfold.revision') != self.revision):
                raise OpsError(f'foreign container occupies {name}')
            if not re.fullmatch(r'[0-9a-f]{64}', record['Id']):
                raise ValueError('invalid container id')
            return record
        except (ValueError, TypeError, KeyError) as error:
            raise OpsError(f'invalid runtime inspection for {name}') from error

    def healthy(self, timeout=None):
        def fetch(endpoint):
            limit = timeout or self.http_timeout
            if self.deadline is not None:
                limit = min(limit, self.deadline - self.clock())
                if limit <= 0:
                    raise TimeoutError('readiness deadline reached')
            return self.http(self.base_url + endpoint, limit)

        try:
            health = fetch('/health')
            models = fetch('/v1/models')
            return (isinstance(health, dict) and (health.get('ok') is True or health.get('status') in ('ok', 'healthy'))
                    and health.get('ok') is not False and not health.get('fatal')
                    and health.get('status') not in ('stalled', 'draining', 'error')
                    and isinstance(models, dict) and isinstance(models.get('data'), list)
                    and any(isinstance(item, dict) and item.get('id') for item in models['data']))
        except (OSError, ValueError, TimeoutError):
            return False

    def ready(self, pair, timeout=None):
        return all(item and item.get('State', {}).get('Running') is True for item in pair) and self.healthy(timeout)

    def preflight(self):
        self.policy(check_stop=False)
        local = self.launch_command(0, '--identity')
        remote = self.launch_command(1, '--identity')
        if not re.fullmatch(r'[0-9a-f]{64}', local) or local != remote:
            raise OpsError('rank identities differ or are invalid')
        self.containers()
        return local

    @contextmanager
    def marker_locked(self):
        self.stop_file.parent.mkdir(parents=True, exist_ok=True)
        with self.stop_file.with_name(self.stop_file.name + '.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def hold(self, reason):
        with self.marker_locked():
            durable_json(self.stop_file, {'reason': reason, 'time': self.wall(), 'token': os.urandom(16).hex()})
            return self.stop_file.read_bytes()

    def stop_owned(self):
        errors = []
        # Inspect each rank independently so one unreachable runtime cannot prevent safe peer cleanup.
        for rank in (0, 1):
            name = f'{self.prefix}-r{rank}'
            try:
                names = self.command(rank, ['docker', 'container', 'ls', '--all', '--format', '{{.Names}}']).splitlines()
                if name not in names:
                    continue
                item = self.inspect_owned(rank)
                self.command(rank, ['docker', 'container', 'stop', '--time', str(self.stop_timeout), item['Id']],
                             timeout=max(self.command_timeout, self.stop_timeout + 10))
            except (OpsError, ValueError, TypeError, KeyError) as error:
                errors.append(str(error))
        if errors:
            raise OpsError('; '.join(errors))

    def start_locked(self, explicit=False, expected_stop=None):
        stopped = self.stop_file.read_bytes() if self.stop_file.exists() else None
        if expected_stop is not None and stopped != expected_stop:
            raise OpsError("STOP changed during restart; start refused", 78)
        self.policy(check_stop=not explicit)
        self.preflight()
        if explicit:
            with self.marker_locked():
                current = self.stop_file.read_bytes() if self.stop_file.exists() else None
                if current != stopped:
                    raise OpsError('STOP changed during preflight; start refused', 78)
                self.stop_file.unlink(missing_ok=True)
        if self.ready(self.containers()):
            return
        try:
            self.stop_owned()
            self.policy()
            self.launch_command(1, '--detach')
            self.policy()
            self.launch_command(0, '--detach')
            deadline = self.clock() + self.ready_timeout
            self.deadline = deadline
            while self.clock() < deadline:
                self.policy()
                remaining = deadline - self.clock()
                if remaining <= 0:
                    break
                if self.ready(self.containers(), min(self.http_timeout, remaining / 2)):
                    return
                self.sleep(min(1, max(0, deadline - self.clock())))
            raise OpsError('native pair readiness timed out')
        except Exception:
            self.deadline = None
            self.hold('boot failure; explicit start required')
            try:
                self.stop_owned()
            except OpsError:
                pass
            raise
        finally:
            self.deadline = None

    def start(self):
        with self.locked():
            self.start_locked(explicit=True)

    def stop(self):
        self.hold('operator STOP')
        with self.locked():
            self.stop_owned()

    def restart(self):
        restart_marker = self.hold('operator restart in progress')
        with self.locked():
            self.stop_owned()
            self.start_locked(explicit=True, expected_stop=restart_marker)

    def status(self):
        pair = self.containers()
        return {'stopped': self.stop_file.exists(), 'ranks': [bool(p and p['State']['Running']) for p in pair],
                'ready': self.ready(pair)}

    def tick(self):
        try:
            with self.locked():
                self.policy(operation="watch")
                state = {'failures': 0, 'last_heal': None}
                if self.watch_file.exists():
                    state = json.loads(self.watch_file.read_text())
                pair = self.containers()
                http_ready = self.healthy()
                if all(item and item['State']['Running'] is True for item in pair) and http_ready:
                    state['failures'] = 0
                else:
                    state['failures'] += 1
                due = state['last_heal'] is None or self.wall() - state['last_heal'] >= self.cooldown
                heal = self.env.get('WATCH_HEAL') == '1' and state['failures'] >= self.fails and due
                if heal:
                    state['last_heal'] = self.wall()
                durable_json(self.watch_file, state)
                if heal:
                    self.start_locked()
                return 'healed' if heal else ('healthy' if not state['failures'] else 'unhealthy')
        except OpsError as error:
            if error.code in (75, 78):
                return f'held: {error}'
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('preflight', 'start', 'stop', 'restart', 'status', 'watch'))
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    try:
        ops = Lifecycle()
        if args.operation == 'watch':
            while True:
                try:
                    print(ops.tick(), flush=True)
                except OpsError as error:
                    if args.once:
                        raise
                    print(f'native watch: {error}', file=sys.stderr, flush=True)
                if args.once:
                    break
                ops.sleep(ops.interval)
        else:
            result = getattr(ops, args.operation)()
            if result is not None:
                print(json.dumps(result))
            if args.operation == 'status' and not result['ready']:
                return 1
        return 0
    except (OpsError, OSError, ValueError) as error:
        print(f'native lifecycle: {error}', file=sys.stderr)
        return error.code if isinstance(error, OpsError) else 1


if __name__ == '__main__':
    sys.exit(main())
