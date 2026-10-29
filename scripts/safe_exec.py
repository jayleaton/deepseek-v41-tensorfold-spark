#!/usr/bin/env python3
"""Credential-safe commands with bounded process-group teardown."""

from __future__ import annotations

import codecs
import json
import math
import os
from pathlib import Path
import selectors
import tempfile
import signal
import subprocess
import sys
import threading
import time

from redact import check_argv as _check_argv, known_secrets, redact_text, secret_name, secret_variants
from safe_output import _cli_secrets, _known, _Stream, cli, inherited_known

TERM_GRACE = 1.0
GROUP_ENV = 'TF_SAFE_EXEC_GROUP'
DEADLINE_ENV = 'TF_SAFE_EXEC_DEADLINE'
CONTROL_LIMIT = 8 * 1024 * 1024
OWNER_PID_ENV = 'TF_SAFE_EXEC_OWNER_PID'
OWNER_START_ENV = 'TF_SAFE_EXEC_OWNER_START'
SHELL_STARTUP_NAMES = frozenset({'BASH_ENV', 'ENV', 'PROMPT_COMMAND', 'SHELLOPTS', 'BASHOPTS', 'PS4'})


class ChildEnvironmentError(ValueError):
    """A tainted operational setting cannot safely fall back to a default."""


def _internal_control(name, value):
    if name in {GROUP_ENV, OWNER_PID_ENV, OWNER_START_ENV, 'TF_SAFE_EXEC_PARENT', 'TF_SAFE_REGISTRY_FD'}:
        return str(value).isascii() and str(value).isdigit() and int(value) > 0
    if name == DEADLINE_ENV:
        try:
            return math.isfinite(float(value)) and float(value) > 0
        except ValueError:
            return False
    return False


def child_environment(env=None, secrets=()):
    """Diagnostic children receive no credential names or credential-bearing aliases.

    Validation/redaction remains in the supervisor. Do not carry a secret registry
    in a child environment: even an opaque setting can be printed by diagnostics.
    Serving credentials travel via the separately reviewed server-only transport.
    """
    source = dict(os.environ if env is None else env)
    known = list(secrets) + _known() + known_secrets(source) + _cli_secrets([])
    if source.get('WATCH_RESTART_CMD'):
        raise ChildEnvironmentError('WATCH_RESTART_CMD is removed; use WATCH_RESTART_ARGV JSON argv')
    variants = secret_variants(known)
    result = {}
    for name, value in source.items():
        if name in SHELL_STARTUP_NAMES or name.startswith('BASH_FUNC_') or secret_name(name):
            continue
        if _internal_control(name, value):
            result[name] = value
            continue
        if (any(part in str(value) for part in variants)
                or redact_text(str(value), known, decode=False) != str(value)):
            # Dropping an operational control could select a default and start
            # work the operator never requested. Refuse such configuration;
            # ordinary diagnostic aliases are simply absent in the child.
            if (name.startswith(('WATCH_', 'TF_', 'GLM53_', 'NCCL_', 'WORKER_', 'NATIVE_', 'CHECK_PUBLIC_'))
                    or name in {'PATH', 'CONFIG', 'CONTAINER_PREFIX', 'IMAGE', 'STATE_DIR',
                                'MODEL_DIR', 'ENGRAM_DIR', 'ASSETS_DIR', 'PREPARED_DIR',
                                'CACHE_DIR', 'GUARD_CONFIG', 'STOP_FILE', 'MAINTENANCE_LEASE',
                                'LIFECYCLE_LOCK', 'READY_TIMEOUT', 'OPS_COMMAND_TIMEOUT',
                                'STOP_TIMEOUT', 'HTTP_TIMEOUT', 'HEALTH_URL', 'PORT', 'HOST',
                                'CPUSET_CPUS', 'TENSORFOLD_SOURCE', 'HOME', 'XDG_STATE_HOME'}):
                raise ChildEnvironmentError('credential value refused in child configuration')
            continue
        result[name] = value
    return result


def _process_identity(pid):
    try:
        with open(f'/proc/{pid}/stat') as source:
            fields = source.read().rsplit(')', 1)[1].split()
        return fields[0], int(fields[1]), fields[19]
    except (OSError, ValueError, IndexError):
        return None


def _owner():
    """Accept an owner only when its kernel identity is in our ancestry."""
    pid, start = os.environ.get(OWNER_PID_ENV), os.environ.get(OWNER_START_ENV)
    if pid is None and start is None:
        return None
    try:
        owner = int(pid)
    except (ValueError, TypeError):
        raise ValueError('invalid shell owner') from None
    current = os.getpid()
    while current >= 1:
        identity = _process_identity(current)
        if identity is None:
            break
        state, parent, born = identity
        if current == owner:
            if state not in ('Z', 'X') and born == start:
                return owner, start
            break
        if parent == current:
            break
        current = parent
    raise ValueError('invalid shell owner')


def _owner_alive(owner):
    if owner is None:
        return True
    identity = _process_identity(owner[0])
    return identity is not None and identity[0] not in ('Z', 'X') and identity[2] == owner[1]


class _Control:
    """Hold bounded raw controls until their typed projection is validated."""

    def __init__(self, output, mode, secrets):
        self.output, self.mode, self.secrets = output, mode, secrets
        self.parts = []
        self.size = 0
        self.failed = False
        self.finished = False

    def write(self, text):
        self.size += len(text.encode('utf-8'))
        if self.size > CONTROL_LIMIT:
            self.failed = True
            self.parts.clear()
        elif not self.failed:
            self.parts.append(text)

    def flush(self):
        pass

    def finish(self):
        if self.finished:
            return
        self.finished = True
        try:
            if self.failed:
                raise ValueError
            from watchdog_io import filter_output

            result = filter_output(self.mode, ''.join(self.parts), self.secrets)
            if not isinstance(result, str):
                raise ValueError
        except Exception:
            self.failed = True
            print('control parse failed; output withheld', file=sys.stderr)
        else:
            # cli() already wraps stdout in _Stream. Preserve the parser's
            # validated numbers, field names and protocol framing verbatim.
            self.output.write_sanitized(result)
            self.output.flush()
        finally:
            self.parts.clear()


def _filter(mode, secrets, owner):
    stream = (_Control(sys.stdout, mode, secrets)
              if mode is not None else _Stream(sys.stdout, secrets))
    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    lost_at = None
    # select() also accepts regular redirected files, unlike Linux epoll.
    with selectors.SelectSelector() as selector:
        selector.register(sys.stdin, selectors.EVENT_READ)
        while True:
            if not _owner_alive(owner) and lost_at is None:
                lost_at = time.monotonic()
            # Leave room for command teardown and its final sanitized writes.
            if lost_at is not None and time.monotonic() - lost_at >= 3 * TERM_GRACE:
                break
            if not selector.select(0.05):
                continue
            data = os.read(sys.stdin.fileno(), 65536)
            if not data:
                break
            stream.write(decoder.decode(data))
            stream.flush()
    if lost_at is not None:
        if not isinstance(stream, _Control):
            stream.finish()
        return 128 + signal.SIGTERM
    stream.write(decoder.decode(b'', final=True))
    stream.finish()
    return int(isinstance(stream, _Control) and stream.failed)


def _joined_group():
    # A loose boolean (or stale marker from another session) cannot opt out of
    # ownership. The marker must identify this process's actual kernel group.
    return os.environ.get(GROUP_ENV) == str(os.getpgrp())


def _signal_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
        return True
    except ProcessLookupError:
        return False


def _finish_group(child, grace=TERM_GRACE):
    """Drain the entire owned group, including descendants of an exited leader."""
    if _signal_group(child.pid, signal.SIGTERM):
        until = time.monotonic() + grace
        while time.monotonic() < until:
            child.poll()  # Reap the direct child even when a grandchild remains.
            if not _signal_group(child.pid, 0):
                break
            time.sleep(min(0.02, max(0, until - time.monotonic())))
        _signal_group(child.pid, signal.SIGKILL)
    child.wait()


def _bash_argv(argv):
    """Disable implicit startup files for Bash argv controlled by this adapter."""
    args = list(argv)
    if args and Path(args[0]).name == 'env':
        offset = 1
        if len(args) > offset and args[offset] in {'-i', '--ignore-environment'}:
            offset += 1
        while len(args) > offset and '=' in args[offset] and args[offset].partition('=')[0].isidentifier():
            offset += 1
        return args[:offset] + _bash_argv(args[offset:])
    if args and Path(args[0]).name == 'bash' and args[1:3] != ['--norc', '--noprofile']:
        args[1:1] = ['--norc', '--noprofile']
    return args


def _reviewed_entry(argv, cwd=None):
    """Only same-checkout supervisor scripts receive a private known-value registry."""
    args = list(argv)
    locked = Path(args[0]).name == 'flock'
    if locked:
        if len(args) != 5 or args[1].startswith('-') or args[-1] != '--restart':
            return False
        args = args[2:]
    if Path(args[0]).name == 'env':
        args = args[1:]
        while args and '=' in args[0] and args[0].partition('=')[0].isidentifier():
            args = args[1:]
    if len(args) < 2 or Path(args[0]).name not in {'bash', 'python3', Path(sys.executable).name}:
        return False
    if Path(args[0]).name == 'bash':
        while len(args) > 1 and args[1] in {'--norc', '--noprofile'}:
            args.pop(1)
        if len(args) < 2:
            return False
    entry = (Path(cwd or os.getcwd()) / args[1]).resolve()
    root = Path(__file__).resolve().parents[1]
    entries = {'scripts/' + name for name in ('safe_exec.py', 'native_ops.py', 'serve.sh',
               'run-zig.sh', 'guard-watch-zig.sh', 'build-native.sh', 'check-zig-image.sh')}
    entries.update({'scripts/guard/' + name for name in
                    ('lifecycle.py', 'control.py', 'daemon.py', 'sampler.py', 'profile.py')})
    entries.add('docker/zig/fetch-source.sh')
    if locked:
        return entry == root / 'scripts/safe_exec.py' and args[2:] == ['--restart']
    return entry in {root / name for name in entries}


def _registry_environment(argv, env, cwd=None, secrets=()):
    result = dict(env)
    result.pop('TF_SAFE_REGISTRY_FD', None)
    known = list(secrets) + _known() + _cli_secrets([])
    if not known or not _reviewed_entry(argv, cwd):
        return result, None
    raw = b''.join(value.encode('utf-8') + b'\0' for value in set(known) if value)
    if len(raw) > CONTROL_LIMIT:
        raise ChildEnvironmentError('private registry exceeds limit')
    if hasattr(os, 'memfd_create'):
        fd = os.memfd_create('checked-supervisor', os.MFD_CLOEXEC)
    else:
        with tempfile.TemporaryFile() as stream:
            fd = os.dup(stream.fileno())
    try:
        with os.fdopen(os.dup(fd), 'wb') as stream:
            stream.write(raw)
        os.lseek(fd, 0, os.SEEK_SET)
    except BaseException:
        os.close(fd)
        raise
    result['TF_SAFE_REGISTRY_FD'] = str(fd)
    return result, fd


def _checked_popen(argv, *, env, registry_argv=None, secrets=(), **kwargs):
    argv = _bash_argv(argv)
    env, registry = _registry_environment(registry_argv or argv, env, kwargs.get('cwd'), secrets)
    if registry is not None and registry_argv is None:
        argv = [sys.executable, str(Path(__file__).resolve()), '--group-child', *argv]
    try:
        return subprocess.Popen(argv, env=env, pass_fds=() if registry is None else (registry,), **kwargs)
    finally:
        if registry is not None:
            os.close(registry)


def _spawn(argv, *, env=None, deadline=None, secrets=(), **kwargs):
    # The bootstrap runs inside the newly created session, so it can publish
    # the *actual* pgid before exec. Nested output wrappers retain that group.
    argv = _bash_argv(argv)
    bootstrap = [sys.executable, str(os.path.abspath(__file__)), '--group-child', *argv]
    secrets = list(secrets) + known_secrets(env)
    env = child_environment(env, secrets)
    owner = _owner()
    if owner is not None:
        env[OWNER_PID_ENV], env[OWNER_START_ENV] = str(owner[0]), owner[1]
    env.pop(DEADLINE_ENV, None)
    if deadline is not None:
        env[DEADLINE_ENV] = str(deadline)
    return _checked_popen(bootstrap, registry_argv=argv, env=env, secrets=secrets, start_new_session=True, **kwargs)


def run(argv, *, cwd=None, env=None, capture_output=True, text=True, timeout=None, check=False, input=None):
    """subprocess.run-shaped default runner; timeout returns only after teardown."""
    argv = [os.fsdecode(arg) for arg in argv]
    secrets = _cli_secrets(argv) + _known() + known_secrets(env)
    _check_argv(argv, secrets)
    if not capture_output:
        raise ValueError('supervised commands require captured output')
    owner = _owner()
    deadline = None if timeout is None else time.monotonic() + timeout
    child = _spawn(argv, cwd=cwd, env=env, deadline=deadline, secrets=secrets,
                   stdin=subprocess.PIPE if input is not None else None,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text)
    previous = {}
    pending_input = input

    def interrupted(signum, frame):
        # The lifecycle process itself may be signalled by its outer output
        # wrapper. Its command has a separate owned group; drain it before
        # allowing the lifecycle process's original signal behavior to run.
        _signal_group(child.pid, signum)
        _finish_group(child)
        handler = previous[signum]
        signal.signal(signum, handler)
        if callable(handler):
            handler(signum, frame)
        elif handler == signal.SIG_DFL:
            os.kill(os.getpid(), signum)

    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
                handler = signal.getsignal(signum)
                if handler != signal.SIG_IGN:
                    previous[signum] = handler
                    signal.signal(signum, interrupted)
        while True:
            if not _owner_alive(owner):
                raise RuntimeError('shell owner exited')
            remaining = None if deadline is None else max(0, deadline - time.monotonic())
            interval = remaining
            if owner is not None:
                interval = 0.05 if remaining is None else min(0.05, remaining)
            try:
                out, err = child.communicate(input=pending_input, timeout=interval)
                break
            except subprocess.TimeoutExpired:
                pending_input = None
                if owner is not None and (deadline is None or time.monotonic() < deadline):
                    continue
                _finish_group(child)
                # Detached writers cannot hold the supervisor open indefinitely.
                try:
                    child.communicate(timeout=TERM_GRACE)
                except subprocess.TimeoutExpired:
                    pass
                raise subprocess.TimeoutExpired(argv, timeout) from None
        if check and child.returncode:
            raise subprocess.CalledProcessError(child.returncode, argv, output=out, stderr=err)
        return subprocess.CompletedProcess(argv, child.returncode, out, err)
    finally:
        _finish_group(child)
        child.stdout.close()
        child.stderr.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _restart_argv():
    """Resolve data-only restart argv; only the audited automatic recovery path runs."""
    if os.environ.get('WATCH_RESTART_CMD'):
        raise ValueError('WATCH_RESTART_CMD is removed; use WATCH_RESTART_ARGV JSON argv')
    try:
        values = json.loads(os.environ.get('WATCH_RESTART_ARGV', ''))
    except (ValueError, TypeError):
        raise ValueError('WATCH_RESTART_ARGV must be a JSON argv array') from None
    if not isinstance(values, list) or not values:
        raise ValueError('WATCH_RESTART_ARGV must be a nonempty JSON argv array')
    argv = []
    for value in values:
        if isinstance(value, dict) and set(value) == {'env'}:
            name = value['env']
            if not isinstance(name, str) or not name.isidentifier() or name not in os.environ:
                raise ValueError('restart environment reference unavailable')
            value = os.environ[name]
        if not isinstance(value, str) or '\0' in value:
            raise ValueError('restart argv entries must be strings or environment references')
        argv.append(value)
    _check_argv(argv, _cli_secrets(argv) + _known())
    # No arbitrary shell/interpreter trampoline. A reviewed recovery command also
    # preserves durable STOP: automatic healing may never invoke explicit start.
    serve = Path(__file__).resolve().with_name('serve.sh')
    if (len(argv) != 4 or argv[0] not in ('bash', '/bin/bash', '/usr/bin/bash')
            or Path(argv[1]).resolve() != serve or argv[2:] != ['watch', '--once']):
        raise ValueError('restart argv must select bash scripts/serve.sh watch --once')
    return ['/bin/bash', str(serve), 'watch', '--once']


def main():
    owner = _owner()
    argv = sys.argv[1:]
    if argv and argv[0] == '--group-child':
        # Internal exec bootstrap: no output collection and no new session.
        os.environ[GROUP_ENV] = str(os.getpgrp())
        cutoff = os.environ.get(DEADLINE_ENV)
        if cutoff is not None and time.monotonic() >= float(cutoff):
            print('command deadline expired <redacted>', file=sys.stderr)
            return 124
        _check_argv(argv[1:], _cli_secrets(argv[1:]) + _known())
        environment, registry = _registry_environment(argv[1:], child_environment())
        if registry is not None:
            if registry != 9:
                os.dup2(registry, 9, inheritable=True)
                os.close(registry)
            else:
                os.set_inheritable(9, True)
            environment['TF_SAFE_REGISTRY_FD'] = '9'
        command = _bash_argv(argv[1:])
        os.execvpe(command[0], command, environment)
    timeout = None
    expires = None
    monotonic_expires = None
    control = None
    filtering = False
    while argv and argv[0] in (
        '--timeout', '--deadline', '--monotonic-deadline', '--control', '--filter'
    ):
        if argv[0] == '--filter':
            filtering = True
            argv = argv[1:]
            continue
        option, value = argv[:2]
        if option == '--control':
            control = value
            argv = argv[2:]
            continue
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError('invalid supervisor time budget')
        if option == '--timeout':
            timeout = number
        elif option == '--deadline':
            expires = number
        else:
            monotonic_expires = number
        argv = argv[2:]
    if filtering:
        if argv:
            raise ValueError('unexpected filter arguments')
        return _filter(control, _cli_secrets([]), owner)
    if argv == ['--credential-names']:
        for name in sys.stdin:
            name = name.strip()
            if name.isascii() and name.isidentifier() and secret_name(name):
                sys.stdout.write_sanitized(name + "\n")
        return 0
    if argv == ['--removed-environment-names']:
        retained = child_environment()
        for name in os.environ:
            if name not in retained and name.isascii() and name.isidentifier():
                sys.stdout.write_sanitized(name + "\n")
        return 0
    if argv and argv[0] == '--quote-argv':
        from safe_quote import join_checked
        sys.stdout.write(join_checked(argv[1:]) + '\n')
        return 0
    if argv and argv[0] == '--check-argv':
        if len(argv) == 1:
            raise ValueError('missing argv to check')
        _check_argv(argv[1:], _cli_secrets(argv[1:]) + _known())
        return 0
    if argv == ['--restart']:
        argv = ['--command', *_restart_argv()]
    checked = bool(argv and argv[0] == '--command')
    if checked:
        argv = argv[1:]
    if not argv:
        raise ValueError('missing command')
    secrets = _cli_secrets(argv) + _known()
    _check_argv(argv, secrets)
    for cutoff, clock in ((expires, time.time), (monotonic_expires, time.monotonic)):
        if cutoff is None:
            continue
        remaining = cutoff - clock()
        if remaining <= 0:
            print('command deadline expired <redacted>', file=sys.stderr)
            return 124
        timeout = remaining if timeout is None else min(timeout, remaining)
    deadline = None if timeout is None else time.monotonic() + timeout
    if _joined_group() and os.environ.get(DEADLINE_ENV):
        inherited = float(os.environ[DEADLINE_ENV])
        deadline = inherited if deadline is None else min(deadline, inherited)
    env = child_environment(secrets=secrets)
    env['TF_SAFE_EXEC_PARENT'] = str(os.getpid())
    env.pop(DEADLINE_ENV, None)
    if deadline is not None:
        env[DEADLINE_ENV] = str(deadline)
    owned = not _joined_group()
    if owned:
        child = _spawn(argv, env=env, deadline=deadline, secrets=secrets, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        child = _checked_popen(argv, env=env, secrets=secrets, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    previous = {}
    cancelled = None
    kill_at = None

    def forward(signum, frame):
        nonlocal cancelled, kill_at
        if cancelled is None:
            cancelled = signum
            kill_at = time.monotonic() + TERM_GRACE
            if owned:
                _signal_group(child.pid, signum)
            else:
                # All peers already receive group-directed signals; forwarding
                # to the whole group here would recursively signal ourselves.
                try:
                    child.send_signal(signum)
                except ProcessLookupError:
                    pass

    streams = []
    emitted = False
    timed_out = False
    owner_lost = False
    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
            previous[signum] = signal.signal(signum, forward)
        with selectors.DefaultSelector() as selector:
            for pipe, target in ((child.stdout, sys.stdout), (child.stderr, sys.stderr)):
                stream = (_Control(target, control, secrets)
                          if control is not None and pipe is child.stdout else _Stream(target, secrets))
                streams.append(stream)
                selector.register(pipe, selectors.EVENT_READ,
                                  (stream, codecs.getincrementaldecoder('utf-8')('replace')))
            while selector.get_map() or child.poll() is None:
                now = time.monotonic()
                if not _owner_alive(owner):
                    owner_lost = True
                    forward(signal.SIGTERM, None)
                if deadline is not None and now >= deadline and not timed_out:
                    timed_out = True
                    forward(signal.SIGTERM, None)
                if kill_at is not None and now >= kill_at:
                    if owned:
                        _signal_group(child.pid, signal.SIGKILL)
                    else:
                        child.kill()
                    # Do not let detached writers hold the supervisor open.
                    break
                for key, _ in selector.select(0.05):
                    stream, decoder = key.data
                    data = os.read(key.fd, 65536)
                    if data:
                        emitted = True
                        stream.write(decoder.decode(data))
                        stream.flush()
                    else:
                        stream.write(decoder.decode(b'', final=True))
                        stream.finish()
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
        if owned:
            _finish_group(child, grace=0 if kill_at is not None else TERM_GRACE)
        code = child.wait()
        if timed_out:
            code = 124
        elif owner_lost:
            code = 128 + signal.SIGTERM
        if not code and any(isinstance(stream, _Control) and stream.failed for stream in streams):
            code = 1
        if code and not emitted:
            print('command failed <redacted>', file=sys.stderr)
        return code
    finally:
        if owned:
            _finish_group(child)
        elif child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=TERM_GRACE)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for pipe in (child.stdout, child.stderr):
            pipe.close()
        for stream in streams:
            if not isinstance(stream, _Control):
                stream.finish()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _entry():
    try:
        return main()
    except ChildEnvironmentError as error:
        print(str(error), file=sys.stderr)
        return 78


def _private_registry():
    # A supervisor may receive known values over an anonymous pipe. The fd is
    # consumed here and never inherited by a diagnostic child or shown in argv.
    values = inherited_known()
    if sys.argv[1:2] != ['--known-fd']:
        return values
    fd = int(sys.argv[2])
    del sys.argv[1:3]
    with os.fdopen(fd, 'rb') as source:
        raw = source.read(CONTROL_LIMIT + 1)
    if len(raw) > CONTROL_LIMIT:
        raise ValueError('known-value registry exceeds limit')
    return values + [value.decode('utf-8') for value in raw.split(b'\0') if value]


if __name__ == '__main__':
    try:
        private = _private_registry()
    except Exception:
        print('invalid private credential registry', file=sys.stderr)
        sys.exit(78)
    code = cli(_entry, known=private)
    if code < 0:
        signum = -code
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    sys.exit(code)
