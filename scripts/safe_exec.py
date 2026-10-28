#!/usr/bin/env python3
"""Credential-safe commands with bounded process-group teardown."""

from __future__ import annotations

import codecs
import math
import os
import selectors
import signal
import subprocess
import sys
import threading
import time

from redact import known_secrets, redact_argv, secret_variants
from safe_output import _cli_secrets, _Stream, cli

TERM_GRACE = 1.0
GROUP_ENV = 'TF_SAFE_EXEC_GROUP'
DEADLINE_ENV = 'TF_SAFE_EXEC_DEADLINE'


def _check_argv(argv, secrets):
    variants = secret_variants(secrets)
    if redact_argv(argv, secrets) != argv or any(value in arg for value in variants for arg in argv):
        raise ValueError('credential value refused in command arguments')


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


def _spawn(argv, *, env=None, deadline=None, **kwargs):
    # The bootstrap runs inside the newly created session, so it can publish
    # the *actual* pgid before exec. Nested output wrappers retain that group.
    bootstrap = [sys.executable, str(os.path.abspath(__file__)), '--group-child', *argv]
    env = dict(os.environ if env is None else env)
    env.pop(DEADLINE_ENV, None)
    if deadline is not None:
        env[DEADLINE_ENV] = str(deadline)
    return subprocess.Popen(bootstrap, env=env, start_new_session=True, **kwargs)


def run(argv, *, cwd=None, env=None, capture_output=True, text=True, timeout=None):
    """subprocess.run-shaped default runner; timeout returns only after teardown."""
    argv = [os.fsdecode(arg) for arg in argv]
    _check_argv(argv, _cli_secrets(argv) + known_secrets(env))
    if not capture_output:
        raise ValueError('supervised commands require captured output')
    deadline = None if timeout is None else time.monotonic() + timeout
    child = _spawn(argv, cwd=cwd, env=env, deadline=deadline,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text)
    previous = {}

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
        try:
            remaining = None if deadline is None else max(0, deadline - time.monotonic())
            out, err = child.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            _finish_group(child)
            # SIGKILL closed every owned writer. Do not wait indefinitely for a
            # deliberately detached process that inherited one of the pipes.
            try:
                child.communicate(timeout=TERM_GRACE)
            except subprocess.TimeoutExpired:
                pass
            raise subprocess.TimeoutExpired(argv, timeout) from None
        return subprocess.CompletedProcess(argv, child.returncode, out, err)
    finally:
        _finish_group(child)
        child.stdout.close()
        child.stderr.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == '--group-child':
        # Internal exec bootstrap: no output collection and no new session.
        os.environ[GROUP_ENV] = str(os.getpgrp())
        cutoff = os.environ.get(DEADLINE_ENV)
        if cutoff is not None and time.monotonic() >= float(cutoff):
            print('command deadline expired <redacted>', file=sys.stderr)
            return 124
        os.execvpe(argv[1], argv[1:], os.environ)
    timeout = None
    expires = None
    monotonic_expires = None
    while argv and argv[0] in ('--timeout', '--deadline', '--monotonic-deadline'):
        option, value = argv[:2]
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
    checked = bool(argv and argv[0] == '--command')
    if checked:
        argv = argv[1:]
    if not argv:
        raise ValueError('missing command')
    secrets = _cli_secrets(argv)
    if checked:
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
    env = dict(os.environ)
    env['TF_SAFE_EXEC_PARENT'] = str(os.getpid())
    env.pop(DEADLINE_ENV, None)
    if deadline is not None:
        env[DEADLINE_ENV] = str(deadline)
    owned = not _joined_group()
    if owned:
        child = _spawn(argv, env=env, deadline=deadline, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        child = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
            previous[signum] = signal.signal(signum, forward)
        with selectors.DefaultSelector() as selector:
            for pipe, target in ((child.stdout, sys.stdout), (child.stderr, sys.stderr)):
                stream = _Stream(target, secrets)
                streams.append(stream)
                selector.register(pipe, selectors.EVENT_READ,
                                  (stream, codecs.getincrementaldecoder('utf-8')('replace')))
            while selector.get_map() or child.poll() is None:
                now = time.monotonic()
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
            stream.finish()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == '__main__':
    code = cli(main)
    if code < 0:
        signum = -code
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    sys.exit(code)
