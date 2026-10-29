"""Real process trees; IPC gates timeout until every late-launch client is ready."""

import importlib.util
import json
import os
from pathlib import Path
import resource
import shlex
import signal
import socket
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import safe_exec  # noqa: E402


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


@pytest.fixture
def endpoint(tmp_path):
    with socket.socket(socket.AF_UNIX) as server:
        path = str(tmp_path / 'ready.sock')
        server.bind(path)
        server.listen()
        server.settimeout(15)
        yield server, path


def late_client(tmp_path):
    script = tmp_path / 'late_client.py'
    script.write_text('''import json, os, signal, socket, sys
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with socket.socket(socket.AF_UNIX) as channel:
    channel.connect(sys.argv[1])
    channel.sendall((json.dumps([os.getpid(), os.getpgrp()]) + "\\n").encode())
    if channel.recv(1):
        Path(sys.argv[2]).write_text("late launch")
''')
    return script


def ready(server):
    channel, _ = server.accept()
    channel.settimeout(15)
    record = b''
    while not record.endswith(b'\n'):
        record += channel.recv(1024)
    return channel, json.loads(record)


@pytest.mark.parametrize('kind', ['docker', 'ssh'])
@pytest.mark.parametrize('forged', [False, True])
def test_timeout_kills_nested_term_ignoring_clients_before_cleanup(tmp_path, endpoint, monkeypatch, kind, forged):
    server, address = endpoint
    side_effect = tmp_path / 'launched'
    script = late_client(tmp_path)
    # A nested wrapper must stay in the command's owned group. It must not
    # create a new session that outlives the default runner's timeout.
    argv = [sys.executable, str(ROOT / 'scripts/safe_exec.py'), '--command',
            sys.executable, str(script), address, str(side_effect)]
    if kind == 'ssh':
        proxy = tmp_path / 'ssh'
        proxy.write_text('import subprocess, sys; sys.exit(subprocess.call(sys.argv[1:]))')
        argv = [sys.executable, str(proxy), *argv]
    env = dict(os.environ)
    if forged:
        env[safe_exec.GROUP_ENV] = '1'
    channels = []
    original = subprocess.Popen

    class ControlledTimeout(original):
        def communicate(self, input=None, timeout=None):
            if not channels:
                channel, (_, pgid) = ready(server)
                channels.append(channel)
                assert pgid == self.pid
                # Invoke the real timeout only after the process tree is ready.
                # No assertion depends on Python startup or disk latency.
                timeout = 0
            return super().communicate(input=input, timeout=timeout)

    monkeypatch.setattr(safe_exec.subprocess, 'Popen', ControlledTimeout)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            safe_exec.run(argv, env=env, timeout=60)
        # This models STOP cleanup observing no container, followed by a late
        # client release. EOF proves the real writer died before cleanup.
        assert not side_effect.exists(), kind
        assert channels[0].recv(1) == b''
        assert not side_effect.exists(), kind
    finally:
        for channel in channels:
            channel.close()


@pytest.mark.parametrize('signum', [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT])
def test_unsupervised_wrapper_preserves_signal_status(endpoint, signum, tmp_path):
    server, address = endpoint
    script = tmp_path / 'signal_client.py'
    script.write_text('''import signal, socket, sys, os, json
for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
    signal.signal(number, signal.SIG_DFL)
with socket.socket(socket.AF_UNIX) as channel:
    channel.connect(sys.argv[1])
    channel.sendall((json.dumps([os.getpid(), os.getpgrp()]) + "\\n").encode())
    channel.recv(1)
''')
    child = subprocess.Popen([sys.executable, str(ROOT / 'scripts/safe_exec.py'),
                              sys.executable, str(script), address],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=no_core)
    try:
        channel, _ = ready(server)
        with channel:
            child.send_signal(signum)
            child.communicate(timeout=15)
            assert child.returncode == -signum
            assert channel.recv(1) == b''
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


@pytest.mark.parametrize('code', [0, 2, 43])
@pytest.mark.parametrize('output', ['', 'reported'])
@pytest.mark.parametrize('stream', ['stdout', 'stderr'])
def test_silent_diagnostic_preserves_status(code, output, stream):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/safe_exec.py'),
                             sys.executable, '-c', f'import sys; sys.{stream}.write({output!r}); sys.exit({code})'],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == code
    assert result.stdout == (output if stream == 'stdout' else '')
    if output and stream == 'stderr':
        assert result.stderr == output
    assert ('<redacted>' in result.stderr) == (code != 0 and not output)


@pytest.mark.parametrize('option', ['--deadline', '--monotonic-deadline'])
def test_expired_worker_deadline_refuses_late_ssh_launch(tmp_path, option):
    side_effect = tmp_path / 'late-remote-launch'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/safe_exec.py'),
                             option, '1', '--timeout', '60', '--command',
                             sys.executable, '-c', 'from pathlib import Path; import sys; Path(sys.argv[1]).touch()',
                             str(side_effect)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 124
    assert not side_effect.exists()


def test_default_remote_runner_bounds_worker_before_local_timeout(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('timeout_native_ops', ROOT / 'scripts/native_ops.py')
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    captured = []

    def runner(argv, **kwargs):
        captured.append((argv, kwargs))
        if '-c' in shlex.split(argv[-1]):
            return subprocess.CompletedProcess(argv, 0, '1000', '')
        return subprocess.CompletedProcess(argv, 0, '', '')

    monkeypatch.setattr(ops, 'supervised_run', runner)
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/tensorfold.rev').write_text('a' * 40)
    life = ops.Lifecycle(env={'WORKER_SSH': 'worker.example', 'WORKER_REPO': '/srv/repo space',
                              'WORKER_CONFIG': '/srv/config', 'STATE_DIR': str(tmp_path / 'state')},
                         runner=runner, root=tmp_path)
    life.command(1, ['docker', 'run', 'example'], timeout=20)
    argv, kwargs = captured[1]
    remote = shlex.split(argv[-1])
    assert remote[:3] == ['/usr/bin/env', '-i', 'PATH=/usr/sbin:/usr/bin:/sbin:/bin']
    remote = remote[3:]
    assert remote[:2] == ['/usr/bin/python3', '/srv/repo space/scripts/safe_exec.py']
    assert remote[2] == '--monotonic-deadline'
    assert 1000 < float(remote[3]) <= 1020
    assert remote[4] == '--timeout'
    assert 0 < float(remote[5]) <= 20
    assert remote[6:] == ['--command', 'docker', 'run', 'example']
    assert kwargs['timeout'] == float(remote[5]) + 2 * safe_exec.TERM_GRACE


def test_worker_timeout_drains_detached_remote_tree_before_cleanup(tmp_path, endpoint):
    server, address = endpoint
    side_effect = tmp_path / 'remote-container'
    script = late_client(tmp_path)
    # Only the worker's monotonic clock is advanced, after its real client has
    # announced readiness. The clock step exercises the actual CLI timeout path
    # without putting Python startup inside a small deadline.
    worker = tmp_path / 'worker_supervisor.py'
    worker.write_text('''import os, signal, sys, time
sys.path.insert(0, sys.argv.pop(1))
import safe_exec
original = time.monotonic
offset = 0
def expire(signum, frame):
    global offset
    offset = 120
signal.signal(signal.SIGUSR1, expire)
safe_exec.time.monotonic = lambda: original() + offset
os.environ['TEST_WORKER_SUPERVISOR'] = str(os.getpid())
sys.exit(safe_exec.cli(safe_exec.main))
''')
    script.write_text(script.read_text().replace(
        '[os.getpid(), os.getpgrp()]', '[int(os.environ["TEST_WORKER_SUPERVISOR"]), os.getpgrp()]'))
    # Fake SSH starts a separate worker session: killing the local SSH client
    # alone cannot kill the worker, just as with an actual SSH connection.
    ssh = tmp_path / 'ssh'
    ssh.write_text('''import subprocess, sys
remote = subprocess.Popen(sys.argv[1:], start_new_session=True)
sys.exit(remote.wait())
''')
    child = subprocess.Popen([sys.executable, str(ssh), sys.executable, str(worker),
                              str(ROOT / 'scripts'), '--timeout', '60', '--command',
                              sys.executable, str(ROOT / 'scripts/safe_exec.py'), '--command',
                              sys.executable, str(script), address, str(side_effect)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    remote_pid = None
    try:
        channel, (remote_pid, _) = ready(server)
        with channel:
            os.kill(remote_pid, signal.SIGUSR1)
            _, err = child.communicate(timeout=15)
            assert child.returncode == 124, err
            assert channel.recv(1) == b''
            assert not side_effect.exists()
    finally:
        if remote_pid is not None:
            try:
                os.kill(remote_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if child.poll() is None:
            child.kill()
        child.wait()


@pytest.mark.parametrize('failure', ['disconnect', 'timeout'])
def test_remote_disconnect_waits_for_worker_budget_before_cleanup(tmp_path, monkeypatch, failure):
    spec = importlib.util.spec_from_file_location('disconnect_native_ops', ROOT / 'scripts/native_ops.py')
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    waits = []

    def runner(argv, **kwargs):
        if '-c' in shlex.split(argv[-1]):
            return subprocess.CompletedProcess(argv, 0, '1000', '')
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
        return subprocess.CompletedProcess(argv, 255, '', '')

    monkeypatch.setattr(ops, 'supervised_run', runner)
    monkeypatch.setattr(ops.time, 'monotonic', lambda: 100)
    monkeypatch.setattr(ops.time, 'sleep', waits.append)
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/tensorfold.rev').write_text('a' * 40)
    life = ops.Lifecycle(env={'WORKER_SSH': 'worker.example', 'WORKER_REPO': '/srv/repo',
                              'WORKER_CONFIG': '/srv/config', 'STATE_DIR': str(tmp_path / 'state')},
                         runner=runner, root=tmp_path)
    with pytest.raises(ops.OpsError):
        life.command(1, ['docker', 'run', 'example'], timeout=20)
    assert waits == [20 + 2 * safe_exec.TERM_GRACE]


@pytest.mark.parametrize('signum', [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT])
def test_default_runner_drains_owned_group_on_supervisor_signal(tmp_path, endpoint, signum):
    server, address = endpoint
    script = late_client(tmp_path)
    harness = tmp_path / 'runner.py'
    harness.write_text('''import signal, sys
sys.path.insert(0, sys.argv.pop(1))
import safe_exec
for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
    signal.signal(number, signal.SIG_DFL)
safe_exec.run(sys.argv[1:], timeout=60)
''')
    effect = tmp_path / 'late-container'
    child = subprocess.Popen([sys.executable, str(harness), str(ROOT / 'scripts'),
                              sys.executable, str(ROOT / 'scripts/safe_exec.py'), '--command',
                              sys.executable, str(script), address, str(effect)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=no_core)
    try:
        channel, _ = ready(server)
        with channel:
            child.send_signal(signum)
            child.communicate(timeout=15)
            assert child.returncode == -signum
            assert channel.recv(1) == b''
            assert not effect.exists()
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


@pytest.mark.parametrize('payload', ['known', 'embedded', 'assignment'])
@pytest.mark.parametrize('name', ['API_KEY', 'AUTH_TOKEN', 'TOKEN'])
def test_default_runner_refuses_credentials_before_docker_spawn(tmp_path, monkeypatch, payload, name):
    key = 'timeout' + 'SyntheticCredentialOnly' if name == 'API_KEY' else '200'
    monkeypatch.delenv('API_KEY', raising=False)
    marker = tmp_path / 'docker-called'
    docker = tmp_path / 'docker'
    docker.write_text('#!/bin/sh\n: > "$1"\n')
    docker.chmod(0o700)
    argument = {'known': key, 'embedded': f'--label=prefix-{key}-suffix',
                'assignment': 'API_KEY=' + 'other' + 'SyntheticCredentialOnly'}[payload]
    with pytest.raises(ValueError) as error:
        safe_exec.run([str(docker), str(marker), argument], env={name: key}, timeout=15)
    leaked = key in str(error.value) or argument in str(error.value)
    assert not leaked, 'credential reached command refusal'
    assert str(error.value) == 'credential value refused in command arguments'
    assert not marker.exists()
