"""Observe real SSH processes without saving credential-bearing argv or streams."""

import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from native_ops import Lifecycle, OpsError  # noqa: E402


SHAPES = {'plain': '', 'space': ' ', 'apostrophe': "'", 'double_quote': '"',
          'mixed_quotes': "'\""}
IDENTITY = 'f' * 64

# This observer is a separate executable at the actual spawn boundary. It does
# not import production helpers, execute a remote command, or replace the runner.
# Both the calibration probe and the command return successfully, avoiding the
# deliberate disconnect drain wait in Lifecycle.command().
SSH_OBSERVER = r'''import json
import math
import os
from pathlib import Path
import shlex
import sys
import time

record = {'clock_probe': False, 'command_started': True, 'parse_valid': False,
          'raw_key': False, 'recovered_key': False, 'envelope_valid': False,
          'payload_preserved': False, 'ssh_options_valid': False}
try:
    words = shlex.split(sys.argv[-1])
    record['ssh_options_valid'] = sys.argv[1:-1] == [
        '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'RemoteCommand=none', 'worker.example.invalid']
    assert words[:3] == ['/usr/bin/env', '-i', 'PATH=/usr/sbin:/usr/bin:/sbin:/bin']
    words = words[3:]
    assert words[:2] == ['/usr/bin/python3', str(Path(os.environ['WORKER_REPO']) / 'scripts/safe_exec.py')]
    payload = words[words.index('--command') + 1:]
    probe = payload == ['/usr/bin/python3', '-c', 'import time; print(time.monotonic())']
    record['clock_probe'] = probe
    record['command_started'] = not probe
    if probe:
        record['parse_valid'] = True
    else:
        assert 'API_KEY' not in os.environ
        key = FIXTURE_VALUE
        record['raw_key'] = any(key in word for word in sys.argv[1:])
        # Decode successive shell layers without evaluation. In particular the
        # bash -c launch payload has been quoted once before the SSH argv join.
        parts = list(sys.argv[1:])
        for _ in range(8):
            record['recovered_key'] |= any(key in part for part in parts)
            expanded = []
            for part in parts:
                try:
                    expanded.extend(shlex.split(part))
                except ValueError:
                    pass
            parts = expanded
        cutoff, budget = float(words[3]), float(words[5])
        now = time.monotonic()
        record['envelope_valid'] = (
            words[:3] == ['/usr/bin/python3', str(Path(os.environ['WORKER_REPO']) /
                                         'scripts/safe_exec.py'), '--monotonic-deadline']
            and words[4] == '--timeout' and words[6] == '--command'
            and math.isfinite(cutoff) and math.isfinite(budget)
            and 0 < budget <= 10 and now < cutoff <= now + 10)
        payload = words[7:]
        if os.environ['REMOTE_TEST_MODE'] == 'launch':
            nested = shlex.split(payload[2])
            record['payload_preserved'] = (
                payload[:2] == ['bash', '-c'] and len(payload) == 3
                and nested == ['cd', os.environ['WORKER_REPO'], '&&', 'env',
                               'CONFIG=' + os.environ['WORKER_CONFIG'],
                               'CONTAINER_PREFIX=tensorfold-zig', 'bash',
                               'scripts/run-zig.sh', '1', '--identity'])
        else:
            record['payload_preserved'] = payload == [
                'docker', 'container', 'inspect', os.environ['REMOTE_TEST_ARGUMENT']]
        record['parse_valid'] = True
except (ValueError, IndexError, KeyError):
    pass
with Path(os.environ['REMOTE_TEST_OBSERVATIONS']).open('a') as stream:
    stream.write(json.dumps(record) + '\n')
if record['clock_probe']:
    print(time.monotonic())
else:
    print('f' * 64)
'''


@pytest.fixture
def remote(tmp_path):
    binary = tmp_path / 'bin'
    binary.mkdir()
    ssh = binary / 'ssh'
    ssh.write_text(f'#!{sys.executable}\n' + SSH_OBSERVER)
    ssh.chmod(0o700)
    marker = tmp_path / 'observations.jsonl'
    # Deliberately supply credentials only through Lifecycle.env, rather than
    # the test process's ambient environment, to exercise explicit env checking.
    env = {
        'PATH': str(binary) + os.pathsep + os.environ.get('PATH', ''),
        'API_KEY': 'rr4' + 'UnrelatedSyntheticCredentialOnly',
        'WORKER_SSH': 'worker.example.invalid',
        'WORKER_REPO': '/srv/recipe', 'WORKER_CONFIG': '/srv/config.env',
        'STATE_DIR': str(tmp_path / 'state'), 'CONFIG': '/nonexistent',
        'OPS_COMMAND_TIMEOUT': '10',
        'REMOTE_TEST_OBSERVATIONS': str(marker),
        'REMOTE_TEST_MODE': 'launch', 'REMOTE_TEST_ARGUMENT': 'owned-container',
        'CUDA_VISIBLE_DEVICES': '', 'NVIDIA_VISIBLE_DEVICES': 'void',
    }
    return env, marker


def _observations(marker):
    return [json.loads(line) for line in marker.read_text().splitlines()] if marker.exists() else []


def _invoke(env, operation):
    # The independent observer knows its synthetic value from fixture source,
    # never from a production diagnostic child's stripped environment.
    ssh = Path(env['PATH'].split(os.pathsep)[0]) / 'ssh'
    ssh.write_text(f'#!{sys.executable}\n' + SSH_OBSERVER.replace('FIXTURE_VALUE', repr(env['API_KEY'])))
    life = Lifecycle(env=env, root=ROOT)
    # Also exercise the readiness deadline's clamp of a larger command budget.
    life.deadline = life.clock() + 10
    if operation == 'launch':
        return life.launch_command(1, '--identity')
    return life.command(1, ['docker', 'container', 'inspect', env['REMOTE_TEST_ARGUMENT']],
                        timeout=20)


def _assert_refused(remote, operation, capsys):
    env, marker = remote
    refused = False
    error_leaked = False
    try:
        _invoke(env, operation)
    except (ValueError, OpsError) as error:
        refused = str(error) in ('credential value refused in command arguments',
                                 'credential value refused in child configuration')
        error_leaked = env['API_KEY'] in str(error)
    captured = capsys.readouterr()
    stream_leaked = any(env['API_KEY'] in stream for stream in (captured.out, captured.err))
    observations = _observations(marker)
    assert not observations, 'SSH spawned before the requested credential-bearing argv was refused'
    started = any(item['command_started'] for item in observations)
    raw_leaked = any(item['raw_key'] for item in observations)
    recovered = any(item['recovered_key'] for item in observations)
    # Assert only booleans: pytest must never render keys, argv or streams.
    assert not started, 'credential-bearing SSH command started'
    assert not raw_leaked, 'credential appeared in SSH argv'
    assert not recovered, 'credential recoverable from shell-quoted SSH argv'
    assert not error_leaked, 'credential appeared in refusal'
    assert not stream_leaked, 'credential appeared in output'
    assert refused, 'raw remote control was not refused as a credential'


@pytest.mark.parametrize('setting', ['WORKER_REPO', 'WORKER_CONFIG'])
@pytest.mark.parametrize('shape', tuple(SHAPES))
def test_nested_launch_refuses_credentials_before_ssh(remote, setting, shape, capsys):
    env, _ = remote
    key = 'rr4' + SHAPES[shape] + 'RemoteSyntheticCredentialOnly'
    env['API_KEY'] = key
    env[setting] = '/srv/' + key
    _assert_refused(remote, 'launch', capsys)


@pytest.mark.parametrize('shape', ['apostrophe', 'mixed_quotes'])
@pytest.mark.parametrize('source', ['argv', 'supervisor_path'])
def test_direct_command_refuses_credentials_before_ssh(remote, source, shape, capsys):
    env, _ = remote
    key = 'rr4' + SHAPES[shape] + 'RemoteSyntheticCredentialOnly'
    env['API_KEY'] = key
    env['REMOTE_TEST_MODE'] = 'direct'
    if source == 'supervisor_path':
        # The command payload is benign; only the default worker supervisor's
        # script path includes the configured credential through WORKER_REPO.
        env['WORKER_REPO'] = '/srv/' + key
    else:
        env['REMOTE_TEST_ARGUMENT'] = 'container-' + key
    _assert_refused(remote, 'direct', capsys)


@pytest.mark.parametrize('operation', ['launch', 'direct'])
@pytest.mark.parametrize('shape', ['apostrophe', 'mixed_quotes'])
def test_ordinary_quoted_paths_preserve_payload_and_worker_deadline(remote, operation, shape):
    env, marker = remote
    suffix = 'ordinary ' + SHAPES[shape] + ' path'
    env['WORKER_REPO'] = '/srv/recipe/' + suffix
    env['WORKER_CONFIG'] = '/srv/config/' + suffix + '.env'
    env['REMOTE_TEST_ARGUMENT'] = 'container ' + suffix
    env['REMOTE_TEST_MODE'] = operation
    output = _invoke(env, operation)
    observations = _observations(marker)
    calibrated = sum(item['clock_probe'] for item in observations) == 1
    commands = [item for item in observations if item['command_started']]
    single_command = len(commands) == 1
    valid = all(item['parse_valid'] and item['ssh_options_valid'] for item in observations)
    preserved = single_command and commands[0]['payload_preserved']
    bounded = single_command and commands[0]['envelope_valid']
    safe = not any(item['raw_key'] or item['recovered_key'] for item in observations)
    succeeded = output == IDENTITY
    assert succeeded, 'ordinary remote command failed'
    assert calibrated, 'worker monotonic clock was not calibrated'
    assert single_command, 'expected exactly one supervised SSH command'
    assert valid, 'SSH observer could not parse command structure'
    assert preserved, 'shell quoting changed ordinary argument bytes'
    assert bounded, 'worker monotonic deadline or timeout was lost'
    assert safe, 'unrelated credential reached SSH argv'


def test_expired_readiness_deadline_starts_no_ssh_process(remote):
    env, marker = remote
    life = Lifecycle(env=env, root=ROOT)
    life.deadline = life.clock() - 1
    refused = False
    try:
        life.launch_command(1, '--identity')
    except OpsError as error:
        refused = str(error) == 'native pair readiness timed out'
    started = marker.exists()
    assert refused, 'expired readiness deadline was accepted'
    assert not started, 'SSH started after the readiness deadline'
