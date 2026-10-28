"""RR2 real-shell failures and pre-spawn argument refusal, with invented credentials."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/guard-watch-zig.sh'
KEYS = ('rr2' + 'SyntheticCredentialOnly', 'rr2' + 'Quoted\\Credential\nOnlyé')


def fixture(tmp_path, key):
    tools = tmp_path / 'tools'
    tools.mkdir()
    marker = tmp_path / 'spawned.json'
    tool = '''import json,os,sys
from pathlib import Path
key=os.environ.get('API_KEY') or os.environ['EXPECTED_VALUE']
args=sys.argv[1:]
if any(key in arg for arg in args):
 Path(os.environ['SPAWN_MARKER']).write_text(json.dumps({'credential_in_argv':True}))
if 'State.Running' in ' '.join(args): print('false')
elif 'ExitCode' in ' '.join(args): print('0')
else: print('{}')
'''
    for name in ('docker', 'ssh', 'curl'):
        p = tools / name
        p.write_text(f'#!{sys.executable}\n' + tool)
        p.chmod(0o700)
    python = tools / 'python3'
    python.write_text('#!/bin/bash\nif [[ "${1:-}" == scripts/guard/lifecycle.py ]]; then exit 0; fi\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
    python.chmod(0o700)
    env = dict(os.environ, API_KEY=key, PATH=str(tools) + os.pathsep + os.environ['PATH'],
               CONFIG='/nonexistent', WATCH_STATE_DIR=str(tmp_path / 'state'),
               WORKER_SSH='worker.example', GUARD_CONFIG='/fixture-policy',
               WATCH_PROBE='0', WATCH_HEAL='0', SPAWN_MARKER=str(marker))
    return env, marker


def run(env, snippet=None):
    argv = ['bash', str(SCRIPT), '--once'] if snippet is None else [
        'bash', '-c', 'source "$WATCH_SCRIPT"; ' + snippet]
    return subprocess.run(argv, cwd=ROOT, env=dict(env, WATCH_SCRIPT=str(SCRIPT)),
                          capture_output=True, text=True, timeout=30)


def assert_private(key, result):
    forms = (key, key.strip(), repr(key)[1:-1], json.dumps(key, ensure_ascii=True)[1:-1])
    leaked = any(part and part in result.stdout + result.stderr for part in forms)
    assert not leaked, 'known credential escaped the shell boundary'


@pytest.mark.parametrize('key', KEYS, ids=('plain', 'escaped'))
@pytest.mark.parametrize('sourced', [False, True], ids=('direct', 'sourced'))
def test_real_watchdog_numeric_failure_is_private(tmp_path, key, sourced):
    env, _ = fixture(tmp_path, key)
    env['WATCH_FAILS'] = key
    result = run(env, 'watch_tick' if sourced else None)
    assert_private(key, result)
    assert result.returncode == 78
    assert 'invalid watchdog numeric setting' in result.stderr
    assert 'unbound variable' not in result.stderr


@pytest.mark.parametrize('setting', ['WATCH_GRACE', 'WATCH_MIN_HEAL', 'WATCH_FF_MIN_HEAL',
                                     'WATCH_LEASE_MIN', 'WATCH_PROBE_EVERY', 'WATCH_PROBE_TIMEOUT',
                                     'WATCH_PROBE_FAILS', 'WATCH_INTERVAL', 'WATCH_HEAL', 'WATCH_PROBE'])
def test_all_numeric_controls_are_validated_before_arithmetic(tmp_path, setting):
    env, _ = fixture(tmp_path, KEYS[0])
    env[setting] = KEYS[0]
    result = run(env, 'watch_tick')
    assert_private(KEYS[0], result)
    assert result.returncode == 78


@pytest.mark.parametrize('failure', ['redirection', 'nounset', 'command-substitution'])
def test_sourced_operation_contains_new_shell_failure_paths(tmp_path, failure):
    key = KEYS[0]
    env, _ = fixture(tmp_path, key)
    bodies = {'redirection': ': >"/missing/$API_KEY"',
              'nounset': 'local bad=$API_KEY; (( ignored = bad ))',
              'command-substitution': 'local value; value=$( : >"/missing/$API_KEY" ); return $?'}
    # Inject a future shell failure into the operation, without replacing the
    # public entry boundary. This tests containment independently of validators.
    result = run(env, '_watch_watch_tick() { ' + bodies[failure] + '; }; watch_tick')
    assert_private(key, result)
    assert result.returncode != 0
    assert '<redacted>' in result.stderr


@pytest.mark.parametrize('entry', ['watch_run', 'watch_capture json'])
@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('operation', ['logs', 'inspect'])
def test_every_watchdog_command_refuses_key_before_spawn(tmp_path, entry, rank, operation):
    env, marker = fixture(tmp_path, KEYS[0])
    env['CONTAINER_PREFIX'] = KEYS[0]
    command = f'docker {operation} "$NAME-r0"' if rank == 0 else f'ssh "$WORKER_SSH" "docker {operation} $NAME-r1"'
    result = run(env, entry + ' ' + command)
    assert_private(KEYS[0], result)
    assert not marker.exists(), 'credential reached an external process argv'
    assert result.returncode != 0
    assert 'credential value refused' in result.stderr


def test_evidence_entry_refuses_credential_container_prefix(tmp_path):
    env, marker = fixture(tmp_path, KEYS[0])
    env['CONTAINER_PREFIX'] = KEYS[0]
    result = run(env, 'evidence_dump synthetic')
    assert_private(KEYS[0], result)
    assert not marker.exists(), 'credential reached evidence Docker/SSH argv'


@pytest.mark.parametrize('name', ['bearer', 'accessToken'])
@pytest.mark.parametrize('operation', ['log', 'docker'])
def test_unexported_caller_credentials_use_shared_name_classification(tmp_path, name, operation):
    key = KEYS[0]
    env, marker = fixture(tmp_path, key)
    env.pop('API_KEY')
    env.update(EXPECTED_VALUE=key, SOURCED_NAME=name, WATCH_SCRIPT=str(SCRIPT), CONTAINER_PREFIX=key)
    snippet = 'printf -v "$SOURCED_NAME" "%s" "$EXPECTED_VALUE"; source "$WATCH_SCRIPT"; '
    snippet += 'log "${!SOURCED_NAME}"' if operation == 'log' else 'watch_run docker logs "$NAME-r0"'
    result = subprocess.run(['bash', '-c', snippet], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=30)
    assert_private(key, result)
    assert not marker.exists(), 'unexported caller credential reached Docker argv'
    if operation == 'log':
        assert '<redacted>' in result.stderr
    else:
        assert result.returncode != 0


def test_invalid_continuous_interval_exits_instead_of_retrying_without_sleep(tmp_path):
    env, _ = fixture(tmp_path, KEYS[0])
    env['WATCH_INTERVAL'] = KEYS[0]
    result = subprocess.run(['bash', str(SCRIPT)], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=15)
    assert_private(KEYS[0], result)
    assert result.returncode == 78
    assert 'invalid watchdog numeric setting' in result.stderr


def test_rejected_continuous_sleep_terminates_the_observer(tmp_path):
    key = str(6 * 10)
    env, _ = fixture(tmp_path, key)
    env['WATCH_INTERVAL'] = key
    result = subprocess.run(['bash', str(SCRIPT)], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=15)
    assert_private(key, result)
    assert result.returncode != 0
    assert 'credential value refused' in result.stderr
