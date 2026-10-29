"""Host launcher boundaries sanitize real subprocess output without running Docker."""

import json
import os
from pathlib import Path
import shlex
import shutil
import select
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SECRET = 'launcher-fixture-"credential-only'
OUTPUT = '''import json, os, sys
assert 'API_KEY' not in os.environ
value = SECRET_VALUE
data = ('diagnostic ' + value + '\\n').encode()
os.write(1, data[:17])
os.write(1, data[17:])
sys.stderr.write('escaped ' + json.dumps(value) + '\\n')
sys.stderr.write('final ' + value)
sys.exit(37)
'''.replace('SECRET_VALUE', repr(SECRET))


@pytest.fixture
def launchers(tmp_path):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    for name in ('run-zig.sh', 'serve.sh', 'build-native.sh', 'check-zig-image.sh',
                 'safe_exec.py', 'safe_quote.py', 'safe_output.py', 'redact.py', 'safe_shell.sh', 'watchdog_io.py'):
        shutil.copyfile(ROOT / 'scripts' / name, scripts / name)
    docker_dir = tmp_path / 'docker/zig'
    docker_dir.mkdir(parents=True)
    for name in ('Dockerfile', 'fetch-source.sh'):
        shutil.copyfile(ROOT / 'docker/zig' / name, docker_dir / name)
    config_dir = tmp_path / 'config'
    config_dir.mkdir()
    revision = (ROOT / 'config/tensorfold.rev').read_text().strip()
    (config_dir / 'tensorfold.rev').write_text(revision)
    assets = tmp_path / 'assets'
    assets.mkdir()
    (assets / 'sigs.json').write_text('{}')
    settings = {
        'IMAGE': 'fixture', 'TF_TP_MASTER': 'rank0.example', 'NCCL_SOCKET_IFNAME': 'eth0',
        'NCCL_IB_HCA': 'mlx5_0', 'MODEL_DIR': tmp_path / 'model', 'ENGRAM_DIR': tmp_path / 'engram',
        'ASSETS_DIR': assets, 'PREPARED_DIR': tmp_path / 'prepared', 'STATE_DIR': tmp_path / 'state',
        'CACHE_DIR': tmp_path / 'cache',
    }
    config = config_dir / 'prod-zig.env'
    config.write_text(''.join(f'{name}={shlex.quote(str(value))}\n' for name, value in settings.items()))
    for name in ('MODEL_DIR', 'ENGRAM_DIR', 'PREPARED_DIR', 'STATE_DIR', 'CACHE_DIR'):
        settings[name].mkdir()
    (scripts / 'check_native_assets.py').write_text('print("fixture-identity")\n')
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    marker = tmp_path / 'docker-called'
    docker = bindir / 'docker'
    docker.write_text(f'#!{sys.executable}\n' + '''import os, sys
from pathlib import Path
if sys.argv[1] == 'image':
    fmt = sys.argv[4]
    print(os.environ['FIXTURE_REV'] if 'org.tensorfold.revision' in fmt else
          ('linux/arm64' if 'Architecture' in fmt else 'sha256:fixture'))
    sys.exit(0)
Path(os.environ['DOCKER_CALLED']).write_text('called')
''' + OUTPUT)
    docker.chmod(0o700)
    (scripts / 'native_ops.py').write_text(OUTPUT)
    env = dict(os.environ, API_KEY=SECRET, CONFIG=str(config), FIXTURE_REV=revision,
               DOCKER_CALLED=str(marker), PATH=str(bindir) + os.pathsep + os.environ['PATH'])
    env.pop('TF_SAFE_EXEC_PARENT', None)
    return tmp_path, config, env, marker


def invoke(launchers, script, *args, xtrace=False, cwd=None):
    root, _, env, _ = launchers
    return subprocess.run(['bash', *(['-x'] if xtrace else []), str(root / 'scripts' / script), *args],
                          cwd=cwd or root, env=env, capture_output=True, text=True, timeout=15)


def assert_clean(result):
    output = result.stdout + result.stderr
    exposed = any(value in output for value in (SECRET, json.dumps(SECRET)[1:-1]))
    assert not exposed, 'credential reached launcher stdout/stderr'
    assert '<redacted>' in output


@pytest.mark.parametrize(('script', 'args'), [
    ('build-native.sh', []), ('check-zig-image.sh', ['fixture']),
    ('run-zig.sh', ['1', '--check-aot']), ('serve.sh', ['rank', '1', '--check-aot']),
    ('run-zig.sh', ['1']), ('serve.sh', ['rank', '1', '--detach']),
    ('serve.sh', ['status']),
])
def test_launcher_external_output_and_exit_code(launchers, script, args):
    result = invoke(launchers, script, *args)
    assert_clean(result)
    assert result.returncode == 37


@pytest.mark.parametrize('script', ['run-zig.sh', 'serve.sh'])
@pytest.mark.parametrize('failure', ['return', 'syntax'])
def test_config_diagnostics_redacted_before_output(launchers, script, failure, tmp_path):
    _, config, env, _ = launchers
    env.pop('API_KEY')  # The collector must learn this value from CONFIG before Bash sources it.
    prefix = 'set -x\nAPI_KEY=' + shlex.quote(SECRET) + '\nprintf "%s\\n" "$API_KEY"\n'
    suffix = 'return 43\n' if failure == 'return' else "invalid '" + SECRET + '\n'
    config.write_text(prefix + suffix)
    args = ['0', '--dry-run'] if script == 'run-zig.sh' else ['status']
    result = invoke(launchers, script, *args, xtrace=True, cwd=tmp_path.parent)
    assert_clean(result)
    assert result.returncode == (43 if failure == 'return' else 2)


@pytest.mark.parametrize(('script', 'args'), [
    ('build-native.sh', ['--build-arg']), ('check-zig-image.sh', []),
])
def test_known_credential_never_reaches_docker_argv(launchers, script, args):
    result = invoke(launchers, script, *args, SECRET)
    exposed = SECRET in result.stdout + result.stderr
    assert not exposed, 'credential reached rejection output'
    assert result.returncode != 0
    assert not launchers[3].exists()


def test_inherited_boolean_cannot_bypass_boundary(launchers):
    launchers[2]['TF_SAFE_EXEC_PARENT'] = '1'
    result = invoke(launchers, 'build-native.sh')
    assert_clean(result)
    assert result.returncode == 37


@pytest.mark.parametrize('signum', [signal.SIGTERM, signal.SIGINT])
def test_safe_exec_preserves_child_signal(launchers, signum):
    root, _, env, _ = launchers
    result = subprocess.run([sys.executable, str(root / 'scripts/safe_exec.py'), sys.executable,
                             '-c', f'import os, signal; os.kill(os.getpid(), {int(signum)})'],
                            env=env, capture_output=True, timeout=15)
    assert result.returncode == -signum


def test_safe_exec_forwards_operator_signal(launchers):
    root, _, env, _ = launchers
    program = ('import signal, sys; signal.signal(signal.SIGTERM, lambda *args: sys.exit(42)); '
               'print("ready", flush=True); signal.pause()')
    process = subprocess.Popen([sys.executable, str(root / 'scripts/safe_exec.py'),
                                sys.executable, '-c', program], env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        readable, _, _ = select.select([process.stdout], [], [], 10)
        assert readable, 'child output did not stream before exit'
        assert process.stdout.readline() == 'ready\n'
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        assert process.returncode == 42
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


@pytest.mark.parametrize('script', ['run-zig.sh', 'serve.sh'])
def test_key_loaded_during_failed_config_never_reaches_output(launchers, script, tmp_path):
    _, config, env, _ = launchers
    env.pop('API_KEY')
    key_file = tmp_path / 'credential-input'
    key_file.write_text(SECRET)
    env['KEY_FILE'] = str(key_file)
    config.write_text('API_KEY=$(cat "$KEY_FILE")\nset -x\nprintf "%s\\n" "$API_KEY"\nreturn 43\n')
    args = ['0', '--dry-run'] if script == 'run-zig.sh' else ['status']
    result = invoke(launchers, script, *args, xtrace=True)
    assert_clean(result)
    assert result.returncode == 43


def test_known_private_material_is_redacted_across_chunks_and_streams(launchers):
    root, _, env, _ = launchers
    body = "unlabelled-private-material"
    begin = "-----BEGIN " + "PRIVATE KEY-----"
    end = "-----END " + "PRIVATE KEY-----"
    # Known material is isolated from children; the output backstop also masks a
    # fixture deliberately emitting its own copy in chunks on each stream.
    env.update(PRIVATE_KEY=body, OPAQUE_PRIVATE_SETTING=body)
    program = ("import os\n" +
               "assert 'PRIVATE_KEY' not in os.environ\n" +
               "assert 'OPAQUE_PRIVATE_SETTING' not in os.environ\n" +
               "for fd in (1, 2):\n" +
               f" os.write(fd, {begin.encode()!r} + b'\\n')\n" +
               f" os.write(fd, {body[:9].encode()!r})\n" +
               f" os.write(fd, {body[9:].encode()!r} + b'\\n')\n" +
               f" os.write(fd, {end.encode()!r} + b'\\n')\n" +
               " os.write(fd, b'ordinary retry diagnostic\\n')\n")
    emitter = root / 'emit-private-block.py'
    emitter.write_text(program)
    result = subprocess.run([sys.executable, str(root / 'scripts/safe_exec.py'),
                             sys.executable, str(emitter)], env=env, capture_output=True, timeout=15)
    leaked = body.encode() in result.stdout + result.stderr
    assert result.returncode == 0 and not leaked, 'private key body reached a process stream'
    for output in (result.stdout, result.stderr):
        assert output.count(b'<redacted>') >= 2
        assert output.endswith(b'ordinary retry diagnostic\n')
        assert end.encode() in output


@pytest.mark.parametrize('script', ['run-zig.sh', 'serve.sh'])
def test_config_keeps_errexit_when_failure_precedes_valid_assignments(launchers, script, tmp_path):
    _, config, env, _ = launchers
    original = config.read_text()
    reached = tmp_path / 'configuration-continued'
    config.write_text('false\n' + original + 'touch ' + shlex.quote(str(reached)) + '\n')
    args = ['0', '--dry-run'] if script == 'run-zig.sh' else ['status']
    result = invoke(launchers, script, *args)
    assert result.returncode == 1
    assert not reached.exists(), 'failed CONFIG continued after an error'
    leaked = SECRET in result.stdout + result.stderr
    assert not leaked


@pytest.mark.parametrize('mode', ['directory', 'option'])
def test_dynamically_loaded_key_cannot_escape_shell_diagnostics(launchers, tmp_path, mode):
    _, config, env, _ = launchers
    env.pop('API_KEY')
    key_file = tmp_path / 'credential-input'
    key_file.write_text(SECRET)
    env['KEY_FILE'] = str(key_file)
    with config.open('a') as out:
        out.write('API_KEY=$(cat "$KEY_FILE")\nASSETS_DIR=/missing/$API_KEY\n')
    args = ['1'] if mode == 'directory' else ['1', SECRET]
    result = invoke(launchers, 'run-zig.sh', *args)
    leaked = SECRET in result.stdout + result.stderr
    assert not leaked, 'dynamic credential reached shell diagnostics'
    assert result.returncode == (78 if mode == 'directory' else 2)
