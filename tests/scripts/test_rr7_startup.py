"""Implicit shell startup files cannot restore keys after the checked env scrub."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from safe_exec import child_environment  # noqa: E402

HOOKS = ('BASH_ENV', 'ENV', 'PROMPT_COMMAND', 'SHELLOPTS', 'BASHOPTS', 'PS4',
         'BASH_FUNC_restored%%')


def test_shared_child_env_removes_all_shell_startup_hooks(monkeypatch):
    monkeypatch.delenv('CONFIG', raising=False)
    source = {'PATH': os.defpath, **dict.fromkeys(HOOKS, 'ordinary-startup-setting')}
    assert child_environment(source) == {'PATH': os.defpath}


@pytest.mark.parametrize('parent_knows', [False, True])
@pytest.mark.parametrize('shebang', [False, True])
def test_checked_bash_cannot_restore_startup_only_key(tmp_path, parent_knows, shebang):
    value = 'fake-rr7-startup-only-observer-value'
    startup = tmp_path / 'startup.sh'
    startup.write_text('export API_KEY=' + value + '\n')
    marker = tmp_path / 'observer.json'
    observer = tmp_path / 'observe.py'
    observer.write_text('import json, os, sys\nfrom pathlib import Path\n'
                        f'Path({str(marker)!r}).write_text(json.dumps({{"key_present": "API_KEY" in os.environ, "hooks_present": any(n in os.environ for n in {HOOKS!r})}}))\n'
                        'print("diagnostic: " + os.environ.get("API_KEY", "absent"))\n'
                        'print("diagnostic: " + os.environ.get("API_KEY", "absent"), file=sys.stderr)\n'
                        'raise SystemExit(47)\n')
    script = tmp_path / 'observe.sh'
    script.write_text(f'#!/bin/bash\nexec {sys.executable} {observer}\n')
    script.chmod(0o700)
    command = [str(script)] if shebang else ['bash', '-c', 'exec "$1" "$2"',
                                                            'probe', sys.executable, str(observer)]
    env = {'PATH': os.defpath, 'HOME': str(tmp_path), 'BASH_ENV': str(startup)}
    if parent_knows:
        env['API_KEY'] = value
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/safe_exec.py'),
                             '--command', *command], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 47
    assert json.loads(marker.read_text()) == {'key_present': False, 'hooks_present': False}
    assert value not in result.stdout + result.stderr
    assert 'diagnostic: absent' in result.stdout and 'diagnostic: absent' in result.stderr


def test_native_aot_config_cannot_restore_startup_only_key(tmp_path):
    value = 'fake-rr7-config-startup-only-value'
    startup = tmp_path / 'startup.sh'
    startup.write_text('export API_KEY=' + value + '\n')
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    assets = tmp_path / 'assets'
    (assets / 'aot').mkdir(parents=True)
    (assets / 'sigs.json').write_text('{}')
    (assets / 'aot/aot.json').write_text('{}')
    marker = tmp_path / 'child.json'
    revision = (ROOT / 'config/tensorfold.rev').read_text().strip()
    docker = bindir / 'docker'
    docker.write_text('#!/bin/bash\nif [[ "$1" == image ]]; then\n'
                      f' if [[ "$4" == *revision* ]]; then echo {revision}; else echo sha256:{"f" * 64}; fi\n'
                      'else\n'
                      ' if [[ -n "${API_KEY:-}" ]]; then echo true > "$OBSERVATION"; else echo false > "$OBSERVATION"; fi\n'
                      ' printf "diagnostic: %s\\n" "${API_KEY:-absent}"\n'
                      ' printf "diagnostic: %s\\n" "${API_KEY:-absent}" >&2\nexit 47\nfi\n')
    docker.chmod(0o700)
    values = {'IMAGE': 'fixture', 'TF_TP_MASTER': 'rank0.example', 'NCCL_SOCKET_IFNAME': 'eth0',
              'NCCL_IB_HCA': 'mlx5_0', 'ASSETS_DIR': assets, 'BASH_ENV': startup}
    values.update({name: tmp_path / name.lower() for name in
                   ('MODEL_DIR', 'ENGRAM_DIR', 'PREPARED_DIR', 'STATE_DIR', 'CACHE_DIR')})
    config = tmp_path / 'config.env'
    config.write_text(''.join(f'{name}={val}\n' for name, val in values.items()))
    env = {'PATH': str(bindir) + ':' + os.defpath, 'HOME': str(tmp_path),
           'CONFIG': str(config), 'OBSERVATION': str(marker)}
    result = subprocess.run(['bash', str(ROOT / 'scripts/run-zig.sh'), '1', '--check-aot'],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 47
    assert marker.read_text().strip() == 'false'
    assert value not in result.stdout + result.stderr
    assert 'diagnostic: absent' in result.stdout and 'diagnostic: absent' in result.stderr


def test_checked_bash_disables_rc_loading(tmp_path):
    # Startup-only value is never known to the parent output backstop.
    value = 'fake-rr7-login-profile-only-value'
    profile = tmp_path / 'login.sh'
    profile.write_text('export API_KEY=' + value + '\n')
    observer = tmp_path / 'observe.py'
    observer.write_text('import os; print(os.environ.get("API_KEY", "absent")); raise SystemExit(47)\n')
    # --rcfile provides a deterministic owned startup selector without touching
    # account-global profiles or relying on a particular machine home directory.
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/safe_exec.py'), '--command',
         'bash', '--rcfile', str(profile), '-ic',
         'exec "$1" "$2"', 'probe', sys.executable, str(observer)],
        env={'PATH': os.defpath, 'HOME': str(tmp_path)}, cwd=ROOT,
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 47
    assert value not in result.stdout + result.stderr
    assert result.stdout.strip() == 'absent'
