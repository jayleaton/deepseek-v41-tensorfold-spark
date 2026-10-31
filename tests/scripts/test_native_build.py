"""Resolve exact native pins from local Git objects without modifying the source."""
from pathlib import Path
import json
import os
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


@pytest.fixture
def source(tmp_path):
    repo = tmp_path / 'engine'
    repo.mkdir()
    git(repo, 'init', '-q')
    git(repo, 'config', 'user.name', 'JJ Eaton')
    git(repo, 'config', 'user.email', 'fixture@example.com')
    (repo / 'value').write_text('pinned\n')
    git(repo, 'add', 'value')
    git(repo, 'commit', '-qm', 'fixture pin')
    pin = git(repo, 'rev-parse', 'HEAD')
    (repo / 'value').write_text('later\n')
    git(repo, 'commit', '-qam', 'fixture later')
    (repo / 'value').write_text('dirty\n')
    return repo, pin


@pytest.mark.parametrize('kind', ['checkout', 'bundle'])
@pytest.mark.parametrize('docker_exit', [0, 37])
def test_build_resolves_pin_and_cleans_private_context(tmp_path, source, kind, docker_exit):
    repo, pin = source
    source_head = git(repo, 'rev-parse', 'HEAD')
    recipe = tmp_path / 'recipe'
    for name in ('scripts/build-native.sh', 'scripts/safe_exec.py', 'scripts/safe_output.py',
                 'scripts/redact.py', 'scripts/safe_quote.py', 'scripts/safe_shell.sh', 'scripts/watchdog_io.py', 'docker/zig/Dockerfile', 'docker/zig/fetch-source.sh'):
        target = recipe / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (recipe / 'config').mkdir()
    (recipe / 'config/tensorfold.rev').write_text(pin + '\n')
    incoming = repo
    if kind == 'bundle':
        incoming = tmp_path / 'engine.bundle'
        git(repo, 'bundle', 'create', str(incoming), 'HEAD')
    capture = tmp_path / 'capture'
    tools = tmp_path / 'tools'
    tools.mkdir()
    docker = tools / 'docker'
    docker.write_text('''#!/usr/bin/env python3
import json,os,shutil,sys
from pathlib import Path
context=Path(sys.argv[-1])
capture=Path(os.environ['BUILD_CAPTURE'])
shutil.copytree(context,capture)
(capture/'call.json').write_text(json.dumps({'argv':sys.argv[1:],'context':str(context)}))
sys.exit(int(os.environ['DOCKER_EXIT']))
''')
    docker.chmod(0o700)
    result = subprocess.run(['bash', 'scripts/build-native.sh', '-t', 'fixture:local'],
                            cwd=recipe, env=dict(os.environ, TENSORFOLD_SOURCE=str(incoming),
                                                PATH=str(tools) + os.pathsep + os.environ['PATH'],
                                                BUILD_CAPTURE=str(capture), DOCKER_EXIT=str(docker_exit)),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == docker_exit, result.stderr
    call = json.loads((capture / 'call.json').read_text())
    assert f'TENSORFOLD_REV={pin}' in call['argv']
    assert 'TENSORFOLD_REPO=https://github.com/ashhart/TensorFold.git' in call['argv']
    assert not Path(call['context']).parent.exists()
    assert sorted(p.relative_to(capture).as_posix() for p in capture.rglob('*') if p.is_file()) == [
        'call.json', 'docker/zig/Dockerfile', 'docker/zig/fetch-source.sh',
        'docker/zig/source/.keep', 'docker/zig/source/engine.bundle',
        'scripts/redact.py', 'scripts/safe_exec.py', 'scripts/safe_output.py', 'scripts/safe_quote.py',
        'scripts/safe_shell.sh', 'scripts/watchdog_io.py']
    output = tmp_path / 'resolved'
    subprocess.run(['bash', str(capture / 'docker/zig/fetch-source.sh'), pin, str(output),
                    str(capture / 'docker/zig/source/engine.bundle')], check=True, capture_output=True)
    assert git(output, 'rev-parse', 'HEAD') == pin
    assert (output / 'value').read_text() == 'pinned\n'
    assert git(repo, 'rev-parse', 'HEAD') == source_head
    assert (repo / 'value').read_text() == 'dirty\n'


@pytest.mark.parametrize('kind', ['invalid', 'wrong-pin', 'incremental'])
def test_supplied_bundle_fails_without_network_fallback(tmp_path, source, kind):
    repo, pin = source
    bundle = tmp_path / 'bad.bundle'
    if kind == 'invalid':
        bundle.write_text('invalid bundle')
    elif kind == 'wrong-pin':
        git(repo, 'bundle', 'create', str(bundle), 'HEAD')
        pin = '0' * 40
    else:
        git(repo, 'bundle', 'create', str(bundle), 'HEAD', '^' + pin)
        pin = git(repo, 'rev-parse', 'HEAD')
    result = subprocess.run(['bash', str(ROOT / 'docker/zig/fetch-source.sh'), pin,
                             str(tmp_path / 'resolved'), str(bundle)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'github.com' not in result.stderr


def _build_recipe(tmp_path, pin):
    recipe = tmp_path / 'recipe'
    for name in ('scripts/build-native.sh', 'scripts/safe_exec.py', 'scripts/safe_output.py',
                 'scripts/redact.py', 'scripts/safe_quote.py', 'scripts/safe_shell.sh', 'scripts/watchdog_io.py',
                 'docker/zig/Dockerfile', 'docker/zig/fetch-source.sh'):
        target = recipe / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (recipe / 'config').mkdir()
    (recipe / 'config/tensorfold.rev').write_text(pin + '\n')
    tools = tmp_path / 'tools'
    tools.mkdir()
    docker = tools / 'docker'
    docker.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                      "open(os.environ['DOCKER_ARGV'],'w').write(json.dumps(sys.argv[1:]))\n")
    docker.chmod(0o700)
    return recipe, tools


@pytest.mark.parametrize(('repo', 'code'), [('https://github.com/example/TensorFold.git', 0),
                                            ('https://github.com/example/TensorFold.git --upload-pack=x', 2),
                                            ('ssh://example.invalid/TensorFold.git', 2)])
def test_build_fetches_the_pin_from_the_configured_repository(tmp_path, repo, code):
    recipe, tools = _build_recipe(tmp_path, 'a' * 40)
    (recipe / 'config/tensorfold.repo').write_text(repo + '\n')
    argv = tmp_path / 'argv.json'
    env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'], DOCKER_ARGV=str(argv))
    env.pop('TENSORFOLD_SOURCE', None)
    result = subprocess.run(['bash', 'scripts/build-native.sh', '-t', 'fixture:local'], cwd=recipe, env=env,
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == code, result.stderr
    if code == 0:
        assert f'TENSORFOLD_REPO={repo}' in json.loads(argv.read_text())
    else:
        assert not argv.exists()


def test_recipe_pins_a_public_repository_and_the_image_fetches_from_it():
    repo = (ROOT / 'config/tensorfold.repo').read_text().strip()
    assert re.fullmatch(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git', repo)
    dockerfile = (ROOT / 'docker/zig/Dockerfile').read_text()
    assert 'fetch-source.sh "$TENSORFOLD_REV" /ctx/src /engine-source/engine.bundle "$TENSORFOLD_REPO"' in dockerfile


def test_runtime_image_carries_the_compute_lane_interpreter():
    runtime = (ROOT / 'docker/zig/Dockerfile').read_text().split('FROM ${CUDA_RUNTIME}', 1)[1]
    install = runtime.split('RUN apt-get update', 1)[1].split('&& apt-mark', 1)[0]
    assert ' python3 ' in install.replace('\\\n', ' ')
