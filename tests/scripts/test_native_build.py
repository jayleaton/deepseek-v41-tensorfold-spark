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
        'docker/zig/source/.keep', 'docker/zig/source/engine.bundle', 'patches/.keep',
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


def test_recipe_pins_upstream_and_the_image_fetches_from_it():
    # The engine is unmodified upstream TensorFold at the pin, with the recipe's patches/ applied over it.
    assert (ROOT / 'config/tensorfold.repo').read_text().strip() == 'https://github.com/ashhart/TensorFold.git'
    last = (ROOT / 'config/tensorfold.patches').read_text().strip()
    assert re.fullmatch(r'[0-9]{4}', last)
    assert re.fullmatch(r'[0-9a-f]{40}', (ROOT / 'config/tensorfold.tree').read_text().strip())
    numbers = sorted(int(p.name[:4]) for p in (ROOT / 'patches').glob('[0-9][0-9][0-9][0-9]-*.patch'))
    assert numbers[:int(last)] == list(range(1, int(last) + 1))
    dockerfile = (ROOT / 'docker/zig/Dockerfile').read_text()
    assert 'fetch-source.sh "$TENSORFOLD_REV" /ctx/src /engine-source/engine.bundle "$TENSORFOLD_REPO" "$@"' in dockerfile
    assert 'LABEL org.tensorfold.tree="${TENSORFOLD_TREE}"' in dockerfile


@pytest.fixture
def series(tmp_path, source):
    """The fixture pin plus a two-patch series over it, and the tree the series gives."""
    repo, pin = source
    work = tmp_path / 'series-work'
    subprocess.run(['git', 'clone', '-q', str(repo), str(work)], check=True)
    git(work, 'checkout', '-q', pin)
    git(work, 'config', 'user.name', 'Fixture Author')
    git(work, 'config', 'user.email', 'fixture@example.com')
    for n in (1, 2):
        (work / f'added{n}').write_text(f'patch {n}\n')
        git(work, 'add', f'added{n}')
        git(work, 'commit', '-qm', f'fixture patch {n}')
    patches = tmp_path / 'patches'
    git(work, 'format-patch', '-q', '--zero-commit', '-o', str(patches), pin + '..HEAD')
    return repo, pin, patches, git(work, 'rev-parse', 'HEAD^{tree}'), git(work, 'rev-parse', 'HEAD~1^{tree}')


def test_fetch_source_applies_the_series_and_requires_its_tree(tmp_path, series):
    repo, pin, patches, tree, first_tree = series
    bundle = tmp_path / 'engine.bundle'
    git(repo, 'bundle', 'create', str(bundle), 'HEAD')
    fetch = ['bash', str(ROOT / 'docker/zig/fetch-source.sh'), pin]
    out = tmp_path / 'all'
    subprocess.run([*fetch, str(out), str(bundle), 'https://github.com/ashhart/TensorFold.git', str(patches), '0002', tree],
                   check=True, capture_output=True, timeout=30)
    assert git(out, 'rev-parse', 'HEAD^{tree}') == tree and git(out, 'rev-parse', 'HEAD~2') == pin
    assert git(out, 'log', '-1', '--format=%an', 'HEAD') == 'Fixture Author'
    # A shorter series is a rollback: 0001 alone, with its own tree.
    subprocess.run([*fetch, str(tmp_path / 'first'), str(bundle), 'https://github.com/ashhart/TensorFold.git',
                    str(patches), '0001', first_tree], check=True, capture_output=True, timeout=30)
    for args, message in (([str(patches), '0002', first_tree], 'differs from config/tensorfold.tree'),
                          ([str(patches), '0003', tree], 'patch 0003 missing'),
                          ([str(patches), '2', tree], 'invalid patch series')):
        bad = tmp_path / ('bad' + args[1])
        result = subprocess.run([*fetch, str(bad), str(bundle), 'https://github.com/ashhart/TensorFold.git', *args],
                                capture_output=True, text=True, timeout=30, check=False)
        assert result.returncode != 0 and message in result.stderr, result.stderr


@pytest.mark.parametrize('partial', [['', '0002'], ['', '0002', 'f' * 40], ['', '', 'f' * 40]])
def test_fetch_source_refuses_partial_patch_arguments(tmp_path, series, partial):
    # An empty patch directory with a count or a (wrong) tree must not pass as the plain pin.
    repo, pin, _, _, _ = series
    bundle = tmp_path / 'engine.bundle'
    git(repo, 'bundle', 'create', str(bundle), 'HEAD')
    result = subprocess.run(['bash', str(ROOT / 'docker/zig/fetch-source.sh'), pin, str(tmp_path / 'out'),
                             str(bundle), 'https://github.com/ashhart/TensorFold.git', *partial],
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode != 0 and 'must be given together' in result.stderr, result.stderr


def test_fetch_source_without_patch_arguments_is_the_plain_pin(tmp_path, series):
    repo, pin, _, _, _ = series
    bundle = tmp_path / 'engine.bundle'
    git(repo, 'bundle', 'create', str(bundle), 'HEAD')
    out = tmp_path / 'plain'
    subprocess.run(['bash', str(ROOT / 'docker/zig/fetch-source.sh'), pin, str(out), str(bundle),
                    'https://github.com/ashhart/TensorFold.git'], check=True, capture_output=True, timeout=30)
    assert git(out, 'rev-parse', 'HEAD') == pin


def test_build_sends_the_series_and_its_tree_to_docker(tmp_path, series):
    repo, pin, patches, tree, _ = series
    recipe, tools = _build_recipe(tmp_path, pin)
    shutil.copytree(patches, recipe / 'patches')
    (recipe / 'patches/0009-not-in-the-series.patch').write_text('ignored\n')
    (recipe / 'config/tensorfold.patches').write_text('0002\n')
    (recipe / 'config/tensorfold.tree').write_text(tree + '\n')
    capture = tmp_path / 'capture'
    docker = tools / 'docker'
    docker.write_text('#!/usr/bin/env python3\nimport json,os,shutil,sys\n'
                      "shutil.copytree(sys.argv[-1], os.environ['BUILD_CAPTURE'])\n"
                      "open(os.environ['DOCKER_ARGV'],'w').write(json.dumps(sys.argv[1:]))\n")
    argv = tmp_path / 'argv.json'
    result = subprocess.run(['bash', 'scripts/build-native.sh', '-t', 'fixture:local'], cwd=recipe,
                            env=dict(os.environ, TENSORFOLD_SOURCE=str(repo), BUILD_CAPTURE=str(capture),
                                     PATH=str(tools) + os.pathsep + os.environ['PATH'], DOCKER_ARGV=str(argv)),
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    call = json.loads(argv.read_text())
    assert 'TENSORFOLD_PATCHES=0002' in call and f'TENSORFOLD_TREE={tree}' in call
    assert sorted(p.name for p in (capture / 'patches').iterdir()) == ['.keep', *sorted(p.name for p in patches.iterdir())]
    (recipe / 'config/tensorfold.tree').write_text('f' * 40 + '\n')
    argv.unlink()
    shutil.rmtree(capture)
    result = subprocess.run(['bash', 'scripts/build-native.sh', '-t', 'fixture:local'], cwd=recipe,
                            env=dict(os.environ, TENSORFOLD_SOURCE=str(repo), BUILD_CAPTURE=str(capture),
                                     PATH=str(tools) + os.pathsep + os.environ['PATH'], DOCKER_ARGV=str(argv)),
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode != 0 and not argv.exists()


def test_runtime_image_carries_the_compute_lane_interpreter():
    runtime = (ROOT / 'docker/zig/Dockerfile').read_text().split('FROM ${CUDA_RUNTIME}', 1)[1]
    install = runtime.split('RUN apt-get update', 1)[1].split('&& apt-mark', 1)[0]
    assert ' python3 ' in install.replace('\\\n', ' ')
