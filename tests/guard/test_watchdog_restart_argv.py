"""Real unhealthy ticks reject interpreter heal bypasses before Docker/SSH spawn."""
import json
from pathlib import Path

import pytest

from test_watchdog_redaction import fixture, no_disclosure, run

KEY = 'rr3' + 'SyntheticCredentialOnly'


def observer(tmp_path):
    env = fixture(tmp_path, KEY)
    marker = tmp_path / 'unsafe-argv'
    for name in ('docker', 'ssh'):
        path = tmp_path / 'bin' / name
        lines = path.read_text().splitlines(keepends=True)
        injected = ('import os,sys\nfrom pathlib import Path\n'
                    'assert "API_KEY" not in os.environ\n'
                    f'if any({KEY!r} in arg for arg in sys.argv[1:]):\n'
                    ' Path(os.environ["ARGV_PROBE"]).touch()\n')
        path.write_text(lines[0] + injected + ''.join(lines[1:]))
    env.update(API_KEY=KEY, WATCH_HEAL='1', WATCH_FAILS='1', ARGV_PROBE=str(marker))
    return env, marker


def tick(env):
    return run(env, 'guard_permit() { return 0; }; watch_tick')


def assert_private(env, result):
    files = [p.read_text() for p in Path(env['WATCH_STATE_DIR']).rglob('*') if p.is_file()]
    no_disclosure(KEY, result.stdout, result.stderr, *files)


@pytest.mark.parametrize('shape', ['literal', 'expanded', 'heredoc'])
def test_old_heal_strings_are_refused_before_evaluation(tmp_path, shape):
    env, marker = observer(tmp_path)
    values = {'literal': 'docker logs ' + KEY,
              'expanded': 'docker logs "$API_KEY"',
              'heredoc': "read -r name <<'EOF'\n" + KEY + '\nEOF\ndocker logs "$name"'}
    env['WATCH_RESTART_CMD'] = values[shape]
    result = tick(env)
    assert_private(env, result)
    assert not marker.exists(), 'old heal shell reached Docker with a credential'
    assert result.returncode == 78
    assert 'WATCH_RESTART_ARGV' in result.stderr
    assert not (tmp_path / 'healed').exists()


@pytest.mark.parametrize('program', ['docker', 'ssh'])
@pytest.mark.parametrize('argument', ['literal', 'environment'])
def test_restart_argv_checks_resolved_values_before_spawn(tmp_path, program, argument):
    env, marker = observer(tmp_path)
    arg = KEY if argument == 'literal' else {'env': 'API_KEY'}
    env['WATCH_RESTART_ARGV'] = json.dumps([program, 'logs', arg])
    result = tick(env)
    assert_private(env, result)
    assert not marker.exists(), 'structured heal reached a process with a credential'
    assert result.returncode != 0
    log = tmp_path / 'state/heal.log'
    # Literal values may now fail before a heal starts. Environment references
    # to stripped secret names fail as unavailable at the checked restart.
    if log.exists():
        assert ('credential value refused' in log.read_text()
                or 'restart environment reference unavailable' in log.read_text())
    else:
        assert result.returncode == 78
        assert 'credential value refused' in result.stderr


@pytest.mark.parametrize('argv', [
    ['bash', '-c', 'docker logs "$API_KEY"'],
    ['python3', '-c', 'import os; os.system("docker logs $API_KEY")'],
    ['bash', 'scripts/serve.sh', 'start'],
    ['bash', 'scripts/serve.sh', 'restart'],
    ['bash', '/tmp/unknown-launcher', 'watch', '--once'],
])
def test_restart_argv_cannot_select_an_interpreter_or_stop_clearing_path(tmp_path, argv):
    env, marker = observer(tmp_path)
    env['WATCH_RESTART_ARGV'] = json.dumps(argv)
    result = tick(env)
    assert_private(env, result)
    assert not marker.exists()
    assert result.returncode != 0
    assert not (tmp_path / 'healed').exists()


@pytest.mark.parametrize('exit_code', [0, 47])
def test_reviewed_restart_is_synchronous_and_preserves_failure_status(tmp_path, exit_code):
    env, marker = observer(tmp_path)
    env['RECIPE_LAUNCHER'] = str(Path(__file__).resolve().parents[2] / 'scripts/serve.sh')
    env['WATCH_RESTART_ARGV'] = json.dumps(['bash', {'env': 'RECIPE_LAUNCHER'}, 'watch', '--once'])
    env['HEAL_EXIT'] = str(exit_code)
    lock_probe = tmp_path / 'lock-held'
    env['LOCK_PROBE'] = str(lock_probe)
    tool = Path(env['HEAL_EXEC'])
    source = tool.read_text().replace('elif role == "heal-tool":', 'elif role == "heal-tool":\n'
        '    import fcntl\n'
        '    with Path(os.environ["WATCH_STATE_DIR"], "lock").open("a") as lock:\n'
        '        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n'
        '        except BlockingIOError: Path(os.environ["LOCK_PROBE"]).touch()')
    tool.write_text(source)
    result = tick(env)
    assert_private(env, result)
    assert not marker.exists()
    assert (tmp_path / 'healed').exists(), 'tick returned before the reviewed heal completed'
    assert lock_probe.exists(), 'restart did not retain its lock through execution'
    assert result.returncode == (1 if exit_code == 0 else exit_code)
