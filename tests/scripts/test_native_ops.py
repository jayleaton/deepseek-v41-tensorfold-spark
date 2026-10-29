"""Native supervision with isolated runtimes, clock and HTTP; never invokes Docker or SSH."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location('native_ops', Path(__file__).resolve().parents[2] / 'scripts/native_ops.py')
ops = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ops)
REV = 'a' * 40
IDENTITY = 'b' * 64


class World:
    def __init__(self, tmp_path, prefix):
        self.time = 0
        self.calls = []
        self.stop_calls = []
        self.ranks = [None, None]
        self.unavailable = set()
        self.guard = 0
        self.boot_failure = False
        self.identities = [IDENTITY, IDENTITY]
        self.prefix = prefix
        self.http_ok = True
        self.before_launch = None
        (tmp_path / 'config').mkdir()
        (tmp_path / 'config/tensorfold.rev').write_text(REV)
        self.env = {'WORKER_SSH': 'worker.example', 'WORKER_REPO': '/srv/recipe space',
                    'WORKER_CONFIG': '/srv/config with spaces.env', 'STATE_DIR': str(tmp_path / 'state'),
                    'CONTAINER_PREFIX': prefix, 'READY_TIMEOUT': '3', 'WATCH_HEAL': '1',
                    'WATCH_FAILS': '2', 'WATCH_MIN_HEAL': '60'}
        self.ops = ops.Lifecycle(self.env, runner=self.run, clock=lambda: self.time,
                                 wall=lambda: self.time, sleep=self.sleep, http=self.http, root=tmp_path)

    def sleep(self, seconds):
        self.time += seconds

    def http(self, url, timeout):
        if not self.http_ok:
            raise OSError('unreachable')
        return {'status': 'ok'} if url.endswith('/health') else {'data': [{'id': 'model'}]}

    def item(self, rank, **labels):
        return {'Name': f'/{self.prefix}-r{rank}', 'Id': str(rank + 1) * 64,
                'Config': {'Labels': {'org.tensorfold.rank': str(rank), 'org.tensorfold.revision': REV, **labels}},
                'State': {'Running': True}}

    def run(self, argv, **kwargs):
        rank = int(argv[0] == 'ssh')
        if rank:
            assert argv[1:5] == ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
            assert argv[5] == 'worker.example'
            argv = shlex.split(argv[6])
        self.calls.append((rank, argv))
        if argv[:2] == ['bash', '-c']:
            command = shlex.split(argv[2])
            assert command[:3] == ['cd', '/srv/recipe space', '&&']
            argv = command[3:]
            assert argv[1] == 'CONFIG=/srv/config with spaces.env'
        code, out = 0, ''
        if '--identity' in argv:
            out = self.identities[rank]
        elif '--detach' in argv:
            if self.before_launch:
                self.before_launch(rank)
            if self.boot_failure and rank == 0:
                code = 1
            else:
                self.ranks[rank] = self.item(rank)
        elif 'scripts/guard/lifecycle.py' in argv:
            code = self.guard
        elif argv[:3] == ['docker', 'container', 'ls']:
            code = 1 if rank in self.unavailable else 0
            out = '\n'.join(['unrelated', f'{self.prefix}-r99'] +
                            ([self.ranks[rank]['Name'][1:]] if self.ranks[rank] else []))
        elif argv[:3] == ['docker', 'container', 'inspect']:
            assert argv[3] == f'{self.prefix}-r{rank}'
            out = json.dumps([self.ranks[rank]])
        elif argv[:3] == ['docker', 'container', 'stop']:
            assert self.ops.stop_file.exists() or self.ops.lock_file.exists()
            assert argv[-1] == self.ranks[rank]['Id']
            self.stop_calls.append((rank, int(argv[4]), kwargs['timeout']))
            self.ranks[rank] = None
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, code, out, '')

    def launches(self):
        return [rank for rank, argv in self.calls if '--detach' in argv or
                (argv[:2] == ['bash', '-c'] and '--detach' in argv[2])]


@pytest.fixture(params=['tensorfold-zig', 'custom.pair'])
def world(tmp_path, request):
    return World(tmp_path, request.param)


def test_start_orders_worker_before_head_and_verifies_http(world):
    world.ops.start()
    assert world.launches() == [1, 0]
    assert world.ops.status()['ready']
    assert sum('--identity' in str(argv) for _, argv in world.calls) == 2


def test_identity_mismatch_never_launches(world):
    world.identities[1] = 'c' * 64
    with pytest.raises(ops.OpsError, match='identities'):
        world.ops.start()
    assert world.launches() == []


def test_one_rank_loss_heals_pair_after_persistent_threshold(world):
    world.ranks = [world.item(0), None]
    assert world.ops.tick() == 'unhealthy'
    assert world.launches() == []
    # New timer invocation reads the same durable counter.
    world.ops = ops.Lifecycle(world.env, runner=world.run, clock=lambda: world.time,
                              wall=lambda: world.time, sleep=world.sleep, http=world.http, root=world.ops.root)
    assert world.ops.tick() == 'healed'
    assert world.launches() == [1, 0]
    world.ranks[1] = None
    assert world.ops.tick() == 'unhealthy'
    assert world.launches() == [1, 0]


@pytest.mark.parametrize('rank', [0, 1])
def test_unavailable_runtime_is_not_missing_container(world, rank):
    world.unavailable.add(rank)
    for _ in range(3):
        with pytest.raises(ops.OpsError):
            world.ops.tick()
    assert not world.launches()
    assert not any('stop' in argv for _, argv in world.calls)


@pytest.mark.parametrize('http_failure', [False, True])
def test_boot_failure_stops_pair_and_holds(world, http_failure):
    world.boot_failure = not http_failure
    world.http_ok = not http_failure
    with pytest.raises(ops.OpsError):
        world.ops.start()
    assert world.ops.stop_file.exists()
    assert world.ranks == [None, None]
    launches = world.launches()
    assert world.ops.tick().startswith('held:')
    assert world.launches() == launches
    assert world.time <= 3


def test_operator_stop_is_durable_before_signals_and_watch_stands_down(world):
    world.ranks = [world.item(0), world.item(1)]
    original = world.ops.command

    def command(rank, argv, **kwargs):
        if argv[:3] == ['docker', 'container', 'stop']:
            assert json.loads(world.ops.stop_file.read_text())['reason'] == 'operator STOP'
        return original(rank, argv, **kwargs)

    world.ops.command = command
    world.ops.stop()
    assert world.ranks == [None, None]
    assert world.ops.tick().startswith('held:')
    assert not world.launches()


@pytest.mark.parametrize('labels', [{'org.tensorfold.rank': '9'}, {'org.tensorfold.revision': 'f' * 40}])
def test_foreign_container_untouched_but_owned_peer_stopped(world, labels):
    foreign = world.item(0, **labels)
    world.ranks = [foreign, world.item(1)]
    with pytest.raises(ops.OpsError, match='foreign'):
        world.ops.stop()
    assert world.ranks == [foreign, None]
    assert not world.launches()


@pytest.mark.parametrize('code', [75, 78])
def test_guard_hold_never_heals_or_clears_stop(world, code):
    world.ops.env['GUARD_CONFIG'] = '/srv/guard.json'
    world.guard = code
    assert world.ops.tick().startswith('held:')
    world.ops.hold('operator STOP')
    with pytest.raises(ops.OpsError) as error:
        world.ops.start()
    assert error.value.code == code
    assert world.ops.stop_file.exists()
    assert not world.launches()


def test_maintenance_and_flock_stand_down(world, tmp_path):
    world.ops.lease = tmp_path / 'maintenance'
    world.ops.lease.touch()
    assert world.ops.tick().startswith('held:')
    with pytest.raises(ops.OpsError) as error:
        world.ops.start()
    assert error.value.code == 75
    world.ops.lease.unlink()
    with world.ops.locked():
        assert world.ops.tick().startswith('held:')
    assert not world.launches()


def test_healing_is_opt_in(world):
    world.ops.env['WATCH_HEAL'] = '0'
    for _ in range(5):
        assert world.ops.tick() == 'unhealthy'
    assert not world.launches()


def test_stop_between_rank_launches_prevents_head(world):
    world.before_launch = lambda rank: world.ops.hold('operator STOP')
    with pytest.raises(ops.OpsError):
        world.ops.start()
    assert world.launches() == [1]
    assert world.ranks == [None, None]


def test_ssh_option_target_is_rejected(world):
    with pytest.raises(ops.OpsError, match='SSH target'):
        ops.Lifecycle({**world.env, 'WORKER_SSH': '-oProxyCommand=unsafe'}, root=world.ops.root)


def test_concurrent_operator_stop_during_identity_is_preserved(world):
    original = world.ops.launch_command

    def launch(rank, mode):
        result = original(rank, mode)
        if rank == 1 and mode == '--identity':
            world.ops.hold('operator STOP')
        return result

    world.ops.launch_command = launch
    with pytest.raises(ops.OpsError, match='STOP changed'):
        world.ops.start()
    assert world.ops.stop_file.exists()
    assert not world.launches()


def test_unavailable_worker_during_stop_still_stops_owned_head(world):
    world.ranks = [world.item(0), world.item(1)]
    world.unavailable.add(1)
    with pytest.raises(ops.OpsError):
        world.ops.stop()
    assert world.ranks[0] is None
    assert world.ranks[1] is not None
    assert world.ops.stop_file.exists()


@pytest.mark.parametrize('health,models', [({'ok': False, 'status': 'ok'}, {'data': [{'id': 'x'}]}),
                                         ({'ok': True}, {'data': []}),
                                         ({'ok': True, 'status': 'stalled'}, {'data': [{'id': 'x'}]}),
                                         ({}, {'data': [{'id': 'x'}]})])
def test_readiness_requires_positive_health_and_models(world, health, models):
    world.ops.http = lambda url, timeout: health if url.endswith('/health') else models
    assert not world.ops.healthy()


def test_file_auth_start_uses_public_health_and_preserves_rank_checks(world, tmp_path):
    # A nonexistent path proves lifecycle never opens the serving key file.
    world.ops.env['NATIVE_AUTH_FILE'] = str(tmp_path / 'server-only-keys')
    calls = []

    def public_http(url, timeout):
        calls.append(url)
        assert url.endswith('/health'), 'lifecycle attempted an authenticated model listing'
        return {'status': 'ok'}  # Exact keyed response at the engine pin.

    world.ops.http = public_http
    world.ops.start()
    assert world.launches() == [1, 0]
    assert world.ops.status()['ready']
    assert sum('--identity' in str(argv) for _, argv in world.calls) == 2
    assert calls
    world.ranks[1] = None
    assert not world.ops.status()['ready']


@pytest.mark.parametrize('health', [{}, {'status': 'unavailable'}, {'status': 'draining'},
                                  {'status': 'stalled'}, {'status': 'ok', 'ok': False},
                                  {'status': 'ok', 'fatal': True}])
def test_file_auth_does_not_bypass_unhealthy_engine(world, health):
    world.ops.env['NATIVE_AUTH_FILE'] = '/server-only-keys'
    world.ops.http = lambda url, timeout: health
    assert not world.ops.healthy()


def test_file_auth_does_not_bypass_identity_mismatch(world):
    world.ops.env['NATIVE_AUTH_FILE'] = '/server-only-keys'
    world.identities[1] = 'c' * 64
    with pytest.raises(ops.OpsError, match='identities'):
        world.ops.start()
    assert not world.launches()


def test_runtime_return_after_outage_can_recover(world):
    world.unavailable.add(1)
    with pytest.raises(ops.OpsError):
        world.ops.tick()
    world.unavailable.clear()
    assert world.ops.tick() == 'unhealthy'
    assert world.ops.tick() == 'healed'
    assert world.ops.status()['ready']


def test_http_timeouts_share_readiness_deadline(world):
    def slow_http(url, timeout):
        world.time += timeout
        raise TimeoutError('injected')

    world.ops.http = slow_http
    with pytest.raises(ops.OpsError, match='readiness timed out'):
        world.ops.start()
    # One initial health observation may precede boot, then READY_TIMEOUT bounds polling.
    assert world.time <= world.ops.ready_timeout
    assert world.ranks == [None, None]


def test_watch_observes_http_even_when_one_rank_is_missing(world):
    calls = []
    original = world.ops.http

    def http(url, timeout):
        calls.append(url)
        return original(url, timeout)

    world.ops.http = http
    world.ranks = [world.item(0), None]
    assert world.ops.tick() == 'unhealthy'
    assert [url.rsplit('/', 1)[-1] for url in calls] == ['health', 'models']


@pytest.mark.parametrize('settings,expected', [({}, 80),
                                               ({'TF_DSV41_DRAIN_S': '45'}, 105),
                                               ({'TF_DSV41_DRAIN_S': '45.5'}, 106),
                                               ({'STOP_TIMEOUT': '120'}, 120),
                                               ({'STOP_TIMEOUT': '5'}, 80),
                                               ({'TF_DSV41_DRAIN_S': '90', 'STOP_TIMEOUT': '100'}, 150)])
def test_stop_budget_covers_native_drain_and_subprocess_wait(world, settings, expected):
    world.ops = ops.Lifecycle({**world.env, **settings}, runner=world.run, root=world.ops.root)
    world.ranks = [world.item(0), world.item(1)]
    world.ops.stop()
    assert world.stop_calls == [(0, expected, expected + 10), (1, expected, expected + 10)]
    assert world.ranks == [None, None]


def test_guard_uses_scripts_path_and_start_operation(world):
    world.ops.env['GUARD_CONFIG'] = '/srv/guard.json'
    world.ops.policy()
    assert world.calls == [(0, [ops.sys.executable, 'scripts/guard/lifecycle.py', '--config',
                               '/srv/guard.json', 'check', '--operation', 'start'])]


def test_watch_uses_guard_watch_operation_to_hold_pending_bootstrap(world):
    world.ops.env['GUARD_CONFIG'] = '/srv/guard.json'
    world.ops.tick()
    checks = [argv for _, argv in world.calls if 'scripts/guard/lifecycle.py' in argv]
    assert checks and checks[0][-1] == 'watch'


def test_operator_stop_during_restart_drain_is_preserved(world):
    world.ranks = [world.item(0), world.item(1)]
    original = world.ops.stop_owned

    def drain():
        original()
        world.ops.hold('operator STOP during drain')

    world.ops.stop_owned = drain
    with pytest.raises(ops.OpsError, match='STOP changed'):
        world.ops.restart()
    assert json.loads(world.ops.stop_file.read_text())['reason'] == 'operator STOP during drain'
    assert not world.launches()


@pytest.mark.parametrize('already_running', [True, False])
def test_operator_stop_during_successful_readiness_drains_pair(world, already_running):
    if already_running:
        world.ranks = [world.item(0), world.item(1)]
    ready = world.ops.ready
    markers, exits = [], []

    def stop_during_ready(pair, timeout=None):
        result = ready(pair, timeout)
        if result:
            try:
                world.ops.stop()
            except ops.OpsError as error:
                exits.append(error.code)
            markers.append(world.ops.stop_file.read_bytes())
        return result

    world.ops.ready = stop_during_ready
    with pytest.raises(ops.OpsError, match='STOP'):
        world.ops.start()
    assert exits == [75]
    assert world.ranks == [None, None]
    assert len(world.stop_calls) == 2
    assert world.ops.stop_file.read_bytes() == markers[0]
    assert json.loads(markers[0])['reason'] == 'operator STOP'
    assert world.ops.tick().startswith('held:')


def test_stop_at_lifecycle_unlock_is_acknowledged_by_lock_holder(world):
    world.ranks = [world.item(0), world.item(1)]
    marker = None
    with world.ops.locked():
        with pytest.raises(ops.OpsError) as error:
            world.ops.stop()
        assert error.value.code == 75
        marker = world.ops.stop_file.read_bytes()
    assert world.ranks == [None, None]
    assert world.ops.stop_file.read_bytes() == marker



def test_stop_after_lock_acquisition_before_preflight_is_not_cleared(world):
    world.ranks = [world.item(0), world.item(1)]
    locked = world.ops.locked
    marker = None

    @contextmanager
    def stop_before_body(**kwargs):
        nonlocal marker
        with locked(**kwargs):
            with pytest.raises(ops.OpsError) as error:
                world.ops.stop()
            assert error.value.code == 75
            marker = world.ops.stop_file.read_bytes()
            yield

    world.ops.locked = stop_before_body
    with pytest.raises(ops.OpsError, match='STOP changed'):
        world.ops.start()
    assert world.ranks == [None, None]
    assert world.ops.stop_file.read_bytes() == marker
    assert not world.launches()
