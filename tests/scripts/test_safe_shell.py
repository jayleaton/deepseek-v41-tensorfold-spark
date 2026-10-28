"""Exercise the shared boundary through actual Python and Bash processes."""

import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXEC = ROOT / "scripts/safe_exec.py"
SHELL = ROOT / "scripts/safe_shell.sh"


def invoke(*args, payload="", key="rr2SyntheticBoundaryValue"):
    return subprocess.run(
        [sys.executable, str(EXEC), *args],
        input=payload,
        capture_output=True,
        text=True,
        env=dict(os.environ, API_KEY=key),
        timeout=20,
    )


def assert_private(result, key):
    text = result.stdout + result.stderr
    variants = (key, repr(key)[1:-1], json.dumps(key)[1:-1])
    leaked = any(value in text for value in variants)
    assert not leaked, "shared boundary disclosed a configured credential"
    assert "Traceback (most recent call last)" not in text


@pytest.fixture(params=["bash", "bash32"])
def bash(request):
    if request.param == "bash":
        return "bash"
    executable = os.environ.get("BASH32")
    if not executable:
        pytest.skip("set BASH32 to exercise real GNU Bash 3.2")
    version = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, timeout=20
    )
    assert "version 3.2." in version.stdout
    return executable


def shell(bash, body, *, key="rr2SyntheticBoundaryValue"):
    return subprocess.run(
        [bash, "-c", "source " + shlex.quote(str(SHELL)) + "\n" + body],
        capture_output=True,
        text=True,
        env=dict(os.environ, API_KEY=key),
        timeout=20,
    )


@pytest.mark.parametrize("key", ["rr2SyntheticBoundaryValue", 'rr2"\\BoundaryValue'])
def test_text_filter_masks_both_literal_and_escaped_records(key):
    result = invoke(
        "--filter", payload=key + "\n" + json.dumps({"echo": key}) + "\n", key=key
    )
    assert_private(result, key)
    assert result.returncode == 0
    assert "<redacted>" in result.stdout


@pytest.mark.parametrize(
    "mode,payload,key,expected",
    [
        ("uint", "200\n", "200", "200\n"),
        ("running", "true\n", "true", "true\n"),
        ("choices", '{"choices": []}', "choices", "1\n"),
        ("healthy", '{"ok": true, "errors": 0}', "true", "1\n"),
    ],
)
@pytest.mark.parametrize("command", [False, True])
def test_control_parses_raw_before_sanitizing(mode, payload, key, expected, command):
    args = ["--control", mode]
    if command:
        args += [
            "--command", sys.executable, "-c",
            "import sys; sys.stdout.write(sys.stdin.read())",
        ]
    else:
        args += ["--filter"]
    result = invoke(*args, payload=payload, key=key)
    assert result.returncode == 0
    assert result.stdout == expected
    assert result.stderr == ""


def test_control_projection_masks_only_diagnostics():
    key = "choices"
    result = invoke(
        "--filter", "--control", "http_control",
        payload=json.dumps({"choices": [], "diagnostic": key}) + "\n200\n", key=key,
    )
    assert result.returncode == 0
    code, document = result.stdout.split("\n", 1)
    value = json.loads(document)
    assert code == "200"
    assert value["choices"] == []
    assert value["diagnostic"] == "<redacted>"


def test_control_command_sanitizes_stderr_independently():
    result = invoke(
        "--control", "uint", "--command", sys.executable, "-c",
        'import os,sys; sys.stdout.write(sys.stdin.read()); '
        'sys.stderr.write(os.environ["API_KEY"])',
        payload="200\n", key="200",
    )
    assert result.returncode == 0
    assert result.stdout == "200\n"
    assert result.stderr == "<redacted>"


def test_control_accepts_exact_buffer_limit():
    result = invoke(
        "--filter", "--control", "uint", payload=" " * (8 * 1024 * 1024 - 1) + "7"
    )
    assert result.returncode == 0
    assert result.stdout == "7\n"
    assert result.stderr == ""


@pytest.mark.parametrize("valid", [False, True])
def test_control_finish_is_idempotent(valid, monkeypatch, capsys):
    # Exercise the actual writer without process timing or a second EOF race.
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("rr2_safe_exec", EXEC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from safe_output import _Stream

    output = io.StringIO()
    stream = module._Control(_Stream(output, []), "uint", [])
    stream.write("200" if valid else "invalid")
    stream.finish()
    first = output.getvalue(), capsys.readouterr().err
    stream.finish()
    assert output.getvalue() == first[0]
    assert capsys.readouterr().err == ""
    assert stream.failed is not valid
    assert bool(first[0]) is valid
    assert bool(first[1]) is not valid


@pytest.mark.parametrize("command", [False, True])
@pytest.mark.parametrize("oversized", [False, True])
def test_control_failure_withholds_raw_output(command, oversized):
    key = "rr2SyntheticBoundaryValue"
    payload = key if not oversized else "x" * (8 * 1024 * 1024 + 1)
    args = ["--control", "uint"]
    args += (["--command", sys.executable, "-c",
              "import sys; sys.stdout.write(sys.stdin.read())"] if command else ["--filter"])
    result = invoke(*args, payload=payload, key=key)
    assert_private(result, key)
    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr == "control parse failed; output withheld\n"


@pytest.mark.parametrize("helper", ["safe_command", "safe_capture uint"])
def test_shell_commands_refuse_secret_arguments_before_spawn(bash, helper):
    result = shell(bash, helper + ' python3 -c \'print("spawned")\' "$API_KEY"')
    assert_private(result, "rr2SyntheticBoundaryValue")
    assert result.returncode != 0
    assert "spawned" not in result.stdout
    assert "refused" in result.stderr


@pytest.mark.parametrize(
    "body",
    [
        'set -u; local count=$API_KEY; (( count >= 3 ))',
        'printf x > "/nonexistent/$API_KEY/output"',
        '"$API_KEY"',
        'printf "%s\\n" "$1" >&2; return 43',
    ],
)
def test_shell_contains_interpreter_and_function_diagnostics(bash, body):
    result = shell(
        bash, 'body() { ' + body + '; }\nsafe_shell text body "$API_KEY"'
    )
    assert_private(result, "rr2SyntheticBoundaryValue")
    assert result.returncode != 0
    assert "<redacted>" in result.stderr


@pytest.mark.parametrize("code", [0, 43])
def test_shell_waits_for_both_filters_and_preserves_status(bash, code):
    result = shell(
        bash,
        'body() { printf "%s\\n" "$API_KEY"; '
        'printf "%s\\n" "$API_KEY" >&2; return ' + str(code) + '; }\n'
        'safe_shell text body',
    )
    assert_private(result, "rr2SyntheticBoundaryValue")
    assert result.returncode == code
    assert result.stdout == "<redacted>\n"
    assert result.stderr == "<redacted>\n"


def test_shell_control_keeps_functions_and_raw_arguments_in_process(bash):
    result = shell(
        bash,
        'guard_permit() { printf "true\\n"; }\n'
        'body() { printf "%s\\n" "$1" >&2; guard_permit; }\n'
        'safe_shell running body "$API_KEY"',
        key="true",
    )
    assert result.returncode == 0
    assert result.stdout == "true\n"
    assert result.stderr == "<redacted>\n"


def test_shell_rejects_external_program_body(bash):
    result = shell(bash, "safe_shell text printf forbidden")
    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr == "shell boundary requires a function\n"


def test_shell_library_can_be_sourced_twice_with_readonly_path(bash):
    result = shell(
        bash,
        'source ' + shlex.quote(str(SHELL)) + '\n'
        'declare -p safe_exec\n'
        'safe_command python3 -c \'print("ready")\'',
    )
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.startswith("declare -r safe_exec=")
    assert result.stdout.endswith("ready\n")


def test_main_bootstrap_covers_full_script(bash, tmp_path):
    script = tmp_path / "entry.sh"
    script.write_text(
        'source ' + shlex.quote(str(SHELL)) + '\n'
        'safe_main "${BASH_SOURCE[0]}" "$@"\n'
        'printf "%s\\n" "$API_KEY"\n'
        'printf "%s\\n" "$API_KEY" >&2\n'
        'exit 43\n'
    )
    result = subprocess.run(
        [bash, str(script)], capture_output=True, text=True,
        env=dict(os.environ, API_KEY="rr2" + "SyntheticBoundaryValue"), timeout=20,
    )
    assert_private(result, "rr2SyntheticBoundaryValue")
    assert result.returncode == 43
    assert result.stdout == "<redacted>\n"
    assert result.stderr == "<redacted>\n"


@pytest.mark.parametrize("failure", ["missing", "explicit"])
def test_main_bootstrap_preserves_malformed_argument_status(bash, tmp_path, failure):
    script = tmp_path / "invalid.sh"
    script.write_text(
        'source ' + shlex.quote(str(SHELL)) + '\n'
        'if [[ ${TEST_SKIP_BOOTSTRAP:-0} != 1 ]]; then\n'
        '    safe_main "${BASH_SOURCE[0]}" "$@"\n'
        'fi\n'
        'set -u\n'
        + ('argument=${1:?usage: invalid.sh ARG}\n' if failure == "missing" else
           'printf "%s\\n" "$API_KEY" >&2; exit 2\n')
    )
    baseline = subprocess.run(
        [bash, str(script)], capture_output=True, text=True,
        env=dict(os.environ, API_KEY="rr2" + "SyntheticBoundaryValue", TEST_SKIP_BOOTSTRAP="1"),
        timeout=20,
    )
    result = subprocess.run(
        [bash, str(script)], capture_output=True, text=True,
        env=dict(os.environ, API_KEY="rr2" + "SyntheticBoundaryValue"), timeout=20,
    )
    assert_private(result, "rr2SyntheticBoundaryValue")
    assert baseline.returncode != 0
    assert result.returncode == baseline.returncode
    assert result.stdout == ""
    assert result.stderr


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGKILL])
@pytest.mark.parametrize("kind", ["command", "capture", "shell", "run"])
def test_sourced_owner_death_drains_owned_groups(bash, tmp_path, signum, kind):
    if not Path("/proc/self/stat").exists():
        pytest.skip("Linux kernel process identities required")
    address = str(tmp_path / "owner.sock")
    effect = tmp_path / "late-effect"
    client = tmp_path / "client.py"
    client.write_text('''import json, os, signal, socket, sys
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGHUP, signal.SIG_IGN)
os.fork()
with socket.socket(socket.AF_UNIX) as channel:
    channel.connect(sys.argv[1])
    channel.sendall((json.dumps([os.getpid(), os.getpgrp()]) + "\\n").encode())
    if channel.recv(1):
        Path(sys.argv[2]).write_text("late effect")
''')
    runner = tmp_path / "runner.py"
    runner.write_text(
        'import sys\n'
        'sys.path.insert(0, ' + repr(str(ROOT / "scripts")) + ')\n'
        'from safe_exec import run\n'
        # An explicit child env must still propagate verified ownership.
        'run(sys.argv[1:], env={})\n'
    )
    command = 'python3 ' + shlex.quote(str(client)) + ' ' + shlex.quote(address) + ' ' + shlex.quote(str(effect))
    if kind == "command":
        body = 'safe_command ' + command
    elif kind == "capture":
        body = 'safe_capture uint ' + command
    elif kind == "shell":
        body = 'heal() { safe_command ' + command + '; }\nsafe_shell text heal'
    else:
        body = 'python3 ' + shlex.quote(str(runner)) + ' ' + command
    channels, groups = [], set()
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(address)
        server.listen()
        server.settimeout(20)
        caller = subprocess.Popen(
            [bash, '-c', 'source ' + shlex.quote(str(SHELL)) + '\n' + body + '\n:'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=dict(os.environ, API_KEY="rr2" + "SyntheticBoundaryValue"),
            start_new_session=True,
        )
        try:
            for _ in range(2):
                channel, _ = server.accept()
                channels.append(channel)
                channel.settimeout(20)
                record = b''
                while not record.endswith(b'\n'):
                    data = channel.recv(1024)
                    assert data, "client exited before readiness"
                    record += data
                _, group = json.loads(record)
                groups.add(group)
                assert group != caller.pid, "command did not get an owned group"
            # Kill only the sourcing Bash, with its descendants demonstrably live.
            # wait() deliberately comes after EOF to cover zombie-owner detection.
            caller.send_signal(signum)
            for channel in channels:
                assert channel.recv(1) == b'', "owned writer survived sourcing Bash"
            assert not effect.exists()
            out, err = caller.communicate(timeout=20)
            leaked = b"rr2SyntheticBoundaryValue" in out + err
            assert not leaked, "owner cancellation disclosed a credential"
            assert caller.returncode == -signum
            assert not effect.exists()
        finally:
            if caller.poll() is None:
                caller.kill()
            for group in groups | {caller.pid}:
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            caller.communicate(timeout=20)
            for channel in channels:
                channel.close()


@pytest.mark.parametrize("forged", ["start", "unrelated"])
def test_owner_identity_cannot_be_forged(forged):
    unrelated = None
    try:
        if forged == "unrelated":
            # PID 1 may legitimately be pytest itself inside the CPU container.
            # Create a live sibling of the supervisor, never borrow another job.
            unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
            owner = unrelated.pid
        else:
            owner = os.getpid()
        with open(f'/proc/{owner}/stat') as source:
            fields = source.read().rsplit(')', 1)[1].split()
        environment = dict(os.environ)
        environment['TF_SAFE_EXEC_OWNER_PID'] = str(owner)
        environment['TF_SAFE_EXEC_OWNER_START'] = '0' if forged == "start" else fields[19]
        result = subprocess.run(
            [sys.executable, str(EXEC), '--command', sys.executable, '-c', 'print("spawned")'],
            capture_output=True, text=True, env=environment, timeout=20,
        )
        assert result.returncode != 0
        assert result.stdout == ''
        assert 'invalid shell owner' in result.stderr
    finally:
        if unrelated is not None:
            unrelated.terminate()
            unrelated.wait(timeout=10)


def test_namespace_init_can_be_the_actual_shell_owner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("safe_exec_owner", EXEC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('TF_SAFE_EXEC_OWNER_PID', '1')
    monkeypatch.setenv('TF_SAFE_EXEC_OWNER_START', 'fixture-start')
    monkeypatch.setattr(module.os, 'getpid', lambda: 42)
    identities = {42: ('R', 1, 'child-start'), 1: ('S', 0, 'fixture-start')}
    monkeypatch.setattr(module, '_process_identity', identities.get)
    assert module._owner() == (1, 'fixture-start')


def test_inprocess_runner_rejects_credentials_loaded_by_outer_cli(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import safe_output
    import safe_exec
    key = 'rr2' + 'ConfiguredBoundaryOnly'
    config = tmp_path / 'control.json'
    config.write_text(json.dumps({'api_key': key}))
    marker = tmp_path / 'spawned'
    tool = tmp_path / 'docker'
    tool.write_text('#!/bin/sh\n: > "$1"\n')
    tool.chmod(0o700)
    monkeypatch.setattr(sys, 'argv', ['control', '--config', str(config)])
    def operation():
        with pytest.raises(ValueError):
            safe_exec.run([str(tool), str(marker), key], timeout=10)
    assert safe_output.cli(operation) == 0
    assert not marker.exists(), 'configured credential reached Docker argv'


def test_preflight_docker_log_reader_uses_argument_boundary(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import preflight
    key = 'rr2' + 'PreflightContainerOnly'
    marker = tmp_path / 'spawned'
    tool = tmp_path / 'docker'
    tool.write_text('#!/bin/sh\n: > "' + str(marker) + '"\n')
    tool.chmod(0o700)
    monkeypatch.setenv('API_KEY', key)
    monkeypatch.setattr(preflight.shutil, 'which', lambda name: str(tool))
    with pytest.raises(ValueError):
        preflight.boot_log_text(None, key)
    assert not marker.exists(), 'preflight disclosed a credential in Docker argv'
