"""Exercise watchdog disclosure boundaries with invented keys and fake external tools."""

import json
import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/guard-watch-zig.sh"
# Every value is invented. Assertions intentionally never include captured output
# or the value itself, even when run against the leaking baseline.
KEYS = ("rr1SyntheticCredentialOnly", "rr1'Quoted\\Credential\nOnlyé", "q7!z")
TOOL = r"""
import datetime
import json
import os
import sys
from pathlib import Path

controls = {}  # fixture-protocol-controls
role = Path(sys.argv[0]).name
args = " ".join(sys.argv[1:])
assert "AUTH_TOKEN" not in os.environ and "API_KEY" not in os.environ
assert "OPAQUE_SETTING" not in os.environ
key = FIXTURE_KEY
def echoes(label):
    print(label + " diagnostic")
    print(label + " error diagnostic", file=sys.stderr)
    print(label + " opaque " + json.dumps(key, ensure_ascii=False))
    print(label + " escaped " + json.dumps(key, ensure_ascii=True))
    print(label + " repr " + repr(key), file=sys.stderr)
    print(label + " error " + json.dumps(key, ensure_ascii=False), file=sys.stderr)

if role in ("docker", "ssh") and "State.Running" in args:
    print(controls.get("FAKE_RUNNING", "true"))
elif role == "docker" and "StartedAt" in args:
    print("started-at diagnostic " + json.dumps(key, ensure_ascii=False), file=sys.stderr)
    print(controls.get("FAKE_STARTED") or (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)).isoformat())
elif role in ("docker", "ssh") and "ExitCode" in args:
    print(controls.get("FAKE_EXIT", "0"))
elif role == "curl":
    if "/chat/completions" in args:
        print(controls.get("FAKE_COMPLETION", '{"choices":[]}'))
    elif controls.get("FAKE_STALE_ONCE") and not Path(controls["FAKE_STALE_ONCE"]).exists():
        Path(controls["FAKE_STALE_ONCE"]).touch()
        print(json.dumps({"ok": False, "errors": 0, "status": "stalled",
                          "stall_reason": "no round for 300 s with work in flight",
                          "oldest_s": 1.5, "rounds": {"last_age_s": 300}, "inflight": 1,
                          "opaque": key}))
        print("503")
        sys.exit(0)
    elif controls.get("FAKE_HEALTHY"):
        print(json.dumps({"ok": True, "errors": 0, "inflight": 200,
                          "requests_running": int(controls.get("FAKE_REQUESTS", "200")), "streams": {"max": 200}, "opaque": key}))
    else:
        print(json.dumps({"ok": False, "status": "stalled", "opaque": key}))
    print("health stderr " + repr(key), file=sys.stderr)
    if "-w" in sys.argv:
        print("200" if controls.get("FAKE_HEALTHY") else "503")
elif role == "alert-tool":
    Path(os.environ["ALERT_CAPTURE"]).write_text(sys.argv[1])
    echoes("alert-tool")
elif role == "heal-tool":
    echoes("heal-tool")
    Path(os.environ["HEAL_MARKER"]).touch()
    sys.exit(int(os.environ.get("HEAL_EXIT", "0")))
else:
    echoes(role)
"""


def fixture(tmp_path, key):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("curl", "docker", "ssh", "alert-tool", "heal-tool"):
        path = bindir / name
        path.write_text(f"#!{sys.executable}\n" + TOOL.replace("FIXTURE_KEY", repr(key)))
        path.chmod(0o700)
    # Fake the external native lifecycle Python command, keeping the production
    # reviewed serve.sh and all of its subprocess boundaries unchanged.
    python = bindir / "python3"
    python.write_text('#!/bin/bash\nif [[ "${1:-}" == scripts/native_ops.py ]]; then\nexec "${HEAL_EXEC}"\nfi\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
    python.chmod(0o700)
    config = tmp_path / "config.env"
    config.write_text("API_KEY=" + shlex.quote(key) + "\n")
    config.chmod(0o600)
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "CONFIG": str(config),
        "OPAQUE_SETTING": key,
        "WATCH_STATE_DIR": str(tmp_path / "state"),
        "WORKER_SSH": "worker.example.invalid",
        "WATCH_ALERT": str(bindir / "alert-tool"),
        "ALERT_CAPTURE": str(tmp_path / "alert.txt"),
        "HEAL_MARKER": str(tmp_path / "healed"),
        "HEAL_EXEC": str(bindir / "heal-tool"),
        "WATCH_PROBE": "0",
        "WATCH_HEAL": "0",
        "WATCH_RESTART_ARGV": json.dumps(["bash", "scripts/serve.sh", "watch", "--once"]),
        "GUARD_CONFIG": "",
    }
    return env


def run(env, snippet):
    # Runtime controls are fixture data, independent of process environments.
    # A numeric credential may equal an HTTP code; exercise typed protocols
    # without reintroducing that credential into a diagnostic child's env.
    import re
    controls = {name: value for name, value in env.items() if name.startswith("FAKE_")}
    for name in ("docker", "ssh", "curl", "alert-tool", "heal-tool"):
        path = Path(env["PATH"].split(os.pathsep)[0]) / name
        if path.is_file():
            try:
                source = path.read_text()
            except UnicodeDecodeError:
                continue
            if '# fixture-protocol-controls' in source:
                source = re.sub(r'^controls = .*  # fixture-protocol-controls$',
                                lambda _: repr(controls).join(('controls = ', '  # fixture-protocol-controls')),
                                source, flags=re.M)
                path.write_text(source)
    return subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}" || exit $?; {snippet}'],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def no_disclosure(key, *outputs):
    forms = {key, key.strip(), repr(key)[1:-1], json.dumps(key)[1:-1]}
    # A static failure message keeps even baseline failure receipts secret-free.
    assert not any(value in output for value in forms for output in outputs if value), (
        "watchdog disclosed a synthetic credential"
    )


@pytest.mark.parametrize("key", KEYS, ids=("plain", "escaped", "short"))
def test_evidence_redacts_both_ranks_and_both_streams_before_writing(tmp_path, key):
    env = fixture(tmp_path, key)
    # Diagnostic display values use one physical encoded record. Raw multiline
    # environment values are covered by the child-isolation tests.
    env['EVIDENCE_REASON'] = json.dumps('opaque ' + key)
    result = run(env, 'evidence_dump "$EVIDENCE_REASON"')
    assert result.returncode == 0, "evidence collection failed"
    dirs = list((tmp_path / "state/heal-evidence").iterdir())
    assert len(dirs) == 1
    directory = dirs[0]
    expected = {
        "why.txt",
        "health.json",
        "r0.log",
        "r0-state.json",
        "r1.log",
        "r1-state.json",
    }
    assert {p.name for p in directory.iterdir()} == expected
    contents = [p.read_text() for p in directory.iterdir()]
    no_disclosure(key, result.stdout, result.stderr, *contents)
    for name in expected:
        assert "<redacted>" in (directory / name).read_text(), (
            "missing sanitized evidence"
        )
    for name in ("r0.log", "r0-state.json", "r1.log", "r1-state.json"):
        assert "error" in (directory / name).read_text(), "stderr evidence was dropped"
    for path in (tmp_path / "state", directory.parent, directory, *directory.iterdir()):
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)


@pytest.mark.parametrize("heal", (False, True), ids=("observe", "heal"))
def test_health_alert_journal_and_heal_are_sanitized(tmp_path, heal):
    key = KEYS[1]
    env = fixture(tmp_path, key)
    env["WATCH_HEAL"] = "1" if heal else "0"
    result = run(
        env, 'guard_permit() { return 0; }; watch_tick; code=$?; wait; exit "$code"'
    )
    assert result.returncode == 1, "unhealthy tick did not complete"
    assert "ALERT:" in result.stderr, "journal alert missing"
    alert = tmp_path / "alert.txt"
    assert alert.exists(), "alert callback missing"
    outputs = [p.read_text() for p in (tmp_path / "state").rglob("*") if p.is_file()]
    no_disclosure(key, result.stdout, result.stderr, alert.read_text(), *outputs)
    assert "<redacted>" in alert.read_text()
    assert (tmp_path / "healed").exists() == heal
    if heal:
        log = tmp_path / "state/heal.log"
        assert "heal-tool" in log.read_text()
        assert "error" in log.read_text(), "heal stderr was dropped"
        assert stat.S_IMODE(log.stat().st_mode) == 0o600


def test_external_failure_and_redactor_failure_never_release_raw_output(tmp_path):
    key = KEYS[0]
    env = fixture(tmp_path, key)
    result = run(env, 'wssh synthetic; alert "opaque $API_KEY"')
    no_disclosure(key, result.stdout, result.stderr)
    # Even if the filter prints a partial unfiltered value before failing, its
    # output must be withheld. Its own stderr is never a raw fallback path.
    python = tmp_path / "bin/python3"
    python.write_text(
        f"#!{sys.executable}\nimport os,sys\n"
        "if any('sys.stdin.read()' in arg for arg in sys.argv):\n"
        " print(os.environ['API_KEY'])\n"
        " print(os.environ['API_KEY'], file=sys.stderr)\n"
        " sys.exit(23)\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
    )
    python.chmod(0o700)
    result = run(env, 'evidence_dump "opaque $API_KEY"; alert "opaque $API_KEY"')
    outputs = [p.read_text() for p in (tmp_path / "state").rglob("*") if p.is_file()]
    no_disclosure(key, result.stdout, result.stderr, *outputs)


def test_default_disabled_still_stands_down(tmp_path):
    env = fixture(tmp_path, KEYS[0])
    result = run(env, "watch_tick")
    assert result.returncode == 0
    assert not (tmp_path / "healed").exists()
    assert not (tmp_path / "alert.txt").exists()
    no_disclosure(KEYS[0], result.stdout, result.stderr)


def test_health_credentials_never_reach_python_diagnostic_arguments(tmp_path):
    key = KEYS[0]
    env = fixture(tmp_path, key)
    marker = tmp_path / "unsafe-argv"
    env["ARGV_MARKER"] = str(marker)
    python = tmp_path / "bin/python3"
    python.write_text(
        f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\n"
        f"key = {key!r}\n"
        "forms = (key, json.dumps(key)[1:-1], repr(key)[1:-1])\n"
        "if key and any(value in arg for value in forms for arg in sys.argv[1:]):\n"
        " Path(os.environ['ARGV_MARKER']).touch()\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
    )
    python.chmod(0o700)
    result = run(
        env,
        "guard_permit() { return 0; }; watch_tick; "
        'printf -v health \'{"ok":true,"errors":0,"opaque":"%s"}\' "$API_KEY"; '
        'startup_stale_idle 503 "$health"; startup_recheck_healthy 200 "$health"',
    )
    no_disclosure(key, result.stdout, result.stderr)
    assert not marker.exists(), "raw health reached a diagnostic process argument"


def test_config_cannot_leave_child_shell_tracing_or_startup_hooks_enabled(tmp_path):
    key = KEYS[0]
    env = fixture(tmp_path, key)
    hook = tmp_path / "startup-hook"
    hook.write_text('printf "%s\\n" "$API_KEY"; set -x\n')
    config = Path(env["CONFIG"])
    with config.open("a") as stream:
        stream.write("set -x\nBASH_ENV=" + shlex.quote(str(hook)) + "\n")
        stream.write('printf "%s\\n" "$API_KEY"\n')
    env["WATCH_HEAL"] = "1"
    result = run(env, "guard_permit() { return 0; }; watch_tick; wait")
    outputs = [p.read_text() for p in (tmp_path / "state").rglob("*") if p.is_file()]
    no_disclosure(key, result.stdout, result.stderr, *outputs)
    assert (tmp_path / "healed").exists()


@pytest.mark.parametrize("probe", (False, True), ids=("health-only", "busy-probe"))
@pytest.mark.parametrize("credential_name", ("API_KEY", "AUTH_TOKEN"))
def test_numeric_key_preserves_health_controls_without_false_heal(
    tmp_path, probe, credential_name
):
    env = fixture(tmp_path, "200")
    if credential_name != "API_KEY":
        env.pop("API_KEY", None)
        env[credential_name] = "200"
        Path(env["CONFIG"]).write_text(credential_name + "=200\n")
    env.update(
        FAKE_HEALTHY="1",
        FAKE_STARTED="2033-05-18T03:30:00Z",
        WATCH_HEAL="1",
        WATCH_FAILS="1",
        WATCH_PROBE="1" if probe else "0",
        WATCH_PROBE_EVERY="0",
    )
    date = tmp_path / "bin/date"
    date.write_text("#!/bin/sh\nprintf '%s\\n' 2000000000\n")
    date.chmod(0o700)
    result = run(
        env, 'guard_permit() { return 0; }; watch_tick; code=$?; wait; exit "$code"'
    )
    assert result.returncode == 0, "numeric credential corrupted a healthy tick"
    assert (tmp_path / "state/fails").read_text().strip() == "0"
    assert not (tmp_path / "state/last_heal").exists()
    assert not (tmp_path / "healed").exists()
    assert not (tmp_path / "alert.txt").exists()
    assert "bad tick" not in result.stderr
    # Each line is masked in place; the numeric key survives nowhere.
    assert "200" not in result.stderr and "<redacted>" in result.stderr
    assert ("every request slot busy" in result.stderr) == probe
    result = run(env, 'watch_capture json curl "$BASE/health"')
    assert json.loads(result.stdout)["inflight"] == 200
    assert json.loads(result.stdout)["opaque"] == "<redacted>"
    result = run(env, 'watch_capture http_control curl -w "\\n%{http_code}" "$BASE/health"')
    code, body = result.stdout.split("\n", 1)
    control = json.loads(body)
    assert code == "200"
    assert control["inflight"] == control["requests_running"] == 200
    assert json.loads(control["diagnostic"])["opaque"] == "<redacted>"


@pytest.mark.parametrize("key", ("true", "70", "200"))
def test_inspect_controls_are_parsed_before_literal_redaction(tmp_path, key):
    env = fixture(tmp_path, key)
    env.update(FAKE_EXIT="70", FAKE_STARTED="1970-01-01T00:03:20Z")
    result = run(
        env,
        "running 0; running 1; "
        "watch_capture uint docker inspect -f '{{.State.ExitCode}}' r0; "
        "wssh_control uint \"docker inspect -f '{{.State.ExitCode}}' r1\"; "
        "watch_capture started docker inspect -f '{{.State.StartedAt}}' r0",
    )
    assert result.returncode == 0
    assert result.stdout.splitlines() == ["true", "true", "70", "70", "200"]
    no_disclosure(key, result.stderr)


@pytest.mark.parametrize("mode", ("uint", "running", "started", "stamp"))
def test_invalid_control_stdout_is_withheld(tmp_path, mode):
    key = KEYS[1]
    env = fixture(tmp_path, key)
    result = run(env, f"watch_capture {mode} heal-tool")
    assert result.returncode == 1
    assert not result.stdout
    assert 'control parse failed; output withheld' in result.stderr
    assert (tmp_path / 'healed').exists(), 'fixture failed before its invalid diagnostic'
    no_disclosure(key, result.stderr)


@pytest.mark.parametrize("busy", (False, True), ids=("idle", "busy"))
def test_field_name_credential_does_not_turn_successful_probe_into_heal(tmp_path, busy):
    env = fixture(tmp_path, "choices")
    env.update(
        FAKE_HEALTHY="1", FAKE_REQUESTS="200" if busy else "0",
        WATCH_PROBE="1", WATCH_PROBE_EVERY="0", WATCH_PROBE_FAILS="1",
        WATCH_HEAL="1", WATCH_FAILS="1",
    )
    result = run(
        env, 'guard_permit() { return 0; }; watch_tick; code=$?; wait; exit "$code"'
    )
    assert result.returncode == 0, "protocol-field credential caused false recovery"
    assert (tmp_path / "state/fails").read_text().strip() == "0"
    assert not (tmp_path / "state/last_heal").exists()
    assert not (tmp_path / "healed").exists()
    assert not (tmp_path / "alert.txt").exists()
    if not busy:
        assert (tmp_path / "state/probe_fails").read_text().strip() == "0"
    assert ("every request slot busy" in result.stderr) == busy


@pytest.mark.parametrize("key", ("ok", "status", "stalled"))
def test_protocol_credentials_preserve_startup_recheck_and_stall_policy(tmp_path, key):
    env = fixture(tmp_path, key)
    env.update(
        FAKE_HEALTHY="1", FAKE_STALE_ONCE=str(tmp_path / "stale-seen"),
        WATCH_HEAL="1", WATCH_FAILS="1",
    )
    result = run(
        env, 'guard_permit() { return 0; }; watch_tick; code=$?; wait; exit "$code"'
    )
    assert result.returncode == 0, "protocol credential broke a healthy startup retry"
    assert (tmp_path / "stale-seen").exists()
    assert "resolved by actual healthy 200" in result.stderr
    assert (tmp_path / "state/fails").read_text().strip() == "0"
    assert not (tmp_path / "healed").exists()
    assert not (tmp_path / "alert.txt").exists()
    # A persistent stall still uses the immediate failure policy when the enum
    # or its property name happens to equal the configured credential.
    env.pop("FAKE_STALE_ONCE")
    env.pop("FAKE_HEALTHY")
    env["WATCH_HEAL"] = "0"
    env["WATCH_FAILS"] = "99"
    result = run(env, 'guard_permit() { return 0; }; watch_tick')
    assert result.returncode == 1
    # Exit 1 with WATCH_FAILS=99 proves the immediate stall policy; the
    # message contains the configured credential, masked in place.
    assert key not in result.stderr and "<redacted>" in result.stderr
    assert (tmp_path / "alert.txt").exists()
    assert not (tmp_path / "healed").exists()


@pytest.mark.parametrize("completion", ('{"choices":"text"}', '{"opaque":"choices"}'))
def test_probe_success_requires_actual_choices_array(tmp_path, completion):
    env = fixture(tmp_path, "choices")
    env.update(
        FAKE_HEALTHY="1", FAKE_REQUESTS="0", FAKE_COMPLETION=completion,
        WATCH_PROBE="1", WATCH_PROBE_EVERY="0", WATCH_PROBE_FAILS="1",
    )
    result = run(env, 'guard_permit() { return 0; }; watch_tick')
    assert result.returncode == 1
    # The message contains the configured credential, masked in place.
    assert "choices" not in result.stderr and "<redacted>" in result.stderr
    assert not (tmp_path / "healed").exists()
