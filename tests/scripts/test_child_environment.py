"""Real children must not inherit credentials, including innocuously named aliases.

Observers retain only presence booleans and benign controls. The module also runs
on the pre-fix revision: it imports the existing runner, not the new helper.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import safe_exec  # noqa: E402

# Invented values exercise whitespace, implicit pprint string concatenation,
# escaping and recognizable tokens without putting credentials in receipts.
VALUES = (
    "unrelated-invented-long-credential-" + "ab " * 60,
    "Bearer " + "invented-pprint-value " * 25,
    "invented-'quoted'-\"value\"-\\-café\nsecond-line",
    "hf_" + "InventedObserverValue" * 3,
)
SECRET_NAMES = ("API_KEY", "HF_TOKEN", "EMPTY_SECRET")
ALIAS_NAMES = ("OPAQUE_SETTING", "DEBUG", "DEBUG_ESCAPED", "NESTED_DEBUG", "SHAPED_VALUE")
BENIGN = {
    "BENIGN_SETTING": "ready for worker's next batch",
    "MODEL": "example-model",
    "TF_DSV41_POOL_TOKENS": "2100000",
    "API_KEY_FILE": "/tmp/operator-key-path",
}

# The child never writes inherited values, even when run against the vulnerable
# base. JSON booleans in this private sidecar bypass diagnostic redaction so an
# output filter cannot make an inheritance failure look like a successful test.
OBSERVER = r'''
import json, os, pathlib, sys
secret_names = ("API_KEY", "HF_TOKEN", "EMPTY_SECRET")
alias_names = ("OPAQUE_SETTING", "DEBUG", "DEBUG_ESCAPED", "NESTED_DEBUG", "SHAPED_VALUE")
expected = {
    "BENIGN_SETTING": "ready for worker's next batch",
    "MODEL": "example-model",
    "TF_DSV41_POOL_TOKENS": "2100000",
    "API_KEY_FILE": "/tmp/operator-key-path",
}
try:
    os.fstat(9)
    registry_fd_closed = False
except OSError:
    registry_fd_closed = True
record = {
    "registry_fd_closed": registry_fd_closed,
    "names_absent": all(name not in os.environ for name in secret_names),
    "aliases_absent": all(name not in os.environ for name in alias_names),
    "benign_preserved": all(os.environ.get(k) == v for k, v in expected.items()),
    "other_benign_preserved": all(os.environ.get(k) == v for k, v in expected.items()
                                  if k != "BENIGN_SETTING"),
    "path_preserved": bool(os.environ.get("PATH")),
    "joined_group": os.environ.get("TF_SAFE_EXEC_GROUP") == str(os.getpgrp()),
    "kind": pathlib.Path(sys.argv[0]).name,
}
with open(os.environ["PROBE_REPORT"], "a") as output:
    output.write(json.dumps(record) + "\n")
if pathlib.Path(sys.argv[0]).name == "git":
    if "rev-parse" in sys.argv:
        print("1" * 40)
    elif "rev-list" in sys.argv:
        print("1" * 40)
    elif "ls-tree" in sys.argv:
        sys.stdout.write("100644 blob " + "2" * 40 + "\tprobe.txt\0")
    elif "cat-file" in sys.argv:
        for line in sys.stdin:
            sys.stdout.write(line.strip() + " blob 8\ncontrol\n\n")
            sys.stdout.flush()
elif pathlib.Path(sys.argv[0]).name == "grep":
    if "-HnIiE" in sys.argv:
        print(sys.argv[-1] + ":1:/home/user/")
else:
    print("ordinary child diagnostic")
    print("worker couldn't find the requested artifact", file=sys.stderr)
sys.exit(int(os.environ.get("PROBE_STATUS", "47")))
'''


def fixture(tmp_path, value=VALUES[0]):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("docker", "ssh", "git", "grep", "diagnostic"):
        executable = bindir / name
        executable.write_text(f"#!{sys.executable}\n" + OBSERVER)
        executable.chmod(0o700)
    env = {
        "PATH": str(bindir) + os.pathsep + os.defpath,
        "HOME": str(tmp_path),
        "LANG": "C.UTF-8",
        "PROBE_REPORT": str(tmp_path / "observations.jsonl"),
        "PROBE_SAFE_EXEC": str(ROOT / "scripts/safe_exec.py"),
        "PROBE_PYTHON": sys.executable,
        "API_KEY": value,
        "HF_TOKEN": "fake-distinct-secondary-credential",
        "EMPTY_SECRET": "",
        "OPAQUE_SETTING": value,
        "DEBUG": "prefix=" + value + "; suffix",
        "DEBUG_ESCAPED": json.dumps(value)[1:-1],
        "NESTED_DEBUG": json.dumps(json.dumps(value))[1:-1],
        "SHAPED_VALUE": "hf_" + "InventedStandaloneValue" * 3,
        **BENIGN,
    }
    return env


def invoke(argv, env):
    return subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)


def observations(env, count=1):
    path = Path(env["PROBE_REPORT"])
    assert path.exists(), "observer child was not executed"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == count, "unexpected number of observer children"
    assert all(r["registry_fd_closed"] for r in records), "private registry fd reached a child"
    assert all(r["names_absent"] for r in records), "child inherited a credential-named variable"
    assert all(r["aliases_absent"] for r in records), "child inherited a credential-bearing alias"
    assert all(r["benign_preserved"] and r["path_preserved"] for r in records), "child lost benign configuration"
    return records


@pytest.mark.parametrize("value", VALUES, ids=("long-spaces", "bearer-pprint", "escaped-unicode", "token"))
@pytest.mark.parametrize("argv", (("docker", "logs", "rank0"), ("docker", "inspect", "rank0"),
                                  ("ssh", "worker.example.invalid", "diagnostic"),
                                  ("git", "fetch", "fixture", "missing-ref")))
def test_checked_cli_strips_child_environment(tmp_path, value, argv):
    env = fixture(tmp_path, value)
    result = invoke([sys.executable, str(ROOT / "scripts/safe_exec.py"), "--command", *argv], env)
    assert result.returncode == 47, "checked CLI changed the child exit status"
    observations(env)


@pytest.mark.parametrize("mode", ("unchecked", "bootstrap", "nested-group"))
def test_cli_bootstrap_and_joined_group_strip_environment(tmp_path, mode):
    env = fixture(tmp_path)
    prefix = [sys.executable, str(ROOT / "scripts/safe_exec.py")]
    if mode == "unchecked":
        argv = [*prefix, "diagnostic"]
    elif mode == "bootstrap":
        argv = [*prefix, "--group-child", "diagnostic"]
    else:
        # The outer supervisor creates the real process group. Reintroduce
        # credentials inside that group to exercise the joined Popen branch.
        payload = ('export API_KEY="$BENIGN_SETTING" DEBUG="$BENIGN_SETTING"; '
                   '"$PROBE_PYTHON" "$PROBE_SAFE_EXEC" --command diagnostic')
        argv = [*prefix, "--command", "bash", "-c", payload]
    result = invoke(argv, env)
    assert result.returncode == 47, "CLI changed the child exit status"
    if mode == "nested-group":
        # BENIGN_SETTING became a configured credential only for this nested
        # invocation; require other benign settings to survive independently.
        records = [json.loads(line) for line in Path(env["PROBE_REPORT"]).read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["names_absent"] and records[0]["aliases_absent"]
        assert records[0]["path_preserved"]
        assert records[0]["other_benign_preserved"] and records[0]["joined_group"]
    else:
        observations(env)


@pytest.mark.parametrize("explicit", ("inherited", "explicit", "aliases-only"))
def test_python_runner_strips_environment(tmp_path, monkeypatch, explicit):
    env = fixture(tmp_path)
    monkeypatch.setattr(os, "environ", env.copy())
    supplied = env.copy()
    if explicit == "aliases-only":
        for name in SECRET_NAMES:
            supplied.pop(name)
    result = safe_exec.run([str(Path(env["PATH"].split(os.pathsep)[0]) / "diagnostic")],
                           env=None if explicit == "inherited" else supplied, timeout=10)
    assert result.returncode == 47, "Python runner changed child status"
    assert "ordinary child diagnostic" in result.stdout
    assert "couldn't find" in result.stderr
    observations(env)
    assert env["API_KEY"] == VALUES[0], "runner mutated caller's environment"


@pytest.mark.parametrize("helper", ("watch_run docker logs rank0", "watch_capture json docker inspect rank0"))
def test_sourced_watchdog_helpers_strip_environment(tmp_path, helper):
    env = fixture(tmp_path)
    env["CONFIG"] = str(tmp_path / "absent.env")
    result = invoke(["bash", "-c", 'source scripts/guard-watch-zig.sh; ' + helper], env)
    assert result.returncode == 47, "watchdog helper changed child status"
    observations(env)


def test_benchmark_provenance_git_children_strip_environment(tmp_path, monkeypatch):
    env = fixture(tmp_path)
    env["PROBE_STATUS"] = "0"
    monkeypatch.setattr(os, "environ", env.copy())
    spec = importlib.util.spec_from_file_location("child_environment_speed", ROOT / "bench/speed.py")
    speed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(speed)
    identity = speed.tool_identity()
    assert identity["git"] == "1" * 40
    assert identity["dirty"] is False
    assert all(r["kind"] == "git" for r in observations(env, count=2))


def test_benchmark_cli_provenance_children_strip_environment(tmp_path):
    env = fixture(tmp_path)
    env["PROBE_STATUS"] = "0"
    output = tmp_path / "empty-suite.json"
    result = invoke([sys.executable, str(ROOT / "bench/speed.py"), "--suite", "", "--quiet",
                     "--out", str(output)], env)
    assert result.returncode == 0, "benchmark diagnostic invocation failed"
    assert json.loads(output.read_text())["identity"]["git"] == "1" * 40
    observations(env, count=2)


@pytest.mark.parametrize("checker,count", (("tree", 1), ("secret-history", 1), ("public-history", 5)))
def test_public_checker_children_strip_environment(tmp_path, checker, count):
    env = fixture(tmp_path)
    env["PROBE_STATUS"] = "0"
    if checker == "public-history":
        argv = [sys.executable, str(ROOT / "scripts/check_public_history.py"),
                "home directories|/home/|"]
    else:
        argv = [sys.executable, str(ROOT / "scripts/check_secrets.py")]
        if checker == "secret-history":
            argv.append("--history")
    result = invoke(argv, env)
    assert result.returncode == 0, "public checker failed with benign fixture input"
    observations(env, count=count)


@pytest.mark.parametrize("setting", ["WATCH_INTERVAL", "WATCH_FAILS", "STOP_FILE", "STATE_DIR",
                                     "LIFECYCLE_LOCK", "MAINTENANCE_LEASE", "HEALTH_URL",
                                     "HOME", "XDG_STATE_HOME", "TF_SAFE_EXEC_DEADLINE"])
def test_tainted_operational_setting_cannot_select_a_default(tmp_path, setting):
    env = fixture(tmp_path)
    env[setting] = env["API_KEY"]
    result = invoke([sys.executable, str(ROOT / "scripts/safe_exec.py"),
                     "--command", "diagnostic"], env)
    assert result.returncode == 78
    assert "credential value refused in child configuration" in result.stderr
    assert not Path(env["PROBE_REPORT"]).exists(), "tainted configuration spawned a child"


def test_watchdog_unexports_configured_keys_and_aliases_but_keeps_private_validation(tmp_path):
    env = fixture(tmp_path)
    config = tmp_path / "config.env"
    config.write_text('export API_KEY="$API_KEY" OPAQUE_SETTING="$API_KEY"\n')
    env["CONFIG"] = str(config)
    marker = tmp_path / "shell-environment.json"
    env["SHELL_ENVIRONMENT_MARKER"] = str(marker)
    payload = ('source scripts/guard-watch-zig.sh; '
               'if [[ $(declare -p API_KEY) == "declare -x "* || '
               '$(declare -p OPAQUE_SETTING) == "declare -x "* ]]; then '
               'printf false > "$SHELL_ENVIRONMENT_MARKER"; else '
               'printf true > "$SHELL_ENVIRONMENT_MARKER"; fi; '
               'watch_run diagnostic')
    result = invoke(["bash", "-c", payload], env)
    assert result.returncode == 47
    assert marker.read_text() == "true", "watchdog retained exported secret values"
    observations(env)


def test_authenticated_watchdog_refuses_unauthorized_completion_probe(tmp_path):
    env = fixture(tmp_path)
    env.update(CONFIG=str(tmp_path / "absent.env"), NATIVE_AUTH_FILE=str(tmp_path / "server.keys"),
               WATCH_PROBE="1", WATCH_HEAL="1")
    result = invoke(["bash", "-c", "source scripts/guard-watch-zig.sh; watch_tick"], env)
    assert result.returncode == 78
    assert "requires WATCH_PROBE=0" in result.stderr
    assert not Path(env["PROBE_REPORT"]).exists(), "unauthorized probe spawned a child"


def test_benchmark_cli_key_is_known_before_git_child_environment_is_built(tmp_path):
    env = fixture(tmp_path)
    env["PROBE_STATUS"] = "0"
    cli_key = "rr6-distinct-commandline-credential"
    env["DEBUG"] = cli_key
    output = tmp_path / "empty-suite.json"
    result = invoke([sys.executable, str(ROOT / "bench/speed.py"), "--suite", "", "--quiet",
                     "--api-key", cli_key, "--out", str(output)], env)
    assert result.returncode == 0
    observations(env, count=2)


def test_publication_shell_git_child_has_no_credential_environment(tmp_path):
    env = fixture(tmp_path)
    env["PROBE_STATUS"] = "0"
    result = invoke(["bash", "scripts/check-public.sh"], env)
    # The empty fake index is an intentional scanner error, not a clean scan.
    assert result.returncode == 2
    observations(env)


def test_agent_socket_is_a_pointer_kept_for_ssh_children(tmp_path):
    # RR8-2: SSH_AUTH_SOCK names a socket path; dropping it broke agent-only SSH.
    from redact import secret_name
    assert not secret_name("SSH_AUTH_SOCK")
    env = {"PATH": "/usr/bin:/bin", "CONFIG": str(tmp_path / "absent.env"),
           "SSH_AUTH_SOCK": str(tmp_path / "agent.sock"), "API_KEY": "fake-rr8-agent-credential"}
    child = safe_exec.child_environment(env)
    assert child["SSH_AUTH_SOCK"] == str(tmp_path / "agent.sock")
    assert "API_KEY" not in child
    # A key value in the socket variable is still an alias, and is still dropped.
    assert "SSH_AUTH_SOCK" not in safe_exec.child_environment({**env, "SSH_AUTH_SOCK": env["API_KEY"]})
