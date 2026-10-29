"""Exercise real SSH session credentials at the maintained worker boundaries."""

import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import socket
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from safe_quote import join_checked, quote_checked  # noqa: E402

PATH = "/usr/sbin:/usr/bin:/sbin:/bin"


@pytest.fixture
def worker(tmp_path, monkeypatch):
    sshd = shutil.which("sshd", path=PATH)
    if sshd is None:
        pytest.skip("real SSH regression requires sshd")
    clean = {"PATH": PATH}
    for name in ("host", "client"):
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(tmp_path / name)],
                       env=clean, check=True, capture_output=True)
    authorized = tmp_path / "authorized"
    authorized.write_text((tmp_path / "client.pub").read_text())
    authorized.chmod(0o600)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    # This value exists only in the server session; neither caller env nor
    # command argv contains it. Keep failure messages free of captured output.
    key = "fake-rr7-worker-session-credential-only"
    config = tmp_path / "sshd_config"
    config.write_text(
        f"Port {port}\nListenAddress 127.0.0.1\nHostKey {tmp_path}/host\n"
        f"PidFile {tmp_path}/pid\nAuthorizedKeysFile {authorized}\nStrictModes no\n"
        "PermitRootLogin yes\nPubkeyAuthentication yes\nPasswordAuthentication no\n"
        "KbdInteractiveAuthentication no\nUsePAM no\n"
        f"SetEnv API_KEY={key} OTHER_SESSION_VALUE=worker-only\nLogLevel ERROR\n"
    )
    profile_marker = tmp_path / "profile-ran"
    client = tmp_path / "ssh_config"
    client.write_text(
        f"Host *\n HostName 127.0.0.1\n Port {port}\n User {pwd.getpwuid(os.getuid()).pw_name}\n"
        f" IdentityFile {tmp_path}/client\n IdentitiesOnly yes\n StrictHostKeyChecking no\n"
        " UserKnownHostsFile /dev/null\n LogLevel ERROR\n"
        f" RemoteCommand touch {quote_checked(str(profile_marker))}\n"
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    ssh = shutil.which("ssh", path=PATH)
    wrapper = bindir / "ssh"
    wrapper.write_text(f"#!/bin/sh\nexec {quote_checked(ssh)} -F {quote_checked(str(client))} \"$@\"\n")
    wrapper.chmod(0o700)
    monkeypatch.setenv("PATH", str(bindir) + ":" + PATH)
    repo = tmp_path / "worker checkout's space"
    repo.mkdir()
    (repo / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
    diagnostic = tmp_path / "diagnostic with spaces.py"
    observed = tmp_path / "observed.jsonl"
    diagnostic.write_text(
        "import json,os,sys\n"
        f"with open({str(observed)!r}, 'a') as out:\n"
        " out.write(json.dumps({'credential': 'API_KEY' in os.environ, "
        "'session': 'OTHER_SESSION_VALUE' in os.environ, 'path': os.environ.get('PATH')})+'\\n')\n"
        "print('47' if sys.argv[1:] == ['uint'] else 'diagnostic: '+os.environ.get('API_KEY','absent'))\n"
        "print('diagnostic: '+os.environ.get('API_KEY','absent'),file=sys.stderr)\n"
        "raise SystemExit(47)\n"
    )
    with (tmp_path / "sshd.log").open("w") as log:
        daemon = subprocess.Popen([sshd, "-D", "-e", "-f", str(config)], env=clean, stdout=log, stderr=log)
        try:
            for _ in range(100):
                assert daemon.poll() is None, "isolated SSH server exited before readiness"
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.02)
            else:
                pytest.fail("isolated SSH server did not become ready")
            yield {"repo": repo, "diagnostic": diagnostic, "observed": observed,
                   "key": key, "profile_marker": profile_marker, "client": client, "ssh": ssh}
        finally:
            daemon.terminate()
            try:
                daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait(timeout=5)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def assert_isolated(worker, result, stdout="diagnostic: absent"):
    assert result.returncode == 47
    assert worker["key"] not in result.stdout + result.stderr, "worker credential disclosed"
    assert stdout in result.stdout
    assert "diagnostic: absent" in result.stderr
    rows = [json.loads(line) for line in worker["observed"].read_text().splitlines()]
    assert rows == [{"credential": False, "session": False, "path": PATH}]
    assert not worker["profile_marker"].exists()


@pytest.mark.parametrize("typed", [False, True], ids=["logs", "typed-exit-code"])
def test_watchdog_real_ssh_scrubs_server_environment(worker, monkeypatch, typed):
    monkeypatch.setenv("CONFIG", "/nonexistent/rr7-config")
    monkeypatch.setenv("WORKER_SSH", "worker.example")
    monkeypatch.setenv("WORKER_REPO", str(worker["repo"]))
    command = join_checked(["/usr/bin/python3", str(worker["diagnostic"]), *(["uint"] if typed else [])])
    call = 'wssh_control uint "$1"' if typed else 'wssh "$1"'
    result = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c",
         'source scripts/guard-watch-zig.sh || exit $?; ' + call, "watch", command],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    assert_isolated(worker, result, stdout="47" if typed else "diagnostic: absent")


def test_native_real_ssh_clock_and_command_scrub_server_environment(worker, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    ops = module("rr7_native_ops", ROOT / "scripts/native_ops.py")
    lifecycle = ops.Lifecycle({"PATH": os.environ["PATH"],
                               "WORKER_SSH": "worker.example", "WORKER_REPO": str(worker["repo"]),
                               "WORKER_CONFIG": "/srv/config.env", "STATE_DIR": str(tmp_path / "state")})
    with pytest.raises(ops.OpsError) as error:
        lifecycle.command(1, ["/usr/bin/python3", str(worker["diagnostic"])], timeout=5)
    assert error.value.code == 47
    assert json.loads(worker["observed"].read_text()) == {"credential": False, "session": False, "path": PATH}
    assert not worker["profile_marker"].exists()


def test_backend_real_ssh_scrubs_server_environment(worker, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.syspath_prepend(str(ROOT / "scripts/guard"))
    backend = module("rr7_backend", ROOT / "scripts/guard/backend.py")
    instance = backend.Backend.__new__(backend.Backend)
    instance.config = {"worker_ssh": "worker.example", "worker_repo": str(worker["repo"])}
    with pytest.raises(subprocess.CalledProcessError) as error:
        instance.run("1", ["/usr/bin/python3", str(worker["diagnostic"])])
    assert_isolated(worker, subprocess.CompletedProcess([], error.value.returncode,
                                                       error.value.stdout, error.value.stderr))


def test_sampler_real_ssh_scrubs_server_environment(worker, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.syspath_prepend(str(ROOT / "scripts/guard"))
    sampler = module("rr7_sampler", ROOT / "scripts/guard/sampler.py")
    monkeypatch.setattr(sampler, "NODE_SAMPLE", f"import runpy; runpy.run_path({str(worker['diagnostic'])!r})")
    with pytest.raises(subprocess.CalledProcessError) as error:
        sampler.worker_sample({"worker_ssh": "worker.example", "worker_repo": str(worker["repo"])})
    assert_isolated(worker, subprocess.CompletedProcess([], error.value.returncode,
                                                       error.value.stdout, error.value.stderr))


def test_real_ssh_control_proves_server_credential_source(worker):
    result = subprocess.run(
        [worker["ssh"], "-F", str(worker["client"]), "-o", "RemoteCommand=none", "worker.example",
         join_checked(["/usr/bin/python3", "-c", "import os; print('API_KEY' in os.environ)"])],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "True"


def test_evidence_collection_uses_checked_remote_adapters():
    source = (ROOT / "scripts/guard-watch-zig.sh").read_text()
    body = source.split("_watch_evidence_dump()", 1)[1].split("_watch_watch_tick()", 1)[0]
    assert re.search(r'^\s*wssh "docker logs .*\$NAME-r1', body, re.MULTILINE)
    assert re.search(r'^\s*wssh_control state "docker inspect .*\$NAME-r1', body, re.MULTILINE)
    assert not re.search(r"^\s*(?:watch_run\s+|watch_capture\s+\w+\s+)?ssh\s", body, re.MULTILINE)
