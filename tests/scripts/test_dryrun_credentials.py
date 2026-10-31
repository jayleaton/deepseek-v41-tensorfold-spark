"""Refuse credential-bearing raw argv before the real launcher's dry-run quoting."""

import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pytest

from test_native_launcher import ROOT, launcher  # noqa: F401


SHAPES = {
    "space": " ",
    "metacharacters": ";&|()>*?[]{}!",
    "single_quote": "'",
    "double_quote": '"',
    "backslash": "\\",
    "newline": "\n",
    "tab": "\t",
    "dollar": "$",
    "backticks": "`",
}
MOUNTS = ("MODEL_DIR", "ENGRAM_DIR", "ASSETS_DIR", "STATE_DIR")
CONTROLS = ("HOST", "PORT", "IMAGE", "CPUSET_CPUS")


def _fixture_key(shape):
    # Construct at runtime so the publication scan never sees a credential literal.
    return "rr3" + SHAPES[shape] + "DryrunCredentialOnly"


def _shell_words(text):
    """Decode printf %q's ANSI-C words, then use POSIX shlex; never evaluate output."""
    escapes = dict(zip("abefnrtv", "\a\b\x1b\f\n\r\t\v"))
    escapes.update({"E": "\x1b", "\\": "\\", "'": "'", '"': '"', "?": "?"})

    def unescape(match):
        value = match.group()[1:]
        if value.startswith("x"):
            return chr(int(value[1:], 16))
        if value[0] in "01234567":
            return chr(int(value, 8))
        return escapes.get(value, "\\" + value)

    def ansi_word(match):
        value = re.sub(r"\\(?:[0-7]{1,3}|x[0-9a-fA-F]{1,2}|.)", unescape,
                       match[1], flags=re.DOTALL)
        return shlex.quote(value)

    try:
        normalized = re.sub(r"\$'((?:\\.|[^'\\])*)'", ansi_word, text, flags=re.DOTALL)
        return shlex.split(normalized)
    except ValueError:
        raise AssertionError("dry-run output could not be reconstructed safely") from None


@pytest.fixture
def dryrun_launcher(request):
    env, marker, values = request.getfixturevalue("launcher")
    # Observe every Docker call, including inspect; do not persist argv or secrets.
    docker = Path(env["PATH"].split(os.pathsep)[0]) / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import os\nfrom pathlib import Path\n"
        "Path(os.environ['LAUNCHER_TEST_ARGV']).write_text('called')\n"
        "raise SystemExit(44)\n"
    )
    for name, value in values.items():
        if name.endswith("_DIR"):
            Path(value).mkdir(exist_ok=True)
    return env, marker, values


def _run(dryrun_launcher, setting, key, rank):
    env, marker, _ = dryrun_launcher
    env = dict(env, API_KEY=key)
    value = "/fixture/" + key if setting.endswith("_DIR") else key
    with Path(env["CONFIG"]).open("a") as config:
        config.write(setting + "=" + shlex.quote(value) + "\n")
    try:
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/run-zig.sh"), str(rank), "--dry-run"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise AssertionError("dry-run launcher did not settle") from None
    # Compute booleans before asserting so pytest never renders the key or streams.
    spawned = marker.exists()
    leaked_stdout = key in result.stdout
    leaked_stderr = key in result.stderr
    recovered = any(key in word for stream in (result.stdout, result.stderr)
                    for word in _shell_words(stream))
    refused = result.returncode != 0
    assert not spawned, "dry-run invoked Docker"
    assert not leaked_stdout, "credential appeared on stdout"
    assert not leaked_stderr, "credential appeared on stderr"
    assert not recovered, "credential recoverable from shell-quoted output"
    assert refused, "credential-bearing argv was accepted"


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("setting", MOUNTS)
@pytest.mark.parametrize("shape", tuple(SHAPES))
def test_dryrun_refuses_credentials_in_mounts(dryrun_launcher, rank, setting, shape):
    _run(dryrun_launcher, setting, _fixture_key(shape), rank)


@pytest.mark.parametrize("setting", CONTROLS)
@pytest.mark.parametrize("shape", tuple(SHAPES))
def test_dryrun_refuses_credentials_in_controls(dryrun_launcher, setting, shape):
    _run(dryrun_launcher, setting, _fixture_key(shape), 0)


@pytest.mark.parametrize("setting", ["PREPARED_DIR", "CACHE_DIR", "TF_TP_MASTER", "TF_TP_PORT",
                                    "TF_COMM_BACKEND"])
def test_dryrun_refuses_other_raw_arguments(dryrun_launcher, setting):
    _run(dryrun_launcher, setting, _fixture_key("space"), 1)


@pytest.mark.parametrize("setting", ["TF_DSV41_SLOTS", "TF_DSV41_CONTEXT", "NCCL_SOCKET_IFNAME",
                                    "NCCL_IB_HCA", "TF_TP_ROCE_HCA"])
def test_existing_forwarded_environment_refusal(dryrun_launcher, setting):
    # Negative controls already fail closed on the base; report them separately.
    _run(dryrun_launcher, setting, _fixture_key("space"), 0)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("shape", tuple(SHAPES))
def test_safe_dryrun_preserves_shell_arguments(dryrun_launcher, rank, shape):
    env, marker, _ = dryrun_launcher
    value = "/fixture/benign" + SHAPES[shape] + "mount"
    with Path(env["CONFIG"]).open("a") as config:
        config.write("MODEL_DIR=" + shlex.quote(value) + "\n")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run-zig.sh"), str(rank), "--dry-run"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    succeeded = result.returncode == 0
    spawned = marker.exists()
    words = _shell_words(result.stdout)
    preserved = value + ":/model:ro" in words
    correct_command = words[:2] == ["docker", "run"]
    assert succeeded, "safe dry-run was refused"
    assert not spawned, "safe dry-run invoked Docker"
    assert preserved, "shell argument reconstruction lost mount bytes"
    assert correct_command, "safe dry-run did not describe Docker run"
