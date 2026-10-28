"""Exercise the real CLI process, including failures before a report exists.

Keep captured streams out of assertion diagnostics: a regression must fail without
copying the synthetic credential into pytest logs.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEED = ROOT / "bench/speed.py"


def invoke(args, key, source="separate", code=None):
    env = dict(os.environ)
    env.pop("API_KEY", None)
    flags = []
    if source == "env":
        env["API_KEY"] = key
    elif source == "equals":
        flags = ["--api-key=" + key]
    elif source == "abbreviation":
        flags = ["--api-k", key]
    else:
        flags = ["--api-key", key]
    command = [sys.executable, str(SPEED)] if code is None else [sys.executable, "-c", code]
    try:
        return subprocess.run(
            command + flags + args, cwd=ROOT, env=env, capture_output=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("CLI invocation timed out", pytrace=False)


def assert_safe_failure(result, key):
    variants = {key}
    for _ in range(3):
        variants.update(json.dumps(v)[1:-1] for v in list(variants))
        variants.update(repr(v)[1:-1] for v in list(variants))
    # Also detect partial disclosure when a trailing newline/control is omitted.
    variants.add(key.rstrip())
    leaked = any(
        v.encode("utf-8") in stream
        for v in variants if v
        for stream in (result.stdout, result.stderr)
    )
    traceback = b"Traceback (most recent call last)" in result.stderr
    failed = result.returncode != 0
    diagnostic = bool(result.stderr.strip())
    assert failed, "CLI failure unexpectedly succeeded"
    assert not leaked, "credential reached a captured stream"
    assert not traceback, "CLI failure escaped the sanitized boundary"
    assert diagnostic, "CLI failure needs a safe diagnostic"


@pytest.mark.parametrize("source", ["separate", "equals", "env", "abbreviation"])
@pytest.mark.parametrize("suffix", ["\n", "\r\n", "\x7f", "\u2603"], ids=["newline", "crlf", "del", "nonencodable"])
def test_invalid_header_rejected_before_health(source, suffix):
    key = "rr1" + "SyntheticCredentialOnly" + suffix
    result = invoke([
        "--url", "http://127.0.0.1:1", "--health", "--suite", "decode",
    ], key, source)
    assert_safe_failure(result, key)


@pytest.mark.parametrize("source", ["separate", "equals", "env", "abbreviation"])
@pytest.mark.parametrize("failure", [
    "port", "bracket", "scheme", "argparse", "depths", "lengths", "streams", "suite",
    "reference", "compare", "output",
])
def test_cli_setup_and_file_failures_are_sanitized(tmp_path, source, failure):
    key = "rr1" + 'Synthetic"quote\\Credential'
    args = ["--url", "http://127.0.0.1:1", "--suite", ""]
    if failure == "port":
        args += ["--url", "http://localhost:" + key]
    elif failure == "bracket":
        args += ["--url", "http://[" + key]
    elif failure == "scheme":
        args += ["--url", key]
    elif failure == "argparse":
        args += ["--reps", key]
    elif failure in ("depths", "lengths", "streams", "suite"):
        args += ["--" + failure, key]
    elif failure == "reference":
        args += ["--exact-ref", str(tmp_path / key)]
    elif failure == "compare":
        args += ["--compare", str(tmp_path / key)]
    else:
        # Real output-open error after summary generation, with the key in its path.
        target = tmp_path / key
        target.mkdir()
        args += ["--out", str(target)]
    assert_safe_failure(invoke(args, key, source), key)


@pytest.mark.parametrize("stage", [
    "health_before", "health_after", "metrics_before", "metrics_after", "suite",
    "report", "identity", "compare", "reference", "output",
])
@pytest.mark.parametrize("representation", ["literal", "repr", "json", "nested_json"])
def test_cli_runtime_failures_have_one_sanitized_boundary(stage, representation):
    # Run in a fresh interpreter so the process exit and both actual streams are checked.
    code = r'''
import builtins
import importlib.util
import json
import sys
spec = importlib.util.spec_from_file_location("speed", "bench/speed.py")
speed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speed)
stage, representation = sys.argv[-2:]
args = sys.argv[1:-2]
key = args[1]
value = {"literal": key, "repr": repr(key), "json": json.dumps(key),
         "nested_json": json.dumps(json.dumps(key))}[representation]
def fail(*args, **kwargs):
    raise RuntimeError("synthetic failure: " + value)
calls = 0
def snapshot(*args, **kwargs):
    global calls
    calls += 1
    if calls == (2 if stage.endswith("after") else 1):
        fail()
    return {}
if stage.startswith("health"):
    speed.Client.get_json = snapshot
    args += ["--health"]
elif stage.startswith("metrics"):
    speed.metrics_text = snapshot
    args += ["--metrics"]
elif stage == "suite":
    speed.Run.decode = fail
    args += ["--suite", "decode"]
elif stage == "report":
    speed.report = fail
elif stage == "identity":
    speed.tool_identity = fail
elif stage == "compare":
    speed.load_run = fail
    args += ["--compare", "synthetic.json"]
else:
    builtins.open = fail
    args += ["--exact-ref" if stage == "reference" else "--out", "synthetic.json"]
sys.exit(speed.main(args))
'''
    key = "rr1" + 'Synthetic"quote\\Credential'
    result = invoke(["--suite", "", stage, representation], key, code=code)
    assert_safe_failure(result, key)


@pytest.mark.parametrize("mode", ["help", "parse_exit", "string_exit"])
def test_cli_system_exit_preserves_status_and_redacts(tmp_path, mode):
    key = "rr1" + "SyntheticCredentialOnly"
    if mode == "help":
        result = invoke(["--help"], key)
        ok = result.returncode == 0 and bool(result.stdout) and not result.stderr
        assert ok, "help must remain successful"
    elif mode == "parse_exit":
        result = invoke(["--tokens", key], key)
        assert_safe_failure(result, key)
        assert result.returncode == 2
    else:
        # load_run raises a message-bearing SystemExit for a foreign report.
        path = tmp_path / key
        path.write_text("{}")
        result = invoke(["--compare", str(path)], key)
        assert_safe_failure(result, key)


def test_numeric_key_report_is_valid_json_with_numeric_metrics(tmp_path):
    path = tmp_path / "report.json"
    result = invoke(["--url", "http://127.0.0.1:1", "--suite", "", "--reps", "1", "--out", str(path)], "1")
    assert result.returncode == 0
    doc = json.loads(path.read_text())
    assert doc["settings"]["reps"] == 1
    assert isinstance(doc["settings"]["tokens"], int)
