"""Check actual writes and process streams without exposing captured credentials in failures."""

import builtins
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/guard"))
import safe_output  # noqa: E402
from redact import MASK, known_secrets, redact_text  # noqa: E402


def load(relative):
    spec = importlib.util.spec_from_file_location("security_" + Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contains_secret(text, key):
    forms = {key, key.strip(), repr(key)[1:-1], json.dumps(key)[1:-1]}
    forms.add(json.dumps(json.dumps(key)[1:-1])[1:-1])
    return any(value in text for value in forms if value)


class CheckedWriter:
    """Reject a raw credential at the write call, before it can reach any file."""
    def __init__(self, stream, key):
        self.stream, self.key = stream, key

    def write(self, text):
        if contains_secret(text, self.key):
            raise AssertionError("credential passed to file write")
        return self.stream.write(text)

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)


def check_writes(monkeypatch, key):
    original_open, original_path_open, original_fdopen = builtins.open, Path.open, os.fdopen

    def checked_open(*args, **kwargs):
        return CheckedWriter(original_open(*args, **kwargs), key)

    def checked_path_open(*args, **kwargs):
        return CheckedWriter(original_path_open(*args, **kwargs), key)

    def checked_fdopen(*args, **kwargs):
        return CheckedWriter(original_fdopen(*args, **kwargs), key)

    monkeypatch.setattr(builtins, "open", checked_open)
    monkeypatch.setattr(Path, "open", checked_path_open)
    monkeypatch.setattr(os, "fdopen", checked_fdopen)


@pytest.mark.parametrize("client", ["stress", "structured", "quality", "soak", "tooleval/chains"])
def test_legacy_cli_argparse_failure_masks_both_streams(client):
    key = "rr1" + "CliCredentialOnly"
    result = subprocess.run(
        [sys.executable, str(ROOT / "bench" / (client + ".py")), "--unknown=" + key],
        env=dict(os.environ, API_KEY=key), capture_output=True, timeout=15,
    )
    leaked = contains_secret((result.stdout + result.stderr).decode("utf-8", "replace"), key)
    assert not leaked, "CLI parse failure disclosed a credential"
    assert result.returncode == 2


def test_legacy_cli_invalid_port_masks_both_streams():
    key = "rr1" + "PortCredentialOnly"
    result = subprocess.run(
        [sys.executable, str(ROOT / "bench/stress.py"), "admit", "--base", "http://localhost:" + key],
        env=dict(os.environ, API_KEY=key), capture_output=True, timeout=15,
    )
    leaked = contains_secret((result.stdout + result.stderr).decode("utf-8", "replace"), key)
    traceback = b"Traceback (most recent call last)" in result.stderr
    assert not leaked, "CLI URL conversion disclosed a credential"
    assert not traceback, "CLI URL conversion escaped its error boundary"
    assert result.returncode != 0


@pytest.mark.parametrize("client", ["structured", "quality", "soak"])
def test_legacy_reports_mask_before_first_write(client, tmp_path, monkeypatch):
    key = "rr1" + 'Echo"\\Only'
    monkeypatch.setenv("API_KEY", key)
    module = load("bench/" + client + ".py")
    # This regression is about persistence; even a failing baseline must not echo
    # its synthetic diagnostic into pytest's captured-output report.
    monkeypatch.setattr(module, "print", lambda *args, **kwargs: None, raising=False)
    output = tmp_path / "report.json"
    args = ["--out", str(output)]
    if client == "structured":
        monkeypatch.setattr(module, "suite_schemas", lambda args: ({"echo": key, "tokens": 128}, []))
        args += ["--suites", "schemas"]
    elif client == "quality":
        data = tmp_path / "questions.jsonl"
        data.write_text(json.dumps({"subject": "test", "question": "test", "choices": ["x"], "answer": 0}))
        monkeypatch.setattr(module, "chat", lambda *args: (key, 128))
        args = ["mmlu", "--data", str(data), *args]
    else:
        def result(self):
            return {"pass": True, "by_kind": {"echo": key}, "error_samples": [key], "requests": 128}
        monkeypatch.setattr(module.Soak, "run", result)
    check_writes(monkeypatch, key)
    module.main(args)
    doc = json.loads(output.read_text())
    leaked = contains_secret(output.read_text(), key)
    assert not leaked, "saved report disclosed a credential"
    if client == "quality":
        assert doc["items"][0]["prompt_tokens"] == 128
    elif client == "soak":
        assert doc["requests"] == 128
    else:
        assert doc["schemas"]["cases"]["tokens"] == 128


def test_guard_atomic_and_events_mask_before_write(tmp_path, monkeypatch):
    common, policy = load("scripts/guard/common.py"), load("scripts/guard/policy.py")
    key = "rr1" + 'Guard"\\Only'
    monkeypatch.setenv("API_KEY", key)
    check_writes(monkeypatch, key)
    state, events = tmp_path / "state.json", tmp_path / "events.jsonl"
    common.atomic(state, {"echo": key, "tokens": 128})
    instance = policy.Policy(tmp_path / "policy.json", events, now=0)
    instance.event("TEST", 128, echo=json.dumps(key), tokens=128)
    assert json.loads(state.read_text())["tokens"] == 128
    row = json.loads(events.read_text())
    assert row["tokens"] == row["utc_epoch"] == 128
    leaked = contains_secret(state.read_text() + events.read_text(), key)
    assert not leaked, "guard persistence disclosed a credential"


def test_guard_inference_environment_excludes_credentials_without_losing_knobs():
    common = load("scripts/guard/common.py")
    key = "rr1" + "InferenceCredential"
    item = {"Config": {"Env": [
        "TF_API_KEY=" + key, "NCCL_PASSWORD=" + key, "CUDA_AUTH_TOKEN=" + key,
        "TF_DSV41_POOL_TOKENS=2100000", "TF_CUDA_SCHED=blocking", "NCCL_IB_HCA=mlx5_0",
        "TF_DSV41_LAUNCH_T0=123", "PATH=/bin",
    ]}}
    result = common.inference_environment(item)
    leaked = contains_secret(json.dumps(result), key)
    assert not leaked, "credential entered an inference identity"
    assert result == {"TF_DSV41_POOL_TOKENS": "2100000", "TF_CUDA_SCHED": "blocking", "NCCL_IB_HCA": "mlx5_0"}


@pytest.mark.parametrize("key", ["z7", "1"], ids=["short_text", "short_numeric"])
def test_short_configured_api_key_masks_text_and_preserves_numeric_metrics(key, monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY", key)
    masked = redact_text("diagnostic " + key, known_secrets())
    assert MASK in masked, "short configured key was ignored"
    check_writes(monkeypatch, key)
    value = {"reply": "echo " + key, "tokens": 128, "accuracy": 1.0, "enabled": True}
    output = io.StringIO()
    # Numeric metrics legitimately contain a numeric key; check strings before serialization.
    safe_output.dump(value, output)
    doc = json.loads(output.getvalue())
    assert doc["reply"] == "echo " + MASK
    assert doc["tokens"] == 128 and doc["accuracy"] == 1.0 and doc["enabled"] is True
    path = tmp_path / "report.json"
    # Do not use CheckedWriter here: it cannot distinguish numeric measurements from strings.
    with monkeypatch.context() as context:
        context.setattr(Path, "open", lambda path, *args, **kwargs: io.open(path, *args, **kwargs))
        safe_output.write_text(path, json.dumps(value))
    doc = json.loads(path.read_text())
    assert doc["tokens"] == 128 and doc["accuracy"] == 1.0
    assert doc["reply"] == "echo " + MASK


def test_cli_explicit_key_reaches_worker_persistence(monkeypatch, tmp_path, capsys):
    key = "rr1" + "ConfiguredCredential"
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["tool", "--api-key", key])
    check_writes(monkeypatch, key)
    path = tmp_path / "report.json"

    def main():
        import threading
        def worker():
            safe_output.write_text(path, json.dumps({"echo": key, "tokens": 128}))
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        sys.stdout.write(key[:8])
        sys.stdout.flush()
        sys.stdout.write(key[8:])
        sys.stderr.write(repr(key))
        return 3

    assert safe_output.cli(main) == 3
    captured = capsys.readouterr()
    leaked = contains_secret(captured.out + captured.err + path.read_text(), key)
    assert not leaked, "CLI-configured key escaped a worker or stream boundary"
    assert json.loads(path.read_text())["tokens"] == 128


def test_pretty_json_cli_keeps_numeric_measurements(monkeypatch, capsys):
    monkeypatch.setenv("API_KEY", "1")
    monkeypatch.setattr(sys, "argv", ["tool"])
    value = {"echo": "1", "tokens": 128, "accuracy": 1.0, "counts": [1, 128]}

    def main():
        safe_output.dump(value, sys.stdout, indent=1)

    assert safe_output.cli(main) == 0
    captured = capsys.readouterr()
    doc = json.loads(captured.out)
    assert doc == {**value, "echo": MASK}


@pytest.mark.parametrize('indent', [1, 2, '\t'])
def test_pretty_json_file_masks_before_serialization(monkeypatch, tmp_path, indent):
    key = "invented'quoted\\value\nnext café"
    monkeypatch.setenv('API_KEY', key)
    value = {'items': [{'reply': key, 'prompt_tokens': 128}], 'accuracy': 1.0,
             'aliases': [key], 'mapping': {key: 'ordinary'}}
    path = tmp_path / 'report.json'
    # Inspect the actual file write, then parse the saved artifact independently.
    check_writes(monkeypatch, key)
    safe_output.write_text(path, json.dumps(value, indent=indent) + '\n')
    saved = path.read_text()
    assert json.loads(saved) == {
        'items': [{'reply': MASK, 'prompt_tokens': 128}], 'accuracy': 1.0,
        'aliases': [MASK], 'mapping': {MASK: 'ordinary'},
    }
    assert saved.startswith('{\n' + str(indent if isinstance(indent, str) else ' ' * indent))
    assert saved.endswith('\n')


@pytest.mark.parametrize("split", ["before_newline", "after_newline", "inside_suffix", "whole" ])
def test_stream_masks_encoded_newline_keys_and_flushes_ordinary_records(split):
    key = "rr1" + "FirstLine\nSecondLineCredential"
    encoded = json.dumps(key)
    index = {"before_newline": encoded.index('\\n'), "after_newline": encoded.index('\\n') + 2,
             "inside_suffix": encoded.index('\\n') + 5, "whole": len(encoded)}[split]
    output = io.StringIO()
    stream = safe_output._Stream(output, [key])
    stream.write("ready\n")
    stream.flush()
    assert output.getvalue() == "ready\n", "ordinary log waited for process exit"
    stream.write("tool: " + encoded[:index])
    stream.flush()
    stream.write(encoded[index:] + "\n")
    stream.flush()
    leaked = contains_secret(output.getvalue(), key)
    assert not leaked, "known key crossed record boundaries"
    assert MASK in output.getvalue()
    for _ in range(100):
        stream.write("ordinary log record\n")
    assert not stream.pending, "completed ordinary logs accumulated in memory"
    stream.finish()


def test_stream_releases_complete_lines_even_when_they_match_multiline_prefix():
    key = "rr1" + "FirstLine\nSecondLineCredential"
    prefix = key[:key.index("\n") + 1]
    output = io.StringIO()
    stream = safe_output._Stream(output, [key])
    stream.write("ready\n" + prefix)
    stream.flush()
    assert output.getvalue() == "ready\n" + prefix
    assert stream.pending == ''
    stream.write("different continuation\n")
    stream.flush()
    assert not stream.pending
    assert output.getvalue().endswith("different continuation\n")


def test_shared_cli_filter_handles_known_internal_newline():
    key = "rr1" + "FirstLine\nSecondLineCredential"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/redact.py"), "text"],
        input="ready\nencoded: " + json.dumps(key) + "\n", text=True,
        env=dict(os.environ, API_KEY=key), capture_output=True, timeout=15,
    )
    leaked = contains_secret(result.stdout + result.stderr, key)
    assert not leaked, "line filter disclosed an internal-newline credential"
    assert result.returncode == 0
    assert result.stdout.startswith("ready\n") and MASK in result.stdout


@pytest.mark.parametrize("multiline", [False, True], ids=["single_line", "multiline"])
def test_cli_uses_repository_default_config_when_config_is_absent(tmp_path, monkeypatch, multiline):
    key = "rr1" + "DefaultConfigCredential" + ("\nSecondLine" if multiline else "")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (tmp_path / "config").mkdir()
    (tmp_path / "config/prod-zig.env").write_text("export API_KEY='" + key + "' # comment\n")
    monkeypatch.delenv("CONFIG", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(safe_output, "__file__", str(scripts / "safe_output.py"))
    collected = key in safe_output._cli_secrets([])
    assert collected, "default repository config key was not collected"


@pytest.mark.parametrize("mode", ["teb", "sb", "both", "chains"])
def test_third_party_report_paths_fail_closed_without_adapter(mode, tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "bench/tooleval/run.sh"), mode, "off"],
        env=dict(os.environ, OUT=str(tmp_path / "reports")), capture_output=True, timeout=15,
    )
    assert result.returncode == 78
    assert b"pre-write sanitization is unavailable" in result.stderr
    assert not (tmp_path / "reports").exists()


def test_tooleval_snapshots_are_sanitized_before_persistence(tmp_path):
    # Fake only the external HTTP/client commands. Inspect all bytes piped to a snapshot writer.
    key = "rr1" + "SnapshotCredential"
    source = (ROOT / "bench/tooleval/run.sh").read_text()
    functions = source[source.index('die()'):source.index('cmd=${1:-}')]
    code = 'set -euo pipefail\nROOT=$TOOL_ROOT\nBASE_URL=http://localhost/v1\n' + functions + r'''
api() { printf '%s\n' "$API_KEY"; printf '%s\n' "$API_KEY" >&2; }
snapshot "$SNAPSHOT_DIR" test
'''
    result = subprocess.run(
        ["bash", "-c", code], env=dict(os.environ, API_KEY=key, SNAPSHOT_DIR=str(tmp_path),
        TOOL_ROOT=str(ROOT)), cwd=ROOT, capture_output=True, timeout=15,
    )
    ok = result.returncode == 0
    assert ok, "snapshot harness failed"
    text = "".join(p.read_text() for p in tmp_path.glob("*.json"))
    leaked = contains_secret(text + (result.stdout + result.stderr).decode("utf-8", "replace"), key)
    assert not leaked, "snapshot or diagnostic disclosed a credential"
    assert text.count(MASK) == 2


def test_canary_error_replies_are_sanitized(monkeypatch, capsys):
    key = 'rr1' + 'CanaryCredentialOnly'
    monkeypatch.setenv('API_KEY', key)
    module = load('scripts/canary.py')
    monkeypatch.setattr(sys, 'argv', ['canary'])

    def failed_reply(*args, **kwargs):
        raise ValueError(key)

    monkeypatch.setattr(module, 'post', failed_reply)
    assert module.main() == 1
    captured = capsys.readouterr()
    leaked = contains_secret(captured.out + captured.err, key)
    assert not leaked, 'canary failure disclosed a credential'


@pytest.mark.parametrize(('script', 'arguments'), [
    ('preflight.py', ['--probe', '{key}']),
    ('pack_engram.py', ['--src', '/missing/{key}', '--list']),
    ('check_native_assets.py', ['--needs', '/missing/needs', '--aot', '/missing/{key}']),
    ('guard/control.py', ['--config', '/missing/{key}', 'authorize-profile', '--sha', 'fixture']),
    ('guard/profile.py', ['--config', '/missing/{key}']),
    ('guard/lifecycle.py', ['--config', '/missing/{key}', 'check']),
    ('guard/sampler.py', ['--config', '/missing/{key}']),
    ('guard/daemon.py', ['--config', '/missing/{key}']),
    ('native_ops.py', ['status']),
])
def test_host_tool_setup_errors_mask_both_streams(script, arguments, tmp_path):
    key = 'rr1' + 'HostSetupCredentialOnly'
    env = dict(os.environ, API_KEY=key, WORKER_SSH='worker.example',
               WORKER_REPO=str(ROOT), WORKER_CONFIG='/missing/config.env',
               STATE_DIR=str(tmp_path), READY_TIMEOUT=key)
    result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script),
                             *[value.format(key=key) for value in arguments]],
                            cwd=ROOT, env=env, capture_output=True, timeout=15)
    leaked = contains_secret((result.stdout + result.stderr).decode('utf-8', 'replace'), key)
    assert result.returncode != 0
    assert not leaked, 'host-tool setup failure disclosed a credential'


@pytest.mark.parametrize('name', ['AUTH_TOKEN', 'API_TOKEN', 'TOKEN', 'SHORT_TOKEN'])
def test_numeric_token_environment_is_redacted_before_persistence(name, monkeypatch):
    monkeypatch.delenv('API_KEY', raising=False)
    monkeypatch.setenv(name, '200')
    out = io.StringIO()
    safe_output.dump({'opaque': '200', 'inflight': 200}, out)
    result = json.loads(out.getvalue())
    assert result['opaque'] == '<redacted>'
    assert result['inflight'] == 200
