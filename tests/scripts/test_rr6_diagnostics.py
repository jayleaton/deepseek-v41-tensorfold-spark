"""Real diagnostics survive the credential boundary; environment values do not."""

import ast
import io
import json
import os
from pathlib import Path
import pprint
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import safe_output  # noqa: E402
from redact import MASK, _filter, redact_text  # noqa: E402


def supervisor(command, tmp_path):
    env = {'PATH': os.environ['PATH'], 'CONFIG': '/nonexistent',
           'HOME': str(tmp_path), 'API_KEY': 'test-rr6-unrelated-' + 'synthetic'}
    return subprocess.run([sys.executable, str(ROOT / 'scripts/safe_exec.py'),
                           '--command', *command], env=env, capture_output=True,
                          text=True, timeout=30)


def test_real_git_missing_ref_diagnostic(tmp_path):
    repository = tmp_path / 'empty'
    subprocess.run(['git', 'init', '-q', str(repository)], check=True, capture_output=True)
    command = ['git', '-C', str(repository), 'fetch', str(repository),
               'refs/heads/rr6-missing']
    raw = subprocess.run(command, capture_output=True, text=True)
    checked = supervisor(command, tmp_path)
    assert raw.returncode == checked.returncode == 128
    assert "couldn't find remote ref refs/heads/rr6-missing" in raw.stderr
    assert checked.stderr == raw.stderr


def test_tracked_result_log_replayed_through_supervisor(tmp_path):
    path = ROOT / 'results/campaign/G1-20261002/kbench-l3.log'
    raw = path.read_text()
    checked = supervisor(['cat', str(path)], tmp_path)
    assert checked.returncode == 0
    assert sum(line.startswith('R=') for line in raw.splitlines()) == 7
    assert checked.stdout == raw


@pytest.mark.parametrize('text', [
    'progress=50\nerror: missing aot/kernel.fatbin\n',
    "can't connect to worker.example.invalid\n",
    "NAME=value\nerror: missing aot/kernel.fatbin\n",
])
def test_ordinary_assignments_and_trailing_errors(text):
    secrets = ['unrelated-synthetic-value']
    assert safe_output.clean(text, secrets) == text
    output = io.StringIO()
    stream = safe_output._Stream(output, secrets)
    for char in text:
        stream.write(char)
    stream.finish()
    assert output.getvalue() == text


@pytest.mark.parametrize('sink', ['clean', 'stream', 'file', 'print', 'dump', 'dump-stream', 'filter'])
def test_single_line_python_environment_at_sinks(monkeypatch, tmp_path, sink):
    secret = ('rr6 private value with spaces ' * 6) + 'SyntheticOnly'
    ordinary = 'ordinary-environment-value'
    monkeypatch.setenv('API_KEY', secret)
    monkeypatch.setenv('ORDINARY_SETTING', ordinary)
    value = {'API_KEY': secret, 'ORDINARY_SETTING': ordinary}
    text = pprint.pformat(value, width=1000) + '\n'
    assert ast.literal_eval(text) == value
    assert secret in text
    output = io.StringIO()
    if sink == 'clean':
        sanitized = safe_output.clean(text)
    elif sink == 'stream':
        stream = safe_output._Stream(output, [secret])
        for char in text:
            stream.write(char)
        stream.finish()
        sanitized = output.getvalue()
    elif sink == 'file':
        path = tmp_path / 'pprint.log'
        safe_output.write_text(path, text)
        sanitized = path.read_text()
    elif sink == 'print':
        safe_output.safe_print(text, end='', file=output)
        sanitized = output.getvalue()
    elif sink == 'filter':
        _filter(io.StringIO(text), output, [secret])
        sanitized = output.getvalue()
    else:
        stream = safe_output._Stream(output, [secret]) if sink == 'dump-stream' else output
        safe_output.dump({'diagnostic': text}, stream)
        if sink == 'dump-stream':
            stream.finish()
        sanitized = json.loads(output.getvalue())['diagnostic']
    # Parse inertly: checking only the literal secret misses pprint's split value.
    try:
        recovered = ast.literal_eval(sanitized)
    except (SyntaxError, ValueError):
        recovered = None
    assert recovered != value, 'pprint credential was reconstructable'
    assert secret not in sanitized and MASK in sanitized
    # Ordinary settings are not credentials; the backstop leaves them readable.
    assert ordinary in sanitized


def test_ordinary_multiline_assignment_and_following_error_are_verbatim(monkeypatch):
    value = "ordinary first line\nsecond line with ' unmatched quote\nlast line"
    monkeypatch.setenv('ACTUAL_ENV_SETTING', value)
    text = 'ACTUAL_ENV_SETTING=' + value + '\nerror: missing aot/kernel.fatbin\n'
    for streamed in (False, True):
        if streamed:
            output = io.StringIO()
            stream = safe_output._Stream(output, [])
            for char in text:
                stream.write(char)
            stream.finish()
            result = output.getvalue()
        else:
            result = safe_output.clean(text)
        assert result == text


def test_environment_assignment_does_not_hide_following_function(monkeypatch):
    monkeypatch.setenv('ACTUAL_ENV_SETTING', 'ordinary-value')
    text = ('ACTUAL_ENV_SETTING=ordinary-value\nhelper () \n{\n'
            '  echo ordinary-function-value\n}\nerror: missing kernel\n')
    output = io.StringIO()
    stream = safe_output._Stream(output, [])
    for char in text:
        stream.write(char)
    stream.finish()
    assert 'ordinary-function-value' in output.getvalue()
    assert output.getvalue().endswith('error: missing kernel\n')


def test_large_benign_decoding_does_not_trigger_limit_masking():
    text = json.dumps({'measurements': [f'ordinary-{i}' for i in range(300)]})
    assert redact_text(text, ['unrelated-synthetic-value']) == text


@pytest.mark.parametrize('record', [
    "{'payload': b'ordinary-bytes'}",
    "{'payload': {'ordinary-set-member'}}",
    "{('ordinary-tuple-key', 2): 'ordinary-value'}",
    "[b'ordinary-bytes', {'ordinary-set-member'}, {('tuple-key',): (1+2j)}, ...]",
])
def test_python_diagnostic_types_preserve_following_output_and_exit(tmp_path, record):
    # A real child emits a Python diagnostic on each stream, then an independent
    # error. Both lines are released verbatim; nothing is reparsed or reserialized.
    script = tmp_path / 'python-diagnostic.py'
    script.write_text(
        'import sys\n'
        'for stream in (sys.stdout, sys.stderr):\n'
        f'    print({record!r}, file=stream, flush=True)\n'
        "    print('error: missing aot/kernel.fatbin', file=stream, flush=True)\n"
        'raise SystemExit(47)\n'
    )
    checked = supervisor([sys.executable, str(script)], tmp_path)
    assert checked.returncode == 47
    for output in (checked.stdout, checked.stderr):
        lines = output.splitlines()
        assert lines == [record, 'error: missing aot/kernel.fatbin']


def test_python_only_types_are_masked_in_place(monkeypatch):
    secret = 'test-rr6-private-' + 'normalization-value'
    monkeypatch.setenv('API_KEY', secret)
    record = repr({(secret,): [secret.encode(), {secret}, 1 + 2j]})
    cleaned = safe_output.clean(record)
    assert secret not in cleaned
    assert cleaned.count(MASK) == 3 and cleaned.endswith('(1+2j)]}')
