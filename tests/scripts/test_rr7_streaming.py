"""Complete diagnostic lines arrive before child exit; credentials stay out of children."""

import io
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import safe_output  # noqa: E402
from redact import split_lines  # noqa: E402


@pytest.mark.parametrize('line', [
    "[worker] couldn't open aot/kernel.fatbin\n",
    "[worker] 'unfinished quote\n",
    '{incomplete container\n',
    'OPAQUE_SETTING="unfinished\n',
    '-----BEGIN ' + 'PRIVATE KEY-----\n',
    'known first line\n',
])
def test_complete_lines_never_wait_for_syntax_or_secret_continuation(line):
    secrets = ['known first line\nknown second line']
    output = io.StringIO()
    stream = safe_output._Stream(output, secrets)
    stream.write(line + 'partial')
    stream.flush()
    assert output.getvalue() == safe_output.clean(line, secrets)
    assert stream.pending == 'partial'
    stream.write(' retry\n')
    assert output.getvalue().endswith('partial retry\n')
    assert stream.pending == ''
    assert split_lines(line + 'partial') == (line, 'partial')


def test_assignment_lines_are_released_verbatim_and_independently(monkeypatch):
    monkeypatch.setenv('ACTUAL_SETTING', 'ordinary environment value')
    text = 'ACTUAL_SETTING=ordinary environment value\nRESULT=ready\nR=7\n'
    assert safe_output.clean(text) == text
    output = io.StringIO()
    stream = safe_output._Stream(output, [])
    for line in text.splitlines(keepends=True):
        stream.write(line)
        assert output.getvalue().endswith(line)
    assert stream.pending == ''


@pytest.mark.parametrize('terminator', ['\r', '\n'])
def test_progress_carriage_returns_release_each_update(terminator):
    output = io.StringIO()
    stream = safe_output._Stream(output, [])
    stream.write('progress 10%' + terminator + 'progress 20%')
    assert output.getvalue() == 'progress 10%' + terminator
    assert stream.pending == 'progress 20%'


def test_unterminated_output_is_released_at_the_line_limit():
    output = io.StringIO()
    stream = safe_output._Stream(output, [])
    stream.write('x' * stream.LINE_LIMIT)
    assert output.getvalue() == 'x' * stream.LINE_LIMIT
    assert stream.pending == ''


def test_real_supervisor_delivers_both_streams_before_exit():
    # stdin is the release handshake: no filesystem marker or disk polling.
    text = ("[worker] couldn't open aot/kernel.fatbin\n"
            'normal retry in 5 seconds\n'
            '{unfinished container\n'
            'error: missing kernel\n')
    program = (
        'import sys\n'
        f'text = {text!r}\n'
        'for stream in (sys.stdout, sys.stderr):\n'
        '    stream.write(text)\n'
        '    stream.flush()\n'
        "assert sys.stdin.buffer.read(1) == b'!'\n"
        'raise SystemExit(47)\n'
    )
    # A configured, unrelated credential must not change ordinary framing.
    env = {'PATH': os.environ['PATH'], 'CONFIG': '/nonexistent',
           'API_KEY': 'test-rr8-unrelated-' + 'synthetic'}
    child = subprocess.Popen(
        [sys.executable, str(ROOT / 'scripts/safe_exec.py'), '--command',
         sys.executable, '-c', program], env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    received = {'stdout': b'', 'stderr': b''}
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ, 'stdout')
            selector.register(child.stderr, selectors.EVENT_READ, 'stderr')
            deadline = time.monotonic() + 10
            while not all(data == text.encode() for data in received.values()):
                remaining = deadline - time.monotonic()
                assert remaining > 0, 'complete diagnostics waited for child exit'
                for event, _ in selector.select(remaining):
                    data = os.read(event.fd, 65536)
                    assert data, 'stream closed before release handshake'
                    received[event.data] += data
            assert child.poll() is None, 'child exited before release handshake'
        out, err = child.communicate(input=b'!', timeout=10)
        assert child.returncode == 47
        assert received['stdout'] + out == text.encode()
        assert received['stderr'] + err == text.encode()
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


@pytest.mark.parametrize('width', [20, 45, 80])
@pytest.mark.parametrize('secret', [
    'invented private words ' * 20,
    "invented'quoted\nsecond line\nlast line",
    '-----BEGIN ' + 'PRIVATE KEY-----\ninvented-body\n-----END ' + 'PRIVATE KEY-----',
])
def test_real_child_cannot_reconstruct_multiline_or_pprint_credentials(width, secret):
    program = (
        'import json, os, pprint, sys\n'
        "names = ('API_KEY', 'OPAQUE_SETTING', 'NESTED_SETTING')\n"
        "print(json.dumps({'absent': all(n not in os.environ for n in names)}), flush=True)\n"
        'for stream in (sys.stdout, sys.stderr):\n'
        f'    pprint.pprint(dict(os.environ), width={width}, stream=stream)\n'
        "    print('ordinary retry diagnostic', file=stream, flush=True)\n"
        'raise SystemExit(47)\n'
    )
    env = {'PATH': os.environ['PATH'], 'CONFIG': '/nonexistent', 'API_KEY': secret,
           'OPAQUE_SETTING': secret, 'NESTED_SETTING': json.dumps(secret)}
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/safe_exec.py'), '--command',
         sys.executable, '-c', program], env=env, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 47
    assert json.loads(result.stdout.splitlines()[0]) == {'absent': True}
    for output in (result.stdout, result.stderr):
        assert secret not in output
        assert json.dumps(secret)[1:-1] not in output
        assert 'ordinary retry diagnostic' in output
