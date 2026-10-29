"""Environment structure is recognized before any credential-name redaction."""
import json
import sys
import subprocess
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from safe_output import clean, _clean_text  # noqa: E402


@pytest.mark.parametrize('field', ['Env', 'env', 'ENV', 'environment', 'Environment'])
@pytest.mark.parametrize('mapping', [False, True])
def test_environment_values_are_never_saved_even_when_field_name_is_a_known_value(field, mapping):
    ordinary = 'ordinary-environment-value-that-is-not-a-secret'
    env = {'OPAQUE': ordinary} if mapping else ['OPAQUE=' + ordinary]
    result = clean(json.dumps({'Config': {field: env}}), [field])
    leaked = ordinary in result
    assert not leaked, 'environment value escaped structural projection'
    assert json.loads(result)['Config']['EnvNames'] == ['OPAQUE']


def test_ordinary_record_separators_are_preserved():
    assert clean("first\nsecond\n") == "first\nsecond\n"
    assert clean("first\nsecond") == "first\nsecond"


def test_unterminated_quoted_record_does_not_duplicate_physical_newline():
    text = "unterminated 'record\n"
    assert _clean_text(text, []) == text


def test_state_evidence_projects_only_typed_operational_fields():
    from watchdog_io import filter_output
    value = {'Running': True, 'ExitCode': 47, 'Status': 'running', 'Pid': 123,
             'StartedAt': '2026-10-29T00:00:00Z', 'Error': 'opaque-error-value',
             'Env': ['OPAQUE=ordinary-private-value'],
             'Config': {'Env': ['OPAQUE=nested-private-value']}}
    result = json.loads(filter_output('state', json.dumps(value)))
    assert result == {key: value[key] for key in
                      ('Running', 'ExitCode', 'Status', 'Pid', 'StartedAt')}


def test_retired_transport_harness_never_executes_tools_or_echoes_arguments(tmp_path):
    marker = tmp_path / 'unexpected-tool'
    tools = tmp_path / 'bin'
    tools.mkdir()
    for name in ('ssh', 'nvidia-smi', 'hostname', 'timeout'):
        tool = tools / name
        tool.write_text('#!/bin/sh\nprintf called > "$UNEXPECTED_TOOL"\n')
        tool.chmod(0o700)
    key = 'rr5' + "'TransportSyntheticOnly"
    path = Path(__file__).resolve().parents[2] / 'engine/zig/zig/src/fabric/tp/run_transport.sh'
    result = subprocess.run(['/bin/bash', str(path), key, key, str(tmp_path)],
                            env={'PATH': str(tools), 'API_KEY': key,
                                 'UNEXPECTED_TOOL': str(marker)}, capture_output=True, text=True)
    leaked = key in result.stdout + result.stderr
    assert result.returncode == 78
    assert not leaked
    assert not marker.exists()


@pytest.mark.parametrize('value', ["/srv/with'quote", '/srv/with\"double', "/srv/with'and\"quotes"])
def test_raw_checked_argv_does_not_treat_literal_path_quotes_as_output_syntax(value):
    from redact import check_argv
    check_argv([value], ['unrelated-credential'])
    with pytest.raises(ValueError):
        check_argv([value], [value])


def test_typed_raw_scratch_path_supports_literal_apostrophes():
    from watchdog_io import filter_output
    value = "/tmp/ordinary'quoted-directory"
    assert filter_output('path', value, ['unrelated-credential']) == value + '\n'
    with pytest.raises(ValueError):
        filter_output('path', value, [value])
