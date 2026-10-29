"""Known credentials cannot survive shell diagnostic encoding at shared text sinks."""

from __future__ import annotations

import io
import json
import sys
import subprocess
import os
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from redact import MASK, _filter, redact_text, split_safe_records  # noqa: E402
from safe_output import _Stream, clean, write_text  # noqa: E402


@pytest.mark.parametrize("record,secret", [
    ("OPAQUE_SETTING='Randy'\\''s credential'", "Randy's credential"),
    ('OPAQUE_SETTING=pre\'Randy\'"\'"\'s "mixed"\'post', 'preRandy\'s "mixed"post'),
    (r"OPAQUE_SETTING=$'Randy\047s \u2603\nnext'", "Randy's ☃\nnext"),
    (r"OPAQUE_SETTING=$'ab\x27cd\U0001f512'", "ab'cd🔒"),
    (r"OPAQUE_SETTING='ab'$'\x27'cd", "ab'cd"),
    (r"OPAQUE_SETTING=$'ab\ccde'", "ab\x03de"),
    (r"OPAQUE_SETTING=$'Randy\047s \342\230\203'", "Randy's ☃"),
])
def test_shell_diagnostics_mask_whole_record(record, secret):
    assert redact_text(record + "\n", [secret]) == MASK + "\n"
    assert MASK in clean(record, [secret])


@pytest.mark.parametrize("wrapper", [json.dumps, repr, ascii])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_nested_string_encodings(wrapper, depth):
    secret = "Randy's ☃\nnext"
    record = r"OPAQUE_SETTING=$'Randy\047s \u2603\nnext'"
    for _ in range(depth):
        record = wrapper(record)
    assert redact_text(record, [secret]) == MASK


@pytest.mark.parametrize("record", [
    "OPAQUE_SETTING='Randy'\\''s benign setting'",
    r"OPAQUE_SETTING=$'benign\n\u2603'",
    'OPAQUE_SETTING=pre\'a b\'"c d"post',
    '{"status": 200, "message": "benign quoted value"}',
])
def test_benign_quoted_values_are_preserved(record):
    assert redact_text(record, ["unrelated credential"]) == record


@pytest.mark.parametrize("record", [
    "OPAQUE_SETTING='unfinished",
    "OPAQUE_SETTING=$'broken\\uXYZ'",
    "OPAQUE_SETTING=$'broken\\cß'",
    "OPAQUE_SETTING=unfinished\\",
    "tool diagnostic: 'Randy'\\''s\nunfinished",
])
def test_malformed_assignments_without_known_values_are_preserved(record):
    assert redact_text(record, ["configured credential"]) == record


def test_no_shell_evaluation(tmp_path):
    marker = tmp_path / "must-not-exist"
    record = f"OPAQUE_SETTING=$(touch {marker})`touch {marker}`"
    assert redact_text(record, ["unrelated credential"]) == record
    assert not marker.exists()


@pytest.mark.parametrize("prefix", ["OPAQUE_SETTING=", "", "tool diagnostic: "])
def test_multiline_shell_syntax_releases_each_complete_physical_line(prefix):
    secret = "Randy's\nsecond fragment"
    record = prefix + "'Randy'\\''s\nsecond fragment'\n"
    pending = ""
    released = ""
    for char in record:
        ready, pending = split_safe_records(pending + char, [secret])
        released += ready
        assert '\n' not in pending
        assert not ready or ready.endswith('\n')
    assert pending == ""
    assert released == record


def test_benign_multiline_assignment_is_preserved():
    record = "OPAQUE_SETTING='benign\nquoted setting'\n"
    ready, pending = split_safe_records(record, ["unrelated credential"])
    assert ready == record
    assert pending == ""
    output = io.StringIO()
    _filter(io.StringIO(record), output, ["unrelated credential"])
    assert output.getvalue() == record


def test_stream_and_file_sinks_use_shared_decoder(tmp_path):
    secret = "Randy's ☃\nnext"
    record = r"OPAQUE_SETTING=$'Randy\047s \u2603\nnext'"
    output = io.StringIO()
    stream = _Stream(output, [secret])
    for char in record + "\n":
        stream.write(char)
    stream.finish()
    assert output.getvalue() == MASK + " environment variable: OPAQUE_SETTING\n"
    # write_text obtains configured credentials from the environment.
    from unittest.mock import patch
    with patch.dict("os.environ", {"API_KEY": secret}):
        path = tmp_path / "diagnostic.log"
        write_text(path, record)
    assert MASK in path.read_text()


def test_numeric_json_protocol_values_remain_numeric():
    record = '{"status": 200, "count": 7, "message": "benign"}'
    assert json.loads(clean(record, ["200"])) == {
        "status": 200, "count": 7, "message": "benign",
    }


def test_safe_parsers_cannot_echo_source_in_warnings():
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        redact_text(r"OPAQUE_SETTING=$'benign\cA'", ["unrelated credential"])
    assert not emitted


def test_prose_apostrophe_cannot_absorb_following_encoded_assignment():
    secret = "rr5 alpha"
    prose = "can't connect\n"
    assignment = "OPAQUE_SETTING=rr5\\ alpha\n"
    text = prose + assignment
    assert redact_text(text, [secret]) == prose + MASK + "\n"
    assert clean(text, [secret]) == prose + MASK + " environment variable: OPAQUE_SETTING\n"
    output = io.StringIO()
    stream = _Stream(output, [secret])
    stream.write(prose)
    assert output.getvalue() == prose
    stream.write(assignment)
    stream.finish()
    assert output.getvalue() == prose + MASK + " environment variable: OPAQUE_SETTING\n"
    filtered = io.StringIO()
    _filter(io.StringIO(text), filtered, [secret])
    assert filtered.getvalue() == prose + MASK + "\n"


def test_ambiguous_prose_with_encoded_value_fails_closed():
    assert redact_text("can't connect rr5\\ alpha\n", ["rr5 alpha"]) == MASK + "\n"


def test_prose_is_preserved_without_consuming_line_separators():
    text = "can't connect\n\nRandy's ordinary prose\nnext line\n"
    ready, pending = split_safe_records(text, ["unrelated credential"])
    assert ready == text
    assert pending == ""
    assert redact_text(text, ["unrelated credential"]) == text
    assert clean(text, ["unrelated credential"]) == text


def test_word_internal_half_quotes_without_known_values_are_preserved():
    text = "a'b\nc'\n"
    assert redact_text(text, ["configured credential"]) == text


@pytest.mark.parametrize('encoding', ['utf-8', 'latin-1'])
@pytest.mark.parametrize('wrapper', [str, json.dumps, repr])
def test_shell_quoted_byte_diagnostics_are_decoded_before_publication(encoding, wrapper):
    secret = "rr5" + chr(39) + chr(34) + "café\nOnly"
    encoded = repr(secret.encode(encoding))
    result = subprocess.run(['/bin/bash', '-c', 'printf "diagnostic %q\\n" "$OPAQUE"'],
                            env={'PATH': os.environ['PATH'], 'OPAQUE': encoded},
                            capture_output=True, text=True, check=True)
    text = wrapper(result.stdout.rstrip('\n'))
    masked = redact_text(text, [secret]) == MASK
    assert masked, 'shell-quoted byte diagnostic escaped decode-aware sink'
