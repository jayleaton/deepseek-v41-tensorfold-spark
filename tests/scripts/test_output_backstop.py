"""The output backstop contract (docs/SECURITY.md): stateless lines, fixed renderings, prose untouched."""

import io
import json
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import safe_output  # noqa: E402
from redact import MASK, redact_line, redact_text  # noqa: E402
from safe_quote import quote_checked  # noqa: E402

SECRET = "rr8-invented key/with 'quote' + space"
RENDERINGS = {
    "literal": SECRET,
    "json": json.dumps(SECRET)[1:-1],
    "repr": repr(SECRET)[1:-1],
    "bytes": repr(SECRET.encode())[2:-1],
    "shell": quote_checked(SECRET),
    "url": urllib.parse.quote(SECRET, safe=""),
}


@pytest.mark.parametrize("rendering", RENDERINGS.values(), ids=RENDERINGS.keys())
def test_fixed_renderings_are_masked_on_their_line(rendering):
    line = f"worker: value={rendering} retry\n"
    masked = redact_line(line, [SECRET])
    assert SECRET not in masked and rendering not in masked
    assert masked.startswith("worker: value=") and masked.endswith(" retry\n")


@pytest.mark.parametrize("line", [
    "fatal: couldn't find remote ref refs/heads/missing\n",
    "[worker] can't connect to worker.example.invalid\n",
    "R=7 rows=128 progress=50\n",
    "worker said 'ready' and \"steady\"; price $5, café\n",
    "{unfinished container\n",
])
def test_ordinary_lines_are_verbatim_with_a_configured_key(line):
    assert redact_line(line, [SECRET]) == line
    output = io.StringIO()
    stream = safe_output._Stream(output, [SECRET])
    stream.write(line)
    assert output.getvalue() == line


def test_a_line_never_depends_on_another_line():
    lines = ["OPAQUE='unfinished\n", "error: missing aot/kernel.fatbin\n", SECRET + "\n"]
    output = io.StringIO()
    stream = safe_output._Stream(output, [SECRET])
    for line in lines:
        stream.write(line)
    assert output.getvalue() == lines[0] + lines[1] + MASK + "\n"


def test_value_split_across_lines_is_not_a_backstop_promise():
    # Documented limit: only complete documents are masked across lines. Streams
    # rely on the child never holding the value (child_environment).
    first, second = SECRET[:10], SECRET[10:]
    assert redact_text(first + "\n" + second + "\n", [first + "\n" + second]) == MASK + "\n"
    output = io.StringIO()
    stream = safe_output._Stream(output, [first + "\n" + second])
    stream.write(first + "\n" + second + "\n")
    assert output.getvalue() == first + "\n" + second + "\n"


def test_complete_json_documents_keep_numbers_and_syntax():
    document = json.dumps({"inflight": 200, "opaque": "200", "token_source": None, "api_key": "x"})
    cleaned = json.loads(safe_output.clean_document(document, ["200"]))
    assert cleaned == {"inflight": 200, "opaque": MASK, "token_source": None, "api_key": MASK}


def test_serving_keys_in_the_auth_file_are_known_to_the_backstop(tmp_path):
    # A process that prints the serving key (as a misbehaving engine log line
    # would) through the real supervisor: the key never reaches either stream.
    keys = tmp_path / "auth-keys"
    keys.write_text("# operator keys\nrr8-invented-plain-serving-key\nci[inference]: rr8-invented-labelled-key\n")
    keys.chmod(0o600)
    program = ("import sys; text=open(sys.argv[1]).read()\n"
               "print(text, flush=True); print(text, file=sys.stderr, flush=True); raise SystemExit(47)\n")
    env = {"PATH": os.environ["PATH"], "CONFIG": "/nonexistent", "NATIVE_AUTH_FILE": str(keys)}
    result = subprocess.run([sys.executable, str(ROOT / "scripts/safe_exec.py"), "--command",
                             sys.executable, "-c", program, str(keys)],
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 47
    for output in (result.stdout, result.stderr):
        assert "rr8-invented-plain-serving-key" not in output
        assert "rr8-invented-labelled-key" not in output
        assert "# operator keys" in output and output.count(MASK) == 2
