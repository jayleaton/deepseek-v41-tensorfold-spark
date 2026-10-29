"""RR5: real Bash diagnostics must not survive any publishing boundary.

Run only with invented credentials in a minimal environment. Raw diagnostics stay
in memory; even baseline failures report booleans rather than captured records.
"""

import ast
import codecs
import io
import json
import os
from pathlib import Path
import random
import re
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests/guard"))
import safe_output  # noqa: E402
from safe_exec import child_environment  # noqa: E402
from test_watchdog_redaction import fixture as watchdog_fixture, run as watchdog_run  # noqa: E402

BENIGN = "worker said 'ready' and \"steady\"; price $5, path C:\\work, café"
ORDINARY = "ordinary-diagnostic-value-without-a-credential"
# Each command emits one encoding only, with all eight values batched together.
# Bash creates its own shell representations at runtime.
ENCODERS = {
    "set": "set",
    "declare": 'declare -p "${!OPAQUE_SETTING_@}"',
    "printf-q": r"""for name in "${!OPAQUE_SETTING_@}"; do printf '%s=%q\n' "$name" "${!name}"; done""",
    "json": """"$PROBE_PYTHON" -c 'import json,os; print(json.dumps({k:v for k,v in os.environ.items() if k.startswith("OPAQUE_SETTING_")}))' """,
    "repr": """"$PROBE_PYTHON" -c 'import os; print(repr({k:v for k,v in os.environ.items() if k.startswith("OPAQUE_SETTING_")}))' """,
}


def both_streams(encoding):
    command = 'printf "%s\\n" "$BENIGN_MESSAGE"\n' + ENCODERS[encoding]
    return "{\n" + command + "\n}; {\n" + command + "\n} >&2; exit 47"


ANSI = re.compile(r"\$'((?:\\.|[^'\\])*)'", re.DOTALL)


def decoded_strings(text):
    """Independent inert decoder: JSON/repr containers and shell word syntax.

    Never execute captured text. Bash ANSI-C byte escapes are decoded before
    POSIX shlex tokenization, including octal UTF-8 produced under LC_ALL=C.
    """
    yield text
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            value = None
    if isinstance(value, dict):
        for child in value.values():
            if isinstance(child, str):
                yield from decoded_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            if isinstance(child, str):
                yield from decoded_strings(child)

    def ansi_word(match):
        body = match[1].replace(r"\'", "'").replace(r"\e", r"\033")

        def escape(part):
            token = part[0]
            if (
                token[1:2]
                not in (
                    b"\\",
                    b"'",
                    b'"',
                    b"a",
                    b"b",
                    b"f",
                    b"n",
                    b"r",
                    b"t",
                    b"v",
                    b"x",
                )
                and not token[1:2].isdigit()
            ):
                raise ValueError("unsupported diagnostic escape")
            return codecs.escape_decode(token)[0]

        raw = re.sub(
            rb"\\(?:[0-7]{1,3}|x[0-9a-fA-F]{1,2}|.)", escape, body.encode("utf-8")
        )
        return shlex.quote(raw.decode("utf-8"))

    for line in text.splitlines():
        # Serialization may contain several independent JSON/repr records.
        if line != text and line[:1] in "{[":
            yield from decoded_strings(line)
        try:
            yield from shlex.split(ANSI.sub(ansi_word, line))
        except (ValueError, UnicodeError):
            pass


def recoverable(text, values):
    return any(
        value in decoded for decoded in decoded_strings(text) for value in values
    )


def assert_private(text, values):
    leaked = recoverable(text, values)
    assert not leaked, "recoverable synthetic credential reached an output sink"


def minimal_env(tmp_path, values):
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "CONFIG": "/nonexistent",
        "LC_ALL": "C",
        "PROBE_PYTHON": sys.executable,
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "OPAQUE_ORDINARY": ORDINARY,
        "BENIGN_MESSAGE": BENIGN,
    }
    for index, value in enumerate(values):
        env[f"API_KEY_{index}"] = value
        env[f"OPAQUE_SETTING_{index}"] = value
    return env


@pytest.fixture(scope="module")
def credentials():
    rng = random.Random(49237)
    alphabet = "abcXYZ019'\" $\\\néλ雪"
    return tuple(
        "rr5-"
        + str(index)
        + "-"
        + "".join(rng.sample("'\" $\\\néλ雪", 9))
        + "".join(rng.choice(alphabet) for _ in range(20))
        + "-end"
        for index in range(8)
    )


def raw_batch(env, command):
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", command],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 47, "diagnostic emitter did not preserve exit 47"
    return result


@pytest.fixture(scope="module")
def encoded_batches(tmp_path_factory, credentials):
    env = minimal_env(tmp_path_factory.mktemp("encoding-oracle"), credentials)
    # Each encoding must independently reconstruct every complete credential.
    # A broken oracle must never make the privacy assertions vacuously pass.
    batches = {}
    for encoding in ENCODERS:
        result = raw_batch(env, both_streams(encoding))
        for text in (result.stdout, result.stderr):
            for value in credentials:
                verified = recoverable(text, (value,))
                assert verified, (
                    "independent decoder did not recover an input credential"
                )
            benign_preserved = text.startswith(BENIGN + "\n")
            assert benign_preserved, "emitter did not prefix the benign control"
        batches[encoding] = result.stdout
    return batches


@pytest.fixture(scope="module", params=tuple(ENCODERS))
def batch(request, encoded_batches):
    return request.param, encoded_batches[request.param]


@pytest.mark.parametrize(
    "sink", ("clean", "stream", "write_text", "dump", "safe_print", "dump-stream")
)
def test_seeded_encoded_batch_at_shared_output_sinks(
    tmp_path, monkeypatch, batch, credentials, sink
):
    encoding, text = batch
    env = minimal_env(tmp_path, credentials)
    if encoding == 'set':
        # Raw multiline shell output is protected at the child boundary. Other
        # encoders below still exercise masking of the original seeded values.
        isolated = child_environment(env)
        assert all(f'API_KEY_{i}' not in isolated and f'OPAQUE_SETTING_{i}' not in isolated
                   for i in range(len(credentials)))
        text = raw_batch(isolated, both_streams(encoding)).stdout
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    if sink == "clean":
        output = safe_output.clean(text)
    elif sink == "stream":
        output_stream = io.StringIO()
        stream = safe_output._Stream(output_stream, list(credentials))
        # Split shell quotes, escapes and physical multiline records across writes.
        for offset in range(0, len(text), 7):
            stream.write(text[offset : offset + 7])
            stream.flush()
        stream.finish()
        output = output_stream.getvalue()
    elif sink == "write_text":
        path = tmp_path / "sanitized.txt"
        safe_output.write_text(path, text)
        output = path.read_text()
        path.unlink()
    elif sink == "safe_print":
        output_stream = io.StringIO()
        safe_output.safe_print(text, end="", file=output_stream)
        output = output_stream.getvalue()
    else:
        output_stream = io.StringIO()
        stream = (
            safe_output._Stream(output_stream, list(credentials))
            if sink == "dump-stream"
            else output_stream
        )
        safe_output.dump({"diagnostic": text}, stream)
        if sink == "dump-stream":
            stream.finish()
        output = json.loads(output_stream.getvalue())["diagnostic"]
    assert_private(output, credentials)
    if encoding == "set":
        ordinary_leaked = ORDINARY in output
        assert not ordinary_leaked, (
            "ordinary environment assignment value was published"
        )
    benign_preserved = BENIGN in output
    assert benign_preserved, "benign quoted diagnostic was lost"


@pytest.mark.parametrize("boundary", ("safe_exec", "watch_run", "watch_capture"))
def test_seeded_batch_through_real_supervisors(tmp_path, credentials, batch, boundary):
    encoding, _ = batch
    env = minimal_env(tmp_path, credentials)
    script = tmp_path / "emit.sh"
    # A real child checks inheritance before emitting diagnostics. Exit 99 cannot
    # be disguised by output masking, including for raw multiline Bash values.
    isolation_check = (
        '"$PROBE_PYTHON" -c \'import os,sys; '
        'sys.exit(99 if any(n.startswith(("API_KEY_", "OPAQUE_SETTING_")) '
        'for n in os.environ) else 0)\' || exit 99\n'
    )
    script.write_text(isolation_check + both_streams(encoding))
    env["DUMP_SCRIPT"] = str(script)
    if boundary == "safe_exec":
        # The CLI publishes child output. The internal run() API captures data
        # for typed parsing before it reaches a publishing boundary.
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/safe_exec.py"),
                "bash",
                "--noprofile",
                "--norc",
                str(script),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    else:
        env["SAVED_OUTPUT"] = str(tmp_path / "saved-output")
        entry = "watch_run" if boundary == "watch_run" else "watch_capture json"
        result = watchdog_run(
            env,
            entry + ' bash --noprofile --norc "$DUMP_SCRIPT" > "$SAVED_OUTPUT"; '
            'code=$?; watch_run cat "$SAVED_OUTPUT"; exit "$code"',
        )
        saved = Path(env["SAVED_OUTPUT"]).read_text()
        Path(env["SAVED_OUTPUT"]).unlink()
        assert_private(saved, credentials)
    assert result.returncode == 47, "supervisor changed failing child exit 47"
    for output in (result.stdout, result.stderr):
        assert_private(output, credentials)
        if encoding == "set":
            ordinary_leaked = ORDINARY in output
            assert not ordinary_leaked, (
                "ordinary environment dump value escaped supervisor"
            )
        benign_preserved = BENIGN in output
        assert benign_preserved, "supervisor lost the leading benign quoted message"


@pytest.mark.parametrize("boundary", ("safe_exec", "watch_run", "watch_capture"))
@pytest.mark.parametrize("shape", ("apostrophe", "mixed"))
def test_original_randy_set_poc(tmp_path, boundary, shape):
    key = (
        "rr5'ShellDumpSyntheticOnly"
        if shape == "apostrophe"
        else "rr5'\"ShellDumpSyntheticOnly"
    )
    env = minimal_env(tmp_path, (key,))
    env["API_KEY"] = key
    env["OPAQUE_SETTING"] = key
    command = "set; set >&2; exit 47"
    if boundary == "safe_exec":
        argv = [
            sys.executable,
            str(ROOT / "scripts/safe_exec.py"),
            "--command",
            "bash",
            "-c",
            command,
        ]
        result = subprocess.run(
            argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30
        )
    else:
        env["DUMP_COMMAND"] = command
        entry = "watch_run" if boundary == "watch_run" else "watch_capture json"
        result = watchdog_run(env, entry + ' bash -c "$DUMP_COMMAND"')
    assert result.returncode == 47, "original POC exit changed"
    disclosures = {
        "stdout": recoverable(result.stdout, (key,)),
        "stderr": recoverable(result.stderr, (key,)),
    }
    assert not any(disclosures.values()), "original shell POC disclosed a credential"


@pytest.mark.parametrize(
    "payload", (*ENCODERS, "docker-inspect", "randy-apostrophe", "randy-mixed")
)
def test_actual_evidence_dump_all_four_rank_files(
    tmp_path, credentials, encoded_batches, payload
):
    if payload.startswith("randy"):
        key = (
            "rr5'ShellEvidenceSyntheticOnly"
            if payload.endswith("apostrophe")
            else "rr5'\"ShellEvidenceSyntheticOnly"
        )
        credentials = (key,)
    fixture_env = watchdog_fixture(tmp_path, credentials[0])
    keep = {
        "PATH",
        "CONFIG",
        "WATCH_STATE_DIR",
        "WORKER_SSH",
        "WATCH_ALERT",
        "ALERT_CAPTURE",
        "HEAL_MARKER",
        "HEAL_EXEC",
        "WATCH_PROBE",
        "WATCH_HEAL",
        "WATCH_RESTART_ARGV",
        "GUARD_CONFIG",
    }
    env = {name: value for name, value in fixture_env.items() if name in keep}
    env.update(minimal_env(tmp_path, credentials))
    env["PATH"] = fixture_env["PATH"]
    if payload.startswith("randy"):
        env.update(API_KEY=credentials[0], OPAQUE_SETTING=credentials[0])
        tool = "#!/bin/bash\nset\nset >&2\nexit 47\n"
    elif payload in ENCODERS:
        assert payload in encoded_batches, "missing independently verified encoding"
        tool = "#!/bin/bash\n" + both_streams(payload) + "\n"
    else:
        # Real Docker-shaped inspect JSON contains ordinary values too. The
        # evidence projection must omit Config.Env irrespective of credential names.
        tool = (
            f"#!{sys.executable}\n"
            + """import json,os,sys
value={"State":{"Running":False,"ExitCode":47},"Config":{"Env":[
    k+"="+v for k,v in os.environ.items() if k.startswith("OPAQUE_")]}}
print(json.dumps(value, ensure_ascii=True))
print(json.dumps(value, ensure_ascii=True), file=sys.stderr)
sys.exit(47)
"""
        )
    for name in ("docker", "ssh"):
        (tmp_path / "bin" / name).write_text(tool)
    result = watchdog_run(env, "evidence_dump reason")
    assert result.returncode == 0, "actual evidence collection failed"
    expected = {"r0.log", "r0-state.json", "r1.log", "r1-state.json"}
    files = {
        path.name: path
        for path in (tmp_path / "state/heal-evidence").rglob("*")
        if path.is_file()
    }
    assert expected <= files.keys(), "actual rank evidence files missing"
    disclosures = {}
    ordinary_values = {}
    for name, path in files.items():
        text = path.read_text()
        path.unlink()
        if name in expected:
            disclosures[name] = recoverable(text, credentials)
            if payload in ("set", "docker-inspect") or payload.startswith("randy"):
                ordinary_values[name] = ORDINARY in text
            if payload in ENCODERS and name.endswith(".log"):
                benign_preserved = BENIGN in text
                assert benign_preserved, (
                    "rank log lost the leading benign quoted message"
                )
    assert not any(disclosures.values()), (
        "recoverable credential persisted in rank evidence"
    )
    assert not any(ordinary_values.values()), (
        "ordinary dump/Config.Env value persisted in evidence"
    )
    assert_private(result.stdout + result.stderr, credentials)
