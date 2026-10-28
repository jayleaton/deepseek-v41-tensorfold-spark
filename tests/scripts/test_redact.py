"""scripts/redact.py: what is masked in argv, env, text and JSON, and what stays readable."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from redact import (
    MASK,
    _filter,
    known_secrets,
    mask_values,
    redact,
    redact_argv,
    redact_env,
    redact_text,
    scrub,
    secret_name,
)

# built at runtime: no credential literal in the tree
KEY = "DUMMY" + "-k3y-" + "0123456789abcdef"


def test_secret_names():
    for name in (
        "--api-key",
        "API_KEY",
        "TENSORFOLD_API_KEY",
        "--hf-token",
        "x-api-key",
        "apiKey",
        "HF_TOKEN",
        "client_secret",
        "--password",
        "Authorization",
        "SPARK_BENCH_API_KEY",
        "--token",
    ):
        assert secret_name(name), name
    for name in (
        "--max-tokens",
        "--tokens",
        "--tokenizer",
        "TF_DSV41_POOL_TOKENS",
        "--model",
        "--api-key-file",
        "TOKEN_PATH",
        "--keyword",
        "monkey",
        "--decode-tokens",
    ):
        assert not secret_name(name), name


def test_argv_both_flag_forms_env_assignments_and_headers():
    argv = [
        "bench",
        "--api-key",
        KEY,
        "--api-key=" + KEY,
        "--model",
        "m",
        "--max-tokens",
        "5",
        "API_KEY=" + KEY,
        "-H",
        "Authorization: Bearer " + KEY,
        "--header=x-api-key: " + KEY,
        "--api-key-file",
        "/run/key",
        "--url",
        "https://user:" + KEY + "@host/v1",
        "--hf-token",
        "--quiet",
    ]
    out = redact_argv(argv)
    assert KEY not in " ".join(out)
    assert out[:5] == ["bench", "--api-key", MASK, "--api-key=" + MASK, "--model"]
    assert out[5:9] == ["m", "--max-tokens", "5", "API_KEY=" + MASK]
    # a pointer at a secret stays readable; a following flag is not taken as the value
    assert "--api-key-file" in out and "/run/key" in out
    assert out[-2:] == ["--hf-token", "--quiet"]


def test_env_and_json():
    env = redact_env(
        {
            "API_KEY": KEY,
            "TF_DSV41_POOL_TOKENS": "65536",
            "PATH": "/bin",
            "NOTE": "key is " + KEY,
        },
        known=[KEY],
    )
    assert env == {
        "API_KEY": MASK,
        "TF_DSV41_POOL_TOKENS": "65536",
        "PATH": "/bin",
        "NOTE": "key is " + MASK,
    }
    doc = redact(
        {
            "argv": ["--api-key", KEY],
            "api_key": KEY,
            "max_tokens": 5,
            "token": "Ġthe",
            "tokens": [1, 2],
            "nested": [{"headers": {"Authorization": "Bearer " + KEY}}],
        }
    )
    assert KEY not in repr(doc)
    assert doc["max_tokens"] == 5 and doc["token"] == "Ġthe" and doc["tokens"] == [1, 2]


def test_text_masks_credentials_keeps_placeholders():
    fake_sk = "sk-" + "a1" * 12
    text = (
        f'curl -H "x-api-key: {KEY}" --api-key={KEY} "api_key": "{KEY}" {fake_sk} HF_TOKEN={KEY}\n'
        'bos_token="<s>" --api-key "$API_KEY" password: <you> max_tokens=128 token count 42\n'
    )
    out = redact_text(text)
    assert KEY not in out and fake_sk not in out
    assert (
        'bos_token="<s>"' in out
        and '"$API_KEY"' in out
        and "<you>" in out
        and "max_tokens=128" in out
    )
    assert "token count 42" in out


def test_known_values_and_scrub(tmp_path):
    known = known_secrets({"API_KEY": KEY, "SHORT_TOKEN": "1", "MODEL": "x" * 20})
    assert known == [KEY]
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "report.json").write_text(
        '{"echo": "the tool said ' + KEY + '"}'
    )
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe" + KEY.encode())
    assert scrub(tmp_path, known) == 1
    assert KEY not in (tmp_path / "sub" / "report.json").read_text()


def test_cli_argv_and_text_filter():
    env = {"API_KEY": KEY, "PATH": "/usr/bin:/bin"}
    cmd = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "redact.py"),
            "argv",
            "--",
            "tool",
            "--api-key",
            KEY,
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout
    assert cmd.strip() == f"tool --api-key '{MASK}'"
    pem = "-----BEGIN " + "PRIVATE KEY-----\nMIIBVQ\n-----END " + "PRIVATE KEY-----\n"
    log = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "redact.py"), "text"],
        input="echo " + KEY + "\n" + pem,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout
    assert KEY not in log and "MIIBVQ" not in log


def test_mask_values_only_masks_given_values():
    text = '{"reply": "password: hunter22", "echo": "' + KEY + '"}'
    assert (
        mask_values(text, [KEY, ""])
        == '{"reply": "password: hunter22", "echo": "' + MASK + '"}'
    )


def render_known(value, rendering):
    if rendering == "raw":
        return value
    if rendering == "repr":
        return repr(value)[1:-1]
    if rendering == "ascii":
        return ascii(value)[1:-1]
    if rendering == "json":
        return json.dumps(value)[1:-1]
    if rendering == "unicode_json":
        return json.dumps(value, ensure_ascii=False)[1:-1]
    if rendering == "nested_json":
        return json.dumps(json.dumps(value)[1:-1])[1:-1]
    if rendering == "nested_repr":
        return repr(repr(value)[1:-1])[1:-1]
    return repr(value.encode("utf-8"))[2:-1]


@pytest.mark.parametrize("suffix", ["", "\n", "\r\n", "\u00e9", "\u2603", "\U0001f512"],
                         ids=["plain", "newline", "crlf", "latin1", "nonlatin1", "nonbmp"])
@pytest.mark.parametrize("rendering", [
    "raw", "repr", "ascii", "json", "unicode_json", "nested_json", "nested_repr", "bytes",
])
@pytest.mark.parametrize("trimmed", [False, True], ids=["original", "trimmed"])
def test_shared_known_secret_renderings(suffix, rendering, trimmed):
    key = KEY + '"quote\\' + suffix
    echoed = key.strip() if trimmed else key
    rendered = render_known(echoed, rendering)
    text = "tool diagnostic: " + rendered + " end"
    out = redact_text(text, known=[key])
    leaked = rendered in out
    masked = MASK in out
    assert not leaked, "known credential rendering survived redaction"
    assert masked, "known credential rendering was not masked"


def test_line_filter_handles_trimmed_and_escaped_copied_keys():
    key = KEY + '"quote\\\u2603\n'
    forms = [render_known(key, name) for name in ("repr", "json", "nested_json", "bytes")]
    forms.append(key.strip())
    out = io.StringIO()
    _filter(io.StringIO("\n".join("tool: " + v for v in forms) + "\n"), out, [key])
    leaked = any(v in out.getvalue() for v in forms)
    assert not leaked, "line filter disclosed a known credential"
    assert out.getvalue().count(MASK) == len(forms)


def test_cli_text_filter_expands_known_environment_key():
    key = KEY + '"quote\\\u2603\n'
    forms = [render_known(key, name) for name in ("repr", "json", "nested_json", "bytes")]
    forms.append(key.strip())
    env = dict(os.environ, API_KEY=key)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/redact.py"), "text"],
        input="\n".join("tool: " + v for v in forms) + "\n",
        capture_output=True, text=True, env=env, timeout=15,
    )
    leaked = any(v in result.stdout or v in result.stderr for v in forms)
    ok = result.returncode == 0
    assert ok, "CLI filter failed"
    assert not leaked, "CLI filter disclosed a known credential"
    assert result.stdout.count(MASK) == len(forms)


def test_structured_and_scrub_boundaries_expand_known_values(tmp_path):
    key = KEY + '"quote\\\u2603\n'
    escaped = render_known(key, "nested_json")
    doc = redact({"message": escaped}, known=[key])
    env = redact_env({"DIAGNOSTIC": escaped}, known=[key])
    argv = redact_argv(["tool", escaped], known=[key])
    masked = all(MASK in v for v in (doc["message"], env["DIAGNOSTIC"], argv[1]))
    assert masked, "structured boundaries did not use shared expansion"
    path = tmp_path / "diagnostic.txt"
    path.write_text("tool: " + escaped)
    changed = scrub(path, known=[key])
    assert changed == 1
    leaked = escaped in path.read_text()
    assert not leaked, "scrub disclosed a known credential"
