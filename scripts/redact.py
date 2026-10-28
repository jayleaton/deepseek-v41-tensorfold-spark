#!/usr/bin/env python3
"""Mask credentials before a command line, environment, log or result is saved or printed (stdlib only).

Every tool that persists argv, env or a third-party tool's output goes through this one helper:

  from redact import redact_argv, redact_env, redact_text, redact    (scripts/ on sys.path)
  redact.py argv -- CMD ARG...    print the command shell-quoted with secret values masked (a saved .cmd file)
  redact.py text                  copy stdin to stdout, masked line by line (put it in front of `tee x.log`)
  redact.py scrub PATH...         mask every text file under PATH in place (a bench's own reports and dumps)

A value is masked when its flag or variable name has a part (split on - _ . and case) that names a credential: key,
token, secret, password, authorization, bearer, auth, credential, cookie (--api-key X, --api-key=X, API_KEY=X,
{"hf_token": X}); names ending in file / path / dir point at a secret and stay readable. Also masked anywhere in text:
Authorization / x-api-key / Cookie header values, Bearer tokens, URL passwords, PEM private keys, the known token
shapes (sk-, hf_, gh*_, github_pat_, xox*-, AKIA), and the literal value of every credential-named variable in the
environment, including short explicit keys, so a tool that echoes $API_KEY in its own words is still caught.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path

MASK = "<redacted>"
SECRET_WORDS = frozenset(
    [
        "key",
        "apikey",
        "token",
        "secret",
        "password",
        "passwd",
        "authorization",
        "auth",
        "bearer",
        "credential",
        "credentials",
        "cookie",
    ]
)
POINTER_WORDS = frozenset({"file", "path", "dir"})
# alone, too common in prose and data to mask in text (a tokenizer's "token")
GENERIC_WORDS = frozenset({"key", "token", "auth", "cookie"})

# Credential shapes, shared with scripts/check_secrets.py: (name, pattern).
TOKEN_PATTERNS = (
    ("OpenAI / Anthropic key", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    (
        "GitHub token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})"),
    ),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
)
PEM_BEGIN = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
PEM_BLOCK = re.compile(
    r"-----BEGIN ([A-Z0-9 ]*)PRIVATE KEY-----.*?(?:-----END \1PRIVATE KEY-----|\Z)",
    re.DOTALL,
)

_VALUE = r"""("[^"\n]*"|'[^'\n]*'|[^\s"'&,;]+)"""
_HEADER = re.compile(
    r"(?i)\b((?:proxy-)?authorization|x-api-key|api-key|x-auth-token|cookie)(\s*:[ \t]*)"
    r"(?!\s*" + re.escape(MASK) + ")([^\r\n\"'\\\\]+)"
)
_BEARER = re.compile(
    r"(?i)\b(bearer)(\s+)(?!" + re.escape(MASK) + r")([A-Za-z0-9._~+/=-]{6,})"
)
_FLAG = re.compile(r"(?<![\w-])(--?[A-Za-z][\w.-]*)(=|[ \t]+)" + _VALUE)
_ASSIGN = re.compile(
    r"""(?<![\w-])(["']?)([A-Za-z_][\w.-]*)\1([ \t]*[:=][ \t]*)""" + _VALUE
)
_URL_PASSWORD = re.compile(
    r"(\b[a-z][a-z0-9+.-]*://[^/\s:@]+:)([^/\s@]+)(@)", re.IGNORECASE
)


def secret_name(name: str) -> bool:
    """True when a flag, variable or field name names a credential (API_KEY, --hf-token, x-api-key, apiKey)."""

    if re.fullmatch(r"(?:bytes|bits)_(?:a|per)_token", name, re.IGNORECASE):
        return False
    parts = [
        p.lower()
        for p in re.split(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])", name.lstrip("-"))
        if p
    ]
    return (
        bool(parts)
        and parts[-1] not in POINTER_WORDS
        and any(p in SECRET_WORDS for p in parts)
    )


def known_secrets(environ: Mapping[str, str] | None = None) -> list[str]:
    """Configured credentials, including short keys; numeric token-control flags stay readable."""

    env = os.environ if environ is None else environ
    return sorted(
        {v for k, v in env.items() if v and secret_name(k)},
        key=len,
        reverse=True,
    )


def _masked(value: str) -> str:
    """MASK, keeping the quotes of a quoted value; a placeholder (<you>, $API_KEY, ${KEY}) stays as it is."""

    quoted = len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'"
    inner = value[1:-1] if quoted else value
    if inner[:1] in ("<", "$") or inner == MASK:
        return value
    return value[0] + MASK + value[0] if quoted else MASK


def _text_name(name: str) -> bool:
    """A credential name in free text: a bare key / token / auth is too common there (a tokenizer's "token")."""

    return secret_name(name) and name.lower().lstrip("-") not in GENERIC_WORDS


def mask_values(text: str, values: Iterable[str]) -> str:
    """``text`` with each given value (a key the caller holds) replaced by MASK, longest first; nothing else."""

    for value in sorted({v for v in values if v}, key=len, reverse=True):
        text = text.replace(value, MASK)
    return text


@lru_cache(maxsize=256)
def _secret_variants(value: str) -> tuple[str, ...]:
    """Known values as tools render them, including copied trailing whitespace and byte diagnostics."""
    variants = {v for v in (value, value.strip()) if v}
    # HTTP/encoding errors can show a bytes repr rather than the original Unicode.
    for v in tuple(variants):
        for encoding in ("utf-8", "latin-1"):
            try:
                variants.add(repr(v.encode(encoding))[2:-1])
            except UnicodeEncodeError:
                continue
    # Three layers cover repr/JSON in exceptions, JSON reports and nested tool output.
    for _ in range(3):
        for v in tuple(variants):
            variants.update((json.dumps(v)[1:-1], json.dumps(v, ensure_ascii=False)[1:-1],
                             repr(v)[1:-1], ascii(v)[1:-1]))
    return tuple(sorted(variants, key=len, reverse=True))


def secret_variants(values: Iterable[str]) -> list[str]:
    """Expand explicitly known credentials once for every text/CLI/report boundary."""
    return sorted({variant for value in values if value for variant in _secret_variants(value)},
                  key=len, reverse=True)


def split_safe_records(pending: str, known: Iterable[str]) -> tuple[str, str]:
    """Release complete lines, retaining suffixes that may continue a known multiline credential."""
    multiline = [value for value in secret_variants(known) if "\n" in value]
    # Mask complete multiline matches before splitting them into individual records.
    pending = mask_values(pending, multiline)
    hold = 0
    for value in multiline:
        for size in range(min(len(value) - 1, len(pending)), hold, -1):
            if value.startswith(pending[-size:]):
                hold = size
                break
    boundary = pending.rfind("\n", 0, len(pending) - hold) + 1
    return pending[:boundary], pending[boundary:]


def redact_text(text: str, known: Iterable[str] = ()) -> str:
    """``text`` with every credential this module recognises replaced by MASK."""

    text = mask_values(text, secret_variants(known))
    text = PEM_BLOCK.sub(MASK, text)
    for _, pattern in TOKEN_PATTERNS:
        text = pattern.sub(MASK, text)
    text = _URL_PASSWORD.sub(lambda m: m[1] + MASK + m[3], text)
    text = _HEADER.sub(lambda m: m[1] + m[2] + MASK, text)
    text = _BEARER.sub(lambda m: m[1] + m[2] + MASK, text)
    text = _FLAG.sub(
        lambda m: m[1] + m[2] + _masked(m[3]) if secret_name(m[1]) else m[0], text
    )
    return _ASSIGN.sub(
        lambda m: (
            m[1] + m[2] + m[1] + m[3] + _masked(m[4]) if _text_name(m[2]) else m[0]
        ),
        text,
    )


def redact_argv(argv: Iterable[str], known: Iterable[str] = ()) -> list[str]:
    """A command line with the values of credential flags (--api-key X and --api-key=X), credential env assignments
    (API_KEY=X cmd) and credential text inside any argument (-H "Authorization: Bearer X") masked."""

    known = list(known)
    out: list[str] = []
    hide_next = False
    for arg in map(str, argv):
        if hide_next and not arg.startswith("--"):
            out.append(MASK)
            hide_next = False
            continue
        hide_next = False
        name, eq, _ = arg.partition("=")
        if arg.startswith("-") and secret_name(name):
            out.append(name + "=" + MASK if eq else arg)
            hide_next = not eq
        elif eq and re.fullmatch(r"[A-Za-z_]\w*", name) and secret_name(name):
            out.append(name + "=" + MASK)
        else:
            out.append(redact_text(arg, known))
    return out


def redact_env(env: Mapping[str, str], known: Iterable[str] = ()) -> dict[str, str]:
    """An environment snapshot with credential-named variables masked and secrets inside other values masked."""

    known = list(known)
    return {
        k: MASK if secret_name(k) and v else redact_text(str(v), known)
        for k, v in env.items()
    }


def redact(obj, known: Iterable[str] = ()):
    """A JSON-like value with credential-named fields (not a bare "key" / "token") masked, string lists treated as
    argv, and strings as text."""

    known = list(known)
    if isinstance(obj, Mapping):
        return {
            redact_text(k, known) if isinstance(k, str) else k: MASK
            if isinstance(k, str)
            and _text_name(k)
            and isinstance(v, (str, int, float))
            and v != ""
            else redact(v, known)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        if obj and all(isinstance(x, str) for x in obj):
            return redact_argv(obj, known)
        return [redact(x, known) for x in obj]
    if isinstance(obj, str):
        return redact_text(obj, known)
    return obj


def scrub(path: Path, known: Iterable[str] = ()) -> int:
    """Mask every UTF-8 text file under ``path`` in place; the number of files changed."""

    known = list(known)
    files = (
        sorted(p for p in path.rglob("*") if p.is_file() and not p.is_symlink())
        if path.is_dir()
        else [path]
    )
    changed = 0
    for f in files:
        try:
            text = f.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        masked = redact_text(text, known)
        if masked != text:
            f.write_text(masked)
            changed += 1
    return changed


def _filter(src, dst, known: list[str]) -> None:
    """Copy src to dst line by line, masked; a PEM private key is masked from its BEGIN line to its END line."""

    in_key = False

    def emit(line):
        nonlocal in_key
        if in_key or PEM_BEGIN.search(line):
            in_key = "-----END" not in line
            dst.write(MASK + "\n")
        else:
            dst.write(redact_text(line, known))
        dst.flush()

    pending = ""
    for line in src:
        ready, pending = split_safe_records(pending + line, known)
        for record in ready.split("\n")[:-1]:
            emit(record + "\n")
    if pending:
        emit(pending)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    known = known_secrets()
    if args[:1] == ["argv"]:
        rest = args[2:] if args[1:2] == ["--"] else args[1:]
        print(shlex.join(redact_argv(rest, known)))
        return 0
    if args == ["text"]:
        _filter(sys.stdin, sys.stdout, known)
        return 0
    if args[:1] == ["scrub"] and len(args) > 1:
        n = sum(scrub(Path(p), known) for p in args[1:] if Path(p).exists())
        print(
            f"redact: masked credentials in {n} file(s)"
            if n
            else "redact: nothing to mask",
            file=sys.stderr,
        )
        return 0
    print(__doc__.split("\n\n")[1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
