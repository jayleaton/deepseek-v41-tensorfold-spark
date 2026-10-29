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
Authorization / x-api-key / Cookie header values, Bearer tokens, URL passwords, PEM private-key markers, the known token
shapes (sk-, hf_, gh*_, github_pat_, xox*-, AKIA), and the literal value of every credential-named variable in the
environment, including short explicit keys, so a tool that echoes $API_KEY in its own words is still caught.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import sys
import warnings
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


_LITERAL = re.compile(r'''(?:[bBuUrR])?(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')''', re.DOTALL)
_DECODE_DEPTH = 6
_DECODE_LIMIT = 256


def _shell_records(text: str):
    """Yield physical lines and their newline status, independent of diagnostic syntax."""
    start = 0
    for match in re.finditer("\n", text):
        yield text[start:match.end()], True
        start = match.end()
    if start < len(text):
        yield text[start:], False


def iter_safe_records(text: str):
    """Iterate physical lines, without their terminating newline."""
    for record, terminated in _shell_records(text):
        yield record[:-1] if terminated else record


def _ansi_c(text: str, start: int) -> tuple[str, int]:
    """Decode one Bash ANSI-C string without executing substitutions or shell code."""
    out = bytearray()
    i = start
    escapes = {"a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f",
               "n": "\n", "r": "\r", "t": "\t", "v": "\v", "\\": "\\", "'": "'", '"': '"', "?": "?"}
    while i < len(text):
        char = text[i]
        i += 1
        if char == "'":
            return out.decode("utf-8", errors="surrogateescape"), i
        if char != "\\":
            out.extend(char.encode("utf-8", errors="surrogateescape"))
            continue
        if i == len(text):
            break
        char = text[i]
        i += 1
        if char in escapes:
            out.extend(escapes[char].encode())
        elif char in "01234567xuU":
            octal = char in "01234567"
            width = 3 if octal else {"x": 2, "u": 4, "U": 8}[char]
            begin = i - 1 if octal else i
            digits = re.match(r"[0-7]+" if octal else r"[0-9a-fA-F]+", text[begin:begin + width])
            if not digits:
                raise ValueError("invalid ANSI-C escape")
            i = begin + len(digits[0])
            value = int(digits[0], 8 if octal else 16)
            if octal or char == "x":
                out.append(value % 256)
            else:
                out.extend(chr(value).encode("utf-8"))
        elif char == "c":
            if i == len(text):
                raise ValueError("incomplete ANSI-C control escape")
            if not text[i].isascii():
                raise ValueError("non-ASCII ANSI-C control escape")
            out.append(ord(text[i].upper()) ^ 64)
            i += 1
        else:
            out.extend(("\\" + char).encode("utf-8", errors="surrogateescape"))
    raise ValueError("unterminated ANSI-C string")


def _shell_words(text: str, *, prose: bool = False) -> list[str]:
    """Use POSIX shlex concatenation, replacing ANSI-C fragments with opaque placeholders first."""
    parts, replacements = [], {}
    quote = ""
    i = 0
    marker = "__redact_fragment_"
    while marker in text:
        marker += "_"
    while i < len(text):
        char = text[i]
        if (prose and not quote and char == "'" and i > 0 and i + 1 < len(text)
                and text[i - 1].isalpha() and text[i + 1].isalpha()):
            parts.append("\\'")
            i += 1
            continue
        if not quote and text.startswith("$'", i):
            value, i = _ansi_c(text, i + 2)
            placeholder = marker + str(len(replacements)) + "__"
            replacements[placeholder] = value
            parts.append(placeholder)
            continue
        if quote != "'" and text.startswith("\\\n", i):
            i += 2
            continue
        parts.append(char)
        i += 1
        if char == "\\" and quote != "'" and i < len(text):
            parts.append(text[i])
            i += 1
        elif quote:
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
    words = shlex.split("".join(parts), comments=False, posix=True)
    if replacements:
        fragments = re.compile("|".join(re.escape(key) for key in replacements))
        words = [fragments.sub(lambda match: replacements[match[0]], word) for word in words]
    return words


def _decoded_secret(record: str, known: list[str]) -> bool:
    """Inspect bounded inert decoding layers, masking only a matching known value."""
    pending = [(record, 0)]
    seen = set()
    while pending:
        value, depth = pending.pop()
        if value in seen:
            continue
        seen.add(value)
        if any(secret in value for secret in known):
            # Raw occurrences retain the existing fine-grained masking behavior.
            if depth:
                return True
        if depth >= _DECODE_DEPTH or len(seen) >= _DECODE_LIMIT:
            continue
        if not any(char in value for char in "\"'\\"):
            continue
        try:
            words = _shell_words(value)
        except (ValueError, OverflowError, UnicodeError):
            try:
                # Prose punctuation must not prevent decoding a later escaped
                # value on the same line ("can't connect rr5\\ alpha").
                words = _shell_words(value, prose=True)
            except (ValueError, OverflowError, UnicodeError):
                words = []
        children = [word for word in words if word != value]
        literals = [value] + [match[0] for match in _LITERAL.finditer(value)]
        for literal in literals:
            for parse in (json.loads, ast.literal_eval):
                try:
                    # Parser warnings can echo untrusted source, bypassing the output boundary.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        parsed = parse(literal)
                except (ValueError, SyntaxError, TypeError, RecursionError):
                    continue
                # Only traverse strings and containers; numeric JSON protocol keys stay numeric.
                containers = [parsed]
                count = 0
                while containers:
                    item = containers.pop()
                    count += 1
                    if count > _DECODE_LIMIT:
                        break
                    if isinstance(item, str) and item != value:
                        children.append(item)
                    elif isinstance(item, bytes):
                        # A parsed byte diagnostic may use HTTP latin-1 or UTF-8.
                        # Inspect the decoded data, never another escaped-text spelling.
                        children.append(item.decode("utf-8", errors="replace"))
                        children.append(item.decode("latin-1"))
                    elif isinstance(item, dict):
                        containers.extend(item.keys())
                        containers.extend(item.values())
                    elif isinstance(item, (list, tuple, set)):
                        containers.extend(item)
        pending.extend((child, depth + 1) for child in children if child not in seen)
    return False


def _mask_decoded_records(text: str, known: list[str]) -> str:
    if not known:
        return text
    known = [value for secret in known for value in (secret, secret.strip()) if value]
    return "".join(
        MASK + ("\n" if record.endswith("\n") else "") if _decoded_secret(record, known) else record
        for record, _ in _shell_records(text)
    )


def split_safe_records(pending: str, known: Iterable[str]) -> tuple[str, str]:
    """Release every newline-terminated line; retain only an incomplete final line."""
    boundary = pending.rfind("\n") + 1
    return pending[:boundary], pending[boundary:]


def redact_text(text: str, known: Iterable[str] = (), *, decode: bool = True) -> str:
    """``text`` with every credential this module recognises replaced by MASK."""
    known = list(known)
    return ''.join(
        _redact_line(record[:-1] if terminated else record, known, decode)
        + ('\n' if terminated else '')
        for record, terminated in _shell_records(text)
    )


def _redact_line(text: str, known: list[str], decode: bool) -> str:
    """Apply literal, variant and inert decode checks within one physical line."""
    if decode:
        text = _mask_decoded_records(text, known)
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


def redact_argv(argv: Iterable[str], known: Iterable[str] = (), *, decode: bool = True) -> list[str]:
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
            out.append(redact_text(arg, known, decode=decode))
    return out


def check_argv(argv, secrets):
    variants = secret_variants(secrets)
    # Raw argv has not been shell-encoded. Output-only decoding must not mistake
    # a literal apostrophe in a path for malformed child diagnostics.
    if redact_argv(argv, secrets, decode=False) != argv or any(value in arg for value in variants for arg in argv):
        raise ValueError('credential value refused in command arguments')



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
    """Copy src to dst with independent redaction of each physical line."""

    def emit(line):
        dst.write(redact_text(line, known))
        dst.flush()

    pending = ""
    for line in src:
        ready, pending = split_safe_records(pending + line, known)
        for record in iter_safe_records(ready):
            emit(record + "\n")
    if pending:
        emit(pending)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    known = known_secrets()
    if args[:1] == ["argv"]:
        rest = args[2:] if args[1:2] == ["--"] else args[1:]
        from safe_quote import join_checked
        print(join_checked(redact_argv(rest, known)))
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
