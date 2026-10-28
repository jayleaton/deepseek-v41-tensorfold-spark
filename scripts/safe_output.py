"""Credential-safe terminal and diagnostic-file boundaries for host tooling."""

from __future__ import annotations

import builtins
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import shlex
import sys

from redact import (
    MASK,
    PEM_BEGIN,
    known_secrets,
    redact,
    redact_text,
    secret_name,
    split_safe_records,
)

_configured_secrets = []


def _known():
    return _configured_secrets + known_secrets()


def _value(value, secrets):
    # Sanitize strings and object keys before serialization so short/numeric keys
    # cannot turn a numeric measurement into a string or invalid JSON.
    value = redact(value, secrets)
    if isinstance(value, dict):
        return {
            redact_text(k, secrets) if isinstance(k, str) else k: _value(v, secrets)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_value(v, secrets) for v in value]
    return value


def clean(text, known=()):
    text = str(text)
    secrets = list(known) + _known()
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        value = None
    if isinstance(value, (dict, list)):
        return json.dumps(_value(value, secrets), ensure_ascii=True)
    return redact_text(text, secrets)


def safe_print(*args, **kwargs):
    # Sanitize the complete record, including any escaped JSON, before writing.
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    builtins.print(
        clean((" " if sep is None else sep).join(str(arg) for arg in args)),
        end=redact_text("\n" if end is None else end, _known()),
        **kwargs,
    )


def dump(value, stream, **kwargs):
    text = json.dumps(_value(value, _known()), **kwargs)
    if isinstance(stream, _Stream):
        # Already sanitized structurally: another pass over pretty-printed lines
        # would mistake digits in numeric fields for short credential strings.
        stream.write_sanitized(text)
    else:
        stream.write(text)


def write_text(path, text, **kwargs):
    return Path(path).write_text(clean(text), **kwargs)


class _Stream:
    def __init__(self, output, secrets):
        self.output, self.secrets, self.pending = output, secrets, ""
        self.in_key = False

    def write(self, text):
        ready, self.pending = split_safe_records(
            self.pending + text, self.secrets + _known()
        )
        for line in ready.split("\n")[:-1]:
            self._record(line, "\n")
        return len(text)

    def _record(self, text, end):
        if self.in_key or PEM_BEGIN.search(text):
            self.in_key = "-----END" not in text
            self.output.write(MASK + end)
        else:
            self.output.write(_clean_record(text, self.secrets) + end)

    def flush(self):
        # Hold partial records so a credential split over writes is never exposed.
        self.output.flush()

    def finish(self):
        if self.pending:
            self._record(self.pending, "")
            self.pending = ""
        self.output.flush()

    def write_sanitized(self, text):
        if self.pending or self.in_key:
            self.write(text)
        else:
            self.output.write(text)

    def isatty(self):
        return False


def _clean_record(text, secrets):
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        value = None
    secrets = secrets + _known()
    if isinstance(value, (dict, list)):
        return json.dumps(_value(value, secrets), ensure_ascii=True)
    return redact_text(text, secrets)


def _cli_secrets(argv):
    values = known_secrets()
    config = os.environ.get("CONFIG") or str(
        Path(__file__).resolve().parents[1] / "config/prod-zig.env"
    )
    for i, arg in enumerate(argv):
        name, _, value = arg.partition("=")
        if name.startswith("--") and secret_name(name):
            values.append(
                value if "=" in arg else (argv[i + 1] if i + 1 < len(argv) else "")
            )
        if name == "--config":
            config = value if "=" in arg else (argv[i + 1] if i + 1 < len(argv) else "")
    if config:
        try:
            text = Path(config).read_text()
            if text.lstrip().startswith("{"):

                def visit(value):
                    if isinstance(value, dict):
                        for key, child in value.items():
                            if secret_name(key) and isinstance(child, str):
                                values.append(child)
                            else:
                                visit(child)
                    elif isinstance(value, list):
                        for child in value:
                            visit(child)

                visit(json.loads(text))
            else:
                # Parse quotes/comments and multiline values without evaluating shell code.
                tokens = shlex.shlex(text, posix=True, punctuation_chars=";&|()")
                tokens.whitespace_split = True
                for token in tokens:
                    name, equals, value = token.partition("=")
                    if equals and secret_name(name):
                        values.append(value)
        except (OSError, ValueError):
            pass  # Configuration errors are handled inside the sanitized boundary.
    return [value for value in values if value]


def cli(main, error_code=1):
    """Cover parsing, setup, worker diagnostics and unhandled errors before any output."""
    global _configured_secrets
    secrets = _cli_secrets(sys.argv[1:])
    previous, _configured_secrets = _configured_secrets, secrets
    out, err = _Stream(sys.stdout, secrets), _Stream(sys.stderr, secrets)
    try:
        with redirect_stdout(out), redirect_stderr(err):
            try:
                result = main()
                return 0 if result is None else result
            except SystemExit as error:
                if error.code is None or isinstance(error.code, int):
                    return error.code or 0
                safe_print(error.code, file=sys.stderr)
                return 1
            except Exception as error:
                safe_print(f"{type(error).__name__}: {error}", file=sys.stderr)
                return error_code
    finally:
        try:
            out.finish()
            err.finish()
        finally:
            _configured_secrets = previous
