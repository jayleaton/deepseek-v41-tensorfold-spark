"""Credential-safe terminal and diagnostic-file boundaries for host tooling."""

from __future__ import annotations

import builtins
from contextlib import redirect_stderr, redirect_stdout
import json
import os
import re
from pathlib import Path
import shlex
import sys

from redact import (
    MASK,
    PEM_BEGIN,
    known_secrets,
    _shell_records,
    redact,
    redact_text,
    redact_argv,
    _text_name,
    secret_name,
    split_safe_records,
)

_configured_secrets = []


def _known():
    return _configured_secrets + known_secrets()


def _environment_name(text):
    # Shell set/env/export/declare diagnostics differ in quoting, but their
    # variable record prefix is a structural name, never a value to publish.
    match = re.match(r"^(?:(?:declare|typeset|export)(?:\s+-[A-Za-z-]+)*\s+)?([A-Za-z_][A-Za-z0-9_]*)=", text)
    return match.group(1) if match else None


def _environment_record(name, secrets):
    return MASK + " environment variable: " + redact_text(name, secrets)


def _clean_text(text, secrets):
    records = []
    in_environment = False
    for record, terminated in _shell_records(text):
        line = record[:-1] if terminated else record
        ending = "\n" if terminated else ""
        name = _environment_name(line)
        if name is not None:
            in_environment = True
            records.append(_environment_record(name, secrets) + ending)
        elif in_environment:
            # A dump can contain unquoted multiline values or function bodies.
            # Once recognized, withhold remaining non-name records fail closed.
            records.append(MASK + ending)
        else:
            records.append(redact_text(line, secrets) + ending)
    return ''.join(records)


def _environment_names(value, secrets):
    if isinstance(value, dict):
        names = value.keys()
    elif isinstance(value, list):
        names = (item.partition('=')[0] for item in value if isinstance(item, str))
    else:
        return []
    return [redact_text(name, secrets) for name in names
            if isinstance(name, str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name)]


def _value(value, secrets):
    # Recognize structural fields before redacting their names. A configured
    # value equal to "Env" must not hide the field from environment projection.
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if isinstance(key, str) and key.lower() in {'env', 'environment'}:
                result['EnvNames'] = _environment_names(child, secrets)
                continue
            name = redact_text(key, secrets) if isinstance(key, str) else key
            sensitive = (isinstance(key, str) and _text_name(key)
                         and isinstance(child, (str, int, float)) and child != "")
            result[name] = MASK if sensitive else _value(child, secrets)
        return result
    if isinstance(value, (list, tuple)):
        # Preserve credential flag/value pairing in saved command displays.
        if value and all(isinstance(item, str) for item in value):
            return [_clean_text(item, secrets) for item in redact_argv(value, secrets)]
        return [_value(item, secrets) for item in value]
    if isinstance(value, str):
        return _clean_text(value, secrets)
    return redact(value, secrets)


def clean(text, known=()):
    text = str(text)
    secrets = list(known) + _known()
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        value = None
    if isinstance(value, (dict, list)):
        return json.dumps(_value(value, secrets), ensure_ascii=True)
    return _clean_text(text, secrets)


def safe_print(*args, **kwargs):
    # Sanitize the complete record, including any escaped JSON, before writing.
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    builtins.print(
        clean((" " if sep is None else sep).join(str(arg) for arg in args)),
        end=_clean_text("\n" if end is None else end, _known()),
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
        self.in_environment = False

    def write(self, text):
        ready, self.pending = split_safe_records(
            self.pending + text, self.secrets + _known()
        )
        for record, terminated in _shell_records(ready):
            self._record(record[:-1] if terminated else record, "\n" if terminated else "")
        return len(text)

    def _record(self, text, end):
        name = _environment_name(text)
        if name is not None:
            self.in_environment = True
            self.output.write(_environment_record(name, self.secrets + _known()) + end)
        elif self.in_environment:
            self.output.write(MASK + end)
        elif self.in_key or PEM_BEGIN.search(text):
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
    return _clean_text(text, secrets)


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
