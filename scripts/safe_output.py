"""Credential-safe terminal and diagnostic-file boundaries for host tooling."""

from __future__ import annotations

import builtins
from contextlib import redirect_stderr, redirect_stdout
import json
import os
import re
from pathlib import Path
import shlex
import stat
import sys

from redact import (
    MASK,
    known_secrets,
    redact,
    redact_argv,
    redact_lines,
    redact_text,
    _text_name,
    secret_name,
    split_lines,
)

_configured_secrets = []


def _known():
    return _configured_secrets + known_secrets()


def _value(value, secrets):
    # Mask a complete JSON value field by field, so redaction cannot break its syntax.
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if isinstance(key, str):
                name = redact_text(key, secrets)
            elif key is None or isinstance(key, (bool, int, float)):
                name = key
            else:
                name = redact_text(repr(key), secrets)
            sensitive = (isinstance(key, str) and _text_name(key)
                         and isinstance(child, (str, int, float)) and child != "")
            result[name] = MASK if sensitive else _value(child, secrets)
        return result
    if isinstance(value, (list, tuple)):
        # Preserve credential flag/value pairing in saved command displays.
        if value and all(isinstance(item, str) for item in value):
            return [_value(item, secrets) for item in redact_argv(value, secrets)]
        return [_value(item, secrets) for item in value]
    if isinstance(value, str):
        # A parsed string is a complete value, including embedded newlines.
        return redact_text(value, secrets)
    if value is None or isinstance(value, (bool, int, float)):
        return redact(value, secrets)
    return redact_text(repr(value), secrets)


def sanitize(value, known=()):
    """A JSON-like value masked field by field, ready to serialize; numbers stay numbers."""
    return _value(value, list(known) + _known())


def clean(text, known=()):
    text = str(text)
    secrets = list(known) + _known()
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


def clean_document(text, known=()):
    """A complete JSON document masked field by field; any other document line by line.

    Only complete documents (a file, an HTTP body) are parsed. Live streams never
    are, so no line waits for, or is masked because of, another line.
    """
    text = str(text)
    secrets = list(known) + _known()
    try:
        value = json.loads(text)
    except (ValueError, TypeError, RecursionError):
        value = None
    if not isinstance(value, (dict, list)):
        return redact_text(text, secrets)
    # Mask values before serialization so redaction cannot invalidate JSON.
    indentation = re.search(r'\n([ \t]+)\S', text)
    sanitized = json.dumps(_value(value, secrets), ensure_ascii=True,
                           indent=indentation[1] if indentation else None)
    return sanitized + ('\n' if text.endswith('\n') else '')


def write_text(path, text, **kwargs):
    return Path(path).write_text(clean_document(text), **kwargs)


class _Stream:
    """Line-based output backstop: each complete line is redacted and released at once.

    Only an unterminated tail waits, for its terminator (\\n or \\r), LINE_LIMIT or finish().
    No line ever waits for, or is masked because of, any other line.
    """

    LINE_LIMIT = 64 * 1024

    def __init__(self, output, secrets):
        self.output, self.secrets, self.pending = output, secrets, ""

    def write(self, text):
        ready, self.pending = split_lines(self.pending + text)
        if len(self.pending) >= self.LINE_LIMIT:
            ready, self.pending = ready + self.pending, ""
        if ready:
            self.output.write(redact_lines(ready, self.secrets + _known()))
        return len(text)

    def flush(self):
        self.output.flush()

    def finish(self):
        if self.pending:
            self.output.write(redact_lines(self.pending, self.secrets + _known()))
            self.pending = ""
        self.output.flush()

    def write_sanitized(self, text):
        if self.pending:
            self.write(text)
        else:
            self.output.write(text)

    def isatty(self):
        return False


AUTH_FILE_LIMIT = 1 << 20


def auth_file_keys(path):
    """Serving keys in the engine's key-file grammar, so the backstop can mask them.

    Lines are trimmed; empty lines and # comments are skipped; ``label[scopes]: key``
    yields the key. Unreadable or unsafe files yield nothing: the launcher, not the
    backstop, enforces the file's safety. The keys stay in this process's memory.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except (OSError, TypeError, ValueError):
        return []
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > AUTH_FILE_LIMIT:
            return []
        raw = os.read(fd, AUTH_FILE_LIMIT + 1)
    except OSError:
        return []
    finally:
        os.close(fd)
    keys = []
    for line in raw.decode("utf-8", errors="replace").replace("\r", "\n").split("\n"):
        line = line.strip(" \t\x0b\x0c\x1c\x1d\x1e\x1f")
        if not line or line.startswith("#"):
            continue
        keys.append(line.partition(":")[2].strip(" \t") if ":" in line else line)
    return [key for key in keys if key]


def _cli_secrets(argv):
    values = known_secrets()
    auth_files = [os.environ.get("NATIVE_AUTH_FILE", "")]
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
                    elif equals and name == "NATIVE_AUTH_FILE":
                        auth_files.append(value)
        except (OSError, ValueError):
            pass  # Configuration errors are handled inside the sanitized boundary.
    for path in dict.fromkeys(auth_files):
        if path:
            values.extend(auth_file_keys(path))
    return [value for value in values if value]


def inherited_known():
    """Consume the private registry passed only to reviewed supervisor entries."""
    number = os.environ.pop('TF_SAFE_REGISTRY_FD', None)
    if number is None:
        return []
    fd = int(number)
    if fd < 3:
        raise ValueError('invalid private registry descriptor')
    with os.fdopen(fd, 'rb') as source:
        raw = source.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError('private registry exceeds limit')
    return [value.decode('utf-8') for value in raw.split(b'\0') if value]


def cli(main, error_code=1, known=()):
    """Cover parsing, setup, worker diagnostics and unhandled errors before any output."""
    global _configured_secrets
    secrets = list(known) + inherited_known() + _cli_secrets(sys.argv[1:])
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
