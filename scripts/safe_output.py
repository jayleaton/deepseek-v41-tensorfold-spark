"""Credential-safe terminal and diagnostic-file boundaries for host tooling."""

from __future__ import annotations

import ast
import builtins
from contextlib import redirect_stderr, redirect_stdout
import json
import os
import re
from pathlib import Path
import shlex
import sys
import warnings

from redact import (
    MASK,
    known_secrets,
    mask_values,
    _shell_records,
    redact,
    redact_text,
    redact_argv,
    _text_name,
    secret_name,
    secret_variants,
    split_safe_records,
    _decoded_secret,
)

_configured_secrets = []


def _known():
    return _configured_secrets + known_secrets()


def _environment_name(text, secrets=()):
    # Shell set/env/export/declare diagnostics differ in quoting, but their
    # variable record prefix is a structural name, never a value to publish.
    match = re.match(r"^(?:(?:declare|typeset|export)(?:\s+-[A-Za-z-]+)*\s+)?([A-Za-z_][A-Za-z0-9_]*)=", text)
    if not match:
        return None
    name = match[1]
    # An assignment alone can be a progress metric. Recognize actual inherited
    # names, explicit shell declarations and encoded known credentials instead.
    declared = not text.startswith(name + "=")
    if (declared or name in os.environ or secret_name(name)
            or MASK in text or any(secret and secret in text[match.end():] for secret in secrets)
            or _decoded_secret(text, list(secrets))):
        return name
    return None


def _environment_record(name, secrets):
    return MASK + " environment variable: " + redact_text(name, secrets)


def _clean_text(text, secrets):
    records = []
    for record, terminated in _shell_records(text):
        line = record[:-1] if terminated else record
        ending = "\n" if terminated else ""
        name = _environment_name(line, secrets)
        if name is not None:
            records.append(_environment_record(name, secrets) + ending)
        else:
            value = _container(line)
            if isinstance(value, (dict, list)):
                records.append(json.dumps(_value(value, secrets), ensure_ascii=True) + ending)
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


def _container(text):
    """Parse JSON or Python containers inertly, including pprint's adjacent strings."""
    if not text.lstrip().startswith(("{", "[")):
        return None
    for parse in (json.loads, ast.literal_eval):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = parse(text)
        except (ValueError, SyntaxError, TypeError, RecursionError):
            continue
        if isinstance(value, (dict, list)):
            return value
    return None


def _environment_mapping(value, secrets):
    """A bare environment mapping needs corroborating names or known values."""
    if not value or not all(isinstance(k, str) and re.fullmatch(r'[A-Za-z_]\w*', k)
                            and isinstance(v, str) for k, v in value.items()):
        return False
    return any((k in os.environ and os.environ[k] == v)
               or (k.upper() == k and (secret_name(k)
                   or any(secret and secret in v for secret in secrets)))
               for k, v in value.items())


def _value(value, secrets):
    # Recognize structural fields before redacting their names. A configured
    # value equal to "Env" must not hide the field from environment projection.
    if isinstance(value, dict):
        if _environment_mapping(value, secrets):
            return {'EnvNames': _environment_names(value, secrets)}
        result = {}
        for key, child in value.items():
            if isinstance(key, str) and key.lower() in {'env', 'environment'}:
                result['EnvNames'] = _environment_names(child, secrets)
                continue
            # Python diagnostics permit tuple/bytes keys, whereas JSON permits
            # scalar keys only. Preserve their sanitized spelling as a string.
            if isinstance(key, str):
                name = redact_text(mask_values(key, secret_variants(secrets)), secrets)
            elif key is None or isinstance(key, (bool, int, float)):
                name = key
            else:
                name = redact_text(repr(key), secrets)
            sensitive = (isinstance(key, str) and _text_name(key)
                         and isinstance(child, (str, int, float)) and child != "")
            result[name] = MASK if sensitive else _value(child, secrets)
        return result
    if isinstance(value, (set, frozenset)):
        return [_value(item, secrets) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        # Preserve credential flag/value pairing in saved command displays.
        if value and all(isinstance(item, str) for item in value):
            return [_value(item, secrets) for item in redact_argv(value, secrets)]
        return [_value(item, secrets) for item in value]
    if isinstance(value, str):
        # A parsed string is a complete value, including embedded newlines.
        return _clean_text(mask_values(value, secret_variants(secrets)), secrets)
    if value is None or isinstance(value, (bool, int, float)):
        return redact(value, secrets)
    # literal_eval also accepts bytes, complex numbers and Ellipsis. A safe
    # representation keeps these diagnostics JSON-serializable without another
    # container parse, which would recurse on the same unsupported object.
    return redact_text(repr(value), secrets)


def clean(text, known=()):
    text = str(text)
    secrets = list(known) + _known()
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
    """Sanitize a complete JSON document structurally; other text uses independent lines."""
    text = str(text)
    try:
        value = json.loads(text)
    except (ValueError, TypeError, RecursionError):
        value = None
    if isinstance(value, (dict, list)):
        # A complete file has a structural boundary, unlike a live child stream.
        # Mask values before serialization so redaction cannot invalidate JSON.
        indentation = re.search(r'\n([ \t]+)\S', text)
        sanitized = json.dumps(_value(value, _known()), ensure_ascii=True,
                               indent=indentation[1] if indentation else None)
        if text.endswith('\n'):
            sanitized += '\n'
    else:
        sanitized = clean(text)
    return Path(path).write_text(sanitized, **kwargs)


class _Stream:
    def __init__(self, output, secrets):
        self.output, self.secrets, self.pending = output, secrets, ""

    def write(self, text):
        ready, self.pending = split_safe_records(
            self.pending + text, self.secrets + _known()
        )
        for record, terminated in _shell_records(ready):
            self._record(record[:-1] if terminated else record, "\n" if terminated else "")
        return len(text)

    def _record(self, text, end):
        self.output.write(_clean_text(text, self.secrets + _known()) + end)

    def flush(self):
        # Hold partial records so a credential split over writes is never exposed.
        self.output.flush()

    def finish(self):
        if self.pending:
            self._record(self.pending, "")
            self.pending = ""
        self.output.flush()

    def write_sanitized(self, text):
        if self.pending:
            self.write(text)
        else:
            self.output.write(text)

    def isatty(self):
        return False


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
