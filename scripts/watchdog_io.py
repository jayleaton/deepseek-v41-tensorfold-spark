"""Typed watchdog control projection, shared by the sanitizing command boundary."""

import datetime
import json
import math
import re

from safe_output import clean
from redact import known_secrets, check_argv


def filter_output(mode, raw, known=()):
    known = list(known) + known_secrets()
    value = raw.strip()
    if mode == "names":
        names = raw.splitlines()
        if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in names):
            raise ValueError("invalid environment names")
        return "\n".join(names) + ("\n" if names else "")
    if mode == "state":
        # Evidence receives only operational State fields, never Config.Env or
        # arbitrary diagnostic text returned by a failed Docker/SSH command.
        try:
            state = json.loads(raw)
        except (ValueError, TypeError):
            state = None
        if not isinstance(state, dict):
            return json.dumps({"diagnostic": "<redacted>", "unavailable": True}) + "\n"
        allowed = {}
        for name in ('Running', 'Paused', 'Restarting', 'OOMKilled', 'Dead'):
            if type(state.get(name)) is bool:
                allowed[name] = state[name]
        for name in ('Pid', 'ExitCode'):
            if type(state.get(name)) is int:
                allowed[name] = state[name]
        status = state.get('Status')
        if (isinstance(status, str) and status in {'created', 'running', 'paused', 'restarting', 'exited', 'removing', 'dead'}
                and clean(status, known) == status):
            allowed['Status'] = status
        for name in ('StartedAt', 'FinishedAt'):
            value = state.get(name)
            if isinstance(value, str) and re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})', value) and clean(value, known) == value:
                allowed[name] = value
        return json.dumps(allowed or {"diagnostic": "<redacted>", "unavailable": True}) + "\n"
    if mode == "revision":
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError("invalid engine revision")
        return value + "\n"
    if mode == "image_id":
        if re.fullmatch(r"sha256:[A-Za-z0-9]+", value) is None:
            raise ValueError("invalid image identity")
        return value + "\n"
    if mode == "path":
        # Used only for mktemp-created private scratch; reject control characters.
        if not value.startswith("/") or any(ord(c) < 32 for c in value):
            raise ValueError("invalid temporary directory")
        check_argv([value], known)
        return value + "\n"

    def number(v):
        return type(v) is int or (type(v) is float and math.isfinite(v))

    def project(text):
        try:
            d = json.loads(text)
        except (ValueError, TypeError):
            d = {}
        if not isinstance(d, dict):
            d = {}
        streams = d.get("streams")
        rounds = d.get("rounds")
        maximum = streams.get("max") if isinstance(streams, dict) else None
        age = rounds.get("last_age_s") if isinstance(rounds, dict) else None
        reason = d.get("stall_reason")
        # Only canonical names, typed values and recognized enums cross the control
        # boundary. In particular, never redact protocol field names or enum strings.
        control = {
            "ok": d.get("ok") if type(d.get("ok")) is bool else None,
            "errors": d.get("errors") if type(d.get("errors")) is int else None,
            "fatal": bool(d.get("fatal")),
            "status": "stalled" if d.get("status") == "stalled" else None,
            "stall_reason": "no round for 0 s with work in flight"
            if isinstance(reason, str)
            and re.fullmatch(r"no round for [0-9]+ s with work in flight", reason)
            else None,
            "oldest_s": d.get("oldest_s") if number(d.get("oldest_s")) else None,
            "inflight": d.get("inflight") if type(d.get("inflight")) is int else None,
            "requests_running": d.get("requests_running")
            if type(d.get("requests_running")) is int
            else None,
            "streams": {"max": maximum if number(maximum) else None},
            "rounds": {"last_age_s": age if number(age) else None},
            "choices": [] if isinstance(d.get("choices"), list) else None,
            "code": "watchdog_deferred"
            if d.get("code") == "watchdog_deferred"
            else None,
        }
        # Derive all controls above from raw JSON before sanitizing diagnostics.
        # Opaque/unknown strings exist only in this sanitized diagnostic payload.
        diagnostic = clean(
            d.get("diagnostic") if isinstance(d.get("diagnostic"), str) else text, known
        )
        return {"diagnostic": diagnostic, **control}

    def predicate(mode, d):
        if mode == "healthy":
            return d["ok"] is True and d["errors"] == 0 and not d["fatal"]
        if mode == "stalled":
            return d["status"] == "stalled"
        if mode == "choices":
            return isinstance(d["choices"], list)
        if mode == "deferred":
            return d["code"] == "watchdog_deferred"
        if mode == "busy":
            running = d["requests_running"]
            maximum = d["streams"]["max"]
            return (
                running is not None
                and maximum is not None
                and maximum > 0
                and running >= maximum
            )
        if mode == "stale":
            oldest = d["oldest_s"]
            age = d["rounds"]["last_age_s"]
            active = d["inflight"]
            return (
                d["ok"] is False
                and d["errors"] == 0
                and not d["fatal"]
                and d["status"] == "stalled"
                and d["stall_reason"] is not None
                and oldest is not None
                and 0 <= oldest <= 3
                and age is not None
                and age > 120
                and active is not None
                and active > 0
            )
        return False

    if mode == "uint":
        if re.fullmatch(r"[0-9]{1,18}", value) is None:
            raise ValueError("invalid watchdog control")
        return str(int(value)) + "\n"
    elif mode == "running":
        if value not in ("true", "false", "absent", "unreachable"):
            raise ValueError("invalid watchdog control")
        return value + "\n"
    elif mode == "started":
        if (
            re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})",
                value,
            )
            is None
        ):
            raise ValueError("invalid watchdog control")
        return (
            str(
                int(
                    datetime.datetime.fromisoformat(
                        value.replace("Z", "+00:00")
                    ).timestamp()
                )
            )
            + "\n"
        )
    elif mode == "stamp":
        if re.fullmatch(r"[0-9]{8}-[0-9]{6}", value) is None:
            raise ValueError("invalid watchdog control")
        return (
            datetime.datetime.strptime(value, "%Y%m%d-%H%M%S").strftime("%Y%m%d-%H%M%S")
            + "\n"
        )
    elif mode == "health_result":
        if not raw:
            return ""
        code, sep, body = raw.partition("\n")
        if not sep or re.fullmatch(r"[0-9]{3}", code) is None:
            raise ValueError("invalid startup health framing")
        return code + "\n" + json.dumps(project(body), allow_nan=False) + "\n"
    elif mode in ("http", "http_control"):
        body, sep, code = raw.rstrip("\n").rpartition("\n")
        if not sep or re.fullmatch(r"[0-9]{3}", code) is None:
            body, code = raw, "000"
        # Separate curl framing before sanitizing the JSON body structurally. clean()
        # masks opaque string echoes, including multiline keys, but preserves numbers.
        sanitized = (
            json.dumps(project(body), allow_nan=False)
            if mode == "http_control"
            else clean(body, known)
        )
        return code + "\n" + sanitized + "\n"
    elif mode == "project":
        return json.dumps(project(raw), allow_nan=False) + "\n"
    elif mode in ("healthy", "stale", "busy", "stalled", "choices", "deferred"):
        return str(int(predicate(mode, project(raw)))) + "\n"
    elif mode == "json":
        return clean(raw, known) + "\n"
    else:
        raise ValueError("invalid watchdog control")
