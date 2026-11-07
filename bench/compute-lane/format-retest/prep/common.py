"""Strict manifest bytes and externally pinned prospective specification."""
import hashlib
import json
from pathlib import Path

SPEC_SHA256 = "b47ede697eed3a38d244ad5ff7447690732bd12945ff4592554d45bc2d8527ec"
FAMILIES = ("integer", "modular", "division", "dp", "graph", "aggregation")
PLAIN_FAMILIES = ("dp", "graph", "aggregation")
OLD_INTEGER = "Final answer format: a JSON integer."
OLD = {family: OLD_INTEGER for family in FAMILIES}
OLD.update(division="Final answer format: the JSON array [q,r].",
           aggregation="Final answer format: a JSON object with keys A through F and integer values.")
NEW = {family: "Final answer: only the integer, written as a bare JSON number (for example -42): no object, no key, no quotes, no code fence." for family in FAMILIES}
NEW.update(division="Final answer: only the JSON array [q,r] itself, with nothing around it and no code fence.",
           aggregation="Final answer: only the JSON object with keys A through F and integer values, on its own line, with no code fence.")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)


def pinned(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("explicit SHA256 required")
    data = Path(path).read_bytes()
    if sha(data) != expected:
        raise ValueError("SHA256 mismatch: " + str(path))
    return data


def spec_gate(path, expected):
    if expected != SPEC_SHA256:
        raise ValueError("external FORMAT spec SHA differs from frozen specification")
    return pinned(path, expected)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_tasks(data):
    tasks = [strict_json(line) for line in data.decode("utf-8").splitlines()]
    if not tasks:
        raise ValueError("empty task manifest")
    seen = set()
    slots = set()
    for task in tasks:
        required = {"id", "suite", "family", "slot", "attempt", "generator_label", "inputs", "prompt"}
        if type(task) is not dict or set(task) != required:
            raise ValueError("malformed task fields")
        family, slot, attempt = task["family"], task["slot"], task["attempt"]
        if family not in FAMILIES or type(slot) is not int or not 0 <= slot < 10 or type(attempt) is not int or not 0 <= attempt < 1000:
            raise ValueError("malformed family/slot/attempt")
        if task["suite"] != "compute_dl" or task["id"] != f"compute-format-v1-{family}-{slot + 1:03d}":
            raise ValueError("malformed task identity")
        if task["generator_label"] != f"compute-format-v1|20261107|{family}|{slot}|{attempt}":
            raise ValueError("generation label mismatch")
        if type(task["inputs"]) is not dict or not task["inputs"] or not isinstance(task["prompt"], str) or not task["prompt"].endswith(OLD[family]):
            raise ValueError("malformed inputs/old prompt")
        if task["id"] in seen or (family, slot) in slots:
            raise ValueError("duplicate task identity/slot")
        seen.add(task["id"])
        slots.add((family, slot))
    if slots != {(family, slot) for family in FAMILIES for slot in range(10)}:
        raise ValueError("exactly sixty tasks, ten per family required")
    return tasks


def create(path, data, mode=0o600):
    import os
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
