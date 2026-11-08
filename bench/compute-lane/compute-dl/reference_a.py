"""Independent reference A. Run only on the authorized bounded node CPU.

solve(public_record) -> exact typed answer. The CLI reads public JSONL and writes
a private JSON object keyed by ID plus a source/input/output SHA256 receipt.
Malformed inputs, invariant failures and duplicate IDs stop; never regenerate.
Small handcrafted examples are supported; production size checks belong to the
preparation owner. No imports of reference B or generation are used.
"""

import argparse
import hashlib
import heapq
import json
import os
import platform
from pathlib import Path

SPEC_SHA256 = "c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141"


def _int(value, name, lo=None, hi=None):
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer, not bool/float")
    if lo is not None and value < lo or hi is not None and value > hi:
        raise ValueError(f"{name} out of range")
    return value


def _fields(inputs, names):
    if type(inputs) is not dict or set(inputs) != set(names):
        raise ValueError("unexpected input fields")


def solve(record):
    """Validate typed inputs and compute using A's specified algorithms."""
    if type(record) is not dict:
        raise ValueError("record must be an object")
    family = record.get("family")
    x = record.get("inputs")
    if family == "integer":
        _fields(x, ("a", "b", "c"))
        a = _int(x["a"], "a", 1)
        b = _int(x["b"], "b", 1)
        c = _int(x["c"], "c")
        return a * b + c
    if family == "modular":
        _fields(x, ("a", "e", "m"))
        a = _int(x["a"], "a", 1)
        e = _int(x["e"], "e", 0, 5000)
        m = _int(x["m"], "m", 1)
        answer = pow(a, e, m)
        if not 0 <= answer < m:
            raise ValueError("modular range invariant")
        return answer
    if family == "division":
        _fields(x, ("a", "b"))
        a = _int(x["a"], "a", 1)
        b = _int(x["b"], "b", 1)
        q, r = divmod(a, b)
        if a != q * b + r or not 0 <= r < b:
            raise ValueError("division identity/range invariant")
        return [q, r]
    if family == "dp":
        _fields(x, ("rows", "cols", "matrix"))
        rows = _int(x["rows"], "rows", 1, 10)
        cols = _int(x["cols"], "cols", 1, 10)
        matrix = x["matrix"]
        if type(matrix) is not list or len(matrix) != rows:
            raise ValueError("invalid grid rows")
        if type(matrix[0]) is not list or len(matrix[0]) != cols:
            raise ValueError("invalid grid columns")
        for row in matrix:
            if type(row) is not list or len(row) != cols:
                raise ValueError("grid must be rectangular")
            for cell in row:
                _int(cell, "cell", 0, 99)
        previous = [None] * cols
        for i, row in enumerate(matrix):
            for j, cell in enumerate(row):
                if i == j == 0:
                    previous[j] = cell
                elif i == 0:
                    previous[j] = previous[j - 1] + cell
                elif j == 0:
                    previous[j] += cell
                else:
                    previous[j] = min(previous[j], previous[j - 1]) + cell
        return previous[-1]
    if family == "graph":
        _fields(x, ("n", "edges"))
        n = _int(x["n"], "n", 2, 14)
        if type(x["edges"]) is not list:
            raise ValueError("edges must be a list")
        adj = [[] for _ in range(n)]
        seen = set()
        for edge in x["edges"]:
            if type(edge) is not list or len(edge) != 3:
                raise ValueError("edge must be [u,v,w]")
            u = _int(edge[0], "u", 0, n - 1)
            v = _int(edge[1], "v", 0, n - 1)
            w = _int(edge[2], "w", 1, 40)
            if u == v or (u, v) in seen:
                raise ValueError("self or duplicate edge")
            seen.add((u, v))
            adj[u].append((v, w))
        if any((i, i + 1) not in seen for i in range(n - 1)):
            raise ValueError("missing backbone edge")
        distances = [None] * n
        distances[0] = 0
        heap = [(0, 0)]
        while heap:
            distance, u = heapq.heappop(heap)
            if distance != distances[u]:
                continue
            for v, weight in adj[u]:
                candidate = distance + weight
                if distances[v] is None or candidate < distances[v]:
                    distances[v] = candidate
                    heapq.heappush(heap, (candidate, v))
        if distances[-1] is None:
            raise ValueError("unreachable target")
        return distances[-1]
    if family == "aggregation":
        _fields(x, ("count", "records"))
        count = _int(x["count"], "count", 6, 60)
        rows = x["records"]
        if type(rows) is not list or len(rows) != count:
            raise ValueError("invalid record count")
        totals = dict.fromkeys("ABCDEF", 0)
        labels = set()
        for row in rows:
            if type(row) is not list or len(row) != 3:
                raise ValueError("transaction must be [label,quantity,amount]")
            label, quantity, amount = row
            if type(label) is not str or label not in totals:
                raise ValueError("invalid category")
            _int(quantity, "quantity", 1, 20)
            _int(amount, "amount", -999, 1999)
            labels.add(label)
            totals[label] += quantity * amount
        if labels != set(totals):
            raise ValueError("missing category")
        return totals
    raise ValueError("unknown family")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _write_private(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="public task JSONL")
    parser.add_argument("output", help="new private answer JSON object by ID")
    parser.add_argument("--receipt", help="new private receipt; default OUTPUT.receipt.json")
    args = parser.parse_args(argv)
    receipt_path = args.receipt or args.output + ".receipt.json"
    paths = [Path(path).resolve() for path in (args.input, args.output, receipt_path)]
    if len(set(paths)) != 3:
        parser.error("input, output and receipt must be distinct")
    if Path(args.output).exists() or Path(receipt_path).exists():
        parser.error("output and receipt must not already exist")
    raw = Path(args.input).read_bytes()
    receipt = {"reference": "A", "input_sha256": hashlib.sha256(raw).hexdigest(),
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "spec_sha256": SPEC_SHA256, "python_version": platform.python_version(),
               "input_path": str(paths[0]), "output_path": str(paths[1])}
    try:
        answers = {}
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                raise ValueError("blank task record")
            record = json.loads(line, object_pairs_hook=_unique, parse_constant=_nonfinite,
                                parse_float=_nonfinite)
            if type(record) is not dict or type(record.get("id")) is not str or not record["id"]:
                raise ValueError("missing task ID")
            if record["id"] in answers:
                raise ValueError("duplicate task ID")
            answers[record["id"]] = solve(record)
        if not answers:
            raise ValueError("empty input manifest")
        output = (json.dumps(answers, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
        receipt.update(status="PASS", count=len(answers),
                       output_sha256=hashlib.sha256(output).hexdigest())
        _write_private(args.output, output)
    except Exception as error:
        receipt.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        _write_private(receipt_path, (json.dumps(receipt, sort_keys=True) + "\n").encode())
        raise
    _write_private(receipt_path, (json.dumps(receipt, sort_keys=True) + "\n").encode())
    return receipt


if __name__ == "__main__":
    main()
