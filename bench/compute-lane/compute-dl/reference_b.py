"""Independent reference B for the frozen compute-dl protocol.

``solve(record)`` consumes a public record's ``family`` and typed ``inputs`` and
returns only its exact JSON-compatible answer. It never calls reference A or
loads another answer file. Small inputs are supported for handwritten controls;
generation-size minima are not imposed here. Semantic and resource bounds are.

CLI: ``python reference_b.py PUBLIC.jsonl PRIVATE.json [--receipt RECEIPT]``.
The private JSON object maps task IDs to answers. Both it and its receipt are
created exclusively with mode 0600. Run this CLI only in the parent's authorized
node CPU enclosure. Importing this module performs no computations or writes.
"""

import argparse
from decimal import Context, Decimal, Inexact, Rounded, localcontext
import hashlib
from itertools import groupby
import json
from math import comb
import os
from pathlib import Path
import platform


FAMILIES = ("integer", "modular", "division", "dp", "graph", "aggregation")
SPEC_SHA256 = "c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141"
LABELS = tuple("ABCDEF")


def _integer(value, name, lower=None, upper=None):
    if type(value) is not int:
        raise ValueError(name + " must be a JSON integer")
    if lower is not None and value < lower:
        raise ValueError(name + " is below its lower bound")
    if upper is not None and value > upper:
        raise ValueError(name + " exceeds its upper bound")
    return value


def _fields(inputs, expected):
    if type(inputs) is not dict or set(inputs) != set(expected):
        raise ValueError("input fields do not match the family schema")


def _decimal_context():
    context = Context(prec=256)
    context.traps[Inexact] = True
    context.traps[Rounded] = True
    return context


def _product_sum(inputs):
    _fields(inputs, ("a", "b", "c"))
    a = _integer(inputs["a"], "a", 1)
    b = _integer(inputs["b"], "b", 1)
    c = _integer(inputs["c"], "c")
    with localcontext(_decimal_context()):
        result = Decimal(a) * Decimal(b) + Decimal(c)
        if result != result.to_integral_exact():
            raise ArithmeticError("nonintegral Decimal product sum")
        return int(result)


def _residue(inputs):
    _fields(inputs, ("a", "e", "m"))
    a = _integer(inputs["a"], "a", 1)
    exponent = _integer(inputs["e"], "e", 0, 5000)
    modulus = _integer(inputs["m"], "m", 1)
    result = 1
    for _ in range(exponent):
        result = (result * a) % modulus
    # Exponent zero is permitted only for small controls, including modulus 1.
    if exponent == 0:
        result %= modulus
    if not 0 <= result < modulus:
        raise ArithmeticError("modular result outside the canonical range")
    return result


def _quotient_remainder(inputs):
    _fields(inputs, ("a", "b"))
    a = _integer(inputs["a"], "a", 1)
    b = _integer(inputs["b"], "b", 1)
    with localcontext(_decimal_context()):
        dividend, divisor = Decimal(a), Decimal(b)
        quotient = dividend // divisor
        remainder = dividend % divisor
        if quotient != quotient.to_integral_exact():
            raise ArithmeticError("nonintegral Decimal quotient")
        if remainder != remainder.to_integral_exact():
            raise ArithmeticError("nonintegral Decimal remainder")
        if quotient * divisor + remainder != dividend:
            raise ArithmeticError("Decimal division identity failed")
        q, r = int(quotient), int(remainder)
    if a != q * b + r or not 0 <= r < b:
        raise ArithmeticError("independent division identity/range check failed")
    return [q, r]


def _path_minimum(inputs):
    _fields(inputs, ("rows", "cols", "matrix"))
    declared_rows = _integer(inputs["rows"], "rows", 1, 10)
    declared_cols = _integer(inputs["cols"], "cols", 1, 10)
    grid = inputs["matrix"]
    if type(grid) is not list or not 1 <= len(grid) <= 10:
        raise ValueError("matrix must have 1..10 rows")
    if type(grid[0]) is not list or not 1 <= len(grid[0]) <= 10:
        raise ValueError("matrix must have 1..10 columns")
    height, width = len(grid), len(grid[0])
    if height != declared_rows or width != declared_cols:
        raise ValueError("matrix dimensions do not match rows/cols")
    for row in grid:
        if type(row) is not list or len(row) != width:
            raise ValueError("matrix must be rectangular")
        for value in row:
            _integer(value, "cell", 0, 99)
    best = None
    completed = 0

    def enumerate_paths(row, column, preceding_sum):
        nonlocal best, completed
        current_sum = preceding_sum + grid[row][column]
        if row == height - 1 and column == width - 1:
            completed += 1
            if best is None or current_sum < best:
                best = current_sum
            return
        # No memoization, pruning, rolling rows, or shared intermediate tables:
        # every complete right/down path is independently visited and summed.
        if column + 1 < width:
            enumerate_paths(row, column + 1, current_sum)
        if row + 1 < height:
            enumerate_paths(row + 1, column, current_sum)

    enumerate_paths(0, 0, 0)
    expected = comb(height + width - 2, height - 1)
    if completed != expected or completed > 48620 or best is None:
        raise ArithmeticError("full path enumeration invariant failed")
    return best


def _shortest_distance(inputs):
    _fields(inputs, ("n", "edges"))
    n = _integer(inputs["n"], "n", 2, 14)
    edges = inputs["edges"]
    if type(edges) is not list:
        raise ValueError("edges must be an array")
    seen = set()
    distance = [[None for _ in range(n)] for _ in range(n)]
    for vertex in range(n):
        distance[vertex][vertex] = 0
    for edge in edges:
        if type(edge) is not list or len(edge) != 3:
            raise ValueError("each edge must be a three-element array")
        source = _integer(edge[0], "edge source", 0, n - 1)
        target = _integer(edge[1], "edge target", 0, n - 1)
        weight = _integer(edge[2], "edge weight", 1, 40)
        if source == target or (source, target) in seen:
            raise ValueError("self or duplicate directed edge")
        seen.add((source, target))
        distance[source][target] = weight
    if any((vertex, vertex + 1) not in seen for vertex in range(n - 1)):
        raise ValueError("graph backbone edge missing")
    # Floyd-Warshall, with no heap, Dijkstra helper, or external intermediates.
    for via in range(n):
        for source in range(n):
            left = distance[source][via]
            if left is None:
                continue
            for target in range(n):
                right = distance[via][target]
                if right is None:
                    continue
                candidate = left + right
                previous = distance[source][target]
                if previous is None or candidate < previous:
                    distance[source][target] = candidate
    result = distance[0][n - 1]
    if result is None:
        raise ValueError("target is unreachable")
    if any(distance[v][v] != 0 for v in range(n)):
        raise ArithmeticError("shortest-distance diagonal invariant failed")
    return result


def _category_totals(inputs):
    _fields(inputs, ("count", "records"))
    count = _integer(inputs["count"], "count", 6, 60)
    records = inputs["records"]
    if type(records) is not list or not 6 <= len(records) <= 60:
        raise ValueError("records must contain 6..60 transactions")
    if count != len(records):
        raise ValueError("record count does not match count")
    present = set()
    for transaction in records:
        if type(transaction) is not list or len(transaction) != 3:
            raise ValueError("transaction must be a three-element array")
        label, quantity, amount = transaction
        if type(label) is not str or label not in LABELS:
            raise ValueError("unknown category")
        present.add(label)
        _integer(quantity, "quantity", 1, 20)
        _integer(amount, "unit amount", -999, 1999)
    if present != set(LABELS):
        raise ValueError("every category A..F must occur")
    ordered = sorted(records, key=lambda transaction: transaction[0])
    result = {}
    for category, group in groupby(ordered, key=lambda transaction: transaction[0]):
        subtotal = 0
        for transaction in group:
            subtotal += transaction[1] * transaction[2]
        result[category] = subtotal
    return result


def solve(record):
    """Return reference B's exact typed answer; malformed input raises.

    This API does not generate, select, replace, or compare study candidates.
    Call it separately from reference A using the same public manifest bytes.
    """
    if type(record) is not dict:
        raise ValueError("record must be an object")
    family = record.get("family")
    if type(family) is not str or family not in FAMILIES:
        raise ValueError("unknown family")
    inputs = record.get("inputs")
    algorithms = {
        "integer": _product_sum,
        "modular": _residue,
        "division": _quotient_remainder,
        "dp": _path_minimum,
        "graph": _shortest_distance,
        "aggregation": _category_totals,
    }
    return algorithms[family](inputs)


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _not_integer_json(_value):
    raise ValueError("noninteger numeric JSON is forbidden")


def _bytes(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _private_create(path, payload):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    receipt_path = args.receipt or Path(str(args.output) + ".receipt.json")
    paths = [args.input.resolve(), args.output.resolve(), receipt_path.resolve()]
    if len(set(paths)) != 3:
        parser.error("input, output, and receipt must be distinct paths")
    if args.output.exists() or receipt_path.exists():
        parser.error("output and receipt must not already exist")
    source = Path(__file__).read_bytes()
    raw = args.input.read_bytes()
    receipt = {
        "reference": "B",
        "spec_sha256": SPEC_SHA256,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "python_version": platform.python_version(),
        "input_path": str(args.input.resolve()),
        "output_path": str(args.output.resolve()),
    }
    try:
        answers = {}
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                raise ValueError("blank JSONL line at " + str(line_number))
            record = json.loads(line, object_pairs_hook=_no_duplicates,
                                parse_float=_not_integer_json,
                                parse_constant=_not_integer_json)
            if type(record) is not dict:
                raise ValueError("JSONL record must be an object")
            task_id = record.get("id")
            if type(task_id) is not str or not task_id:
                raise ValueError("record ID must be a nonempty string")
            if task_id in answers:
                raise ValueError("duplicate task ID")
            answers[task_id] = solve(record)
        if not answers:
            raise ValueError("empty public manifest")
        payload = _bytes(answers)
        receipt.update(status="PASS", count=len(answers),
                       output_sha256=hashlib.sha256(payload).hexdigest())
        _private_create(args.output, payload)
    except Exception as error:
        receipt.update(status="FAIL", error_type=type(error).__name__,
                       error=str(error))
        _private_create(receipt_path, _bytes(receipt))
        raise
    _private_create(receipt_path, _bytes(receipt))
    return receipt


if __name__ == "__main__":
    main()
