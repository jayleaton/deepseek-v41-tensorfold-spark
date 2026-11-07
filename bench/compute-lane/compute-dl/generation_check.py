"""Handwritten FAKE_NOT_COLLECTION controls; never creates study units.

Run ONLY in the parent's authorized node CPU enclosure, before sealing.
run_checks() returns a JSON-compatible report, including every pass/failure.
main() prints that report and exits nonzero on failure. No generation imports,
random draws, study inputs, or model requests occur here. These checks do not
replace separate A/B CLI runs on the actual pinned public manifest.
"""

from copy import deepcopy
from decimal import Inexact, Rounded
import hashlib
import json
from pathlib import Path

import reference_a
import reference_b

MARKER = "FAKE_NOT_COLLECTION"
FAMILIES = ("integer", "modular", "division", "dp", "graph", "aggregation")


def _typed_answer(family, value):
    if family in ("integer", "modular", "dp", "graph"):
        return type(value) is int
    if family == "division":
        return type(value) is list and len(value) == 2 and all(type(v) is int for v in value)
    if family == "aggregation":
        return (type(value) is dict and set(value) == set("ABCDEF")
                and all(type(v) is int for v in value.values()))
    return False


def verify_pair(record, answer_a, answer_b):
    """Compare exact typed reference outputs; failure MUST stop preparation.

    Does not regenerate, choose an attempt, or check exclusion sets. The parent
    may use this helper after running the two reference CLIs separately.
    """
    family = record.get("family")
    if not _typed_answer(family, answer_a) or not _typed_answer(family, answer_b):
        raise ValueError("malformed reference answer")
    if answer_a != answer_b:
        raise ValueError("reference disagreement; stop preparation")
    return answer_a


def _examples():
    # Literal, independently hand-worked values; never call the generator.
    return [
        ("integer", {"a": 12, "b": 34, "c": -5}, 403),
        ("modular", {"a": 7, "e": 5, "m": 12}, 7),
        ("division", {"a": 12345, "b": 67}, [184, 17]),
        ("dp", {"rows": 2, "cols": 3, "matrix": [[1, 9, 1], [2, 3, 4]]}, 10),
        ("graph", {"n": 4, "edges": [[0, 1, 5], [0, 2, 3], [1, 2, 2],
                                     [1, 3, 1], [2, 3, 4]]}, 6),
        ("aggregation", {"count": 7, "records": [["A", 2, 3], ["B", 3, -2],
                         ["C", 1, 0], ["D", 2, 4], ["E", 1, 9],
                         ["F", 4, -1], ["A", 1, -1]]},
         {"A": 5, "B": -6, "C": 0, "D": 8, "E": 9, "F": -4}),
    ]


def _wrong(value):
    if type(value) is int:
        return value + 1
    result = deepcopy(value)
    if type(result) is list:
        result[1] += 1
    else:
        result["A"] += 1
    return result


def _valid_mutation(record):
    changed = deepcopy(record)
    x = changed["inputs"]
    family = changed["family"]
    if family == "integer":
        x["c"] += 1
    elif family == "modular":
        x["a"] = 2  # 2**5 % 12 == 8, differs from 7
    elif family == "division":
        x["a"] += 1
    elif family == "dp":
        x["matrix"][0][0] += 1  # Every path includes this endpoint.
    elif family == "graph":
        x["edges"][0][2] += 1  # Shortest path remains 0->1->3, cost 7.
    else:
        x["records"][0][2] += 1
    return changed


def _invalid_mutation(record):
    changed = deepcopy(record)
    x = changed["inputs"]
    family = changed["family"]
    if family == "integer":
        x["a"] = True
    elif family == "modular":
        x["m"] = 0
    elif family == "division":
        x["b"] = 0
    elif family == "dp":
        x["matrix"][1].pop()
    elif family == "graph":
        x["edges"].append(deepcopy(x["edges"][0]))
    else:
        x["records"][0][0] = "Z"
    return changed


def _must_reject(operation):
    try:
        operation()
    except (ValueError, ArithmeticError, TypeError):
        return
    raise AssertionError("deliberately invalid control was accepted")


def run_checks():
    """Return all fake control results; never claim actual study validation."""
    results = []

    def check(name, action):
        try:
            action()
        except Exception as error:
            results.append({"name": name, "status": "FAIL",
                            "error_type": type(error).__name__, "error": str(error)})
        else:
            results.append({"name": name, "status": "PASS"})

    for family, inputs, expected in _examples():
        record = {"id": f"{MARKER}-{family}", "family": family,
                  "suite": MARKER, "inputs": inputs}
        for name, solver in (("A", reference_a.solve), ("B", reference_b.solve)):
            def known(solver=solver, record=record, expected=expected):
                verify_pair(record, solver(deepcopy(record)), expected)
            check(f"{family}/{name}/known", known)

            def changed(solver=solver, record=record, expected=expected):
                mutated = _valid_mutation(record)
                actual = solver(mutated)
                # The mutation is valid and must change the result; old answers
                # must be rejected, rather than silently accepted as fixtures.
                if not _typed_answer(record["family"], actual) or actual == expected:
                    raise AssertionError("valid input mutation did not change typed answer")
                _must_reject(lambda: verify_pair(mutated, actual, expected))
            check(f"{family}/{name}/mutated-input-stale-answer", changed)
            check(f"{family}/{name}/malformed-input",
                  lambda solver=solver, record=record:
                  _must_reject(lambda: solver(_invalid_mutation(record))))
        check(f"{family}/wrong-answer",
              lambda record=record, expected=expected:
              _must_reject(lambda: verify_pair(record, expected, _wrong(expected))))
        bad_type = True if type(expected) is int else (
            [184.0, 17] if type(expected) is list else dict(expected, A=True))
        check(f"{family}/wrong-type",
              lambda record=record, expected=expected, bad_type=bad_type:
              _must_reject(lambda: verify_pair(record, expected, bad_type)))

    def decimal_trap():
        # Deliberately outside generated sizes: exact result needs >256 digits.
        record = {"family": "integer", "inputs": {"a": 10**256 + 1, "b": 3, "c": 1}}
        try:
            reference_b.solve(record)
        except (Inexact, Rounded):
            return
        raise AssertionError("Decimal precision loss did not trigger the required traps")
    check("B/Decimal256-rounding-trap", decimal_trap)

    return {"marker": MARKER, "actual_study_units": 0,
            "status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
            "source_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                              for name, path in (("generation_check", __file__),
                                                 ("reference_a", reference_a.__file__),
                                                 ("reference_b", reference_b.__file__))},
            "checks": results}


def main():
    report = run_checks()
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return report


if __name__ == "__main__":
    raise SystemExit(0 if main()["status"] == "PASS" else 1)
