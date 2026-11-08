"""Prospectively pinned FORMAT factory. Importing never generates a candidate."""
import hashlib
import json
from pathlib import Path
import platform
import random

from common import FAMILIES, OLD, encoded, pinned, read_tasks, sha, spec_gate, strict_json
from legacy import Legacy

MASTER_SEED = 20261107
REQUIRED_COVERAGE = frozenset(("W5", "W6", "W7", "W8", "W9", "census", "guide", "trace",
                               "pilot-eleven", "AB3", "guide-S3", "GSM8K", "MATH", "HumanEval",
                               "stdbench", "declaration-control", "compute-dl-v1"))


def verify_pins(*, spec_path, spec_sha256, python_version, source_pins):
    spec_gate(spec_path, spec_sha256)
    if python_version != platform.python_version():
        raise ValueError("Python differs from external pre-generation pin")
    # Pin all prep sources, not just the factory; the parent records these BEFORE use.
    root = Path(__file__).resolve().parent
    expected_names = {path.name for path in root.glob("*.py")}
    if type(source_pins) is not dict or set(source_pins) != expected_names:
        raise ValueError("pre-generation pins must cover every prep Python source")
    for name, digest in source_pins.items():
        pinned(root / name, digest)


def _candidate(family, slot, attempt):
    """Private: caller must pass verify_pins and exclusion gates first.

    Distribution and random draw order copied from compute-dl/generate_tasks.py.
    Only label, slot bounds and ID differ. Never call directly to bypass gates.
    """
    label = f"compute-format-v1|20261107|{family}|{slot}|{attempt}"
    seed = int.from_bytes(hashlib.sha256(label.encode("ascii")).digest(), "big")
    r = random.Random(seed)

    def digits(d):
        return r.randrange(10 ** (d - 1), 10 ** d)

    def embedded(value):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)

    if family == "integer":
        a = digits(r.randint(24, 36))
        b = digits(r.randint(24, 36))
        c = r.randrange(-10**20, 10**20 + 1)
        inputs = {"a": a, "b": b, "c": c}
        body = f"Compute A*B+C exactly for A={a}, B={b}, C={c}. "
    elif family == "modular":
        a = digits(r.randint(14, 20))
        e = r.randint(500, 5000)
        m = digits(r.randint(16, 22))
        inputs = {"a": a, "e": e, "m": m}
        body = (f"Find the least nonnegative residue of A^E modulo M, where A={a}, E={e}, M={m}. "
                "Here ^ denotes exponentiation. ")
    elif family == "division":
        a = digits(80)
        b = digits(29)
        inputs = {"a": a, "b": b}
        body = f"For A={a} and B={b}, find integers q and r such that A=q*B+r and 0<=r<B. "
    elif family == "dp":
        rows = r.randint(7, 10)
        cols = r.randint(7, 10)
        matrix = [[r.randint(0, 99) for _ in range(cols)] for _ in range(rows)]
        inputs = {"rows": rows, "cols": cols, "matrix": matrix}
        body = ("A path starts at the top-left cell of this grid and ends at the bottom-right cell. "
                "Each move is one cell right or one cell down. The path cost is the sum of all visited cells, "
                "including both endpoints. What is the minimum cost? "
                f"Grid: {embedded(matrix)}. ")
    elif family == "graph":
        n = r.randint(9, 14)
        edges = [[i, i + 1, r.randint(1, 40)] for i in range(n - 1)]
        pairs = [(u, v) for u in range(n) for v in range(n) if u != v and v != u + 1]
        r.shuffle(pairs)
        k = r.randint(n, 2 * n)
        edges.extend([u, v, r.randint(1, 40)] for u, v in pairs[:k])
        edges.sort(key=lambda edge: (edge[0], edge[1]))
        inputs = {"n": n, "edges": edges}
        body = (f"A directed graph has vertices 0 through {n - 1}. Each edge [u,v,w] goes from u to v with cost w. "
                f"What is the minimum total cost from vertex 0 to vertex {n - 1}? Edges: {embedded(edges)}. ")
    elif family == "aggregation":
        count = r.randint(40, 60)
        labels = "ABCDEF"
        records = []
        for i in range(count):
            category = labels[i] if i < 6 else labels[r.randrange(6)]
            quantity = r.randint(1, 20)
            amount = r.randint(-999, 1999)
            records.append([category, quantity, amount])
        r.shuffle(records)
        inputs = {"count": count, "records": records}
        body = ("Each transaction is [category,quantity,unit_amount]. Unit amounts are signed integer cents. "
                "For each category A through F, compute the sum of quantity*unit_amount over its transactions. "
                f"Transactions: {embedded(records)}. ")
    else:
        raise ValueError("unknown family")
    return {"id": f"compute-format-v1-{family}-{slot + 1:03d}", "family": family,
            "suite": "compute_dl", "slot": slot, "attempt": attempt,
            "generator_label": label, "inputs": inputs, "prompt": body + OLD[family]}


def inventory(config, legacy):
    """Read only explicitly inventoried authoritative files; no source discovery.

    A parent-signed hash pins the complete inventory, not a claim inferred here.
    Every required source ID must have a nonempty pinned original manifest.
    """
    raw = pinned(config["inventory_path"], config["inventory_sha256"])
    index = strict_json(raw)
    if set(index) != {"complete", "required_source_ids", "sources", "resolutions"} or index["complete"] is not True:
        raise ValueError("complete authoritative exclusion inventory required")
    required = index["required_source_ids"]
    if type(required) is not list or not required or any(type(x) is not str or not x for x in required) or len(set(required)) != len(required):
        raise ValueError("nonempty unique required source IDs required")
    if "compute-dl-v1" not in required:
        raise ValueError("all 120 compute-DL v1 instances mandatory")
    sources = index["sources"]
    if type(sources) is not list or any(type(x) is not dict or set(x) != {"id", "path", "sha256", "count", "coverage"} for x in sources):
        raise ValueError("malformed exclusion source inventory")
    if len(sources) != len(required) or {x["id"] for x in sources} != set(required):
        raise ValueError("missing/duplicate exclusion source")
    coverage = set()
    for source in sources:
        tags = source["coverage"]
        if type(tags) is not list or not tags or any(type(tag) is not str for tag in tags) or len(set(tags)) != len(tags):
            raise ValueError("nonempty unique source coverage tags required")
        coverage.update(tags)
    if not REQUIRED_COVERAGE <= coverage:
        raise ValueError("missing historical source coverage: " + str(sorted(REQUIRED_COVERAGE - coverage)))
    records, audit, by_id = [], [], {}
    for source in sources:
        data = pinned(source["path"], source["sha256"])
        rows = strict_json(data) if data.lstrip().startswith(b"[") else [strict_json(line) for line in data.splitlines()]
        if type(rows) is not list or not rows or type(source["count"]) is not int or len(rows) != source["count"]:
            raise ValueError("empty/count-mismatched historical manifest")
        if source["id"] == "compute-dl-v1":
            if len(rows) != 120 or {r.get("id") for r in rows} != {f"compute-dl-v1-{f}-{n:03d}" for f in FAMILIES for n in range(1, 21)} or any("inputs" not in r for r in rows):
                raise ValueError("incomplete compute-DL v1 typed inputs")
        local_ids = set()
        for original in rows:
            if type(original) is not dict or type(original.get("id")) is not str or not original["id"] or not isinstance(original.get("prompt"), str):
                raise ValueError("historical original requires ID/prompt")
            key = original["id"]
            if key in local_ids:
                raise ValueError("duplicate key within historical source")
            local_ids.add(key)
            # Dedup reused descendants only when mathematical and prompt evidence agrees.
            evidence = {k: original[k] for k in ("id", "prompt", "family", "inputs", "spec") if k in original}
            if key in by_id:
                if encoded(by_id[key]) != encoded(evidence):
                    raise ValueError("historical ID reused with different evidence")
                continue
            by_id[key] = evidence
            records.append(dict(original, source=source["id"], source_path=source["path"], source_sha256=source["sha256"]))
        audit.append(source)
    checker = legacy.load("disjointness").CollisionIndex(records, index["resolutions"])
    return checker, {"inventory_sha256": sha(raw), "sources": audit, "unique_records": len(records)}


def generate(config, retain_collision):
    """Return tasks/A/B/audit only after pins. Callback must durably retain collisions.

    Reference or inventory errors propagate immediately; only documented collision
    authorizes a new attempt. No response/model/performance inputs are accepted.
    """
    verify_pins(**{k: config[k] for k in ("spec_path", "spec_sha256", "python_version", "source_pins")})
    legacy = Legacy()
    if type(config["legacy_source_pins"]) is not dict:
        raise ValueError("legacy source pins required")
    from legacy import PATHS
    for name in ("reference_a", "reference_b", "disjointness"):
        pinned(PATHS[name], config["legacy_source_pins"][name])
    checker, audit = inventory(config, legacy)
    a, b = legacy.load("reference_a"), legacy.load("reference_b")
    tasks, answers_a, answers_b = [], {}, {}
    for family in FAMILIES:
        for slot in range(10):
            for attempt in range(1000):
                task = _candidate(family, slot, attempt)
                collision = checker.check(task)
                if collision:
                    retain_collision({"candidate": task, "collision": collision})
                    continue
                av, bv = a.solve(task), b.solve(task)
                if encoded(av) != encoded(bv):
                    raise ValueError("independent reference disagreement; STOP, never regenerate")
                checker.add(task)
                tasks.append(task)
                answers_a[task["id"]], answers_b[task["id"]] = av, bv
                break
            else:
                raise ValueError("1000 collisions; STOP")
    read_tasks(b"".join(encoded(task) for task in tasks))
    audit.update(spec_sha256=config["spec_sha256"], python_version=config["python_version"],
                 source_pins=config["source_pins"], legacy_source_pins=config["legacy_source_pins"])
    return tasks, answers_a, answers_b, audit
