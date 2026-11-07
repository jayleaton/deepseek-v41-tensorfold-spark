"""Frozen compute-dl-v1 candidate factory; no collection or batch-generation CLI.

generate_candidate(family, slot, attempt) returns ONE public record, never answers.
The preparation owner MUST pin Python (platform.python_version()) and this file's
SHA256 before calling it, verify SPEC_SHA256, and apply its exclusion pipeline.
Only documented collisions authorize a next attempt; verification errors stop.
The parent's collision_checker(candidate) returns True for a collision.
No generate_all API is provided: the owner retains every collision, stops after
1000 collisions per slot, and owns the final 120-record selection and seal.
Calling this factory is not evidence of disjointness or successful verification.
"""

import hashlib
import json
import platform
import random
from pathlib import Path

FAMILIES = ("integer", "modular", "division", "dp", "graph", "aggregation")
SPEC_SHA256 = "c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141"
MASTER_SEED = 20261105


def verify_generation_pin(*, python_version, generator_sha256, spec_path):
    """Verify parent-supplied PRE-generation pins; raise rather than replace.

    Returns an auditable pin receipt. Call before generating ANY study candidate;
    this deliberately neither writes a pin nor chooses a Python version.
    """
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    spec_hash = hashlib.sha256(Path(spec_path).read_bytes()).hexdigest()
    if platform.python_version() != python_version:
        raise ValueError("Python version differs from pre-generation pin")
    if source_hash != generator_sha256:
        raise ValueError("generator source differs from pre-generation pin")
    if spec_hash != SPEC_SHA256:
        raise ValueError("frozen specification SHA256 mismatch")
    return {"python_version": python_version, "generator_sha256": source_hash,
            "spec_sha256": spec_hash}


def _embedded(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def generate_candidate(family, slot, attempt):
    """Return a public candidate for fixed family/slot and collision attempt.

    slot is 0..19; attempt is 0..999 (1000 collisions abort preparation).
    The caller must check collisions and independently verify both references.
    Do not call with a new attempt after ANY verification error.
    """
    if family not in FAMILIES:
        raise ValueError("unknown family")
    if type(slot) is not int or not 0 <= slot < 20:
        raise ValueError("slot must be an integer in 0..19")
    if type(attempt) is not int or not 0 <= attempt < 1000:
        raise ValueError("attempt must be in 0..999; 1000 collisions abort")
    label = f"compute-dl-v1|20261105|{family}|{slot}|{attempt}"
    seed = int.from_bytes(hashlib.sha256(label.encode("ascii")).digest(), "big")
    r = random.Random(seed)

    def digits(d):
        return r.randrange(10 ** (d - 1), 10 ** d)

    if family == "integer":
        a = digits(r.randint(24, 36))
        b = digits(r.randint(24, 36))
        c = r.randrange(-10**20, 10**20 + 1)
        inputs = {"a": a, "b": b, "c": c}
        prompt = (f"Compute A*B+C exactly for A={a}, B={b}, C={c}. "
                  "Final answer format: a JSON integer.")
    elif family == "modular":
        a = digits(r.randint(14, 20))
        e = r.randint(500, 5000)
        m = digits(r.randint(16, 22))
        inputs = {"a": a, "e": e, "m": m}
        prompt = (f"Find the least nonnegative residue of A^E modulo M, where A={a}, E={e}, M={m}. "
                  "Here ^ denotes exponentiation. Final answer format: a JSON integer.")
    elif family == "division":
        a = digits(80)
        b = digits(29)
        inputs = {"a": a, "b": b}
        prompt = (f"For A={a} and B={b}, find integers q and r such that A=q*B+r and 0<=r<B. "
                  "Final answer format: the JSON array [q,r].")
    elif family == "dp":
        rows = r.randint(7, 10)
        cols = r.randint(7, 10)
        matrix = [[r.randint(0, 99) for _ in range(cols)] for _ in range(rows)]
        inputs = {"rows": rows, "cols": cols, "matrix": matrix}
        prompt = ("A path starts at the top-left cell of this grid and ends at the bottom-right cell. "
                  "Each move is one cell right or one cell down. The path cost is the sum of all visited cells, "
                  "including both endpoints. What is the minimum cost? "
                  f"Grid: {_embedded(matrix)}. Final answer format: a JSON integer.")
    elif family == "graph":
        n = r.randint(9, 14)
        edges = [[i, i + 1, r.randint(1, 40)] for i in range(n - 1)]
        pairs = [(u, v) for u in range(n) for v in range(n)
                 if u != v and v != u + 1]
        r.shuffle(pairs)
        k = r.randint(n, 2 * n)
        edges.extend([u, v, r.randint(1, 40)] for u, v in pairs[:k])
        edges.sort(key=lambda edge: (edge[0], edge[1]))
        inputs = {"n": n, "edges": edges}
        prompt = (f"A directed graph has vertices 0 through {n - 1}. Each edge [u,v,w] goes from u to v with cost w. "
                  f"What is the minimum total cost from vertex 0 to vertex {n - 1}? "
                  f"Edges: {_embedded(edges)}. Final answer format: a JSON integer.")
    else:
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
        prompt = ("Each transaction is [category,quantity,unit_amount]. Unit amounts are signed integer cents. "
                  "For each category A through F, compute the sum of quantity*unit_amount over its transactions. "
                  f"Transactions: {_embedded(records)}. "
                  "Final answer format: a JSON object with keys A through F and integer values.")
    return {"id": f"compute-dl-v1-{family}-{slot + 1:03d}", "family": family,
            "suite": "compute_dl", "slot": slot, "attempt": attempt,
            "generator_label": label, "inputs": inputs, "prompt": prompt}
