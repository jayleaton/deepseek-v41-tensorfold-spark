#!/usr/bin/env python3
"""The C4 concurrent cells over HTTP, on m2bench's metric: decode aggregate (all tokens over the first-to-last
token span), the steady aggregate while every stream is live, and the share of the span that is all-live.

Cell names: "mixed4" (code, prose, tweet, edit; every other stream at T 0.7), "<work>4" (one prompt on all four
streams), "<work>4d" (four distinct prompts). Workload texts, caps and the metric follow m2bench (patches/0002:
workloads(), steady(), concurrent()). The steady window apportions each stream's tokens by character share, since
HTTP delivers text in chunks, not tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import threading
import time
import urllib.request

CODE = """Write a Python class `LRUCache` with `get(key)` and `put(key, value)` in O(1) using a dict and a doubly linked
list, with docstrings and type hints, then three unit tests with pytest."""
PROSE = """Write a 400-word essay on why lighthouses were built where they were, how their keepers lived, and what
replaced them. Plain prose, no lists or headings."""
TWEET = """List the first 40 prime numbers as a JSON array, then write a Python one-liner that checks the list, then
a JSON object {"count": ..., "largest": ..., "sum": ...} for it."""
EDIT_SRC = "\n".join(
    f"def handler_{i}(request, session):\n    value = session.get('item_{i}', {i})\n"
    f"    if value > {i * 3}:\n        return request.respond(value - {i})\n    return request.respond(value)\n"
    for i in range(24))
EDIT = "Rename the parameter `session` to `ctx` everywhere in this code and return the whole file, nothing else:\n\n" \
    + EDIT_SRC
CODE_MORE = (
    """Write a Python class `Trie` with `insert(word)`, `search(word)` and `starts_with(prefix)`, with docstrings and
type hints, then three unit tests with pytest.""",
    """Write a Python class `TokenBucket` rate limiter with `allow(now: float) -> bool`, refill rate and burst size,
with docstrings and type hints, then three unit tests with pytest.""",
    """Write a Python function `merge_intervals(intervals)` that merges overlapping [start, end] pairs in
O(n log n), with a docstring and type hints, then three unit tests with pytest.""")
PROSE_MORE = (
    """Write a 400-word essay on why canals were dug where they were, how the people who worked them lived, and what
replaced them. Plain prose, no lists or headings.""",
    """Write a 400-word essay on why early observatories were built where they were, how their astronomers lived, and
what replaced them. Plain prose, no lists or headings.""",
    """Write a 400-word essay on why windmills were built where they were, how their millers lived, and what replaced
them. Plain prose, no lists or headings.""")

WORKLOADS = {"code": (CODE, 384), "prose": (PROSE, 384), "tweet": (TWEET, 384), "edit": (EDIT, 1024)}
MORE = {"code": CODE_MORE, "prose": PROSE_MORE}
MIXED = ("code", "prose", "tweet", "edit")
MOST = 384
HEAD = ("aggregate_tok_s", "decode_aggregate_tok_s", "steady_tok_s")

TEXTS = dict(WORKLOADS)
for w, more in MORE.items():
    for i, msg in enumerate(more, 2):
        TEXTS[f"{w}{i}"] = (msg, 384)


def stream(url: str, model: str, msg: str, temp: float, cap: int, out: dict) -> None:
    body = {"model": model, "messages": [{"role": "user", "content": msg}], "max_tokens": cap,
            "temperature": temp, "stream": True, "stream_options": {"include_usage": True},
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    if temp > 0:
        body.update(top_p=0.95, seed=1234)
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    chunks, usage, tf, text = [], None, None, []
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                usage = d.get("usage") or usage
                tf = d.get("tensorfold") or tf
                if d.get("error"):
                    out["error"] = str(d["error"])[:300]
                for c in d.get("choices") or []:
                    delta = c.get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning_content")
                    if piece:
                        chunks.append((time.perf_counter(), len(piece)))
                        text.append(piece)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    first, last = (chunks[0][0], chunks[-1][0]) if chunks else (None, None)
    n = (usage or {}).get("completion_tokens") or len(chunks)
    out.update(chunks=chunks, first=first, last=last, tokens=n, chars=sum(c[1] for c in chunks),
               ttft_s=round(first - t0, 3) if first is not None else None,
               tok_s=round((n - 1) / (last - first), 2) if len(chunks) > 1 and n > 1 and last > first else None,
               sha=hashlib.sha256("".join(text).encode()).hexdigest()[:16], tf=tf)


def estimate_window_tokens(p: dict, lo: float, hi: float) -> float:
    chars = sum(c for t, c in p["chunks"] if lo < t <= hi)
    return p["tokens"] * chars / p["chars"]


def steady(per: list[dict]) -> dict:
    live = [p for p in per if p.get("first") is not None]
    if len(live) < 2:
        return {"steady_tok_s": None, "per_stream_steady_tok_s": [], "all_live_s": 0.0}
    lo = max(p["first"] for p in live)
    hi = min(p["last"] for p in live)
    if hi <= lo:
        return {"steady_tok_s": None, "per_stream_steady_tok_s": [], "all_live_s": 0.0}
    span = hi - lo
    per_w = [round(estimate_window_tokens(p, lo, hi), 1) for p in live]
    return {"steady_tok_s": round(sum(per_w) / span, 2),
            "per_stream_steady_tok_s": [round(w / span, 2) for w in per_w], "all_live_s": round(span, 3)}


def cell(url: str, model: str, order: list[str], temps: list[float], caps: list[int]) -> dict:
    per = [dict(work=w, temp=temps[i]) for i, w in enumerate(order)]
    th = [threading.Thread(target=stream, args=(url, model, TEXTS[w][0], temps[i], caps[i], per[i]))
          for i, w in enumerate(order)]
    t0 = time.perf_counter()
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.perf_counter() - t0
    errors = [p.get("error") or "no tokens" for p in per if p.get("error") or p.get("first") is None]
    if errors:
        print(f"cell {order} T{temps}: {len(errors)} failed: {errors[:2]}", file=sys.stderr)
    live = [p for p in per if p.get("first") is not None]
    if len(live) < len(order):
        return {"order": order, "failed_streams": len(errors), "errors": errors[:4]}
    first = min(p["first"] for p in live)
    last = max(p["last"] for p in live)
    hi = min(p["last"] for p in live)
    span = last - first
    total = sum(p["tokens"] for p in live)
    st = steady(live)
    return {
        "order": order, "temps": temps, "caps": caps, "failed_streams": 0,
        "tokens": total, "wall_s": round(wall, 2), "decode_span_s": round(span, 3),
        "aggregate_tok_s": round(total / wall, 2),
        "decode_aggregate_tok_s": round(total / span, 2),
        **st,
        "all_live_frac": round(st["all_live_s"] / span, 3),
        "tail_frac": round((last - hi) / span, 3),
        "per_stream": [{"work": p["work"], "temp": p["temp"], "tokens": p["tokens"], "tok_s": p["tok_s"],
                        "ttft_s": p["ttft_s"], "first_off_s": round(p["first"] - first, 3),
                        "last_off_s": round(p["last"] - first, 3), "sha": p["sha"],
                        "rounds": (p["tf"] or {}).get("rounds"), "drafted": (p["tf"] or {}).get("drafted"),
                        "accepted": (p["tf"] or {}).get("accepted"),
                        "tokens_per_round": tokens_per_round(p["tf"])} for p in per],
    }


def tokens_per_round(tf: dict | None) -> float | None:
    tf = tf or {}
    dr = tf.get("drafting") or {}
    if dr.get("tokens_per_round") is not None:
        return dr["tokens_per_round"]
    if not dr and tf.get("rounds") and "accepted" in tf and tf["rounds"]:
        return round((tf["rounds"] + tf["accepted"]) / tf["rounds"], 3)
    return None


def med(reps: list[dict], key: str) -> float | None:
    xs = [r[key] for r in reps if isinstance(r.get(key), (int, float))]
    return round(statistics.median(xs), 2) if xs else None


def spec(name: str) -> tuple[list[str], list[float], list[int]]:
    m = re.fullmatch(r"(mixed|[a-z]+)4d?", name)
    if not m:
        raise SystemExit(f"bad cell name: {name}")
    w, distinct = m.group(1), name.endswith("d")
    if w == "mixed":
        order = list(MIXED)
    elif distinct and MORE.get(w):
        order = [w] + [f"{w}{i}" for i in range(2, len(MORE[w]) + 2)]
        order = (order * 4)[:4]
    else:
        order = [w] * 4
    temps = [0.7 if i % 2 else 0.0 for i in range(4)]
    caps = [min(TEXTS[w][1], MOST) for w in order]
    return order, temps, caps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cells", default="mixed4,code4,code4d,prose4,prose4d")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = {"url": a.url, "model": a.model, "reps": a.reps, "cells": {}}
    for name in [c.strip() for c in a.cells.split(",") if c.strip()]:
        order, temps, caps = spec(name)
        for _ in range(a.warm):
            cell(a.url, a.model, order, temps, caps)
        reps = [r for r in (cell(a.url, a.model, order, temps, caps) for _ in range(a.reps))
                if not r.get("failed_streams")]
        if not reps:
            out["cells"][name] = {"error": "every rep failed"}
            print(f"{name}: every rep failed", file=sys.stderr)
            continue
        medians = {k: med(reps, k) for k in HEAD}
        out["cells"][name] = {"medians": medians, "reps": reps}
        print(f"{name}: agg {medians['aggregate_tok_s']} / decode {medians['decode_aggregate_tok_s']}"
              f" / steady {medians['steady_tok_s']}"
              f" (all live {round(100 * (med(reps, 'all_live_frac') or 0))}%)")
    json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
