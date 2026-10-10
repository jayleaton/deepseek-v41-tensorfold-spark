#!/usr/bin/env python3
"""Zig prod against Python prod on the same greedy requests: where the replies part.

Both servers send a reply's exact ids with ``"return_token_ids": true`` (the ``tensorfold`` stats block: Python's
DSV4.1 app.py, Zig's openai.zig ``sparkStats``), so every request asks for them and the comparison is id by id; the
reply's text (the reasoning kept whole, not cut at 80 characters as requests.sh's log does) is kept beside them, and
re-tokenized only when a server sent no ids. Neither server reports logprobs (Python's ``probability_options`` runs
with ``supported=False``; Zig refuses them): a margin at the first differing token needs the teacher-forced reference
(dsv41_m2b_ref.py), not HTTP.

    # each server, the same requests (prompts.txt as requests.sh sends them, thinking on; the bench's four code and
    # four prose prompts as bench_http.py sends them, thinking off), non-streamed, one at a time:
    python3 diverge.py ask --url http://localhost:8000 --prompts prompts.txt --bench --out py.jsonl
    python3 diverge.py ask --url http://localhost:8000 --prompts prompts.txt --bench --no-draft --out py-nd.jsonl
    python3 diverge.py ask --url http://localhost:8000 --prompts prompts.txt --bench --out zig.jsonl
    # the first differing token of each reply (tokenizer.json from the pack):
    python3 diverge.py compare py.jsonl zig.jsonl --tokenizer /model/tokenizer.json
    # a Zig server's recorded replies (TF_DSV41_TRACE_TOKENS lines, e.g. the 2026-10-07 night's out/a/trace.jsonl
    # lines 42-49 = requests.sh's 8) as `ask` records, to compare with Python's ids without starting Zig again:
    python3 diverge.py fromtrace trace.jsonl --lines 42-49 --prompts prompts.txt --out zig-night.jsonl
    # Python's replies as a trace for `tf-dsv41-m1 generate` (prompt ids from a Zig server's TF_DSV41_TRACE_TOKENS
    # file of the same requests): its "first_differing" is then the Zig engine's first token off Python's reply
    python3 diverge.py trace py.jsonl zig-trace.jsonl --tokenizer /model/tokenizer.json --out py-trace.jsonl
    # multi-turn chats (sessions): every bench prompt a chat of --turns turns (its replies fed back, a follow-up message
    # each turn), --concurrency chats at once, so later turns resume their chat's session while other chats decode;
    # the records compare as `ask`'s do ("prompt" names the chat and turn)
    python3 diverge.py chat --url http://localhost:8000 --turns 3 --out py-chat.jsonl

Standard library only for ``ask``; ``compare`` / ``trace`` need ``tokenizers`` (for the context they print, and for
replies without ids). A reply's text is
``reasoning + "</think>" + content`` when the request thought (the model's own token sequence), else ``content``;
re-tokenizing it gives the server's ids for every reply this check has met (ByteLevel BPE round trip), and
``compare`` says so when it does not (``roundtrip: false``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

BENCH = (  # bench_http.py's PROMPTS, in its order (code, then prose)
    """Write a Python class `LRUCache` with `get(key)` and `put(key, value)` in O(1) using a dict and a doubly linked
list, with docstrings and type hints, then three unit tests with pytest.""",
    """Write a Python class `Trie` with `insert(word)`, `search(word)` and `starts_with(prefix)`, with docstrings and
type hints, then three unit tests with pytest.""",
    """Write a Python class `TokenBucket` rate limiter with `allow(now: float) -> bool`, refill rate and burst size,
with docstrings and type hints, then three unit tests with pytest.""",
    """Write a Python function `merge_intervals(intervals)` that merges overlapping [start, end] pairs in
O(n log n), with a docstring and type hints, then three unit tests with pytest.""",
    """Write a 400-word essay on why lighthouses were built where they were, how their keepers lived, and what
replaced them. Plain prose, no lists or headings.""",
    """Write a 400-word essay on why canals were dug where they were, how the people who worked them lived, and what
replaced them. Plain prose, no lists or headings.""",
    """Write a 400-word essay on why early observatories were built where they were, how their astronomers lived, and
what replaced them. Plain prose, no lists or headings.""",
    """Write a 400-word essay on why windmills were built where they were, how their millers lived, and what replaced
them. Plain prose, no lists or headings.""")
THINK_END = "</think>"


def ask(a) -> int:
    reqs = []
    for line in open(a.prompts, encoding="utf-8") if a.prompts else []:
        if line.strip():
            reqs.append(("requests", line.rstrip("\n"), {}, a.max_tokens))
    if a.bench:
        off = {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
        reqs += [("bench", p, off, a.bench_tokens) for p in BENCH]
    with open(a.out, "w", encoding="utf-8") as out:
        for n, (kind, prompt, extra, most) in enumerate(reqs, 1):
            body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                    "max_tokens": most, "return_token_ids": True, **extra}
            if a.no_draft:
                body["draft"] = False       # both servers: the reply without DSpark (the same ids, by their rule)
            req = urllib.request.Request(f"{a.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"content-type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read())
            msg = (d.get("choices") or [{}])[0].get("message") or {}
            stats = d.get("tensorfold") or (d.get("choices") or [{}])[0].get("tensorfold") or {}
            rec = {"n": n, "kind": kind, "thinking": not extra, "prompt": prompt, "usage": d.get("usage"),
                   "finish": (d.get("choices") or [{}])[0].get("finish_reason"), "seconds": round(time.time() - t0, 2),
                   "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                   "content": msg.get("content") or "", "ids": stats.get("token_ids"), "draft": not a.no_draft}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(json.dumps({k: rec[k] for k in ("n", "kind", "usage", "finish")}), flush=True)
    return 0


FOLLOW = ("Now explain the trickiest part of your answer in more detail.",
          "Give one concrete example, then list two pitfalls.",
          "Summarize everything above in three sentences.")


def chat(a) -> int:
    """Multi-turn chats: each first message (``--prompts`` lines, else the bench's 8 prompts) starts a chat; turn t
    sends the whole conversation (the server's own earlier replies as assistant messages) plus FOLLOW[t - 2]. Thinking
    off, greedy, ``--concurrency`` chats in flight. The server's stats block (``tensorfold``: its session hit, if it
    reports one) is kept beside each reply."""

    import concurrent.futures as cf

    firsts = [l.rstrip("\n") for l in open(a.prompts, encoding="utf-8") if l.strip()] if a.prompts else list(BENCH)
    off = {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}

    def one(c: int) -> list[dict]:
        msgs, recs = [{"role": "user", "content": firsts[c]}], []
        for t in range(1, a.turns + 1):
            body = {"model": a.model, "messages": msgs, "temperature": 0, "max_tokens": a.max_tokens,
                    "return_token_ids": True, **off}
            if a.no_draft:
                body["draft"] = False
            req = urllib.request.Request(f"{a.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"content-type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read())
            msg = (d.get("choices") or [{}])[0].get("message") or {}
            stats = d.get("tensorfold") or (d.get("choices") or [{}])[0].get("tensorfold") or {}
            content = msg.get("content") or ""
            recs.append({"chat": c, "turn": t, "kind": "chat", "thinking": False,
                         "prompt": f"chat {c} turn {t}: {firsts[c][:60]}", "usage": d.get("usage"),
                         "finish": (d.get("choices") or [{}])[0].get("finish_reason"),
                         "seconds": round(time.time() - t0, 2), "reasoning": "", "content": content,
                         "ids": stats.get("token_ids"), "stats": {k: v for k, v in stats.items() if k != "token_ids"},
                         "draft": not a.no_draft})
            msgs += [{"role": "assistant", "content": content}, {"role": "user", "content": FOLLOW[(t - 1) % len(FOLLOW)]}]
        return recs

    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        recs = [r for rs in ex.map(one, range(len(firsts))) for r in rs]
    with open(a.out, "w", encoding="utf-8") as out:
        for n, rec in enumerate(sorted(recs, key=lambda r: (r["chat"], r["turn"])), 1):
            rec["n"] = n
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(json.dumps({k: rec[k] for k in ("n", "chat", "turn", "usage", "finish", "seconds")}), flush=True)
    return 0


def text_of(rec: dict) -> str:
    """The reply as the model produced it: its reasoning closed by </think>, then the content (a reply cut inside its
    reasoning has no content and no </think>)."""

    if rec["thinking"] and (rec["content"] or rec.get("finish") == "stop"):
        return rec["reasoning"] + THINK_END + rec["content"]
    return rec["reasoning"] if rec["thinking"] else rec["content"]


def ids_of(tok, rec: dict) -> tuple[list[int], bool]:
    if rec.get("ids") is not None:          # the server's own ids (return_token_ids)
        return [int(t) for t in rec["ids"]], True
    if tok is None:
        raise SystemExit(f"request {rec['n']}: no ids from the server; pass --tokenizer")
    text = text_of(rec)
    ids = tok.encode(text, add_special_tokens=False).ids
    return ids, tok.decode(ids, skip_special_tokens=False) == text


def compare(a) -> int:
    tok = None
    if a.tokenizer:                         # optional when both files hold ids: the context is then left out
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(a.tokenizer)
    xs = [json.loads(l) for l in open(a.a, encoding="utf-8") if l.strip()]
    ys = [json.loads(l) for l in open(a.b, encoding="utf-8") if l.strip()]
    if [x["prompt"] for x in xs] != [y["prompt"] for y in ys]:
        print("the two files hold different requests", file=sys.stderr)
        return 2
    equal = 0
    for x, y in zip(xs, ys):
        ix, rx = ids_of(tok, x)
        iy, ry = ids_of(tok, y)
        k = next((i for i, (p, q) in enumerate(zip(ix, iy)) if p != q), None)
        if k is None and len(ix) != len(iy):
            k = min(len(ix), len(iy))
        same = k is None and (x["usage"] or {}).get("completion_tokens") == (y["usage"] or {}).get("completion_tokens")
        equal += same
        row = {"n": x["n"], "kind": x["kind"], "equal": same, "first_differing": -1 if k is None else k,
               "tokens": [(x["usage"] or {}).get("completion_tokens"), (y["usage"] or {}).get("completion_tokens")],
               "roundtrip": rx and ry}
        if k is not None:
            row.update(a_id=ix[k] if k < len(ix) else None, b_id=iy[k] if k < len(iy) else None)
        if k is not None and tok is not None:
            ctx = tok.decode(ix[max(0, k - 12):k], skip_special_tokens=False)
            row.update(before=ctx, a=tok.decode(ix[k:k + 6], skip_special_tokens=False),
                       b=tok.decode(iy[k:k + 6], skip_special_tokens=False))
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(f"{equal}/{len(xs)} replies equal")
    return 0 if equal == len(xs) else 1


def trace(a) -> int:
    """Python's replies as `generate` lines: the prompt ids from the Zig server's trace of the same requests (in
    order), the reply ids re-tokenized from Python's text."""

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(a.tokenizer)
    xs = [json.loads(l) for l in open(a.replies, encoding="utf-8") if l.strip()]
    zs = [json.loads(l) for l in open(a.zig_trace, encoding="utf-8") if l.strip()][-len(xs):]
    with open(a.out, "w", encoding="utf-8") as out:
        for x, z in zip(xs, zs):
            ids, ok = ids_of(tok, x)
            if not ok:
                print(f"request {x['n']}: its text does not round-trip; skipped", file=sys.stderr)
                continue
            want = (x["usage"] or {}).get("completion_tokens")
            if x.get("ids") is None and x.get("finish") == "stop":
                ids.append(tok.token_to_id("<｜end▁of▁sentence｜>"))
            if want is not None and len(ids) != want:
                print(f"request {x['n']}: {len(ids)} ids from the text, usage says {want}", file=sys.stderr)
            out.write(json.dumps({"prompt": z["prompt"], "tokens": ids, "sampling": None, "structure": None}) + "\n")
    return 0


def fromtrace(a) -> int:
    lo, hi = (int(v) for v in a.lines.split("-"))
    rows = [json.loads(l) for l in open(a.trace, encoding="utf-8") if l.strip()][lo - 1:hi]
    prompts = [l.rstrip("\n") for l in open(a.prompts, encoding="utf-8") if l.strip()]
    if len(prompts) != len(rows):
        print(f"{len(rows)} trace lines for {len(prompts)} prompts", file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8") as out:
        for n, (z, prompt) in enumerate(zip(rows, prompts), 1):
            rec = {"n": n, "kind": "requests", "thinking": True, "prompt": prompt,
                   "usage": {"completion_tokens": len(z["tokens"])}, "finish": None, "reasoning": "", "content": "",
                   "ids": z["tokens"], "draft": None}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ask")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default="DeepSeek-V4.1-Flash-TF")
    p.add_argument("--prompts", default="")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--bench", action="store_true", help="also bench_http.py's 8 prompts, thinking off")
    p.add_argument("--bench-tokens", type=int, default=384)
    p.add_argument("--no-draft", action="store_true", help='every request with "draft": false')
    p.add_argument("--out", required=True)
    p = sub.add_parser("chat")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default="DeepSeek-V4.1-Flash-TF")
    p.add_argument("--prompts", default="", help="first messages, one a line (default: the bench's 8 prompts)")
    p.add_argument("--turns", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--no-draft", action="store_true")
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--tokenizer", default="", help="the pack's tokenizer.json (the context text; needed for replies without ids)")
    p = sub.add_parser("trace")
    p.add_argument("replies")
    p.add_argument("zig_trace")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("fromtrace")
    p.add_argument("trace")
    p.add_argument("--lines", required=True, help="the trace's lines (1-based, inclusive) of the prompts, e.g. 42-49")
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"ask": ask, "chat": chat, "compare": compare, "trace": trace, "fromtrace": fromtrace}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
