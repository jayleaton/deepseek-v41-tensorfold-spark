#!/usr/bin/env python3
"""Soak test: mixed traffic against our server for --minutes (30), standard library only.

Four worker threads; a controller moves the number of active ones between 1 and 4 every 45-120 s. Each request draws a
kind: code / prose (T = 0 or 0.7), structured (response_format json_schema), tool (tools + tool_choice required),
thinking (effort low), long (an 8K-48K-token document + a question), nonstream. Streamed requests are cancelled at a
random chunk ~15% of the time (the client closes the socket); ~5% are "disconnects" (non-streamed, a 2-5 s client
timeout mid-reply). Both are intentional and not errors.

Checked: errors (HTTP >= 400, exceptions, empty replies of normal requests) = 0; content checks (structured parses,
tool calls present) counted separately; latency (first token, total) per kind; /health every 10 s (inflight,
fatal, stalled); at the end inflight back to 0 within --drain-s (no slot leak) and 17*23 = 391.

  python3 bench/soak.py --base http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-TF --minutes 30 --out soak.json
Exit 0 on PASS (0 errors, drained, 391, no fatal).

``--kinds g16`` (the G16-G19 serving-leak soaks): agent traffic on top of the mix above. ``agent``
is a coding agent's session that grows a turn at a time (40 tools, a tool call and its result appended each turn,
1-4K tokens a turn) up to ``--agent-max-k`` thousand tokens (120), then starts over; each worker keeps its own
session, so up to four grow side by side. ``image`` sends an image part (a 1-pixel PNG). The controller
also idles every worker for 30-90 s with probability ``--idle-p`` (0.15), as a client between turns does.
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import socket
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

KINDS = [("code", 22), ("prose", 18), ("structured", 14), ("tool", 14), ("thinking", 10), ("long", 10),
         ("nonstream", 7), ("disconnect", 5)]
CODE = ["Write a Python function that merges two sorted lists, with a docstring and two doctests.",
        "Implement an LRU cache class in Python with get / put in O(1). Code only.",
        "Write a bash script that rotates log files older than 7 days into a tar.gz archive."]
PROSE = ["Write a short essay about the history of lighthouses.", "Describe a rainy morning in a harbor town.",
         "Explain to a ten-year-old why the sky is blue, in two paragraphs."]
SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}, "year": {"type": "integer"},
                                           "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 4}},
          "required": ["title", "year", "tags"]}
TOOL = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
KINDS_AGENT = [("agent", 30), ("code", 12), ("prose", 10), ("structured", 9), ("tool", 9), ("thinking", 6),
             ("long", 8), ("nonstream", 5), ("disconnect", 4), ("image", 7)]
AGENT_TOOLS = [{"type": "function", "function": {
    "name": f"tool_{i}", "description": f"Workspace operation {i}: reads, edits or searches files. " * 3,
    "parameters": {"type": "object", "properties": {f"arg{j}": {"type": "string", "description": "path, pattern or text"}
                                                    for j in range(4)}, "required": ["arg0"]}}} for i in range(40)]
PNG_1PX = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
WORDS = "amber basin cobalt drift ember fable grove harbor island jetty kelp lumen marsh north orbit pine quay ridge".split()


def _text(rng: random.Random, tokens: int) -> str:
    return " ".join(f"{rng.choice(WORDS)}_{rng.randrange(10000)}" for _ in range(max(1, tokens // 3)))


def agent_body(sess: dict, rng: random.Random, max_k: int) -> dict:
    """The next turn of a worker's agent session (``sess``: its messages, grown here)."""

    msgs = sess.get("messages")
    if not msgs or sess.get("chars", 0) > max_k * 1000 * 4:
        msgs = sess["messages"] = [
            {"role": "system", "content": "You are a coding agent working in a repository. " + _text(rng, 2000)},
            {"role": "user", "content": "Fix the failing tests. " + _text(rng, rng.randrange(200, 2000))}]
        sess["chars"], sess["turn"] = sum(len(m["content"]) for m in msgs), 0
    else:
        sess["turn"] += 1
        cid = f"call_{sess['turn']}_{rng.randrange(1 << 30)}"
        result = _text(rng, rng.randrange(1000, 4000))
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function", "function": {
                     "name": f"tool_{rng.randrange(40)}", "arguments": json.dumps({"arg0": f"src/{rng.choice(WORDS)}.py"})}}]},
                 {"role": "tool", "tool_call_id": cid, "content": result}]
        sess["chars"] += len(result) + 120
    return {"messages": list(msgs), "tools": AGENT_TOOLS, "reasoning_effort": "low", "max_tokens": 400,
            "temperature": 0.6}


def body_for(kind: str, model: str, rng: random.Random, sess: dict | None = None,
             agent_max_k: int = 120) -> tuple[dict, bool]:
    """(request body, streamed)."""

    off = {"chat_template_kwargs": {"enable_thinking": False}}
    t = rng.choice([0.0, 0.7])
    if kind == "agent":
        b = agent_body(sess if sess is not None else {}, rng, agent_max_k)
    elif kind == "image":
        b = dict(off, messages=[{"role": "user", "content": [
            {"type": "text", "text": "What does this image show? One sentence."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG_1PX}}]}], max_tokens=96,
            temperature=t)
    elif kind == "code":
        b = dict(off, messages=[{"role": "user", "content": rng.choice(CODE)}], max_tokens=384, temperature=t)
    elif kind in ("prose", "nonstream", "disconnect"):
        b = dict(off, messages=[{"role": "user", "content": rng.choice(PROSE)}], max_tokens=384, temperature=t)
    elif kind == "structured":
        b = dict(off, messages=[{"role": "user", "content": "Describe a classic film as JSON."}], max_tokens=256,
                 temperature=0, response_format={"type": "json_schema", "json_schema": {"name": "film", "schema": SCHEMA}})
    elif kind == "tool":
        b = dict(off, messages=[{"role": "user", "content": f"Weather in {rng.choice(['Oslo', 'Lima', 'Kyiv'])}?"}],
                 tools=TOOL, tool_choice="required", max_tokens=256, temperature=0)
    elif kind == "thinking":
        b = {"messages": [{"role": "user", "content": "How many weekdays are there in March 2027? Think it through."}],
             "reasoning_effort": "low", "max_tokens": 2048, "temperature": t}
    else:                                           # long: an 8K-48K-token document, one question
        n = rng.choice([8, 16, 32, 48]) * 1024
        doc = " ".join(f"Entry {i}: the {rng.choice(WORDS)} {rng.choice(WORDS)} met the {rng.choice(WORDS)}."
                       for i in range(n // 11))
        b = dict(off, messages=[{"role": "user", "content": doc + "\n\nHow many entries are there? One number."}],
                 max_tokens=32, temperature=0)
    b["model"] = model
    return b, kind not in ("nonstream", "disconnect")


class Soak:
    def __init__(self, a) -> None:
        self.a, self.lock, self.stop = a, threading.Lock(), threading.Event()
        self.active = 1
        self.rec: list[dict] = []
        self.health: list[dict] = []
        self.kinds = KINDS_AGENT if getattr(a, "kinds", "g6") == "g16" else KINDS
        self.sessions = [{} for _ in range(4)]

    def one(self, kind: str, rng: random.Random, worker: int = 0) -> dict:
        body, streamed = body_for(kind, self.a.model, rng, self.sessions[worker], getattr(self.a, "agent_max_k", 120))
        r = {"kind": kind, "t": round(time.time() - self.t0, 1), "error": None, "check": None, "cancel": False}
        cancel_at = rng.randint(1, 20) if streamed and rng.random() < self.a.cancel_p else None
        timeout = rng.uniform(2, 5) if kind == "disconnect" else self.a.timeout
        if streamed:
            body["stream"] = True
        req = urllib.request.Request(self.a.base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0, text, calls, chunks = time.time(), "", 0, 0
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if not streamed:
                    m = json.loads(resp.read())["choices"][0]["message"]
                    text = (m.get("content") or "") + (m.get("reasoning_content") or "")
                    calls = len(m.get("tool_calls") or [])
                else:
                    for raw in resp:
                        line = raw.decode().strip()
                        if not line.startswith("data:") or line == "data: [DONE]":
                            continue
                        c = json.loads(line[5:])
                        if c.get("error"):          # G19: an in-stream error event (e.g. the floor's 503 refusal
                            err = c["error"]        # after the 200 headers) was logged as "empty reply" in G18
                            r["error"] = ("stream error: " + str(err.get("message") if isinstance(err, dict)
                                                                 else err))[:200]
                            break
                        for ch in c.get("choices") or []:
                            d = ch.get("delta") or {}
                            if r.get("ttft_s") is None and (d.get("content") or d.get("reasoning_content")
                                                             or d.get("reasoning") or d.get("tool_calls")):
                                r["ttft_s"] = round(time.time() - t0, 3)
                            text += (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                            calls += len(d.get("tool_calls") or [])
                        chunks += 1
                        if cancel_at is not None and chunks >= cancel_at:
                            r["cancel"] = True
                            break                   # leaving the with-block closes the socket mid-stream
        except urllib.error.HTTPError as exc:
            r["error"] = f"HTTP {exc.code}: {exc.read()[:200]!r}"
        except (socket.timeout, TimeoutError) as exc:
            if kind == "disconnect":
                r["cancel"] = True
            else:
                r["error"] = f"timeout: {exc}"
        except (OSError, http.client.HTTPException, ValueError) as exc:
            r["error"] = f"{type(exc).__name__}: {exc}"[:200]
        r["total_s"] = round(time.time() - t0, 3)
        if r["error"] is None and not r["cancel"]:
            if kind == "structured" and streamed:
                try:
                    json.loads(text)
                except ValueError:
                    r["check"] = f"structured reply not JSON: {text[:80]!r}"
            elif kind == "tool" and not calls:
                r["check"] = "no tool call"
            elif kind not in ("tool",) and not text.strip() and not (kind == "agent" and calls):
                r["error"] = "empty reply"
        return r

    def worker(self, i: int) -> None:
        rng = random.Random(self.a.seed * 100 + i)
        names, weights = zip(*self.kinds)
        while not self.stop.is_set():
            if i >= self.active:
                time.sleep(1)
                continue
            r = self.one(rng.choices(names, weights)[0], rng, i)
            with self.lock:
                self.rec.append(r)
            if r["error"]:
                print(f"[soak] ERROR {r['kind']} at {r['t']} s: {r['error']}", flush=True)

    def control(self) -> None:
        rng = random.Random(self.a.seed)
        while not self.stop.wait(rng.uniform(45, 120)):
            idle = rng.random() < getattr(self.a, "idle_p", 0.0)
            self.active = 0 if idle else rng.randint(1, 4)
            if idle:                                # every worker idles for 30-90 s, then 1-4 again
                if self.stop.wait(rng.uniform(30, 90)):
                    break
                self.active = rng.randint(1, 4)
            print(f"[soak] {round(time.time() - self.t0)} s: {self.active} concurrent; {len(self.rec)} done", flush=True)

    def poll(self) -> None:
        while not self.stop.wait(10):
            self.health.append(dict(health(self.a.base), t=round(time.time() - self.t0)))

    def run(self) -> dict:
        self.t0 = time.time()
        th = [threading.Thread(target=self.worker, args=(i,), daemon=True) for i in range(4)]
        th += [threading.Thread(target=self.control, daemon=True), threading.Thread(target=self.poll, daemon=True)]
        for t in th:
            t.start()
        time.sleep(self.a.minutes * 60)
        self.stop.set()
        for t in th[:4]:
            t.join(self.a.timeout + 30)
        drained, t1 = False, time.time()
        while time.time() - t1 < self.a.drain_s:
            h = health(self.a.base)
            if h.get("inflight") == 0:
                drained = True
                break
            time.sleep(2)
        return self.report(drained, round(time.time() - t1, 1))

    def report(self, drained: bool, drain_s: float) -> dict:
        rec = self.rec
        by: dict[str, dict] = {}
        for k, _ in self.kinds:
            xs = [r for r in rec if r["kind"] == k]
            ok = [r for r in xs if not r["error"] and not r["cancel"]]
            tt = sorted(r["ttft_s"] for r in ok if r.get("ttft_s") is not None)
            tot = sorted(r["total_s"] for r in ok)
            pct = (lambda v, p: v[min(len(v) - 1, int(p * len(v)))] if v else None)    # noqa: E731
            by[k] = {"n": len(xs), "ok": len(ok), "cancelled": sum(r["cancel"] for r in xs),
                     "errors": sum(bool(r["error"]) for r in xs), "check_fails": sum(bool(r["check"]) for r in xs),
                     "ttft_p50": pct(tt, 0.5), "ttft_p95": pct(tt, 0.95),
                     "total_p50": statistics.median(tot) if tot else None, "total_p95": pct(tot, 0.95)}
        ans = ""
        try:
            req = urllib.request.Request(self.a.base + "/v1/chat/completions", json.dumps({
                "model": self.a.model, "messages": [{"role": "user", "content": "What is 17*23? Answer with the number only."}],
                "max_tokens": 64, "temperature": 0, "reasoning_effort": "none"}).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                ans = json.loads(r.read())["choices"][0]["message"].get("content") or ""
        except Exception as exc:                    # noqa: BLE001
            ans = f"ERROR {exc}"
        errors = sum(bool(r["error"]) for r in rec)
        fatal = any(h.get("fatal") for h in self.health)
        out = {"minutes": self.a.minutes, "requests": len(rec), "errors": errors,
               "check_fails": sum(bool(r["check"]) for r in rec), "cancelled": sum(r["cancel"] for r in rec),
               "drained": drained, "drain_s": drain_s, "answer_17x23": ans[:40], "fatal_seen": fatal,
               "stalled_seen": any(h.get("stalled") for h in self.health),
               "inflight_max": max((h.get("inflight") or 0 for h in self.health), default=None),
               "by_kind": by, "health_end": health(self.a.base),
               "error_samples": [r for r in rec if r["error"]][:20], "check_samples": [r for r in rec if r["check"]][:10]}
        out["pass"] = errors == 0 and drained and "391" in ans and not fatal
        return out


def health(base: str) -> dict:
    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as r:
            h = json.loads(r.read())
        inf = h.get("inflight")
        return {"inflight": len(inf) if isinstance(inf, (list, dict)) else inf, "fatal": h.get("fatal"),
                "stalled": h.get("stalled"), "running": h.get("requests_running")}
    except Exception as exc:                        # noqa: BLE001
        return {"error": str(exc)[:120]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash-TF")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--cancel-p", type=float, default=0.15)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--drain-s", type=float, default=120)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--out", default="")
    ap.add_argument("--kinds", choices=("g6", "g16"), default="g6", help="g16: + agent sessions, images, idle spells")
    ap.add_argument("--agent-max-k", type=int, default=120, help="agent sessions start over past this many K tokens")
    ap.add_argument("--idle-p", type=float, default=None, help="chance a control step idles all workers (g16: 0.15)")
    a = ap.parse_args(argv)
    if a.idle_p is None:
        a.idle_p = 0.15 if a.kinds == "g16" else 0.0
    out = Soak(a).run()
    print(json.dumps({k: v for k, v in out.items() if k not in ("by_kind", "error_samples", "check_samples")}), flush=True)
    for k, v in out["by_kind"].items():
        print(f"[soak] {k}: {v}", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
