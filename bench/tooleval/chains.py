#!/usr/bin/env python3
"""Multi-step tool-call chains against an OpenAI endpoint, simulated tools, standard library only.

The shape of tool-eval-bench's category C (multi-step chains) and spark-bench's agentic domain, without an install:
each scenario needs several dependent calls (a result feeds the next call's arguments), the history is sent back the
way agent clients do (assistant ``tool_calls`` + ``reasoning_content`` as returned, ``tool`` messages), and the final
answer is checked. A scenario scores 2 points: 1 for the required calls with the right arguments in a valid order, 1
for the final answer. Malformed calls (unknown tool, arguments that are not a JSON object of the declared keys, markup
in a value) are counted and fail the scenario's call point.

  python3 bench/tooleval/chains.py --base http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-TF --mode off \\
      --out R/chains-off.json [--reps 1] [--temperature 0]
Exit 0 when the score is >= --min-score (default 10 of 12).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
import time
import urllib.request

LEAK = re.compile(r"｜DSML｜|</?think>|<｜[A-Za-z▁]+｜>|</?arg_(key|value)>|</?tool_call>")


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from safe_output import cli, dump as safe_dump, safe_print as print  # noqa: E402


def fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


S = {"type": "string"}
I = {"type": "integer"}
TOOLS = {
    "get_weather": fn("get_weather", "Current weather for a city", {"city": S}, ["city"]),
    "search_flights": fn("search_flights", "Flights between two cities on a date (YYYY-MM-DD)",
                         {"origin": S, "destination": S, "date": S}, ["origin", "destination", "date"]),
    "find_user": fn("find_user", "Look up a customer by email; returns the user id", {"email": S}, ["email"]),
    "get_orders": fn("get_orders", "A user's orders with their amounts", {"user_id": S}, ["user_id"]),
    "list_files": fn("list_files", "List the files of a directory", {"path": S}, ["path"]),
    "read_file": fn("read_file", "Read a file", {"path": S}, ["path"]),
    "write_file": fn("write_file", "Overwrite a file with new content", {"path": S, "content": S}, ["path", "content"]),
    "calc": fn("calc", "Evaluate an arithmetic expression (+ - * / ** and parentheses)", {"expr": S}, ["expr"]),
    "get_stock": fn("get_stock", "Latest price of a stock ticker", {"symbol": S}, ["symbol"]),
    "geocode": fn("geocode", "Coordinates of a place name", {"place": S}, ["place"]),
    "get_timezone": fn("get_timezone", "IANA time zone at coordinates", {"lat": {"type": "number"},
                                                                         "lon": {"type": "number"}}, ["lat", "lon"]),
}


def sim(name: str, args: dict) -> str:
    """The simulated tools (deterministic)."""

    a = {k: (str(v).strip() if isinstance(v, str) else v) for k, v in args.items()}
    if name == "get_weather":
        c = a.get("city", "").lower()
        if c == "pariss":
            return json.dumps({"error": "city not found: Pariss. Did you mean Paris?"})
        return json.dumps({"city": a.get("city"), "sky": {"oslo": "sunny", "paris": "rain"}.get(c, "cloudy"),
                           "temp_c": {"oslo": 12, "paris": 15}.get(c, 18)})
    if name == "search_flights":
        if a.get("origin", "").lower() == "oslo" and a.get("destination", "").lower() == "rome":
            return json.dumps({"flights": [{"number": "SK4711", "dep": "08:15"}, {"number": "AZ609", "dep": "17:40"}]})
        return json.dumps({"flights": []})
    if name == "find_user":
        return json.dumps({"user_id": "u-8842"} if a.get("email", "").lower() == "ana@example.com" else {"error": "no such user"})
    if name == "get_orders":
        if a.get("user_id") == "u-8842":
            return json.dumps({"orders": [{"id": "o1", "amount": 120.50}, {"id": "o2", "amount": 79.25},
                                          {"id": "o3", "amount": 300.00}]})
        return json.dumps({"error": "unknown user id"})
    if name == "list_files":
        return json.dumps({"files": ["README.md", "config.yaml", "main.py"]})
    if name == "read_file":
        if a.get("path", "").endswith("config.yaml"):
            return "port: 8080\nworkers: 4\nlog_level: info\n"
        return "(binary or empty)"
    if name == "write_file":
        return json.dumps({"ok": True, "bytes": len(str(a.get("content", "")))})
    if name == "calc":
        expr = str(a.get("expr", ""))
        if not re.fullmatch(r"[0-9.+\-*/() eE]+", expr):
            return json.dumps({"error": "only numbers and + - * / ** ( ) are allowed"})
        try:
            return json.dumps({"result": round(eval(expr, {"__builtins__": {}}), 6)})   # noqa: S307 - digits only
        except Exception as exc:                    # noqa: BLE001
            return json.dumps({"error": str(exc)})
    if name == "get_stock":
        return json.dumps({"symbol": a.get("symbol"), "price": {"ACME": 187.4, "GLOBEX": 192.1}.get(
            str(a.get("symbol", "")).upper(), None)})
    if name == "geocode":
        return json.dumps({"place": a.get("place"), "lat": 35.68, "lon": 139.69} if "tokyo" in a.get("place", "").lower()
                          else {"error": "not found"})
    if name == "get_timezone":
        lat, lon = float(a.get("lat", 0)), float(a.get("lon", 0))
        return json.dumps({"tz": "Asia/Tokyo"} if abs(lat - 35.68) < 0.5 and abs(lon - 139.69) < 0.5 else {"tz": "UTC"})
    return json.dumps({"error": f"unknown tool {name}"})


def has(calls, name, **want) -> bool:
    return any(c[0] == name and all(str(c[1].get(k, "")).strip().lower() == str(v).lower() for k, v in want.items())
               for c in calls)


def before(calls, a: str, b: str) -> bool:
    names = [c[0] for c in calls]
    return a in names and b in names and names.index(a) < names.index(b)


SCENARIOS = [
    ("weather-then-flight", ["get_weather", "search_flights"],
     "If the weather in Oslo is sunny right now, find me a flight from Oslo to Rome on 2026-11-02 and tell me the "
     "earliest flight number. Use the tools.",
     lambda c: has(c, "get_weather", city="oslo") and has(c, "search_flights", origin="oslo", destination="rome",
                                                          date="2026-11-02") and before(c, "get_weather", "search_flights"),
     lambda t: "SK4711" in t),
    ("user-orders-total", ["find_user", "get_orders", "calc"],
     "What is the total amount of all orders of the customer ana@example.com? Look it up with the tools.",
     lambda c: has(c, "find_user", email="ana@example.com") and has(c, "get_orders", user_id="u-8842")
     and before(c, "find_user", "get_orders"),
     lambda t: "499.75" in t.replace(",", "")),
    ("edit-config", ["list_files", "read_file", "write_file"],
     "In the project directory /srv/app, find the YAML config, read it, and change the number of workers to 8, "
     "keeping every other line. Use the tools, then confirm.",
     lambda c: has(c, "read_file") and any(x[0] == "write_file" and "workers: 8" in str(x[1].get("content", ""))
                                           and "port: 8080" in str(x[1].get("content", "")) for x in c)
     and before(c, "read_file", "write_file"),
     lambda t: "8" in t),
    ("compound-interest", ["calc"],
     "Using the calc tool (not mental arithmetic), compute 10000 * (1 + 0.05) ** 3, then subtract 1000 from that "
     "result with a second calc call. Report the final number rounded to 2 decimals.",
     lambda c: sum(x[0] == "calc" for x in c) >= 2,
     lambda t: "10576.25" in t.replace(",", "")),
    ("retry-after-error", ["get_weather"],
     "What's the weather in Pariss? (Use the tool exactly with the city I wrote first.)",
     lambda c: has(c, "get_weather", city="paris"),
     lambda t: "rain" in t.lower() or "15" in t),
    ("geo-timezone", ["geocode", "get_timezone"],
     "Which IANA time zone is Tokyo in? Find its coordinates with geocode, then ask get_timezone.",
     lambda c: has(c, "geocode") and has(c, "get_timezone") and before(c, "geocode", "get_timezone"),
     lambda t: "Asia/Tokyo" in t),
]


def post(base: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def mode_fields(mode: str) -> dict:
    if mode == "off":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": mode}, "reasoning_effort": mode}


def run(a, sc) -> dict:
    name, tool_names, prompt, calls_ok, answer_ok = sc
    tools = [TOOLS[t] for t in tool_names] + [TOOLS["list_files"] if "list_files" not in tool_names else TOOLS["calc"]]
    msgs = [{"role": "system", "content": "You are an agent. Use the provided tools to do the task; call one tool at a "
                                          "time when a later call needs an earlier result."},
            {"role": "user", "content": prompt}]
    calls, malformed, final, t0, turns = [], [], "", time.time(), 0
    try:
        for turns in range(1, a.max_turns + 1):
            r = post(a.base, dict({"model": a.model, "messages": msgs, "tools": tools, "tool_choice": "auto",
                                   "temperature": a.temperature, "max_tokens": a.max_tokens}, **mode_fields(a.mode)),
                     a.timeout)
            msg = r["choices"][0]["message"]
            tcs = msg.get("tool_calls") or []
            if not tcs:
                final = msg.get("content") or ""
                break
            hist = {"role": "assistant", "content": msg.get("content"), "tool_calls": tcs}
            if msg.get("reasoning_content"):
                hist["reasoning_content"] = msg["reasoning_content"]
            msgs.append(hist)
            for tc in tcs:
                f = tc.get("function") or {}
                try:
                    args = json.loads(f.get("arguments") or "{}")
                    assert isinstance(args, dict)
                except (ValueError, AssertionError):
                    args = {}
                    malformed.append(f"{f.get('name')}: arguments {str(f.get('arguments'))[:80]!r}")
                req = TOOLS.get(f.get("name"), {}).get("function", {}).get("parameters", {}).get("required")
                if req is None:
                    malformed.append(f"unknown tool {f.get('name')!r}")
                elif set(req) - set(args):
                    malformed.append(f"{f.get('name')}: missing {sorted(set(req) - set(args))}")
                if any(isinstance(v, str) and LEAK.search(v) for v in args.values()):
                    malformed.append(f"{f.get('name')}: markup in an argument")
                calls.append((f.get("name"), args))
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": sim(f.get("name"), args)})
    except Exception as exc:                        # noqa: BLE001 - a failed request fails the scenario
        malformed.append(f"request failed: {type(exc).__name__}: {exc}"[:200])
    p_calls = bool(calls_ok(calls)) and not malformed
    p_answer = bool(final) and bool(answer_ok(final))
    return {"scenario": name, "points": int(p_calls) + int(p_answer), "calls_ok": p_calls, "answer_ok": p_answer,
            "turns": turns, "calls": [[c[0], c[1]] for c in calls], "malformed": malformed, "final": final[:300],
            "seconds": round(time.time() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash-TF")
    ap.add_argument("--mode", default="off", choices=("off", "low", "high"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--min-score", type=int, default=10, help="pass line (points of 12, a rep)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    reps = []
    for rep in range(a.reps):
        res = [run(a, sc) for sc in SCENARIOS]
        for r in res:
            print(f"[chains {a.mode} rep {rep}] {r['scenario']}: {r['points']}/2 (calls {r['calls_ok']}, answer "
                  f"{r['answer_ok']}, {len(r['calls'])} calls, {r['turns']} turns, {r['seconds']} s)"
                  + (f" malformed: {r['malformed'][:3]}" if r["malformed"] else ""), flush=True)
        reps.append(res)
    scores = [sum(r["points"] for r in res) for res in reps]
    out = {"mode": a.mode, "model": a.model, "max_points": 2 * len(SCENARIOS), "scores": scores,
           "min_score": a.min_score, "pass": min(scores) >= a.min_score,
           "malformed": sum(len(r["malformed"]) for res in reps for r in res), "reps": reps}
    print(f"[chains {a.mode}] score {scores} of {2 * len(SCENARIOS)} (pass line {a.min_score}): "
          f"{'PASS' if out['pass'] else 'FAIL'}", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            safe_dump(out, f, indent=1)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(cli(main))
