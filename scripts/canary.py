#!/usr/bin/env python3
"""Post-start canary for the V4.1 server (scripts/serve.sh): a short greedy chat, a thinking reply with its
reasoning separated, a forced DSML tool call, a JSON-schema reply and /tokenize. Exit 0 when every probe passes.
Standard library only (runs on the head, outside the image)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import time
import urllib.request


sys.path.insert(0, str(Path(__file__).resolve().parent))
from safe_output import cli, safe_print as print  # noqa: E402


def post(base: str, path: str, body: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    a = ap.parse_args()
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                               "parameters": {"type": "object", "properties": {
                                                   "city": {"type": "string"}}, "required": ["city"]}}}]
    probes = {
        "chat": ({"messages": [{"role": "user", "content": "What is the capital of France? One word."}],
                  "temperature": 0, "max_tokens": 16, "reasoning_effort": "none"},
                 lambda r: "paris" in (r["choices"][0]["message"]["content"] or "").lower()),
        "thinking": ({"messages": [{"role": "user", "content": "What is 17 + 25? Answer with the number."}],
                      "temperature": 0, "max_tokens": 512, "reasoning_effort": "low"},
                     lambda r: "42" in (r["choices"][0]["message"]["content"] or "")
                     and bool(r["choices"][0]["message"].get("reasoning_content"))),
        "tool": ({"messages": [{"role": "user", "content": "What's the weather in Paris?"}], "tools": tools,
                  "tool_choice": "required", "temperature": 0, "max_tokens": 512, "reasoning_effort": "none"},
                 lambda r: r["choices"][0]["finish_reason"] == "tool_calls" and json.loads(
                     r["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]).get("city")),
        "json": ({"messages": [{"role": "user", "content": "Give a JSON object with key ok set to true."}],
                  "response_format": {"type": "json_schema", "json_schema": {"name": "c", "schema": {
                      "type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}}},
                  "temperature": 0, "max_tokens": 64, "reasoning_effort": "none"},
                 lambda r: json.loads(r["choices"][0]["message"]["content"])["ok"] in (True, False)),
    }
    bad = 0
    for name, (body, ok) in probes.items():
        t0 = time.time()
        try:
            r = post(a.base, "/v1/chat/completions", dict(body, model=a.model))
            passed = bool(ok(r))
        except Exception as exc:                # noqa: BLE001
            passed, r = False, {"error": str(exc)}
        usage = r.get("usage", {})
        print(f"[canary] {name}: {'ok' if passed else 'FAILED'} ({time.time() - t0:.1f} s, "
              f"{usage.get('completion_tokens', '?')} tokens)" + ("" if passed else f": {json.dumps(r)[:400]}"))
        bad += not passed
    try:
        t = post(a.base, "/tokenize", {"messages": [{"role": "user", "content": "hi"}]})
        assert t["tokens"][0] == 0 and t["count"] == len(t["tokens"])
        print("[canary] tokenize: ok")
    except Exception as exc:                    # noqa: BLE001
        print(f"[canary] tokenize: FAILED: {exc}")
        bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(cli(main))
