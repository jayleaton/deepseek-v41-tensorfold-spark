"""Structured-output requests to tensorfold-dsv41 (see requests.sh): six chat requests (response_format json_schema /
json_object, tool_choice required / named, strict tools; greedy and seeded sampling, thinking off so the grammar holds
from the first token, 96 tokens), each reply checked against its grammar. Prints "PASS grammar http: N/N replies
follow their grammar" or FAIL with the reason a reply broke it. Stdlib only."""

import argparse
import json
import sys
import urllib.request

WEATHER = {"type": "object", "properties": {"city": {"type": "string", "maxLength": 24},
                                            "unit": {"enum": ["celsius", "fahrenheit"]},
                                            "days": {"type": "integer", "minimum": 1, "maximum": 7}},
           "required": ["city", "unit"], "additionalProperties": False}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather by city", "parameters": WEATHER}},
         {"type": "function", "function": {"name": "note", "parameters": {"type": "object", "properties": {
             "text": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
             "required": ["text"]}}}]
STRICT = [dict(TOOLS[0], function=dict(TOOLS[0]["function"], strict=True)), TOOLS[1]]
ASK = [{"role": "user", "content": "What's the weather in Paris for the next 3 days? Answer briefly."}]
CASES = [
    ("json_schema_greedy", {"response_format": {"type": "json_schema", "json_schema": {"name": "w", "schema": WEATHER}}}, {}),
    ("json_schema_sampled", {"response_format": {"type": "json_schema", "json_schema": {"name": "w", "schema": WEATHER}}},
     {"temperature": 1.0, "top_p": 0.95, "seed": 77}),
    ("tools_required_greedy", {"tools": TOOLS, "tool_choice": "required"}, {}),
    ("tools_named_sampled", {"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "note"}}},
     {"temperature": 0.7, "top_p": 0.9, "seed": 4101}),
    ("tools_strict_greedy", {"tools": STRICT}, {}),
    ("json_object_sampled", {"response_format": {"type": "json_object"}}, {"temperature": 1.3, "top_k": 20, "seed": 9}),
]


def schema_ok(v, s) -> str | None:
    """A small validator for the schemas above (types, enum, required, extra keys, bounds)."""
    if s is True or s == {}:
        return None
    if "enum" in s:
        return None if v in s["enum"] else f"{v!r} not in enum"
    t = s.get("type")
    if t == "object":
        if not isinstance(v, dict):
            return f"{v!r} is not an object"
        for k in s.get("required", []):
            if k not in v:
                return f"missing {k}"
        props = s.get("properties", {})
        if s.get("additionalProperties") is False and set(v) - set(props):
            return f"extra keys {set(v) - set(props)}"
        for k, x in v.items():
            if k in props and (why := schema_ok(x, props[k])):
                return f"{k}: {why}"
        return None
    if t == "string":
        return None if isinstance(v, str) and len(v) <= s.get("maxLength", 1 << 30) else f"{v!r} is not a short string"
    if t == "integer":
        ok = isinstance(v, int) and not isinstance(v, bool) and s.get("minimum", v) <= v <= s.get("maximum", v)
        return None if ok else f"{v!r} out of range"
    if t == "array":
        if not isinstance(v, list) or len(v) > s.get("maxItems", 1 << 30):
            return f"{v!r} is not a short array"
        return next((w for x in v if (w := schema_ok(x, s.get("items", {})))), None)
    return None


def check(name, extra, msg) -> str | None:
    calls = msg.get("tool_calls") or []
    if "response_format" in extra:
        try:
            v = json.loads(msg.get("content") or "")
        except json.JSONDecodeError as exc:
            return f"content is not JSON ({exc.msg}): {msg.get('content')!r:.200}"
        if extra["response_format"]["type"] == "json_object":
            return None if isinstance(v, dict) else "not a JSON object"
        return schema_ok(v, WEATHER)
    if not calls:
        # strict tools under auto may answer in text; required / named must call
        return None if extra.get("tool_choice") is None else "no tool call"
    names = {t["function"]["name"]: t["function"].get("parameters", True) for t in extra["tools"]}
    want = extra.get("tool_choice")
    for c in calls:
        fn = c["function"]
        if fn["name"] not in names:
            return f"call of {fn['name']!r}, not an offered tool"
        if isinstance(want, dict) and fn["name"] != want["function"]["name"]:
            return f"call of {fn['name']!r}, not the named tool"
        try:
            args = json.loads(fn["arguments"])
        except json.JSONDecodeError:
            return f"arguments are not JSON: {fn['arguments']!r:.200}"
        if why := schema_ok(args, names[fn["name"]]):
            return f"{fn['name']} arguments: {why}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=96)
    a = ap.parse_args()
    good = 0
    for name, extra, smp in CASES:
        body = {"model": "dsv41", "messages": ASK, "max_tokens": a.tokens, "chat_template_kwargs": {"enable_thinking": False},
                **extra, **smp}
        req = urllib.request.Request(f"http://localhost:{a.port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                reply = json.load(r)
        except Exception as exc:                       # noqa: BLE001
            print(f"FAIL grammar http {name}: {exc}")
            continue
        with open(f"{a.out}/{name}.json", "w") as f:
            json.dump(reply, f, ensure_ascii=False, indent=1)
        msg = reply["choices"][0]["message"]
        # a reply cut at max_tokens is a grammar's prefix (the pod's backbone prefix may not close it): not checked
        cut = reply["choices"][0].get("finish_reason") == "length"
        why = None if cut else check(name, extra, msg)
        print(f"{name}: finish {reply['choices'][0].get('finish_reason')}, "
              f"{('cut at max_tokens' if cut else 'ok') if why is None else 'BROKEN: ' + why}")
        good += why is None
    ok = good == len(CASES)
    print(f"{'PASS' if ok else 'FAIL'} grammar http: {good}/{len(CASES)} replies follow their grammar")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
