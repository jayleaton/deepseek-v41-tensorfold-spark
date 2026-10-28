#!/usr/bin/env python3
"""Memory stress / cold-boot admission client (stdlib only) for the native `tensorfold-dsv41 serve` endpoint (docs/ZIG-SERVE.md).

  stress.py stress --base URL --long 299000 --dctx 65536 --decode 2048 --out stress.json
      one stream prefills ``--long`` tokens (the 300K prefill) while three streams prefill ``--dctx`` tokens and then
      decode ``--decode`` tokens (``ignore_eos`` + ``min_tokens`` = ``max_tokens``: the decode really runs instead of
      stopping at EOS); the streams start ``--stagger`` s apart. Token ids are random text-range ids (no specials), a
      different seed a stream (no prefix sharing). Reports each stream's first-token / total seconds, completion
      tokens (the server's usage when it sends one, else streamed chunks), its decode tok/s, and its decode tok/s
      WHILE the long stream was still prefilling (chunk times before the long stream's first token), and errors.
      Exit 1 on an error or (``--require-full``) a decode stream short of ``--decode`` tokens.
  stress.py admit --base URL --tokens 512 --max-wait 10
      one request; the seconds to its first token (admission + prefill); exit 1 if over ``--max-wait``.

The pool for every slot is allocated at boot (slots x (context + 64)), so the boot already holds 4 x 300K of KV;
the stress adds what grows with a live long context (the indexer's selection blocks, the session tier, graphs).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
import threading
import time
import urllib.request


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from safe_output import cli, dump as safe_dump, safe_print as print  # noqa: E402


def _model(base: str) -> str:
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
        return json.loads(r.read())["data"][0]["id"]


def _ids(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [0] + [rng.randrange(1000, 120_000) for _ in range(n - 1)]


def _stream(base: str, model: str, ids: list[int], max_tokens: int, out: dict, timeout: float,
            t_ref: float | None = None) -> None:
    body = {"model": model, "prompt": ids, "max_tokens": max_tokens, "min_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    t_ref = time.time() if t_ref is None else t_ref
    out["start_s"] = round(time.time() - t_ref, 2)
    times: list[float] = []
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    out.update(prompt=len(ids), max_tokens=max_tokens, first_token_s=None, tokens=0)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(line[5:])
                except ValueError:
                    chunk = {}
                if chunk.get("usage"):
                    out["usage_completion"] = chunk["usage"].get("completion_tokens")
                if not chunk.get("choices"):
                    continue
                if out["first_token_s"] is None:
                    out["first_token_s"] = round(time.time() - t0, 2)
                out["tokens"] += 1
                times.append(round(time.time() - t_ref, 3))
        out["total_s"] = round(time.time() - t0, 2)
        out["chunks"] = out["tokens"]
        if out.get("usage_completion"):
            out["tokens"] = int(out["usage_completion"])
        out["chunk_times"] = times
        if not out["tokens"]:
            out["error"] = "no streamed tokens (does the server stream /v1/completions?)"
        ft = out["first_token_s"] or out["total_s"]
        out["decode_tok_s"] = round(out["tokens"] / max(out["total_s"] - ft, 1e-6), 2) if out["tokens"] > 1 else None
    except Exception as e:                                   # noqa: BLE001 - reported
        out["error"] = f"{type(e).__name__}: {e}"[:300]
        out["total_s"] = round(time.time() - t0, 2)


def _during(r: dict, end_s: float | None) -> dict:
    """Decode tok/s of a stream while the long stream prefilled: its chunks after its own first one, before
    ``end_s`` (the long stream's first token, seconds from the shared start), scaled from chunks to tokens."""

    times = r.get("chunk_times") or []
    if end_s is None or len(times) < 2:
        return {}
    inside = [t for t in times[1:] if t <= end_s]
    span = min(end_s, times[-1]) - times[0]
    if not inside or span <= 0:
        return {"during_prefill_tokens": 0}
    scale = r["tokens"] / max(r.get("chunks") or len(times), 1)
    return {"during_prefill_tokens": round(len(inside) * scale), "during_prefill_s": round(span, 2),
            "during_prefill_tok_s": round(len(inside) * scale / span, 2)}


def stress(a) -> int:
    model = _model(a.base)
    plan = [(_ids(a.long, 1), a.long_decode)] + [(_ids(a.dctx, 2 + i), a.decode) for i in range(3)]
    res = [dict(role="long" if i == 0 else "decode") for i in range(4)]
    t0 = time.time()
    th = [threading.Thread(target=_stream, args=(a.base, model, ids, mt, res[i], a.timeout, t0), daemon=True)
          for i, (ids, mt) in enumerate(plan)]
    for t in th:
        t.start()
        time.sleep(a.stagger)
    for t in th:
        t.join()
    long = res[0]
    long_ft = None if long.get("first_token_s") is None else long["start_s"] + long["first_token_s"]
    for r in res[1:]:
        r.update(_during(r, long_ft))
    full = all(r.get("tokens", 0) >= a.decode for r in res[1:])
    rep = {"model": model, "long": a.long, "dctx": a.dctx, "decode": a.decode, "wall_s": round(time.time() - t0, 1),
           "long_first_token_s": long_ft, "decode_full": full, "streams": res,
           "ok": all("error" not in r for r in res) and (full or not a.require_full)}
    brief = {k: v for k, v in rep.items() if k != "streams"}
    brief["streams"] = [{k: v for k, v in r.items() if k != "chunk_times"} for r in res]
    print(json.dumps(brief), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            safe_dump(rep, f, indent=1)
    return 0 if rep["ok"] else 1


def admit(a) -> int:
    res: dict = {}
    _stream(a.base, _model(a.base), _ids(a.tokens, 9), 8, res, a.timeout)
    res["pass"] = "error" not in res and (res.get("first_token_s") or 1e9) <= a.max_wait
    print(json.dumps(res), flush=True)
    return 0 if res["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stress")
    s.add_argument("--base", default="http://127.0.0.1:8000")
    s.add_argument("--long", type=int, default=299_000)
    s.add_argument("--dctx", type=int, default=65_536)
    s.add_argument("--decode", type=int, default=2048)
    s.add_argument("--long-decode", type=int, default=16, help="max_tokens of the long stream")
    s.add_argument("--stagger", type=float, default=1.0, help="seconds between the four submits")
    s.add_argument("--require-full", action="store_true", help="exit 1 when a decode stream is short of --decode")
    s.add_argument("--timeout", type=float, default=7200)
    s.add_argument("--out", default="")
    m = sub.add_parser("admit")
    m.add_argument("--base", default="http://127.0.0.1:8000")
    m.add_argument("--tokens", type=int, default=512)
    m.add_argument("--max-wait", type=float, default=10.0)
    m.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args(argv)
    return {"stress": stress, "admit": admit}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(cli(main))
