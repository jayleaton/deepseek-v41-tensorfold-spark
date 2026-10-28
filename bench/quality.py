#!/usr/bin/env python3
"""Answer quality THROUGH THE SERVING PATH (a running server), so CED replay is really exercised.

CED replay (TF_DSV41_PREFILL=replay, engine cuda/replay.py) runs the encoder layers over the whole prompt and the
decoder layers only over the prompt's last 128 rows; for n <= 128 it equals the full prefill bit for bit. So a gate
whose prompts are short, or that is teacher-forced outside the server (gate.py), says nothing about replay. Here:

  quality.py mmlu --base URL --model M --data bench/data/mmlu200.jsonl --out R.json [--shots 0]
      MMLU-200 the way the kit's baseline was scored (thinking off, greedy, max_tokens 8, the reply's
      first A-D letter; the kit scored 87.5%). Records each question's prompt_tokens (replay differs from full only
      above 128).
  quality.py mmlu ... --shots 20
      the same questions after ONE fixed preamble of the first N questions with their answers (~2-3K tokens): every
      prompt is far past the 128-row replay window, so the decoder sees only the question; scored on the other 200-N.
  quality.py needle --base URL --model M --sizes 32768,131072 --out N.json
      a passkey at 1/3 depth of a filler document of about each size (sized with the server's /tokenize).
  quality.py compare --replay R.json --full F.json [--kit 0.875] [--within 1.0]
      accuracy of both, the replay - full delta in points, per-question agreement, vs the kit; exit 1 on a FAIL.

Standard library only (runs on the head, outside the image).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from safe_output import cli, safe_print as print, write_text as safe_write_text  # noqa: E402


LETTERS = "ABCD"
WORDS = ("river mountain lantern copper harbor meadow quartz signal orchard canvas thunder velvet compass ember "
         "glacier timber falcon prairie marble beacon willow cobalt summit ferry saddle tundra pepper violet "
         "anchor bramble cinder delta fjord granite hollow ivory jasmine kettle lagoon mosaic nectar oyster").split()


def post(base: str, path: str, body: dict, timeout: float = 1800) -> dict:
    req = urllib.request.Request(base.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(a, content: str, max_tokens: int) -> tuple[str, int]:
    body = {"model": a.model, "messages": [{"role": "user", "content": content}], "temperature": 0,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    r = post(a.base, "/v1/chat/completions", body)
    return r["choices"][0]["message"].get("content") or "", int(r.get("usage", {}).get("prompt_tokens") or 0)


def question(q: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(q["choices"]))
    return (f"The following is a multiple choice question about {q['subject'].replace('_', ' ')}.\n\n"
            f"{q['question']}\n{opts}\n\nAnswer with the letter of the correct option only.")


def mmlu(a) -> int:
    rows = [json.loads(line) for line in Path(a.data).read_text().splitlines() if line.strip()]
    shots, rest = rows[:a.shots], rows[a.shots:]
    pre = ""
    if shots:
        pre = ("Here are worked examples of multiple choice questions with their answers.\n\n" +
               "\n\n".join(question(q) + f"\nAnswer: {LETTERS[q['answer']]}" for q in shots) +
               "\n\nNow the real question.\n\n")
    t0, correct, items = time.time(), 0, []
    for i, q in enumerate(rest):
        try:
            reply, n = chat(a, pre + question(q), 8)
            m = re.search(r"\b([ABCD])\b", reply)
            got = LETTERS.index(m.group(1)) if m else -1
        except Exception as exc:                    # noqa: BLE001 - recorded, counted wrong
            reply, n, got = f"ERROR {type(exc).__name__}: {exc}"[:200], 0, -2
        correct += got == q["answer"]
        items.append({"i": i + a.shots, "answer": q["answer"], "got": got, "prompt_tokens": n, "reply": reply[:40]})
    acc = correct / max(len(rest), 1)
    rec = {"label": a.label, "model": a.model, "shots": a.shots, "accuracy": acc, "correct": correct, "n": len(rest),
           "errors": sum(x["got"] == -2 for x in items),
           "over_replay_window": sum(x["prompt_tokens"] > 128 for x in items),
           "prompt_tokens_median": sorted(x["prompt_tokens"] for x in items)[len(items) // 2] if items else 0,
           "wall_s": round(time.time() - t0, 1), "items": items}
    print(f"[quality] mmlu {a.label} shots {a.shots}: {acc:.3f} ({correct}/{len(rest)}), errors {rec['errors']}, "
          f"prompts over 128 tokens {rec['over_replay_window']}/{len(rest)}, median {rec['prompt_tokens_median']}",
          flush=True)
    safe_write_text(a.out, json.dumps(rec, indent=1))
    return 0


def filler(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    out = []
    for s in range(max(n_words // 12, 1)):
        w = [rng.choice(WORDS) for _ in range(11)]
        out.append(f"Record {s}: the {w[0]} near the {w[1]} kept its {w[2]} {w[3]}, and the {w[4]} {w[5]} "
                   f"{w[6]} {w[7]} {w[8]} {w[9]} {w[10]}.")
    return " ".join(out)


def count(a, text: str) -> int | None:
    try:
        r = post(a.base, "/tokenize", {"model": a.model, "messages": [{"role": "user", "content": text}]}, 600)
        return int(r.get("count") or len(r["tokens"]))
    except Exception:                               # noqa: BLE001 - sized by estimate then
        return None


def needle(a) -> int:
    res, ok = [], True
    for size in [int(x) for x in a.sizes.split(",") if x]:
        key = str(random.Random(size).randrange(10_000_000, 99_999_999))
        words = int(size * 0.62)
        for _ in range(2):                          # size the document with /tokenize (one correction)
            doc = filler(words, size)
            got = count(a, doc)
            if got is None or abs(got - size) < size * 0.03:
                break
            words = int(words * size / got)
        cut = len(doc) // 3
        doc = doc[:cut] + f" The secret passkey is {key}. Remember it. " + doc[cut:]
        ask = doc + "\n\nWhat is the secret passkey stated in the text above? Answer with the number only."
        t0 = time.time()
        try:
            reply, n = chat(a, ask, 32)
        except Exception as exc:                    # noqa: BLE001
            reply, n = f"ERROR {type(exc).__name__}: {exc}"[:200], 0
        found = key in reply
        ok &= found
        res.append({"size": size, "prompt_tokens": n, "found": found, "reply": reply[:80],
                    "seconds": round(time.time() - t0, 1)})
        print(f"[quality] needle {a.label} {size}: prompt {n} tokens, found {found} ({res[-1]['seconds']} s)", flush=True)
    safe_write_text(a.out, json.dumps({"label": a.label, "needles": res, "ok": ok}, indent=1))
    return 0 if ok else 1


def compare(a) -> int:
    r, f = json.loads(Path(a.replay).read_text()), json.loads(Path(a.full).read_text())
    by = {x["i"]: x["got"] for x in f["items"]}
    same = sum(by.get(x["i"]) == x["got"] for x in r["items"])
    delta = 100 * (r["accuracy"] - f["accuracy"])
    lines = [f"replay {100 * r['accuracy']:.1f}% vs full {100 * f['accuracy']:.1f}% (n {r['n']}, shots {r['shots']}): "
             f"delta {delta:+.1f} points, same answer {same}/{len(r['items'])}"]
    ok = abs(delta) <= a.within and not r.get("errors") and not f.get("errors")
    lines.append(f"PASS replay within {a.within} point of full: {ok}")
    if a.kit:
        for tag, x in (("replay", r), ("full", f)):
            d = 100 * (x["accuracy"] - a.kit)
            good = d >= -a.within
            ok &= good
            lines.append(f"PASS {tag} within {a.within} point of the kit's {100 * a.kit:.1f}% (or above): {good} "
                         f"({d:+.1f})")
    print("\n".join(lines))
    if a.out:
        safe_write_text(a.out, json.dumps({"delta_points": delta, "same": same, "n": len(r["items"]), "ok": ok,
                                           "replay": r["accuracy"], "full": f["accuracy"], "kit": a.kit}, indent=1))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("mmlu", "needle"):
        p = sub.add_parser(name)
        p.add_argument("--base", default="http://127.0.0.1:8000")
        p.add_argument("--model", default="DeepSeek-V4.1-Flash-TF")
        p.add_argument("--label", default="")
        p.add_argument("--out", required=True)
    sub.choices["mmlu"].add_argument("--data", required=True)
    sub.choices["mmlu"].add_argument("--shots", type=int, default=0)
    sub.choices["needle"].add_argument("--sizes", default="32768,131072")
    c = sub.add_parser("compare")
    c.add_argument("--replay", required=True)
    c.add_argument("--full", required=True)
    c.add_argument("--kit", type=float, default=0.0)
    c.add_argument("--within", type=float, default=1.0)
    c.add_argument("--out", default="")
    a = ap.parse_args(argv)
    return {"mmlu": mmlu, "needle": needle, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(cli(main))
