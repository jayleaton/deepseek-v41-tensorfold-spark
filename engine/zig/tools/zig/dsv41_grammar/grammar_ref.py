#!/usr/bin/env python3
"""DeepSeek-V4.1 structured-output reference (gate): the Python engine at TP=2 on two GPUs of one host
(as dsv41_samp_ref.py loads it, or on dsv41_m2b_ref.py's own load with --grammar), each case a prompt from a fresh
slot then serial decoding under prod's grammar, every token chosen as the served engine chooses it:

  prod's ``structured.Dsv41Grammars`` (GLM 0610 ``grammar.Constraint`` over xgrammar 0.2.8) cuts the window (one row:
  the pending token) and fills its row's mask; ``SlotsForward.cand_gather(masks=...)`` sets the disallowed columns
  -inf before each rank's top-k; ``nucleus.finish(..., masks)`` and ``Batcher._choose`` pick (greedy: the first
  candidate); the constraint follows the chosen token. The prompt's last row gives the first token under the
  grammar's start (Python's first window row 0).

Constrained drafted replies equal serial ones (Python's exactness argument), so this is the reference for the Zig
engine's drafted and undrafted replies. Rank 0 writes DIR/trace.jsonl in ``tf-dsv41-m1 generate``'s format, with each
case's grammar: {"prompt", "tokens", "sampling"?, "structure": {"kind", "text"}}.

    python dsv41_grammar/grammar_ref.py --pack PACK --out DIR [--layers 0-24] [--steps 48] [--port 29651]

or ``grammar_ref.run_cases(fw, cfg, rank, out_dir, pack, steps)`` from a reference that holds the model (both ranks).
Needs xgrammar==0.2.8 (prod's image has it). TF_DSV41_EXPERT_TOPP and the forward's knobs come from the environment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

WEATHER = {"type": "object", "properties": {"city": {"type": "string", "maxLength": 24},
                                            "unit": {"enum": ["celsius", "fahrenheit"]},
                                            "days": {"type": "integer", "minimum": 1, "maximum": 7}},
           "required": ["city", "unit"], "additionalProperties": False}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather by city", "parameters": WEATHER}},
         {"type": "function", "function": {"name": "note", "parameters": {"type": "object", "properties": {
             "text": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
             "required": ["text"]}}}]
STRICT = [dict(TOOLS[0], function=dict(TOOLS[0]["function"], strict=True)), TOOLS[1]]

# (name, request body fields, sampling (seed, T, top_k, top_p, min_p) or None: greedy, prompt length)
CASES = [
    ("json_schema_greedy", {"response_format": {"type": "json_schema", "json_schema": {"name": "w", "schema": WEATHER}}}, None, 37),
    ("json_schema_sampled", {"response_format": {"type": "json_schema", "json_schema": {"name": "w", "schema": WEATHER}}}, (77, 1.0, 0, 0.95, 0.0), 50),
    ("tools_required_greedy", {"tools": TOOLS, "tool_choice": "required"}, None, 66),
    ("tools_named_sampled", {"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "note"}}}, (4101, 0.7, 0, 0.9, 0.0), 83),
    ("tools_strict_greedy", {"tools": STRICT}, None, 128),
    ("json_object_sampled", {"response_format": {"type": "json_object"}}, ((1 << 63) + 12345, 1.3, 20, 0.95, 0.05), 292),
]


def run_cases(fw, cfg, rank: int, out: Path, pack: Path, steps: int, seed: int = 4101, log=print) -> None:
    """Each case from a fresh slot, decoded serially under its grammar; rank 0 writes out/trace.jsonl. Both ranks call
    it (each compiles the grammar and walks the same tokens, as prod's two ranks do; the gathers pair up)."""

    import numpy as np
    import torch

    from tensorfold.engine.exact_sampling import MARGIN, Sampling
    from tensorfold.families.deepseek_v41.cuda import nucleus as NU
    from tensorfold.families.deepseek_v41.cuda import structured
    from tensorfold.families.deepseek_v41.cuda.batch import Batcher
    from tensorfold.families.deepseek_v41.cuda.forward import Seg
    from tensorfold.families.deepseek_v41.cuda.slots import SlotsForward
    t0 = time.time()
    for name in ("candidates", "cand_gather", "cand_fetch"):
        setattr(fw, name, types.MethodType(getattr(SlotsForward, name), fw))
    fw.cand_wait = SlotsForward.cand_wait
    nucleus = NU.candidates()
    grammars = structured.Dsv41Grammars.from_model(pack, cfg.vocab_size, (int(cfg.eos_token_id),))
    host = structured.Host.__new__(structured.Host)

    def choose(lg, positions, s, bits):
        """``Batcher._decode``'s count, then ``_sample``'s path for one masked row (both ranks)."""
        masks = [bits] if bits is not None else None
        if s is None:
            c = fw.candidates(lg, [lg.shape[0]], 1 + MARGIN, masks)
            return int(c.ids[0, 0])
        nuc = NU.spec(s) if nucleus else []
        k = int(s.top_k) + MARGIN if s.top_k else (nucleus if nuc else fw.vocab)
        c = fw.candidates(lg, [lg.shape[0]], min(k, fw.vocab), masks)
        if nuc:
            c = NU.finish(fw, lg, c, [lg.shape[0]], [nuc], masks)
        return Batcher._choose(None, c, positions, s)[0]

    lines = []
    for n, (name, body, smp, plen) in enumerate(CASES):
        spec = host.spec(body)
        bound = grammars.bind(spec, grammars.compile(spec), [])     # random prompts: no think block, active at once
        con = grammars.constraint(bound)
        fw.slots[0] = fw.new_slot(contiguous=True)
        g = torch.Generator().manual_seed(seed + n)
        ids = torch.randint(3, 128000, (plen,), generator=g).tolist()
        s = Sampling(*smp) if smp is not None else None

        def mask():
            win = con.fill(con.cut([0]))
            if not win.rows:
                return None
            full = np.full((1, con.words), -1, dtype=np.int32)
            full[0] = win.bits[0]
            return full

        logits = fw.prompt(ids)
        tok = choose(logits[-1:].float(), [len(ids)], s, mask())
        tokens = [tok]
        con.advance([tok])
        for _ in range(steps - 1):
            start = fw.slot.pos
            lg = fw.run([Seg(0, start, (tok,))])
            fw.keep(0, 0)
            tok = choose(lg.float(), [start + 1], s, mask())
            tokens.append(tok)
            con.advance([tok])
        line = {"prompt": ids, "tokens": tokens, "structure": {"kind": spec.kind, "text": spec.text}, "case": name,
                "finished": bool(con.finished)}
        if smp is not None:
            line["sampling"] = dict(zip(("seed", "temperature", "top_k", "top_p", "min_p"), smp))
        lines.append(line)
        if rank == 0:
            log(f"grammar {name}: prompt {plen}, {len(tokens)} tokens, grammar finished {con.finished}, "
                f"{time.time() - t0:.0f} s", flush=True)
    fw.slots[0] = fw.new_slot(contiguous=True)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "trace.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
        log(f"rank 0: {len(lines)} constrained replies into {out / 'trace.jsonl'}", flush=True)


def rank_main(rank: int, a) -> None:
    import torch

    torch.cuda.set_device(rank)
    from dsv41_m1_capture import IndexRows
    from tensorfold.cuda.comm import open_comm
    from tensorfold.families.deepseek_v41.cuda import weights as W
    from tensorfold.families.deepseek_v41.cuda.config import Config
    from tensorfold.families.deepseek_v41.cuda.csa2.backend import Triton
    from tensorfold.families.deepseek_v41.cuda.engram_host import NgramHasher, token_map
    from tensorfold.families.deepseek_v41.cuda.forward import Forward

    t0 = time.time()
    dev = torch.device(f"cuda:{rank}")
    comm = open_comm(rank, 2, "localhost", a.port)
    pack = Path(a.pack)
    cfg = Config.from_file(pack / "config.json")
    tmap, _ = token_map(pack / "tokenizer.json")
    src = W.Shards(pack)
    loader = W.Loader(cfg, src, rank, 2, dev)
    vh = cfg.vocab_size // 2
    embed = loader._dev(src.rows("embed.weight", rank * vh, (rank + 1) * vh).to(torch.bfloat16))
    head, norm = loader.x3("head", "col"), loader.native("norm.weight")
    lo, hi = (int(v) for v in a.layers.split("-"))
    layers = [loader.layer(L) for L in range(lo, hi + 1)]
    tree = W.RankW(rank, 2, rank * vh, embed, layers, norm, head, None, {})
    rows = {L: IndexRows(cfg.engram_head_dim) for L in cfg.engram_layer_ids}
    fw = Forward(cfg, tree, csa2=Triton(128, cfg.num_attention_heads // 2, dev), comm=comm, device=dev,
                 limit=a.limit, chunk=128, engram_rows=rows, hasher=NgramHasher(cfg, tmap), slots=1,
                 contiguous=True, pchunk=2048)
    if rank == 0:
        print(f"loaded layers {lo}-{hi} in {time.time() - t0:.0f} s", flush=True)
    run_cases(fw, cfg, rank, Path(a.out), pack, a.steps, a.seed)
    comm.barrier()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-24")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--limit", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=4101, help="the prompts' ids")
    ap.add_argument("--port", type=int, default=29651)
    a = ap.parse_args()
    import torch.multiprocessing as mp

    mp.spawn(rank_main, args=(a,), nprocs=2, join=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
