#!/usr/bin/env python3
"""DeepSeek-V4.1 M2b reference: the Python engine at TP=2 on two GPUs of one host (NCCL, tensorfold.cuda.comm), the
backbone's layers 0..L and the head, Engram rows a function of the row index (dsv41_m1_capture.IndexRows, as M2a): a
prompt (Python's own prefill), each rank's slot state dumped under our forward's role names, then greedy decoding one
token a window. Recorded: every token (the first is the prompt's greedy pick), each rank's logits digest a step, each
rank's per-block stream digests of the first decode window, the run's Triton variants (rank 0, the decode windows) and
this host's rope tables.

    python dsv41_m2b_ref.py --pack PACK --out DIR [--layers 0-27] [--prompt 1536] [--steps 64] [--port 29611]
Writes DIR/ref.json, DIR/rank<r>/state/*, DIR/rope.bin, DIR/aot/.

``--prefill replay`` (with ``--prompt-tail verify``): the prompt as prod prefills it under ``TF_DSV41_PREFILL=replay``
(CED bounded replay, ``replay.py``): the encoder pass over [0, n - 1), the decoder over the prefilled tail from the
stash, the last token a verify row; ``--aot-prompts`` then run that way too (the decoder pass's Triton variants).
ref.json records ``"prefill"``; the Zig gate (``M2B_PREFILL``) needs ``TF_DSV41_PREFILL=replay`` on its ranks.

``--prompts 2060,2049,...`` (both ranks on this host): one load, each prompt from a fresh slot (Forward.new_slot, the
stores zeroed as a new Forward's) into DIR/p<N>/, a whole reference dir each (the AOT set and rope tables of every
prompt's kernels in each).

M5 (the Python twin with split KV + sessions, tensorfold-decode1 dsv41-midprofile):

- ``--pool``: the forward's slot is a KV pool slot (``slots.SlotsForward`` + ``dsv41_m5_kv.KV``, mode "pool"), bound as
  serving binds it: the compressed rows and index keys read and written through the slot's page table (a page holder
  interleaves its pages, so the table is not the identity; ``TF_DSV41_INDEX_KV`` picks the index family's rows);
- ``--split``: ``TF_DSV41_KV_SPLIT=1`` semantics (set for both ranks): the pool split over the two ranks
  (``pool.Pool(split=2, rank=r)``), each rank's comp families its own pages, ``kvsplit.Exchange`` trading the selected
  rows (dense in windows and short segments, union in 2,048-row prefill segments); ``--compact force`` (or the env
  ``TF_DSV41_KV_SPLIT_COMPACT``): the packed exchange (``kvsplit_pack.pack2``, the variable gather);
- ``--park``: after the prompt (and its state dump), the session saved at the prompt's end (``sessions.Store.save``),
  parked to an NVMe dir (DIR/p<N>/park/, the ``sessdisk`` tier, ``TF_DSV41_SESSION_*`` as set), then the steps decoded
  uninterrupted (the recorded tokens); then the slot emptied, the entry found and restored (the holder takes a page
  first: other physical pages), ``fw.restore``, and the same steps decoded again: rank<r>.json "park" holds those
  tokens / logits digests and "equal" (== the uninterrupted run's, the gate); ``--park`` implies ``--pool`` unless
  ``--split``.
- In these modes the AOT set is recorded on both ranks and merged (rank 1's Triton variants: ``pack2`` specializes on
  the rank), and the state dump (rank<r>/state/) replaces ``s.L<i>.comp.v / .s / .ik`` with the pool's roles (the
  rings ``s.L<i>.swa.v / .s``, the carries ``s.L<i>.carry`` and state.json's pos / tail stay as they are):

  | role (file <role>.bin)  | what                                                                                     |
  | --- | --- |
  | ``s.kv.comp.L<i>``      | kv source i's ``comp`` family tensor as stored: uint8 [rows, 584] (576 value bytes, 8 scale bytes a row); rows = pages x 256 / ratio: replicated (npages + 1 null page), split this rank's local pages (npages / W, then the local null page, then the discard page) |
  | ``s.kv.ik.L<i>``        | kv source i's ``index_k`` family as stored, whole on every rank: uint8 [rows, 132] under ``TF_DSV41_INDEX_KV=fp8``, bf16 [rows, 128] otherwise (state.json kv.families: dtype, shape) |
  | ``s.kv.pt``             | the slot's page table, int32 [pages] (the 1-D row PT the kernels get; unmapped = the null page) |
  | ``s.kv.ct``             | split only: the slot's split table (``Slot.ctable``): an owned logical page -> its local page, any other -> the discard page, unmapped -> the local null page; int32 [pages] |
  | ``s.kv.comp.L<i>.rows`` | the comp rows of logical rows [0, pos // ratio) read through the table, in logical order; split: only the rows this rank owns (logical page k with k % W == rank) through ``s.kv.ct``, which ``s.kv.comp.L<i>.rows.json`` lists ("comp_logical": [a, b) runs) |
  | ``s.kv.ik.L<i>.rows``   | the index keys of logical rows [0, pos // ratio) through ``s.kv.pt`` (every row, every rank) |

  state.json gains "kv" (``dsv41_m5_kv.KV.dump``: mode, page, split, rank, index_kv, the pool's page counts, null /
  local null / discard pages, the slot's physical pages in logical order (and split: their local pages), each
  family's dtype / shape / row bytes, each kv source's shift and row lists) and "shapes" (role -> dtype, shape).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(t) -> str:
    import torch

    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def rank_main(rank: int, a) -> None:
    sys.path.insert(0, str(HERE))
    import shutil

    import torch

    dev_index = rank if a.rank < 0 else 0     # two hosts: each rank on its own GPU 0
    torch.cuda.set_device(dev_index)
    from dsv41_m1_capture import IndexRows, pack_aot, specialization
    from tensorfold.cuda.comm import open_comm
    from tensorfold.families.deepseek_v41.cuda import rope as RO
    from tensorfold.families.deepseek_v41.cuda import weights as W
    from tensorfold.families.deepseek_v41.cuda.config import Config
    from tensorfold.families.deepseek_v41.cuda.csa2.backend import Triton
    from tensorfold.families.deepseek_v41.cuda.engram_host import NgramHasher, token_map
    from tensorfold.families.deepseek_v41.cuda.forward import Forward, Seg, greedy

    t0 = time.time()
    out = Path(a.out)
    dev = torch.device(f"cuda:{dev_index}")
    kvmode = a.kv
    comm = open_comm(rank, 2, a.master, a.port)
    pack = Path(a.pack)
    cfg = Config.from_file(pack / "config.json")
    tmap, _ = token_map(pack / "tokenizer.json")
    src = W.Shards(pack)
    loader = W.Loader(cfg, src, rank, 2, dev)
    vh = cfg.vocab_size // 2
    embed = loader._dev(src.rows("embed.weight", rank * vh, (rank + 1) * vh).to(torch.bfloat16))
    head, norm = loader.x3("head", "col"), loader.native("norm.weight")
    lo, hi = (int(v) for v in a.layers.split("-"))
    layers = []
    for L in range(lo, hi + 1):
        layers.append(loader.layer(L))
        if rank == 0:
            print(f"loaded layer {L} at {time.time() - t0:.0f} s", flush=True)
    tree = W.RankW(rank, 2, rank * vh, embed, layers, norm, head, None, {})
    rows = {L: IndexRows(cfg.engram_head_dim) for L in cfg.engram_layer_ids}
    if a.engram_dir:               # the kit's packed shards (engram_host.ShardRows), as served; --engram-make writes them
        from tensorfold.families.deepseek_v41.cuda.engram_host import open_shards

        if a.engram_make:
            sys.path.insert(0, str(Path(__file__).parent / "dsv41_perf"))
            import engram_shards

            ps = [torch.randint(3, cfg.vocab_size, (n,), generator=torch.Generator().manual_seed(a.seed)).tolist()
                  for n in (a.prompts or [a.prompt])]
            engram_shards.make(a.engram_dir, NgramHasher(cfg, tmap), ps, rank, 2)
        rows = open_shards(a.engram_dir, rank, 2, cfg.engram_layer_ids, dim=cfg.engram_head_dim)
    kv = None
    if kvmode == "contig":
        fw = Forward(cfg, tree, csa2=Triton(128, cfg.num_attention_heads // 2, dev), comm=comm, device=dev,
                     limit=a.limit, chunk=128, engram_rows=rows, hasher=NgramHasher(cfg, tmap),
                     slots=len(a.prompts) if a.batched else 1, contiguous=True, pchunk=2048)
    else:                          # M5: the slot in a (split) KV pool, bound as serving binds it
        import dsv41_m5_kv as M5
        from tensorfold.families.deepseek_v41.cuda.slots import SlotsForward

        fw = SlotsForward(cfg, tree, csa2=Triton(128, cfg.num_attention_heads // 2, dev), comm=comm, device=dev,
                          limit=a.limit, chunk=128, engram_rows=rows, hasher=NgramHasher(cfg, tmap), slots=1,
                          contiguous=False, pchunk=2048)
        kv = M5.KV(fw, pack, kvmode, rank, 2, a.limit, dev, scramble=not a.no_scramble)
    load_s = time.time() - t0
    rec = None
    both = kvmode != "contig" and a.rank < 0     # M5: rank 1's Triton variants too (pack2 specializes on the rank)
    if rank == 0 or a.rank >= 0 or both:   # the AOT set (and rope tables) on each host of a two-host run, the
        import triton_aot_manifest as aot   # prompt's prefill kernels included (Zig's own prefill: M2B_PREFILL)

        rec = aot.Recorder().install()
    dirs = [Path(a.out) / f"p{p}" for p in a.prompts] if a.prompts else [Path(a.out)]
    if a.batched:
        batched(rank, a, fw, cfg, greedy, Seg, digest, dirs, lo, hi, t0, load_s)
    for i, (prompt, out) in enumerate(zip([] if a.batched else (a.prompts or [a.prompt]), dirs)):
        if i > 0:
            if kv is None:
                fw.slots[0] = fw.new_slot(contiguous=True)
            else:
                kv.fresh(fw)
        one(rank, a, fw, cfg, greedy, Seg, digest, prompt, out, lo, hi, t0, load_s, i == len(dirs) - 1, kv)
    # --profile "1 4 8": after the run, windows of each row count on the last prompt's slot: wall time a window (the
    # stream synced; 3 warm + 16 timed, each kept whole) and the CUDA kernel time by name under torch.profiler, into
    # DIR/prof-rank<r>.json (the decode gap's Python side: tf-dsv41-m1 m2b's M2B_PROF_ROWS + TF_DSV41_PROFILE)
    if a.profile:
        prof = {"rows": {}, "kernels": {}}
        tok = 11
        for n in [int(x) for x in a.profile.split()]:
            ts = []
            for i in range(19):
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                fw.run([Seg(0, fw.slot.pos, tuple([tok] * n))])
                torch.cuda.synchronize()
                if i >= 3:
                    ts.append((time.perf_counter() - t1) * 1e3)
                fw.keep(0, n - 1)
            prof["rows"][str(n)] = {"ms_mean": sum(ts) / len(ts), "ms_min": min(ts)}
            from torch.profiler import ProfilerActivity, profile as tprofile

            with tprofile(activities=[ProfilerActivity.CUDA]) as pr:
                for _ in range(8):
                    fw.run([Seg(0, fw.slot.pos, tuple([tok] * n))])
                    fw.keep(0, n - 1)
                torch.cuda.synchronize()
            ks = {}
            for ev in pr.key_averages():
                t = getattr(ev, "device_time_total", None) or getattr(ev, "cuda_time_total", 0)
                if t:
                    ks[ev.key] = {"calls": ev.count / 8, "ms": t / 1e3 / 8}
            prof["kernels"][str(n)] = ks
            if rank == 0:
                print(f"profile: {n}-row windows {prof['rows'][str(n)]['ms_mean']:.3f} ms mean, "
                      f"kernels {sum(v['ms'] for v in ks.values()):.3f} ms a window", flush=True)
        (Path(a.out) / f"prof-rank{rank}.json").write_text(json.dumps(prof, indent=1) + "\n")
    # --profile-prefill N: after the run, the gate's prompt shape (N ids from --seed, as ``one``; served_prompt with
    # --prompt-tail / --prefill) from a fresh slot: once to warm (Triton JIT), once timed (wall, the stream synced),
    # once under torch.profiler (CUDA kernel time by name), into DIR/prof-rank<r>.json's "prefill": the full
    # prefill's Python side of TF_DSV41_PROFILE's "prefill" (tf-dsv41-m1 m2b with M2B_PREFILL=1; gap.py prefill)
    if a.profile_prefill:
        if kv is not None:
            raise SystemExit("--profile-prefill: the contiguous slot only (no --pool / --split / --park)")
        from torch.profiler import ProfilerActivity, profile as tprofile

        g = torch.Generator().manual_seed(a.seed)
        ids = torch.randint(3, cfg.vocab_size, (a.profile_prefill,), generator=g).tolist()

        def prefill_once():
            fw.slots[0] = fw.new_slot(contiguous=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            served_prompt(fw, Seg, ids, a.prompt_tail, a.prefill)
            torch.cuda.synchronize()
            return (time.perf_counter() - t1) * 1e3

        prefill_once()
        wall = prefill_once()
        with tprofile(activities=[ProfilerActivity.CUDA]) as pr:
            prefill_once()
        ks = {}
        for ev in pr.key_averages():
            t = getattr(ev, "device_time_total", None) or getattr(ev, "cuda_time_total", 0)
            if t:
                ks[ev.key] = {"calls": ev.count, "ms": t / 1e3}
        f = Path(a.out) / f"prof-rank{rank}.json"
        doc = json.loads(f.read_text()) if f.exists() else {}
        doc["prefill"] = {"rows": len(ids), "segments": -(-len(ids) // fw.pchunk), "wall_ms": wall,
                          "gpu_ms": sum(v["ms"] for v in ks.values()), "prompt_tail": a.prompt_tail,
                          "prefill": a.prefill, "kernels": ks}
        f.write_text(json.dumps(doc, indent=1) + "\n")
        if rank == 0:
            print(f"profile: prefill of {len(ids)} ({a.prefill}, tail {a.prompt_tail}) {wall:.1f} ms wall, "
                  f"kernels {doc['prefill']['gpu_ms']:.1f} ms", flush=True)
    # --sampled (gate): the keyed sampling gate's reference on this load (dsv41_samp_ref.run_cases: its prompts
    # from fresh slots, serial sampled decoding, DIR/samp/trace.jsonl for tf-dsv41-m1 generate); before the AOT dump, so
    # its prompt lengths' Triton variants are in the set
    if a.sampled:
        if kv is not None:
            raise SystemExit("--sampled: the contiguous slot only (no --pool / --split / --park)")
        import dsv41_samp_ref

        dsv41_samp_ref.run_cases(fw, cfg, rank, Path(a.out) / "samp", dsv41_samp_ref.PROMPTS, a.steps, a.seed)
    # --grammar (gate): the structured-output gate's reference on this load (dsv41_grammar/grammar_ref.py:
    # prod's grammar masks over serial decoding, DIR/grammar/trace.jsonl with each case's structure); the sampled
    # gate's prompt lengths, so the AOT set from --sampled covers them
    if a.grammar:
        if kv is not None:
            raise SystemExit("--grammar: the contiguous slot only (no --pool / --split / --park)")
        sys.path.insert(0, str(Path(__file__).resolve().parent / "dsv41_grammar"))
        import grammar_ref

        grammar_ref.run_cases(fw, cfg, rank, Path(a.out) / "grammar", Path(a.pack), 48, a.seed)
    # --aot-prompts: more prompt lengths, each from a fresh slot, prefilled and decoded a step for their Triton variants
    # alone (Triton specializes integer arguments on % 16 and == 1: rows, key counts and strides of every prefill
    # path, the 64-row CUDA top-k and the `_keys` one; the Spark window's 45-token chat prompt had none). Nothing
    # recorded comes from them, and the slot is fresh again after them.
    if a.aot_taps:      # as a drafting server prefills: DSpark's taps kept (the prefill `_site`'s TAP_ON variants)
        fw.keep_taps = True
    for n in a.aot_prompts:
        if kv is None:
            fw.slots[0] = fw.new_slot(contiguous=True)
        else:
            kv.fresh(fw)
        g = torch.Generator().manual_seed(a.seed + n)
        ids = torch.randint(3, cfg.vocab_size, (n,), generator=g).tolist()
        tok = int(greedy(fw, fw.prompt(ids)[-1:])[0])
        fw.run([Seg(0, fw.slot.pos, (tok,))])
        fw.keep(0, 0)
        if a.prefill == "replay":       # the replay's passes at this length too (encoder segments, the decoder's tail)
            fw.slots[0] = fw.new_slot(contiguous=True)
            tok = int(greedy(fw, replay_prompt(fw, Seg, ids))[0])
            fw.run([Seg(0, fw.slot.pos, (tok,))])
            fw.keep(0, 0)
        torch.cuda.synchronize()
        if rank == 0:
            print(f"aot prompt of {n}: done", flush=True)
    run_s = time.time() - t0 - load_s
    top = Path(a.out)
    cache = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache"))
    if both and rank == 1:          # rank 1's set beside rank 0's, merged by rank 0 after the barrier
        r1 = top / "aot-rank1"
        r1.mkdir(parents=True, exist_ok=True)
        rec.dump(r1 / "launches.json")
        aot.build(cache, r1 / "launches.json", r1 / "manifest.json", None)
        pack_aot(r1 / "manifest.json", cache, r1 / "aot", specialization())
    if both:
        comm.barrier()
    if rank == 0 or a.rank >= 0:   # the AOT set and rope tables on each host of a two-host run
        tabs = RO.tables(cfg, a.limit + 2048)
        (top / "rope.bin").write_bytes(b"DSV41RP1" + struct.pack("<II", *tabs[0].cs.shape)
                                       + tabs[0].cs.contiguous().numpy().tobytes()
                                       + tabs[1].cs.contiguous().numpy().tobytes())
        rec.dump(top / "launches.json")
        aot.build(cache, top / "launches.json", top / "manifest.json", None)
        pack_aot(top / "manifest.json", cache, top / "aot", specialization())
        if both:
            print(f"rank 0: rank 1's AOT set merged: {merge_aot(top / 'aot', top / 'aot-rank1' / 'aot')}", flush=True)
        for d in dirs:
            if d != top:
                shutil.copy(top / "rope.bin", d / "rope.bin")
                shutil.copytree(top / "aot", d / "aot", dirs_exist_ok=True)
        print(f"rank 0: {len(dirs)} prompt(s), load {load_s:.0f} s, run {run_s:.0f} s", flush=True)
    comm.barrier()


def merge_aot(dst: Path, src: Path) -> dict:
    """The kernels of AOT set ``src`` that ``dst`` lacks (by hash), with their cubins (dsv41_aot_merge.py's rule)."""

    import shutil

    a = json.loads((dst / "aot.json").read_text())
    b = json.loads((src / "aot.json").read_text())
    have = {k["hash"] for k in a["kernels"]}
    added = 0
    for k in b["kernels"]:
        if k["hash"] in have:
            continue
        shutil.copyfile(src / "cubins" / f"{k['hash']}.cubin", dst / "cubins" / f"{k['hash']}.cubin")
        a["kernels"].append(k)
        have.add(k["hash"])
        added += 1
    (dst / "aot.json").write_text(json.dumps(a, indent=1) + "\n")
    return {"added": added, "kernels": len(a["kernels"])}


def decode(fw, greedy, Seg, digest, tok, steps, kv, trace) -> tuple[list, list, dict]:
    """``steps`` greedy windows of one row from ``tok``: (tokens, logits digests, the first window's block digests)."""

    tokens, digests, layers_d = [tok], [], {}
    for step in range(steps):
        fw.trace = {} if step == 0 and trace else None
        if kv is not None:
            kv.grow(fw.slot.pos + 1)
        lg = fw.run([Seg(0, fw.slot.pos, (tok,))])
        if step == 0 and trace:
            layers_d = {str(L): digest(t) for L, t in sorted(fw.trace.items())}
        digests.append(digest(lg))
        tok = int(greedy(fw, lg)[0])
        tokens.append(tok)
        fw.keep(0, 0)
    fw.trace = None
    return tokens, digests, layers_d


def served_prompt(fw, Seg, ids, tail: str = "prefill", prefill: str = "full"):
    """The prompt's choice logits [1, V / W] in ``tail``'s shape (the Zig side: prod_knobs.PromptTail). ``verify``:
    prod's server (batch.py ``_pieces`` / ``finals``, rounds.py) prefills [0, n - 1) and runs the last token as the
    pending row of the reply's first verify window, a decode row (decode kernels, the decode expert top-p, R1), kept
    whole; ``prefill``: Forward.prompt, every row prefilled (every reference before this option). ``prefill`` =
    ``replay``: the verify shape with CED's bounded replay (``replay_prompt``)."""

    if prefill == "replay":
        if tail != "verify":
            raise ValueError("--prefill replay: prod's server shape only (--prompt-tail verify)")
        return replay_prompt(fw, Seg, ids)
    if tail == "prefill":
        return fw.prompt(ids)[-1:]
    if len(ids) > 1:
        fw.prompt(ids[:-1])
    lg = fw.run([Seg(0, fw.slot.pos, (ids[-1],))])
    fw.keep(0, 0)
    return lg[-1:]


def replay_prompt(fw, Seg, ids):
    """``TF_DSV41_PREFILL=replay`` as prod's server runs it (``slots.SlotsForward.prefill(mode="replay")`` then
    ``finish_prompt``, rounds.py): ``replay.encode`` over the pieces [0, n - 1) in ``pchunk`` segments (layers 0-19 and
    layer 20's site and compressor, the stash), ``replay.finish`` over the prefilled tail [max(0, n - 128), n - 1)
    (layers 20-39 from the stash, windows from R0), then the last token as the reply's first verify row, kept whole.
    The Zig side: ced.zig (Forward.prefill in replay mode, then Forward.finishPrompt)."""

    from tensorfold.families.deepseek_v41.cuda import replay as RP
    from tensorfold.families.deepseek_v41.cuda.protocol import REPLAY, Piece
    from tensorfold.families.deepseek_v41.cuda.slots import prefill_runs

    if not isinstance(getattr(fw, "stash", None), dict):    # a plain Forward: SlotsForward's stash dict
        fw.stash = {}
    fw.stash.pop(0, None)
    n, pos = len(ids), fw.slot.pos
    if n > 1:
        RP.encode(fw, prefill_runs([Piece(0, pos, tuple(ids[:-1]))], fw.pchunk))
        RP.finish(fw, 0, pos + n - 1, tuple(ids[max(0, n - REPLAY):n - 1]))
    lg = fw.run([Seg(0, fw.slot.pos, (ids[-1],))])
    fw.keep(0, 0)
    return lg[-1:]


def prompt_ids(a, cfg, prompt: int) -> list[int]:
    import torch

    g = torch.Generator().manual_seed(a.seed)
    return torch.randint(3, cfg.vocab_size, (prompt,), generator=g).tolist()


def batched(rank, a, fw, cfg, greedy, Seg, digest, dirs, lo, hi, t0, load_s) -> None:
    """``--batched`` (with ``--prefill replay --prompt-tail verify``): prod's batcher with every prompt admitted in one
    round (batch.py ``_pieces``: each prompt's [0, n - 1) a piece; rounds.py: the round's pieces in one encoder run a
    ``slots.prefill_runs`` run, GLM 0560's one expert pass over every prefilling slot), then each slot's final
    (``replay.finish``), then each prompt as ``one`` writes it from its slot (its first verify window, its state, its
    greedy steps) into its DIR/p<N>. The Zig side: TF_DSV41_PIECE_RUNS=1 (block_prefill.emitMulti)."""

    from tensorfold.families.deepseek_v41.cuda import replay as RP
    from tensorfold.families.deepseek_v41.cuda.protocol import REPLAY, Piece
    from tensorfold.families.deepseek_v41.cuda.slots import prefill_runs

    if a.prefill != "replay" or a.prompt_tail != "verify":
        raise ValueError("--batched: prod's server shape in replay mode only (--prefill replay --prompt-tail verify)")
    idss = [prompt_ids(a, cfg, p) for p in a.prompts]
    if not isinstance(getattr(fw, "stash", None), dict):
        fw.stash = {}
    runs = prefill_runs([Piece(j, 0, tuple(ids[:-1])) for j, ids in enumerate(idss)], fw.pchunk)
    if rank == 0:
        print(f"batched: {len(idss)} prompts, {len(runs)} encoder run(s) of {[len(r) for r in runs]} segments", flush=True)
    RP.encode(fw, runs)
    for j, ids in enumerate(idss):
        n = len(ids)
        RP.finish(fw, j, n - 1, tuple(ids[max(0, n - REPLAY):n - 1]))
    for j, (prompt, out) in enumerate(zip(a.prompts, dirs)):
        fw.slots[0], fw.slots[j] = fw.slots[j], fw.slots[0]     # slot j's state where `one` reads it
        one(rank, a, fw, cfg, greedy, Seg, digest, prompt, out, lo, hi, t0, load_s, j == len(dirs) - 1, prefilled=True)
        fw.slots[0], fw.slots[j] = fw.slots[j], fw.slots[0]


def one(rank, a, fw, cfg, greedy, Seg, digest, prompt, out, lo, hi, t0, load_s, last, kv=None,
        prefilled: bool = False) -> None:
    """One prompt from the slot as it is: its state dump, greedy steps and digests into ``out``. ``prefilled``: the
    prompt's [0, n - 1) is in the slot already (``batched``): its last token's verify window only."""

    import torch

    if kv is not None:
        one_kv(rank, a, fw, cfg, greedy, Seg, digest, prompt, out, lo, hi, t0, load_s, last, kv)
        return
    ids = prompt_ids(a, cfg, prompt)
    t1 = time.time()
    if prefilled:
        logits = fw.run([Seg(0, fw.slot.pos, (ids[-1],))])
        fw.keep(0, 0)
        logits = logits[-1:]
    else:
        logits = served_prompt(fw, Seg, ids, a.prompt_tail, a.prefill)
    tok = int(greedy(fw, logits[-1:])[0])
    torch.cuda.synchronize()
    if rank == 0:
        print(f"prompt of {prompt} at {time.time() - t0:.0f} s, first token {tok} (tail {a.prompt_tail}, prefill {a.prefill})", flush=True)
    # the slot after the prompt, by our forward's role names (block.zig: s.L<i>.swa / comp .v .s, .ik, .carry)
    sd = out / f"rank{rank}" / "state"
    sd.mkdir(parents=True, exist_ok=True)
    roles, s0 = {}, fw.slot

    def put(role, t):
        f = sd / f"{role}.bin"
        f.write_bytes(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        roles[role] = f.name

    for L in range(lo, hi + 1):
        v, sc = s0.swa[L].kernel()
        put(f"s.L{L}.swa.v", v)
        put(f"s.L{L}.swa.s", sc)
        if L in s0.comp:
            v, sc = s0.comp[L].kernel()
            put(f"s.L{L}.comp.v", v)
            put(f"s.L{L}.comp.s", sc)
            put(f"s.L{L}.ik", s0.ikeys[L].kernel())
        if L in s0.carry:
            put(f"s.L{L}.carry", s0.carry[L])
    (sd / "state.json").write_text(json.dumps({"pos": s0.pos, "tail": list(s0.tail), "roles": roles}, indent=1) + "\n")
    tokens, digests, layers_d = [tok], [], {}
    for step in range(a.steps):
        fw.trace = {} if step == 0 else None
        lg = fw.run([Seg(0, fw.slot.pos, (tok,))])
        if step == 0:
            layers_d = {str(L): digest(t) for L, t in sorted(fw.trace.items())}
        digests.append(digest(lg))
        tok = int(greedy(fw, lg)[0])
        tokens.append(tok)
        fw.keep(0, 0)
    fw.trace = None
    # windows of 2.. rows after every recorded step (nothing recorded comes from them): their Triton variants join
    # the AOT set (a 1 specializes), for M3's lanes gates whose verify windows hold drafts
    if last:
        for n in a.extra_rows:
            fw.run([Seg(0, fw.slot.pos, tuple([tok] * n))])
            fw.keep(0, n - 1)
    torch.cuda.synchronize()
    mine = {"rank": rank, "load_s": round(load_s, 1), "run_s": round(time.time() - t1, 1), "tokens": tokens,
            "logits_sha256": digests, "layers_first_window": layers_d, "prompt_ids": ids, "prompt_tail": a.prompt_tail,
            "prefill": a.prefill}
    (out / f"rank{rank}.json").write_text(json.dumps(mine) + "\n")


def one_kv(rank, a, fw, cfg, greedy, Seg, digest, prompt, out, lo, hi, t0, load_s, last, kv) -> None:
    """``one`` on a pool slot (M5): the pages mapped before each run, the state by the pool's roles; ``--park``: the
    session round trip after the prompt, the steps decoded uninterrupted and again after the restore."""

    import torch

    g = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(3, cfg.vocab_size, (prompt,), generator=g).tolist()
    t1 = time.time()
    kv.grow(len(ids))
    logits = fw.prompt(ids)
    tok = int(greedy(fw, logits[-1:])[0])
    torch.cuda.synchronize()
    if rank == 0:
        print(f"prompt of {prompt} at {time.time() - t0:.0f} s, first token {tok} ({kv.mode} pool)", flush=True)
    sd = out / f"rank{rank}" / "state"
    sd.mkdir(parents=True, exist_ok=True)
    roles, shapes, s0 = {}, {}, fw.slot

    def put(role, t):
        f = sd / f"{role}.bin"
        f.write_bytes(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        roles[role] = f.name
        shapes[role] = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape)}

    for L in range(lo, hi + 1):
        v, sc = s0.swa[L].kernel()
        put(f"s.L{L}.swa.v", v)
        put(f"s.L{L}.swa.s", sc)
        if L in s0.carry:
            put(f"s.L{L}.carry", s0.carry[L])
    kvinfo = kv.dump(fw, put, sd)
    (sd / "state.json").write_text(json.dumps({"pos": s0.pos, "tail": list(s0.tail), "roles": roles,
                                               "shapes": shapes, "kv": kvinfo}, indent=1) + "\n")
    park = None
    if a.park:
        import dsv41_m5_kv as M5

        park = M5.Park(kv, out / "park")
        park.save_park(fw, ids)
        if rank == 0:
            print(f"parked {park.info.get('file_bytes', 0) / 2**20:.1f} MiB in {park.info.get('park_s')} s", flush=True)
    tokens, digests, layers_d = decode(fw, greedy, Seg, digest, tok, a.steps, kv, True)
    mine = {"rank": rank, "load_s": round(load_s, 1), "tokens": tokens, "logits_sha256": digests,
            "layers_first_window": layers_d, "prompt_ids": ids, "kv": kv.mode}
    if park is not None:
        park.restore(fw, ids + [tok])
        t2, d2, _ = decode(fw, greedy, Seg, digest, tok, a.steps, kv, False)
        park.info.update(tokens=t2, logits_sha256=d2, equal=bool(t2 == tokens and d2 == digests))
        if rank == 0:
            print(f"restored in {park.info.get('restore_s')} s: {a.steps} steps "
                  f"{'== the uninterrupted run' if park.info['equal'] else 'DIFFER from the uninterrupted run'}",
                  flush=True)
        park.close()
        mine["park"] = park.info
    if last:
        tok = mine["park"]["tokens"][-1] if park is not None else tokens[-1]
        for n in a.extra_rows:
            kv.grow(fw.slot.pos + n)
            fw.run([Seg(0, fw.slot.pos, tuple([tok] * n))])
            fw.keep(0, n - 1)
    torch.cuda.synchronize()
    if kv.kvx is not None:
        mine["exchange"] = kv.kvx.describe()
    mine["run_s"] = round(time.time() - t1, 1)
    (out / f"rank{rank}.json").write_text(json.dumps(mine) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-24")
    ap.add_argument("--prompt", type=int, default=1536)
    ap.add_argument("--engram-dir", default="", help="Engram rows from these packed shards (the Zig ranks' TF_DSV41_ENGRAM_DIR), "
                    "not the index function")
    ap.add_argument("--engram-make", action="store_true", help="write --engram-dir first: sparse shards, records under the prompts' rows")
    ap.add_argument("--profile", default="", help="row counts (\"1 4 8\"): time windows of each after the run")
    ap.add_argument("--profile-prefill", type=int, default=0,
                    help="after the run: an N-id prompt (the gate's ids at N = --prompt) prefilled from a fresh slot, timed "
                         "and under torch.profiler, into DIR/prof-rank<r>.json's \"prefill\" (gap.py prefill)")
    ap.add_argument("--aot-prompts", type=lambda v: [int(x) for x in v.split(",") if x], default=[],
                    help="more prompt lengths after the run, for their Triton variants only (e.g. 1,9,17,45,64,65,173,300,1025,2049,2600)")
    ap.add_argument("--aot-taps", action="store_true",
                    help="--aot-prompts with DSpark's taps kept (Forward taps=True), for a drafting server's prefill variants")
    ap.add_argument("--prompts", type=lambda v: [int(x) for x in v.split(",") if x], default=[],
                    help="several prompts in one load, each from a fresh slot into DIR/p<N> (with --rank: this host's rank)")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--batched", action="store_true",
                    help="--prompts in one batched round (prod's batcher: one encoder run over every prompt's piece, "
                         "slots.prefill_runs), each from its own slot; needs --prefill replay --prompt-tail verify")
    ap.add_argument("--prompt-tail", choices=("prefill", "verify"), default="prefill",
                    help="verify: the prompt as prod's server runs it ([0, n - 1) prefilled, the last token a decode row; "
                         "the Zig side's TF_DSV41_PROMPT_TAIL default); prefill: Forward.prompt (contiguous slot only)")
    ap.add_argument("--prefill", choices=("full", "replay"), default="full",
                    help="replay: prod's TF_DSV41_PREFILL=replay (CED bounded replay; needs --prompt-tail verify); the Zig "
                         "prefill gate's ranks then need TF_DSV41_PREFILL=replay")
    ap.add_argument("--grammar", action="store_true",
                    help="also the structured-output gate's replies (dsv41_grammar/grammar_ref.py) into DIR/grammar/trace.jsonl")
    ap.add_argument("--sampled", action="store_true",
                    help="also the keyed sampling gate's replies (dsv41_samp_ref.run_cases) into DIR/samp/trace.jsonl")
    ap.add_argument("--limit", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=4101)
    ap.add_argument("--port", type=int, default=29611)
    ap.add_argument("--rank", type=int, default=-1, help="two hosts: this host's rank (one process); -1: both ranks here")
    ap.add_argument("--extra-rows", type=lambda v: [int(x) for x in v.split(",") if x], default=[2, 3, 4, 5, 6, 16],
                    help="windows of these rows after the steps (their Triton variants in the AOT set)")
    ap.add_argument("--master", default="localhost", help="rank 0's address (two hosts: the head's CX7 address)")
    ap.add_argument("--pool", action="store_true", help="M5: the slot in a replicated KV pool (paged stores)")
    ap.add_argument("--split", action="store_true", help="M5: TF_DSV41_KV_SPLIT=1, the pool split over the ranks")
    ap.add_argument("--compact", default=None, help="M5 --split: TF_DSV41_KV_SPLIT_COMPACT (0 / 1 / force)")
    ap.add_argument("--park", action="store_true", help="M5: save / park / restore after the prompt (implies --pool)")
    ap.add_argument("--no-scramble", action="store_true", help="M5: no page holder (the slot's pages 0, 1, 2 ...)")
    a = ap.parse_args()
    a.kv = "split" if a.split else ("pool" if a.pool or a.park else "contig")
    if a.prompt_tail == "verify" and a.kv != "contig":
        ap.error("--prompt-tail verify: the contiguous slot only (one_kv prefills with Forward.prompt)")
    if a.prefill == "replay" and a.prompt_tail != "verify":
        ap.error("--prefill replay: prod's server shape only (--prompt-tail verify)")
    if a.compact is not None:
        if not a.split:
            ap.error("--compact needs --split")
        os.environ["TF_DSV41_KV_SPLIT_COMPACT"] = a.compact
    if a.split:
        os.environ["TF_DSV41_KV_SPLIT"] = "1"           # both ranks (spawned with this environment)
    elif a.kv == "pool":
        os.environ.pop("TF_DSV41_KV_SPLIT", None)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.rank >= 0:                         # two hosts: this host's rank alone (the other host runs the other)
        rank_main(a.rank, a)
        for prompt in a.prompts or [a.prompt]:      # --prompts: each prompt's DIR/p<N> as a one-prompt run's DIR
            d = Path(a.out) / f"p{prompt}" if a.prompts else Path(a.out)
            mine = json.loads((d / f"rank{a.rank}.json").read_text())
            ranks = [None, None]
            ranks[a.rank] = {k: mine[k] for k in ("load_s", "run_s", "logits_sha256", "layers_first_window")}
            meta = {"layers": a.layers, "prompt": prompt, "steps": a.steps, "seed": a.seed, "limit": a.limit,
                    "tokens": mine["tokens"], "prompt_ids": mine["prompt_ids"], "host_rank": a.rank,
                    "prompt_tail": mine.get("prompt_tail", "prefill"), "prefill": mine.get("prefill", "full"),
                    "env": {k: v for k, v in os.environ.items() if k.startswith("TF_DSV41")}, "ranks": ranks}
            if a.kv != "contig":
                meta["kv"] = a.kv
                if "park" in mine:
                    ranks[a.rank]["park"] = mine["park"]
            (d / "ref.json").write_text(json.dumps(meta, indent=1) + "\n")
            print(json.dumps({"rank": a.rank, "prompt": prompt, "tokens": len(mine["tokens"]), "first": mine["tokens"][:8]}), flush=True)
        return 0
    import torch.multiprocessing as mp

    mp.spawn(rank_main, args=(a,), nprocs=2, join=True)
    st = 0
    for prompt in a.prompts or [a.prompt]:
        d = Path(a.out) / f"p{prompt}" if a.prompts else Path(a.out)
        r0 = json.loads((d / "rank0.json").read_text())
        r1 = json.loads((d / "rank1.json").read_text())
        if r0["tokens"] != r1["tokens"]:
            print(f"prompt {prompt}: the ranks' greedy tokens differ", flush=True)
            st = 1
            continue
        meta = {"layers": a.layers, "prompt": prompt, "steps": a.steps, "seed": a.seed, "limit": a.limit,
                "tokens": r0["tokens"], "prompt_ids": r0["prompt_ids"], "prompt_tail": r0.get("prompt_tail", "prefill"),
                "prefill": r0.get("prefill", "full"),
                "env": {k: v for k, v in os.environ.items() if k.startswith("TF_DSV41")},
                "ranks": [{k: r[k] for k in ("load_s", "run_s", "logits_sha256", "layers_first_window")} for r in (r0, r1)]}
        if a.kv != "contig":            # M5: the pool mode, the exchange's counts, the park round trip a rank
            meta["kv"] = a.kv
            for m, r in zip(meta["ranks"], (r0, r1)):
                for k in ("park", "exchange"):
                    if k in r:
                        m[k] = r[k]
            if "park" in r0:
                eq = bool(r0["park"].get("equal") and r1.get("park", {}).get("equal"))
                meta["park_equal"] = eq
                if not eq:
                    print(f"prompt {prompt}: the restored session's steps differ from the uninterrupted run", flush=True)
                    st = 1
        (d / "ref.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(json.dumps({"prompt": prompt, "tokens": len(meta["tokens"]), "first": meta["tokens"][:8],
                          **({"park_equal": meta["park_equal"]} if "park_equal" in meta else {})}), flush=True)
    return st


if __name__ == "__main__":
    raise SystemExit(main())
