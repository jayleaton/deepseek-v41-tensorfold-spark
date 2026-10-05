#!/usr/bin/env python3
"""Pack DeepSeek-V4.1-Flash's Engram tables into the per-rank shards the engine reads (TF_DSV41_ENGRAM_DIR).

The EXL3 packs do not carry the Engram tables (layers 1 and 14: ~384M rows of 264 bytes each, ~101 GB a layer).
They come from DeepSeek's original checkpoint, tensors ``layers.{L}.engram.embed.weight`` (FP8 e4m3 [N, 256]) and
``layers.{L}.engram.embed.scale`` (UE8M0 [N, 8]). Each rank reads only its own hash heads (12 of the 24), one
contiguous row range, from a file on its local NVMe:

    engram-l{L}-r{rank}of2.bin = a 4,096-byte header (6 little-endian u64: magic b"DSV41EN1", layer, lo, hi,
                                 the layer's total rows, row bytes 264), then rows lo .. hi - 1,
                                 each row = 256 weight bytes followed by 8 scale bytes

(the engine's ``engram_host.read_header`` / ``ShardRows`` and its test fixture ``write_engram`` define the format;
this script writes it from the source tensors' raw bytes, nothing is converted). Head h of order n reads rows
[offset, offset + prime); the primes are the next unused primes above engram_vocab_size - 1 in (layer, order, head)
order, and rank r owns heads [12 r, 12 r + 12) of every layer.

    # which source shards hold the tables (download only those, plus the index)
    python3 scripts/pack_engram.py --src <dir> --list
    # the head's shards (rank 0), then the worker's (rank 1; or run it on the worker with --rank 1)
    python3 scripts/pack_engram.py --src <dir> --config <pack>/config.json --out <HEAD_ENGRAM> --rank 0
    python3 scripts/pack_engram.py --src <dir> --config <pack>/config.json --out <WORKER_ENGRAM> --rank 1
    # compare existing shards with the source (headers + sampled rows), e.g. shards packed by another tool
    python3 scripts/pack_engram.py --src <dir> --config <pack>/config.json --check <dir with shards> --rank 0

--src is a directory with ``model.safetensors.index.json`` of deepseek-ai/DeepSeek-V4.1-Flash and the shards it
names for the four tensors. Needs numpy only. Writes ~47 GiB a layer a rank (~94 GiB a node); the page cache is
dropped behind the reads and writes (on GB10 it is the GPU's memory).

Written from the format above and not yet run against a full checkpoint in this repository's history (the
measured runs used shards packed earlier): run --check on the result, and the engine's preflight and canary.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = 0x31344E4531565344          # b"DSV41EN1" little-endian
HEADER = 4096
VALUES, SCALES = 256, 8
ROW = VALUES + SCALES
CHUNK_ROWS = 1 << 20                # 264 MiB of output a chunk


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):            # deterministic below 4,759,123,141
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def layout(cfg: dict) -> dict[int, list[int]]:
    """{layer: the primes of its (order, head) columns}, in the engine's order."""

    seen: set[int] = set()
    out = {}
    for layer in cfg["engram_layer_ids"]:
        per = []
        for _ in range(cfg["engram_max_ngram_size"] - 1):
            cur = cfg["engram_vocab_size"] - 1
            for _ in range(cfg["engram_n_heads"]):
                cur += 1
                while not is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                per.append(cur)
        out[layer] = per
    return out


def head_shard(primes: list[int], rank: int, world: int) -> tuple[int, int]:
    part = (len(primes) + world - 1) // world
    h0, h1 = rank * part, min(rank * part + part, len(primes))
    return sum(primes[:h0]), sum(primes[:h1])


class Tensor:
    """One tensor of a safetensors file: its raw bytes by row, read with pread."""

    def __init__(self, path: Path, name: str) -> None:
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n))[name]
        self.dtype, self.shape = meta["dtype"], meta["shape"]
        lo, hi = meta["data_offsets"]
        self.start = 8 + n + lo
        self.row_bytes = (hi - lo) // self.shape[0]
        self.fd = os.open(path, os.O_RDONLY)

    def rows(self, lo: int, hi: int) -> np.ndarray:
        off, length = self.start + lo * self.row_bytes, (hi - lo) * self.row_bytes
        buf = os.pread(self.fd, length, off)
        if len(buf) != length:
            raise IOError(f"{self.path}: short read at {off}")
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(self.fd, off, length, os.POSIX_FADV_DONTNEED)
        return np.frombuffer(buf, dtype=np.uint8).reshape(hi - lo, self.row_bytes)


def sources(src: Path, layer: int) -> tuple[Tensor, Tensor]:
    wmap = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    names = [f"layers.{layer}.engram.embed.weight", f"layers.{layer}.engram.embed.scale"]
    missing = [n for n in names if n not in wmap]
    if missing:
        raise SystemExit(f"{src}: the index has no {missing} (is this deepseek-ai/DeepSeek-V4.1-Flash's index?)")
    w, s = (Tensor(src / wmap[n], n) for n in names)
    if w.row_bytes != VALUES or s.row_bytes != SCALES or w.shape[0] != s.shape[0]:
        raise SystemExit(f"layer {layer}: expected [N, {VALUES}] weight and [N, {SCALES}] scale bytes, got "
                         f"{w.dtype} {w.shape} / {s.dtype} {s.shape}")
    return w, s


def records(w: Tensor, s: Tensor, lo: int, hi: int) -> np.ndarray:
    out = np.empty((hi - lo, ROW), dtype=np.uint8)
    out[:, :VALUES] = w.rows(lo, hi)
    out[:, VALUES:] = s.rows(lo, hi)
    return out


def pack(src: Path, cfg: dict, out: Path, ranks: list[int], world: int) -> None:
    lay = layout(cfg)
    out.mkdir(parents=True, exist_ok=True)
    for layer, primes in lay.items():
        w, s = sources(src, layer)
        total = sum(primes)
        if total != w.shape[0]:
            raise SystemExit(f"layer {layer}: the primes sum to {total} rows, the table has {w.shape[0]}")
        for rank in ranks:
            lo, hi = head_shard(primes, rank, world)
            path = out / f"engram-l{layer}-r{rank}of{world}.bin"
            tmp = path.with_suffix(".bin.tmp")
            head = bytearray(HEADER)
            struct.pack_into("<6Q", head, 0, MAGIC, layer, lo, hi, total, ROW)
            print(f"{path}: rows {lo:,} .. {hi - 1:,} of {total:,} ({(hi - lo) * ROW / 2**30:.1f} GiB)", flush=True)
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.write(fd, bytes(head))
                pos = HEADER
                for a in range(lo, hi, CHUNK_ROWS):
                    b = min(hi, a + CHUNK_ROWS)
                    data = records(w, s, a, b).tobytes()
                    view = memoryview(data)
                    while view:
                        n = os.write(fd, view)
                        view = view[n:]
                    os.fdatasync(fd)
                    if hasattr(os, "posix_fadvise"):
                        os.posix_fadvise(fd, pos, len(data), os.POSIX_FADV_DONTNEED)
                    pos += len(data)
                    print(f"  {(b - lo) / (hi - lo):6.1%}", end="\r", flush=True)
            finally:
                os.close(fd)
            os.replace(tmp, path)
            print(f"  done: {path.stat().st_size:,} bytes")


def check(src: Path, cfg: dict, root: Path, ranks: list[int], world: int, samples: int) -> int:
    lay, bad, rng = layout(cfg), 0, random.Random(0)
    for layer, primes in lay.items():
        w, s = sources(src, layer)
        total = sum(primes)
        for rank in ranks:
            lo, hi = head_shard(primes, rank, world)
            path = root / f"engram-l{layer}-r{rank}of{world}.bin"
            with open(path, "rb") as f:
                magic, ly, a, b, t, rb = struct.unpack("<6Q", f.read(48))
                ok = (magic, ly, a, b, t, rb) == (MAGIC, layer, lo, hi, total, ROW)
                ok &= path.stat().st_size == HEADER + (hi - lo) * ROW
                rows = sorted(rng.randrange(lo, hi) for _ in range(samples)) + [lo, hi - 1]
                for r in rows:
                    f.seek(HEADER + (r - lo) * ROW)
                    if f.read(ROW) != records(w, s, r, r + 1).tobytes():
                        ok = False
                        print(f"{path}: row {r} differs from the source")
                        break
            print(f"{path}: {'ok' if ok else 'MISMATCH'} (header, size, {len(rows)} rows)")
            bad += not ok
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="DeepSeek-V4.1-Flash index + the shards with the tables")
    ap.add_argument("--config", type=Path, help="config.json of the pack you serve (default: --src/config.json)")
    ap.add_argument("--out", type=Path, help="where to write engram-l{L}-r{rank}of2.bin")
    ap.add_argument("--rank", default="all", help="0, 1 or all")
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--list", action="store_true", help="print the source files that hold the tables and exit")
    ap.add_argument("--check", type=Path, help="compare existing shards in this directory with the source")
    ap.add_argument("--samples", type=int, default=256)
    a = ap.parse_args()
    if a.list:
        wmap = json.loads((a.src / "model.safetensors.index.json").read_text())["weight_map"]
        files = sorted({v for k, v in wmap.items() if ".engram.embed." in k})
        print("\n".join(files))
        return 0
    cfg = json.loads((a.config or a.src / "config.json").read_text())
    cfg = {**cfg, **(cfg.get("text_config") or {})}      # the EXL3 packs nest the text model's keys
    ranks = list(range(a.world)) if a.rank == "all" else [int(a.rank)]
    if a.check:
        return check(a.src, cfg, a.check, ranks, a.world, a.samples)
    if not a.out:
        ap.error("--out is required to pack")
    pack(a.src, cfg, a.out, ranks, a.world)
    return 0


if __name__ == "__main__":
    sys.exit(main())
