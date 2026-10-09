"""Engram shards for a pod gate: the kit's packed per-rank layout (engram_host.py's ``write_shard`` header, 264-byte
records), full size but sparse, with records only under the rows a prompt hashes to (``engram_fixture.record``'s
bytes); every other row reads as zeros. The real tables are ~100 GB a rank and live on the Sparks; this exercises the
whole read path (hashes, this rank's heads, O_DIRECT extents, dequant, the gather) on both engines from one set of
files, so the M2b gate compares Zig's rows with Python's ``ShardRows`` bit for bit, the prefill's state included.

Used by dsv41_m2b_ref.py ``--engram-dir DIR --engram-make``: each rank writes its own shards before its prompt.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

MAGIC = 0x31344E4531565344
HEADER = 4096
ROW_BYTES = 264


def record(row: np.ndarray) -> np.ndarray:
    """[n, 264] bytes of global rows ``row`` (engram_fixture.record): hashed values, finite scales 100..139."""

    j = np.arange(264, dtype=np.uint64)[None, :]
    r = row.astype(np.uint64)[:, None]
    h = (r * np.uint64(2654435761) + j * np.uint64(40503)) % np.uint64(1 << 32)
    v = ((h >> np.uint64(7)) & np.uint64(0xFF)).astype(np.uint8)
    v[:, 256:] = (100 + (h[:, 256:] >> np.uint64(9)) % np.uint64(40)).astype(np.uint8)
    return v


def make(out: str | Path, hasher, prompts, rank: int, world: int) -> list[Path]:
    """This rank's shard of every Engram layer under ``out``: sparse, records under the rows of each prompt in
    ``prompts`` (id lists, each hashed from the sequence start as a fresh slot's prefill hashes it)."""

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lay = hasher.layout
    hashes = [hasher(np.asarray(ids, dtype=np.int64)).numpy() for ids in prompts]     # [T, layers, cols] each
    made = []
    for li, layer in enumerate(lay.layer_ids):
        lo, hi, h0, nh = lay.head_shard(li, rank, world)
        rows = np.unique(np.concatenate([h[:, li, h0:h0 + nh].reshape(-1) for h in hashes]))
        path = out / f"engram-l{layer}-r{rank}of{world}.bin"
        with open(path, "wb") as f:
            head = bytearray(HEADER)
            struct.pack_into("<6Q", head, 0, MAGIC, layer, lo, hi, lay.total_rows(li), ROW_BYTES)
            f.write(head)
            f.truncate(HEADER + (hi - lo) * ROW_BYTES)
            recs = record(rows)
            for k, r in enumerate(rows.tolist()):
                f.seek(HEADER + (r - lo) * ROW_BYTES)
                f.write(recs[k].tobytes())
        os.sync()
        made.append(path)
    return made
