"""Engram rows from local NVMe for DeepSeek-V4.1-Flash: the packed per-rank shards, O_DIRECT sector reads, and a
prefetcher that issues a round's reads before the forward needs them.

The tables (layers 1 and 14, 384,006,168 / 384,016,682 rows of 264 bytes: 256 e4m3 values + their scale bytes) are
never pinned or page-cached (95 GiB a node). Each rank owns 12 of the 24 hash heads (3 n-gram orders x 8 heads,
vLLM's head-bucket sharding), i.e. a contiguous row range [lo, hi) of each layer, and reads 12 rows a token a layer.

**File format** (the kit's ``./start.sh pack`` output, reused as is; facts from the files on head, no code):
``engram-l{1,14}-r{rank}of2.bin`` = a 4,096-byte header (little-endian u64: magic ``DSV41EN1``, layer, lo, hi, the
layer's total rows, row bytes = 264) then rows lo .. hi - 1 back to back. A row at byte 4,096 + 264 (row - lo)
spans one or two 512-byte sectors (the NVMe's logical block on both Sparks), so an O_DIRECT read of the covering
sectors fetches it with no page cache: 512-1,024 bytes a row.

**When the reads happen** (the design; ENGINE-PLAN.md section 6):

- Addresses depend only on token ids (the n-gram hashes over the compressed vocabulary, the last 3 ids of
  lookback), never on activations. So a round's rows (the pending token's and every drafted token's) are known when
  the drafter has proposed, before the verify forward starts; DSpark's own forward and layer 0 (~2-4 ms) cover the
  ~2.8 ms tonyd2wild measured for a step's reads done synchronously.
- Prefill: a chunk's rows are known when the chunk is cut; chunk k + 1's reads run under chunk k's forward.
- Rows land in pinned host staging, then one H2D copy a layer before layer 1 / layer 14 needs them; the Engram module
  (reference: vLLM ``common/engram.py``) gathers them on the GPU. Drafted rows that are rejected cost a read and
  nothing else: rows are keyed by id, so the next round re-reads (or re-hits the cache) without any state.
- Optional RAM row cache (``cache_rows``): n-gram rows are Zipfian (bot-lab-21: the top 100M rows cover 92.7%), so a
  small LRU of hot rows cuts reads; it is sized from the memory floor and off by default (our floor is 4-6 GiB).

Exactness: a row's bytes are the file's whatever the read path (direct, buffered, cached), so reads never change a
reply; the tests check every path against a buffered read of the same file.

This module is the CPU reference implementation (Python threads + ``os.preadv``: fine for decode's ~150 rows a
round; prefill's ~50k rows a 2,048-token chunk want the native reader, ENGINE-PLAN.md section 6).
"""

from __future__ import annotations

import mmap
import os
import struct
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = 0x31344E4531565344          # b"DSV41EN1" little-endian
HEADER = 4096
SECTOR = 512
ROW_BYTES = 264


@dataclass(frozen=True)
class Header:
    layer: int
    lo: int
    hi: int
    total: int
    row_bytes: int


def read_header(path: str | Path) -> Header:
    with open(path, "rb") as f:
        raw = f.read(48)
    magic, layer, lo, hi, total, rb = struct.unpack("<6Q", raw)
    if magic != MAGIC:
        raise ValueError(f"{path}: not a packed Engram shard (magic {magic:#x})")
    size = os.path.getsize(path)
    if size != HEADER + (hi - lo) * rb:
        raise ValueError(f"{path}: {size} bytes, the header says {HEADER + (hi - lo) * rb}")
    return Header(layer, lo, hi, total, rb)


def write_shard(path: str | Path, layer: int, lo: int, rows: np.ndarray, total: int) -> Header:
    """A packed shard from rows [n, row_bytes] uint8 (tests and tools; the kit's packer writes the same layout)."""

    n, rb = rows.shape
    head = bytearray(HEADER)
    struct.pack_into("<6Q", head, 0, MAGIC, layer, lo, lo + n, total, rb)
    with open(path, "wb") as f:
        f.write(head)
        f.write(np.ascontiguousarray(rows, dtype=np.uint8).tobytes())
    return Header(layer, lo, lo + n, total, rb)


def extents(rows: np.ndarray, lo: int, row_bytes: int, gap: int = 4096) -> list[tuple[int, int, np.ndarray]]:
    """Sorted unique rows -> sector-aligned reads [(offset, length, rows in it)]: a row's covering sectors, merged
    with the next read when the hole between them is at most ``gap`` bytes (one larger read beats two IOs)."""

    if rows.size == 0:
        return []
    start = HEADER + (rows - lo) * row_bytes
    s0 = start // SECTOR * SECTOR
    s1 = -(-(start + row_bytes) // SECTOR) * SECTOR
    out, a, b, i0 = [], int(s0[0]), int(s1[0]), 0
    for i in range(1, rows.size):
        if int(s0[i]) <= b + gap:
            b = max(b, int(s1[i]))
            continue
        out.append((a, b - a, rows[i0:i]))
        a, b, i0 = int(s0[i]), int(s1[i]), i
    out.append((a, b - a, rows[i0:]))
    return out


def _aligned(nbytes: int) -> tuple[mmap.mmap, memoryview]:
    """A page-aligned buffer (anonymous mmap): what O_DIRECT needs for the destination address."""

    m = mmap.mmap(-1, max(nbytes, SECTOR))
    return m, memoryview(m)


class RowFile:
    """One packed shard, opened O_DIRECT when the filesystem allows it (else buffered + a page-cache drop)."""

    def __init__(self, path: str | Path, *, direct: bool = True) -> None:
        self.path = str(path)
        self.header = read_header(path)
        self.size = os.path.getsize(path)
        flags = os.O_RDONLY
        self.direct = False
        if direct and hasattr(os, "O_DIRECT"):
            try:
                self.fd = os.open(self.path, flags | os.O_DIRECT)
                self.direct = True
            except OSError:
                self.fd = os.open(self.path, flags)
        else:
            self.fd = os.open(self.path, flags)

    def close(self) -> None:
        os.close(self.fd)

    def owns(self, rows: np.ndarray) -> np.ndarray:
        h = self.header
        return (rows >= h.lo) & (rows < h.hi)

    def read_extent(self, off: int, length: int) -> np.ndarray:
        m, view = _aligned(length)
        try:
            got = os.preadv(self.fd, [view[:length]], off)
            if got < min(length, self.size - off):           # the last sector may run past the end of the file
                raise OSError(f"{self.path}: short read at {off} ({got} of {length})")
            out = np.frombuffer(view[:length], dtype=np.uint8).copy()
        finally:
            view.release()
            m.close()
        if not self.direct:                                   # buffered fallback: keep the page cache clean
            os.posix_fadvise(self.fd, off, length, os.POSIX_FADV_DONTNEED)
        return out

    def read(self, rows: np.ndarray, gap: int = 4096) -> dict[int, np.ndarray]:
        """{row: bytes [row_bytes]} for unique sorted ``rows`` this shard owns (synchronous)."""

        h = self.header
        out = {}
        for off, length, rs in extents(rows, h.lo, h.row_bytes, gap):
            buf = self.read_extent(off, length)
            for r in rs.tolist():
                a = HEADER + (r - h.lo) * h.row_bytes - off
                out[r] = buf[a:a + h.row_bytes]
        return out


@dataclass
class Ticket:
    layer: int
    rows: np.ndarray                  # in request order (duplicates allowed)
    futures: list[Future]
    cached: dict[int, np.ndarray]


class Reader:
    """Prefetching reader over a rank's shards: ``submit`` returns at once, ``result`` blocks for the rows."""

    def __init__(self, files: dict[int, RowFile], *, workers: int = 32, cache_rows: int = 0, gap: int = 4096,
                 batch: int = 64) -> None:
        self.files = files
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="engram")
        self.gap, self.batch = gap, batch
        self.cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self.cache_rows = cache_rows
        self.lock = threading.Lock()
        self.stats = {"rows": 0, "unique": 0, "hits": 0, "reads": 0, "bytes": 0}

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        for f in self.files.values():
            f.close()

    def submit(self, layer: int, rows: np.ndarray) -> Ticket:
        rows = np.asarray(rows, dtype=np.int64)
        f = self.files[layer]
        if not bool(f.owns(rows).all()):
            raise ValueError(f"layer {layer}: rows outside this rank's shard [{f.header.lo}, {f.header.hi})")
        uniq = np.unique(rows)
        cached = {}
        if self.cache_rows:
            with self.lock:
                for r in uniq.tolist():
                    v = self.cache.get((layer, r))
                    if v is not None:
                        self.cache.move_to_end((layer, r))
                        cached[r] = v
            uniq = np.array([r for r in uniq.tolist() if r not in cached], dtype=np.int64)
        ext = extents(uniq, f.header.lo, f.header.row_bytes, self.gap)
        futs = []
        for i in range(0, len(ext), self.batch):
            futs.append(self.pool.submit(self._run, f, ext[i:i + self.batch]))
        self.stats["rows"] += int(rows.size)
        self.stats["unique"] += int(uniq.size) + len(cached)
        self.stats["hits"] += len(cached)
        self.stats["reads"] += len(ext)
        self.stats["bytes"] += sum(e[1] for e in ext)
        return Ticket(layer, rows, futs, cached)

    def _run(self, f: RowFile, ext) -> dict[int, np.ndarray]:
        out = {}
        h = f.header
        for off, length, rs in ext:
            buf = f.read_extent(off, length)
            for r in rs.tolist():
                a = HEADER + (r - h.lo) * h.row_bytes - off
                out[r] = buf[a:a + h.row_bytes]
        return out

    def result(self, t: Ticket) -> np.ndarray:
        """Rows [n, row_bytes] uint8 in the ticket's request order."""

        got = dict(t.cached)
        for fu in t.futures:
            got.update(fu.result())
        if self.cache_rows:
            with self.lock:
                for r, v in got.items():
                    self.cache[(t.layer, r)] = v
                    self.cache.move_to_end((t.layer, r))
                while len(self.cache) > self.cache_rows:
                    self.cache.popitem(last=False)
        rb = self.files[t.layer].header.row_bytes
        out = np.empty((t.rows.size, rb), dtype=np.uint8)
        for i, r in enumerate(t.rows.tolist()):
            out[i] = got[r]
        return out


def open_rank(root: str | Path, rank: int, world: int = 2, layers=(1, 14), **kw) -> Reader:
    """The kit's packed layout: ``<root>/engram-l{layer}-r{rank}of{world}.bin``."""

    root = Path(root)
    files = {ly: RowFile(root / f"engram-l{ly}-r{rank}of{world}.bin") for ly in layers}
    for ly, f in files.items():
        if f.header.layer != ly:
            raise ValueError(f"{f.path}: header says layer {f.header.layer}")
    return Reader(files, **kw)
