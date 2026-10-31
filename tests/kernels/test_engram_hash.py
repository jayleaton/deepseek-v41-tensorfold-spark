"""Engram's host side (engine/kernels/engram/hash.py), no GPU:

- ``Tables.rows`` == engine/reference/engram.py's ``NgramHasher`` bit for bit on the real config (16M primes, the
  real multipliers), with image tokens; a sequence hashed in chunks with the carried lookback (DEAD kept) == hashed
  whole; the rank shards == ``EngramLayout.head_shard``; save / load round trip;
- ``Prefetch`` over the serving reader and real packed shard files (tiny config): the staging block holds each
  window row's 12 records in column order, == the file's bytes; issue -> land -> upload.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from engine.kernels.engram import hash as H  # noqa: E402
from engine.reference.config import Config, tiny_config  # noqa: E402
from engine.reference.engram import EngramLayout, NgramHasher  # noqa: E402


@pytest.fixture(scope="module")
def real():
    cfg = Config()
    g = np.random.default_rng(5)
    tmap = g.integers(0, cfg.engram_compressed_vocab_size, size=cfg.vocab_size)
    return cfg, tmap, H.Tables.from_config(cfg, tmap)


def test_rows_equal_reference(real):
    cfg, tmap, t = real
    g = np.random.default_rng(1)
    ids = g.integers(0, cfg.vocab_size, size=300)
    dead = np.zeros(300, dtype=bool)
    dead[[0, 57, 58, 200]] = True
    want = NgramHasher(cfg, tmap)(torch.from_numpy(ids), torch.from_numpy(dead)).numpy()
    assert np.array_equal(t.rows(ids, dead), want)
    assert np.array_equal(t.rows(ids), NgramHasher(cfg, tmap)(torch.from_numpy(ids)).numpy())


def test_chunks_with_lookback_equal_whole(real):
    cfg, _, t = real
    g = np.random.default_rng(2)
    ids = g.integers(0, cfg.vocab_size, size=257)
    dead = g.random(257) < 0.05
    whole = t.rows(ids, dead)
    lb = None
    got = []
    for a, b in ((0, 1), (1, 3), (3, 64), (64, 65), (65, 200), (200, 257)):        # decode-size and prefill chunks
        got.append(t.rows(ids[a:b], dead[a:b], lb))
        lb = H.next_lookback(lb, t, ids[a:b], dead[a:b])
        assert lb.size == min(b, 3)
    assert np.array_equal(np.concatenate(got), whole)


def test_shards_and_tables_io(real, tmp_path):
    cfg, _, t = real
    lay = EngramLayout(cfg)
    for li in range(2):
        for rank in range(2):
            lo, hi, c0, n = t.shard(li, rank, 2)
            assert (lo, hi, c0, n) == lay.head_shard(li, rank, 2)
            rows = t.rows(np.arange(50), None)[:, li, c0:c0 + n]
            assert bool(((rows >= lo) & (rows < hi)).all())
    t.save(tmp_path / "tables.npz")
    t2 = H.Tables.load(tmp_path / "tables.npz")
    ids = np.arange(40) * 977
    assert np.array_equal(t2.rows(ids), t.rows(ids)) and t2.layer_ids == t.layer_ids


def test_prefetch_lands_the_files_bytes(tmp_path):
    from engine.reference import engram_io as E

    cfg = tiny_config()
    tmap = np.arange(cfg.vocab_size) % cfg.engram_compressed_vocab_size
    t = H.Tables.from_config(cfg, tmap)
    li = 0
    lay = t.shard(li, 1, 2)
    total = int(t.primes[li].sum())
    g = np.random.default_rng(3)
    table = g.integers(0, 256, size=(total, H.RECORD), dtype=np.uint8)
    lo, hi = lay[0], lay[1]
    E.write_shard(tmp_path / "engram-l1-r1of2.bin", cfg.engram_layer_ids[0], lo, table[lo:hi], total)
    reader = E.open_rank(tmp_path, 1, 2, layers=(1,), workers=4)
    try:
        pf = H.Prefetch(t, reader, 1, 2, max_rows=8, pin=False)
        lookback = H.next_lookback(None, t, [5, 9])
        window = [11, 30, 30, 7]                       # pending + 3 drafts
        tk = pf.issue(lookback, window)
        pf.land(tk, 0)
        dst = torch.zeros((8,) + tuple(pf.staging.shape[2:]), dtype=torch.uint8)
        pf.upload(0, dst, tk.n)
        rows = t.rows(window, None, lookback)[:, li, lay[2]:lay[2] + lay[3]]
        for i in range(len(window)):
            for c in range(lay[3]):
                assert np.array_equal(dst[i, c].numpy(), table[rows[i, c]])
        assert bool((dst[len(window):] == 0).all())
    finally:
        reader.close()
