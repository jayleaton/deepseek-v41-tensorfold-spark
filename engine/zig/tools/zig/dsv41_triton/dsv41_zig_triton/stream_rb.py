# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""Row-blocked streaming top-k; pf/rb2 retain row arithmetic, rb4 requires the FP64 score and GPU selection gates."""

from __future__ import annotations

import triton
import triton.language as tl

from tensorfold.families.deepseek_v41.cuda.csa2.rows import load_keys, prow
from tensorfold.families.deepseek_v41.cuda.csa2.stream_topk import _compact, _offer, score_key


@triton.jit
def _keys_tile(IK, j, ok, D: tl.constexpr, PT, PSH: tl.constexpr, KFP8: tl.constexpr):
    """score_tile's key load for positions j under mask ok (bf16 [BP, D])."""

    d = tl.arange(0, D)
    jr = prow(tl.where(ok, j, 0).to(tl.int64), PT, PSH, ok)
    return load_keys(IK, jr, ok, d, D, KFP8)


@triton.jit
def _row_scores(qv, w, k, ok, SCALE: tl.constexpr):
    """score_tile's arithmetic on loaded operands: the row's scores at the tile's keys, -inf where not ok."""

    dots = tl.dot(qv, tl.trans(k))                                                                     # [H, BP] fp32
    s = tl.sum(w[:, None] * tl.maximum(dots * SCALE, 0.0), axis=0)
    return tl.where(ok, s, float("-inf"))


@triton.jit
def _row_q(QI, W, w_stride, r, H: tl.constexpr, D: tl.constexpr, WS: tl.constexpr):
    hh = tl.arange(0, H)
    d = tl.arange(0, D)
    qv = tl.load(QI + (r.to(tl.int64) * H + hh[:, None]) * D + d[None, :]).to(tl.bfloat16)
    w = tl.load(W + r.to(tl.int64) * w_stride + hh).to(tl.float32) * WS
    return qv, w


@triton.jit
def _row_end(buf, n, K: tl.constexpr, CAP: tl.constexpr):
    if n > K:
        _compact(buf, n, K, CAP)
        n = K
    kk = tl.arange(0, K)
    tl.store(buf + kk, tl.full((K,), -9223372036854775808, tl.int64), mask=kk >= n)


@triton.jit
def _stream_rb2(QI, W, w_stride, IK, POS, R, BUF, nsplit, RATIO: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                BP: tl.constexpr, WS: tl.constexpr, SCALE: tl.constexpr, SPLIT: tl.constexpr, K: tl.constexpr,
                CAP: tl.constexpr, PT=None, PSH: tl.constexpr = 0, KFP8: tl.constexpr = False):
    """Program (row block b of 2 rows, split c): `_stream` MODE 0 for rows 2b, 2b + 1 (< R)."""

    b = tl.program_id(0)
    c = tl.program_id(1)
    r0 = b * 2
    r1 = r0 + 1
    q0 = tl.load(POS) + r0
    lo = c * SPLIT
    # a row past R sees nothing (its program slot is never stored)
    h0 = tl.minimum(lo + SPLIT, (q0 + 1) // RATIO)
    h1 = tl.where(r1 < R, tl.minimum(lo + SPLIT, (q0 + 2) // RATIO), lo)
    hi = tl.maximum(h0, h1)
    qv0, w0 = _row_q(QI, W, w_stride, r0, H, D, WS)
    qv1, w1 = _row_q(QI, W, w_stride, tl.minimum(r1, R - 1), H, D, WS)
    buf0 = BUF + (r0.to(tl.int64) * nsplit + c) * CAP
    buf1 = BUF + (r1.to(tl.int64) * nsplit + c) * CAP
    n0 = 0
    n1 = 0
    t0_ = tl.full((), -9223372036854775808, tl.int64)
    t1_ = tl.full((), -9223372036854775808, tl.int64)
    for t0 in range(lo, hi, BP):
        j = t0 + tl.arange(0, BP)
        k = _keys_tile(IK, j, j < hi, D, PT, PSH, KFP8)
        ok0 = j < h0
        ok1 = j < h1
        s0 = _row_scores(qv0, w0, k, ok0, SCALE)
        n0, t0_ = _offer(buf0, score_key(s0, j), ok0, n0, t0_, K, CAP)
        if r1 < R:  # its own region: see the module doc (the dot's MMA layout)
            s1 = _row_scores(qv1, w1, k, ok1, SCALE)
            n1, t1_ = _offer(buf1, score_key(s1, j), ok1, n1, t1_, K, CAP)
    _row_end(buf0, n0, K, CAP)
    if r1 < R:
        _row_end(buf1, n1, K, CAP)


@triton.jit
def _stream_rb4(QI, W, w_stride, IK, POS, R, BUF, nsplit, RATIO: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                BP: tl.constexpr, WS: tl.constexpr, SCALE: tl.constexpr, SPLIT: tl.constexpr, K: tl.constexpr,
                CAP: tl.constexpr, PT=None, PSH: tl.constexpr = 0, KFP8: tl.constexpr = False):
    """Score four prompt rows with one multi-row dot and offer each row's keys to its own split buffer."""

    b = tl.program_id(0)
    c = tl.program_id(1)
    r0 = b * 4
    q0 = tl.load(POS) + r0
    lo = c * SPLIT
    h0 = tl.minimum(lo + SPLIT, (q0 + 1) // RATIO)
    h1 = tl.where(r0 + 1 < R, tl.minimum(lo + SPLIT, (q0 + 2) // RATIO), lo)
    h2 = tl.where(r0 + 2 < R, tl.minimum(lo + SPLIT, (q0 + 3) // RATIO), lo)
    h3 = tl.where(r0 + 3 < R, tl.minimum(lo + SPLIT, (q0 + 4) // RATIO), lo)
    hi = tl.maximum(tl.maximum(h0, h1), tl.maximum(h2, h3))
    m = tl.arange(0, 4 * H)
    row = tl.minimum(r0 + m // H, R - 1).to(tl.int64)
    d = tl.arange(0, D)
    qv = tl.load(QI + (row * H + m % H)[:, None] * D + d[None, :]).to(tl.bfloat16)
    wv = tl.load(W + row * w_stride + m % H).to(tl.float32) * WS
    buf0 = BUF + (r0.to(tl.int64) * nsplit + c) * CAP
    buf1 = buf0 + nsplit * CAP
    buf2 = buf1 + nsplit * CAP
    buf3 = buf2 + nsplit * CAP
    n0 = 0
    n1 = 0
    n2 = 0
    n3 = 0
    th0 = tl.full((), -9223372036854775808, tl.int64)
    th1 = tl.full((), -9223372036854775808, tl.int64)
    th2 = tl.full((), -9223372036854775808, tl.int64)
    th3 = tl.full((), -9223372036854775808, tl.int64)
    BT: tl.constexpr = 2 * BP
    for t0 in range(lo, hi, BT):
        j = t0 + tl.arange(0, BT)
        okk = j < hi
        jr = prow(tl.where(okk, j, 0).to(tl.int64), PT, PSH, okk)
        k = load_keys(IK, jr, okk, d, D, KFP8)                                       # [BT, D]
        dots = tl.dot(qv, tl.trans(k))                                               # [4 H, BT]
        t = wv[:, None] * tl.maximum(dots * SCALE, 0.0)
        s = tl.sum(tl.reshape(t, (4, H, BT)), axis=1)                                # [4, BT]
        s = tl.reshape(tl.permute(s, (1, 0)), (BT, 2, 2))                            # [BT, row // 2, row % 2]
        sa, sb = tl.split(s)                                                         # rows {0, 2} / {1, 3}
        s0, s2 = tl.split(sa)
        s1, s3 = tl.split(sb)
        ok0 = j < h0
        n0, th0 = _offer(buf0, score_key(tl.where(ok0, s0, float("-inf")), j), ok0, n0, th0, K, CAP)
        ok1 = j < h1
        n1, th1 = _offer(buf1, score_key(tl.where(ok1, s1, float("-inf")), j), ok1, n1, th1, K, CAP)
        ok2 = j < h2
        n2, th2 = _offer(buf2, score_key(tl.where(ok2, s2, float("-inf")), j), ok2, n2, th2, K, CAP)
        ok3 = j < h3
        n3, th3 = _offer(buf3, score_key(tl.where(ok3, s3, float("-inf")), j), ok3, n3, th3, K, CAP)
    _row_end(buf0, n0, K, CAP)
    if r0 + 1 < R:
        _row_end(buf1, n1, K, CAP)
    if r0 + 2 < R:
        _row_end(buf2, n2, K, CAP)
    if r0 + 3 < R:
        _row_end(buf3, n3, K, CAP)


@triton.jit
def _raw_keys(IK, j, ok, D: tl.constexpr, PT, PSH: tl.constexpr, KFP8: tl.constexpr):
    """load_keys' loads for positions j under mask ok, not yet dequantized: (KFP8) the e4m3 bytes [BP, D] and the
    fp32 scales [BP], else the bf16 rows and zeros."""

    d = tl.arange(0, D)
    jr = prow(tl.where(ok, j, 0).to(tl.int64), PT, PSH, ok)
    if KFP8:
        b = tl.load(IK + jr[:, None] * (D + 4) + d[None, :], mask=ok[:, None], other=0)
        sc = tl.load((IK + jr * (D + 4) + D).to(tl.pointer_type(tl.float32)), mask=ok, other=0.0)
    else:
        b = tl.load(IK + jr[:, None] * D + d[None, :], mask=ok[:, None], other=0.0)
        sc = tl.where(ok, 0.0, 0.0)
    return b, sc


@triton.jit
def _dequant(b, sc, KFP8: tl.constexpr):
    """load_keys' bf16 keys from `_raw_keys`' loads."""

    if KFP8:
        return (b.to(tl.float8e4nv, bitcast=True).to(tl.float32) * sc[:, None]).to(tl.bfloat16)
    return b.to(tl.bfloat16)


@triton.jit
def _stream_pf(QI, W, w_stride, IK, POS, R, BUF, nsplit, RATIO: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
               BP: tl.constexpr, WS: tl.constexpr, SCALE: tl.constexpr, SPLIT: tl.constexpr, K: tl.constexpr,
               CAP: tl.constexpr, PT=None, PSH: tl.constexpr = 0, KFP8: tl.constexpr = False):
    """Program (row r < R, split c): `_stream` MODE 0, statement for statement, with each tile's key loads (the page
    table, then the key bytes and scales) issued one tile ahead: tile t + 1's loads go out before tile t's dequant,
    dot, head sum and offer, so their latency hides behind them (`_stream`'s loads are gathers, which Triton's
    pipeliner leaves in place: its loop has no async copy). The row's q and w are loaded once (the values score_tile
    loads every tile). The dot is score_tile's on the same operands, one row a program (the [2, 4] MMA layout:
    check_stream_rb.py), so every score, every offer and the split buffer are `_stream`'s."""

    r = tl.program_id(0)
    c = tl.program_id(1)
    q = tl.load(POS) + r
    nvis = (q + 1) // RATIO
    buf = BUF + (r.to(tl.int64) * nsplit + c) * CAP
    lo = c * SPLIT
    hi = tl.minimum(lo + SPLIT, nvis)
    qv, w = _row_q(QI, W, w_stride, r, H, D, WS)
    n = 0
    thr = tl.full((), -9223372036854775808, tl.int64)
    j0 = lo + tl.arange(0, BP)
    b, sc = _raw_keys(IK, j0, j0 < hi, D, PT, PSH, KFP8)
    for t0 in range(lo, hi, BP):
        j = t0 + tl.arange(0, BP)
        ok = j < hi
        jn = j + BP
        bn, scn = _raw_keys(IK, jn, jn < hi, D, PT, PSH, KFP8)        # the next tile's (masked past hi: none)
        k = _dequant(b, sc, KFP8)
        s = _row_scores(qv, w, k, ok, SCALE)
        n, thr = _offer(buf, score_key(s, j), ok, n, thr, K, CAP)
        b = bn
        sc = scn
    _row_end(buf, n, K, CAP)


__all__ = ["_stream_pf", "_stream_rb2", "_stream_rb4"]
