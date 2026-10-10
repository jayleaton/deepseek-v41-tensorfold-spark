# SPDX-License-Identifier: MIT
"""Goldens for the Zig image front end (zig/src/families/deepseek_v41/vision): images in every PNG / JPEG / GIF / BMP
flavour Pillow opens (GIF: the first frame; BMP: hand-built headers too), each decoded by Pillow (``decode`` + ``to_rgb``) and preprocessed by the prod engine's own
``vision_prep.preprocess`` (8474f31). Writes ``OUT/<name>`` (the file bytes) and ``OUT/golden.jsonl``:
{name, size, rgb_sha256, grid [n_llm_h, n_llm_w, vit_h, vit_w], patches_sha256, digest, vids_sha256, vids_head}.

Run in the e1e2 image (Pillow 12.3, torch): ``python -I gen_prep_golden.py --py-src <tree>/src --config
<release config.json> --out DIR [--small]`` (``--small``: only the small images, the committed fixture set).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
import zlib
from pathlib import Path


def png_bytes(arr, color: int, depth: int, interlace: bool = False, palette=None, trns=None, filt: int | None = None):
    """A PNG of ``arr`` (h x w x channels, ints already in range) with any colour type / depth (Pillow cannot write
    16-bit RGB, LA16, 2 / 4-bit grey or Adam7)."""

    import numpy as np

    arr = np.asarray(arr)
    h, w = arr.shape[:2]
    ch = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    arr = arr.reshape(h, w, ch)

    def rows(sub):
        sh, sw = sub.shape[:2]
        out = bytearray()
        bpp = max(1, ch * depth // 8)
        prev = bytes((sw * ch * depth + 7) // 8)
        for y in range(sh):
            if depth == 16:
                raw = sub[y].astype(">u2").tobytes()
            elif depth == 8:
                raw = sub[y].astype("u1").tobytes()
            else:
                bits = "".join(format(int(v), f"0{depth}b") for v in sub[y].reshape(-1))
                bits += "0" * (-len(bits) % 8)
                raw = bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
            f = (y % 5) if filt is None else filt
            enc = bytearray(raw)
            for i in range(len(raw)):
                a = raw[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                if f == 1:
                    enc[i] = (raw[i] - a) & 255
                elif f == 2:
                    enc[i] = (raw[i] - b) & 255
                elif f == 3:
                    enc[i] = (raw[i] - ((a + b) >> 1)) & 255
                elif f == 4:
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                    enc[i] = (raw[i] - pr) & 255
            out += bytes([f]) + enc
            prev = raw
        return bytes(out)

    if interlace:
        data = b""
        for x0, y0, dx, dy in ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)):
            sub = arr[y0::dy, x0::dx]
            if sub.shape[0] and sub.shape[1]:
                data += rows(sub)
    else:
        data = rows(arr)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, color, 0, 0, int(interlace)))
    if palette is not None:
        out += chunk(b"PLTE", bytes(palette))
    if trns is not None:
        out += chunk(b"tRNS", bytes(trns))
    z = zlib.compress(data, 6)
    out += chunk(b"IDAT", z[: len(z) // 2]) + chunk(b"IDAT", z[len(z) // 2:])   # two IDATs
    return out + chunk(b"IEND", b"")


def bmp_bytes(arr, bits: int, *, palette=None, compression: int = 0, masks=None, top_down: bool = False,
              core: bool = False, rle: bytes | None = None, header: int = 40):
    """A BMP of ``arr`` (h x w [x channels]): Pillow cannot write 16-bit, bitfields, RLE, top-down or core headers."""

    import numpy as np

    arr = np.asarray(arr)
    h, w = arr.shape[:2]
    stride = ((w * bits + 31) >> 3) & ~3
    rows = []
    for y in range(h):
        r = arr[y]
        if bits == 24:
            row = r[:, ::-1].astype("u1").tobytes()
        elif bits == 32:
            row = r.astype("<u4").tobytes() if r.ndim == 1 else r[:, [2, 1, 0, 3]].astype("u1").tobytes()
        elif bits == 16:
            row = r.astype("<u2").tobytes()
        elif bits == 8:
            row = r.astype("u1").tobytes()
        else:
            bs = "".join(format(int(v), f"0{bits}b") for v in r)
            bs += "0" * (-len(bs) % 8)
            row = bytes(int(bs[i:i + 8], 2) for i in range(0, len(bs), 8))
        rows.append(row + b"\0" * (stride - len(row)))
    data = rle if rle is not None else b"".join(rows if top_down else rows[::-1])
    pal = b""
    if palette is not None:
        pal = b"".join(bytes([c[2], c[1], c[0]]) + (b"" if core else b"\0") for c in palette)
    if core:
        hdr = struct.pack("<IHHHH", 12, w, h, 1, bits)
    else:
        hh = (-h if top_down else h) & 0xFFFFFFFF
        hdr = struct.pack("<IiIHHIIiiII", header, w, 0, 1, bits, compression, len(data), 2835, 2835,
                          len(palette or []), 0)
        hdr = hdr[:8] + struct.pack("<I", hh) + hdr[12:]
        extra = b""
        if masks is not None:
            extra = b"".join(struct.pack("<I", m) for m in masks)
        if header > 40:
            hdr += (extra + b"\0" * 64)[:header - 40]
            extra = b""
        hdr += extra
    off = 14 + len(hdr) + len(pal)
    return b"BM" + struct.pack("<IHHI", off + len(data), 0, 0, off) + hdr + pal + data


def rle8(arr):
    """RLE8 rows (bottom-up): runs of equal bytes, absolute runs (word aligned), end of line / bitmap."""

    out = bytearray()
    for row in arr[::-1]:
        row = [int(v) for v in row]
        i = 0
        while i < len(row):
            j = i
            while j < len(row) and row[j] == row[i] and j - i < 255:
                j += 1
            if j - i >= 3:
                out += bytes([j - i, row[i]])
                i = j
                continue
            k = min(len(row), i + 7)
            run = row[i:k]
            if len(run) >= 3:
                out += bytes([0, len(run)]) + bytes(run) + (b"\0" if len(run) % 2 else b"")
            else:
                for v in run:
                    out += bytes([1, v])
            i = k
        out += b"\0\0"
    return bytes(out + b"\0\1")


def rle4(arr):
    out = bytearray()
    for row in arr[::-1]:
        row = [int(v) for v in row]
        i = 0
        while i < len(row):
            n = min(len(row) - i, 6)
            if n >= 4 and n % 2 == 0:   # an absolute run (n // 2 bytes, word aligned)
                b = bytes((row[i + 2 * t] << 4) | row[i + 2 * t + 1] for t in range(n // 2))
                out += bytes([0, n]) + b + (b"\0" if len(b) % 2 else b"")
            else:
                n = min(n, 2) if n >= 2 else 1
                out += bytes([n, (row[i] << 4) | (row[i + 1] if n > 1 else row[i])])
            i += n
        out += b"\0\0"
    return bytes(out + b"\0\1")


def gif_offset(raw: bytes, grow: int, x0: int, y0: int) -> bytes:
    """A GIF whose screen is ``grow`` px bigger than its first frame, the frame at (x0, y0) (Pillow writes neither)."""

    b = bytearray(raw)
    w, h = struct.unpack("<HH", b[6:10])
    b[6:10] = struct.pack("<HH", w + grow, h + grow)
    i = 13 + ((3 << ((b[10] & 7) + 1)) if b[10] & 128 else 0)
    while b[i] != 0x2C:
        if b[i] == 0x21:
            i += 2
            while b[i]:
                i += 1 + b[i]
            i += 1
        else:
            i += 1
    b[i + 1:i + 5] = struct.pack("<HH", x0, y0)
    return bytes(b)


def images(small: bool):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(4101)

    def field(h, w, c, hi=256):
        y, x = np.mgrid[0:h, 0:w]
        base = (x * 7 + y * 3)[..., None] + np.arange(c)[None, None, :] * 50
        return ((base + rng.integers(0, 40, (h, w, c))) % hi).astype(np.int64)

    def pil(img, fmt, **kw):
        b = io.BytesIO()
        img.save(b, fmt, **kw)
        return b.getvalue()

    out = []
    h, w = (23, 37)
    out.append(("rgb8.png", png_bytes(field(h, w, 3), 2, 8)))
    out.append(("rgb8-adam7.png", png_bytes(field(h, w, 3), 2, 8, interlace=True)))
    out.append(("rgb16.png", png_bytes(field(h, w, 3, 65536), 2, 16)))
    out.append(("rgba8.png", png_bytes(field(h, w, 4), 6, 8)))
    out.append(("rgba16-adam7.png", png_bytes(field(h, w, 4, 65536), 6, 16, interlace=True)))
    out.append(("la8.png", png_bytes(field(h, w, 2), 4, 8)))
    out.append(("la16.png", png_bytes(field(h, w, 2, 65536), 4, 16)))
    for d in (1, 2, 4, 8):
        out.append((f"l{d}.png", png_bytes(field(h, w, 1, 1 << d), 0, d, interlace=d == 2)))
    out.append(("l16.png", png_bytes(field(h, w, 1, 600), 0, 16)))
    pal = rng.integers(0, 256, 3 * 200).tolist()
    for d in (1, 4, 8):
        out.append((f"p{d}.png", png_bytes(field(h, w, 1, 1 << d), 3, d, palette=pal[: 3 * (1 << min(d, 7))])))
    out.append(("p8-trns.png", png_bytes(field(h, w, 1, 200), 3, 8, palette=pal, trns=rng.integers(0, 256, 150).tolist())))
    out.append(("p8-trns1.png", png_bytes(field(h, w, 1, 200), 3, 8, palette=pal, trns=[255] * 7 + [0] + [255] * 3)))
    out.append(("rgb8-trns.png", png_bytes(field(h, w, 3), 2, 8, trns=[0, 1, 0, 2, 0, 3])))
    rgb = Image.fromarray(field(61, 83, 3).astype("u1"), "RGB")
    for sub in (0, 1, 2):
        out.append((f"q{sub}.jpg", pil(rgb, "JPEG", quality=87, subsampling=sub)))
        out.append((f"q{sub}-prog.jpg", pil(rgb, "JPEG", quality=70, subsampling=sub, progressive=True)))
    out.append(("gray.jpg", pil(rgb.convert("L"), "JPEG", quality=90)))
    out.append(("gray-prog.jpg", pil(rgb.convert("L"), "JPEG", quality=90, progressive=True)))
    out.append(("cmyk.jpg", pil(rgb.convert("CMYK"), "JPEG", quality=90)))
    out.append(("q100.jpg", pil(rgb, "JPEG", quality=100, subsampling=0)))
    out.append(("tiny.jpg", pil(Image.fromarray(field(1, 2, 3).astype("u1"), "RGB"), "JPEG", subsampling=2)))
    out.append(("narrow.png", png_bytes(field(400, 3, 3), 2, 8)))
    # GIF (the first frame) as Pillow writes it, then hand-made: a screen bigger than the frame, an offset frame
    pimg = Image.fromarray(field(h, w, 1, 200)[..., 0].astype("u1"), "P")
    pimg.putpalette(pal[:600])
    out.append(("p.gif", pil(pimg, "GIF")))
    out.append(("p-trns.gif", pil(pimg, "GIF", transparency=7)))
    out.append(("p-flat.gif", pil(pimg, "GIF", interlace=False)))
    out.append(("l.gif", pil(Image.fromarray(field(h, w, 1)[..., 0].astype("u1"), "L"), "GIF")))
    out.append(("rgb.gif", pil(Image.fromarray(field(h, w, 3).astype("u1"), "RGB"), "GIF")))
    out.append(("offset.gif", gif_offset(pil(pimg, "GIF", transparency=3), 9, 5, 4)))
    out.append(("offset-opaque.gif", gif_offset(pil(pimg, "GIF", interlace=False), 6, 2, 3)))
    out.append(("bits1.gif", pil(Image.fromarray((field(h, w, 1, 2)[..., 0] * 255).astype("u1"), "L").convert("1"), "GIF")))
    # BMP: Pillow's own writer, then the layouts it reads but does not write
    out.append(("rgb.bmp", pil(Image.fromarray(field(h, w, 3).astype("u1"), "RGB"), "BMP")))
    out.append(("p.bmp", pil(pimg, "BMP")))
    out.append(("l.bmp", pil(Image.fromarray(field(h, w, 1)[..., 0].astype("u1"), "L"), "BMP")))
    out.append(("1.bmp", pil(Image.fromarray((field(h, w, 1, 2)[..., 0] * 255).astype("u1"), "L").convert("1"), "BMP")))
    out.append(("rgba.bmp", pil(Image.fromarray(field(h, w, 4).astype("u1"), "RGBA"), "BMP")))
    pal16 = [tuple(rng.integers(0, 256, 3)) for _ in range(16)]
    out.append(("p4.bmp", bmp_bytes(field(h, w, 1, 16)[..., 0], 4, palette=pal16)))
    out.append(("p1.bmp", bmp_bytes(field(h, w, 1, 2)[..., 0], 1, palette=[(10, 200, 30), (250, 5, 90)])))
    out.append(("p8-core.bmp", bmp_bytes(field(h, w, 1, 200)[..., 0], 8, palette=[tuple(pal[3 * i:3 * i + 3]) for i in range(200)], core=True)))
    out.append(("bgr-top.bmp", bmp_bytes(field(h, w, 3), 24, top_down=True)))
    out.append(("bgr555.bmp", bmp_bytes(field(h, w, 1, 1 << 15)[..., 0], 16)))
    out.append(("bgr565.bmp", bmp_bytes(field(h, w, 1, 1 << 16)[..., 0], 16, compression=3, masks=[0xF800, 0x7E0, 0x1F])))
    out.append(("bgrx.bmp", bmp_bytes(field(h, w, 4), 32)))
    out.append(("rgba-v4.bmp", bmp_bytes(field(h, w, 4), 32, compression=3, masks=[0xFF0000, 0xFF00, 0xFF, 0xFF000000], header=108)))
    out.append(("abgr.bmp", bmp_bytes(field(h, w, 1, 1 << 30)[..., 0] * 4 + 3, 32, compression=3, masks=[0xFF000000, 0xFF0000, 0xFF00, 0xFF], header=56)))
    p8 = field(h, w, 1, 200)[..., 0] // 40 * 40
    out.append(("rle8.bmp", bmp_bytes(p8, 8, palette=[tuple(pal[3 * i:3 * i + 3]) for i in range(200)], compression=1, rle=rle8(p8))))
    p4 = field(h, w, 1, 16)[..., 0] // 4 * 4
    out.append(("rle4.bmp", bmp_bytes(p4, 4, palette=pal16, compression=2, rle=rle4(p4))))
    if small:
        return out
    big = Image.fromarray(field(900, 1500, 3).astype("u1"), "RGB")
    out.append(("big.jpg", pil(big, "JPEG", quality=85)))
    out.append(("big-prog.jpg", pil(big, "JPEG", quality=85, progressive=True, subsampling=1)))
    out.append(("big.png", png_bytes(field(700, 1300, 4), 6, 8)))
    out.append(("tall.png", png_bytes(field(3000, 200, 3), 2, 8)))
    out.append(("wide.jpg", pil(Image.fromarray(field(120, 2600, 3).astype("u1"), "RGB"), "JPEG")))
    out.append(("square544.png", png_bytes(field(544, 544, 3), 2, 8)))
    out.append(("odd.jpg", pil(Image.fromarray(field(333, 517, 3).astype("u1"), "RGB"), "JPEG", subsampling=2, quality=60)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-src", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--bias-vl", action="store_true", help="digests with TF_DSV41_BIAS_VL set")
    a = ap.parse_args()
    sys.path.insert(0, a.py_src)
    import os

    from tensorfold.families.deepseek_v41.cuda import vision_prep as V

    env = dict(os.environ)
    env.pop("TF_DSV41_BIAS_VL", None)
    if a.bias_vl:
        env["TF_DSV41_BIAS_VL"] = "/x"
    cfg_dir = Path(a.config).parent
    s = V.Settings.read(cfg_dir, env)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = V.Registry()
    with open(out / "golden.jsonl", "w") as g:
        for name, raw in images(a.small):
            (out / name).write_bytes(raw)
            img = V.decode(raw, s)
            rgb = img.tobytes()
            p = V.preprocess(img, s)
            vids = reg.vids(p.digest, p.tokens)
            rec = {"name": name, "size": list(p.size), "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
                   "grid": [p.n_llm_h, p.n_llm_w, p.n_vit_h, p.n_vit_w],
                   "patches_sha256": hashlib.sha256(p.patches.view(__import__("torch").int16).numpy().tobytes()).hexdigest(),
                   "digest": p.digest.hex(), "vids_head": vids[:4],
                   "vids_sha256": hashlib.sha256(b"".join(v.to_bytes(4, "little") for v in vids)).hexdigest()}
            g.write(json.dumps(rec) + "\n")
    print(f"wrote {out}/golden.jsonl", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
