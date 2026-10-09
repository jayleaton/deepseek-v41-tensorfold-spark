#!/usr/bin/env python3
"""DeepSeek-V4.1's two RoPE tables (the SWA layers' plain theta, the compressed layers' YaRN), built by the Python
engine's own ``rope.tables`` on THIS host: their bits come from torch's CPU cos / sin, which no port reproduces, so the
Zig engine loads them (run on the serving host's CPU, as the Python engine builds them at start).

    python dsv41_rope_tables.py --config config.json --rows 6144 --out rope.bin
Format (little-endian): magic "DSV41RP1", u32 rows, u32 cols, then the plain table fp32 [rows, cols], then the
compressed one (cos half, then sin half a row: csa2's ``cs`` layout).
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--rows", type=int, default=6144, help="positions: the slot limit + the prefill chunk (4096 + 2048)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tensorfold.families.deepseek_v41.cuda import rope
    from tensorfold.families.deepseek_v41.cuda.config import Config

    cfg = Config.from_file(Path(a.config))
    t = rope.tables(cfg, a.rows)
    plain, comp = t[0].cs.contiguous(), t[1].cs.contiguous()
    head = b"DSV41RP1" + struct.pack("<II", plain.shape[0], plain.shape[1])
    Path(a.out).write_bytes(head + plain.numpy().tobytes() + comp.numpy().tobytes())
    print(f"{a.out}: 2 x [{plain.shape[0]}, {plain.shape[1]}] fp32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
