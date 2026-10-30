# SPDX-License-Identifier: MIT
"""GPU FP64 gate for rb4: recover actual score bits from unmodified stream kernels' packed candidate buffers."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import platform
import re
import sys
from pathlib import Path


def unpack(buf, count, torch):
    """Invert score_key, then restore key-column order; reject missing or duplicate columns."""
    keys = buf[:, 0, :count].cpu()
    positions = 0x7FFFFFFF - (keys & 0xFFFFFFFF)
    order = positions.argsort(dim=1)
    if not torch.equal(positions.gather(1, order), torch.arange(count).expand(len(keys), count)):
        raise AssertionError("incomplete score capture")
    bits = keys >> 32
    bits = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(torch.int32)
    return bits.contiguous().view(torch.float32).gather(1, order).double()


def launch(q, w, ik, pt, count, rows, k, fp8, rb, st, torch):
    """Keep production score constants/options; K >= count exposes every score without compaction."""
    pos = torch.tensor([2 * count - 1], dtype=torch.int32, device="cuda")
    common = {"RATIO": 2, "H": 32, "D": 128, "BP": 64, "WS": 1 / math.sqrt(32),
              "SCALE": 1 / math.sqrt(128), "SPLIT": 16384, "K": k, "CAP": 2 * k,
              "PT": pt, "PSH": 7, "KFP8": fp8}
    buffers = [torch.empty((rows, 1, 2 * k), device="cuda", dtype=torch.int64) for _ in range(2)]
    kernels = []
    kernels.append(st._stream[(rows, 1)](
        q, w, w.stride(0), ik, pos, buffers[0], 0, count, buffers[0], 1,
        MODE=0, BS=8, num_warps=8, num_stages=3, **common))
    kernels.append(rb._stream_rb4[((rows + 3) // 4, 1)](
        q, w, w.stride(0), ik, pos, rows, buffers[1], 1, num_warps=4, num_stages=1, **common))
    torch.cuda.synchronize()
    return buffers, kernels


def inputs(kind, rows, count, seed, fp8, paged, torch):
    """Generate rounded operands on CPU; the oracle independently dequantizes the stored bytes."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn((rows, 32, 128), generator=g).bfloat16()
    w = torch.rand((rows, 32), generator=g)
    keys = torch.randn((count + (rows - 1) // 2, 128), generator=g)
    if kind == "dot-cancellation":
        q[:, :, 64:] = q[:, :, :64]
        keys[:, 64:] = -keys[:, :64]
        keys[:, -1] += 0.125
    elif kind == "head-cancellation":
        q[:, 16:] = q[:, :16]
        w[:, 16:] = -w[:, :16]
        w[:, -1] += 2 ** -20
    elif kind == "near-tie":
        keys[:] = keys[0].clone()
        keys[:, -1] += torch.arange(len(keys)) % 3 * 2 ** -7
    physical = ((len(keys) + 127) // 128) * 128
    if fp8:
        scales = (2.0 ** torch.randint(-3, 4, (len(keys),), generator=g)).float()
        values = keys.to(torch.float8_e4m3fn)
        logical = torch.zeros((physical, 132), dtype=torch.uint8)
        logical[:len(keys), :128] = values.view(torch.uint8)
        logical[:len(keys), 128:] = scales[:, None].contiguous().view(torch.uint8)
        decoded = (values.float() * scales[:, None]).bfloat16()
    else:
        logical = torch.zeros((physical, 128), dtype=torch.bfloat16)
        logical[:len(keys)] = keys.bfloat16()
        decoded = keys.bfloat16()
    pages = physical // 128
    table = torch.arange(pages - 1, -1, -1, dtype=torch.int32) if paged else torch.arange(pages, dtype=torch.int32)
    stored = torch.empty_like(logical)
    for page in range(pages):
        stored[int(table[page]) * 128:(int(table[page]) + 1) * 128] = logical[page * 128:(page + 1) * 128]
    oracle = ((q.double() @ decoded.double().T) / math.sqrt(128)).clamp_min(0)
    oracle = (oracle * (w.double() / math.sqrt(32))[:, :, None]).sum(dim=1)
    for row in range(rows):
        oracle[row, count + row // 2:] = float("-inf")
    return q.cuda(), w.cuda(), stored.cuda(), table.cuda(), oracle


def metrics(scores, oracle, torch):
    error = (scores - oracle).abs()
    return {"max_abs": error.max().item(), "rms": error.square().mean().sqrt().item(),
            "per_column_max_abs": error.amax(dim=0).tolist()}


def arithmetic(kernel):
    """Float instructions exclude integer-only selection/compaction shuffles."""
    ops = re.findall(r"^\s*(mma\.sync\S*|fma\.rn\.f32|mul\.f32|add\.f32|add\.rn\.f32|max\.f32|max\.NaN\.f32)",
                     kernel.asm["ptx"], re.MULTILINE)
    return dict(collections.Counter(ops))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="served Python source package root")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    sys.path[:0] = [args.src, str(Path(__file__).resolve().parent)]
    import torch
    import triton
    from dsv41_zig_triton import stream_rb as rb
    from tensorfold.families.deepseek_v41.cuda.csa2 import stream_topk as st

    torch.set_num_threads(2)
    report = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
              "python": platform.python_version(), "os": platform.system(), "cases": [],
              "reference": "CPU FP64 dot, scale, ReLU, weighted head sum on independently decoded stored operands",
              "scope": "prompt positions rb4 only; decode unchanged; no model-fidelity or performance claim",
              "source_sha256": {name: hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
                                for name, mod in (("stream_rb", rb), ("stream_topk", st))}}
    report["checker_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    failures = 0
    for seed in (7, 23, 101):
        for kind in ("random", "dot-cancellation", "head-cancellation", "near-tie"):
            for fp8, paged, rows in ((False, False, 1), (False, True, 7), (True, True, 3), (True, True, 8)):
                count = 513 if seed == 23 else 1024
                q, w, ik, pt, oracle = inputs(kind, rows, count, seed, fp8, paged, torch)
                full, full_kernels = launch(q, w, ik, pt, count, rows, 2048, fp8, rb, st, torch)
                scores = [unpack(b, count, torch) for b in full]
                errors = [metrics(s, oracle[:, :count], torch) for s in scores]
                base_col = torch.tensor(errors[0]["per_column_max_abs"], dtype=torch.float64)
                new_col = torch.tensor(errors[1]["per_column_max_abs"], dtype=torch.float64)
                worse = (new_col > base_col).nonzero().flatten().tolist()
                top, top_kernels = launch(q, w, ik, pt, count, rows, 512, fp8, rb, st, torch)
                top_cpu = [b[:, 0, :512].cpu().sort(dim=1).values for b in top]
                consistency = all(torch.equal(t, f[:, 0, :2048].cpu().topk(512, dim=1).values.sort(dim=1).values)
                                  for t, f in zip(top_cpu, full, strict=True))
                same_top = torch.equal(*top_cpu)
                same_scores = torch.equal(*scores)
                finite = all(torch.isfinite(s).all().item() for s in scores)
                fp64_top = oracle.argsort(dim=1, descending=True, stable=True)[:, :512].sort(dim=1).values
                differences = []
                position_differences = []
                for t in top_cpu:
                    selected = (0x7FFFFFFF - (t & 0xFFFFFFFF)).sort(dim=1).values
                    bad_rows = (selected != fp64_top).any(dim=1).nonzero().flatten().tolist()
                    differences.append(bad_rows)
                    position_differences.append([
                        {"row": row, "native_only": sorted(set(selected[row].tolist()) - set(fp64_top[row].tolist())),
                         "fp64_only": sorted(set(fp64_top[row].tolist()) - set(selected[row].tolist()))}
                        for row in bad_rows])
                ordered = oracle.sort(dim=1, descending=True).values
                item = {"seed": seed, "kind": kind, "fp8": fp8, "paged": paged, "rows": rows, "columns": count,
                        "stream": errors[0], "rb4": errors[1], "worse_columns": worse,
                        "recovered_scores_equal": same_scores, "top512_equal": same_top,
                        "capture_matches_production_top512": consistency, "finite": finite,
                        "fp64_top512_different_rows": differences, "fp64_position_differences": position_differences,
                        "score_float_ops": [arithmetic(kernel) for kernel in full_kernels],
                        "production_score_float_ops": [arithmetic(kernel) for kernel in top_kernels],
                        "fp64_cutoff_margins": (ordered[:, 511] - ordered[:, 512]).tolist(),
                        "mma_layouts": [[line for line in kernel.asm["ttgir"].splitlines()
                                         if "#ttg.nvidia_mma<" in line] for kernel in full_kernels],
                        "production_mma_layouts": [[line for line in kernel.asm["ttgir"].splitlines()
                                                    if "#ttg.nvidia_mma<" in line] for kernel in top_kernels]}
                layout_equal = all(item["mma_layouts"]) and item["mma_layouts"] == item["production_mma_layouts"]
                item["capture_mma_matches_production"] = layout_equal
                ops_equal = all(item["score_float_ops"]) and item["score_float_ops"] == item["production_score_float_ops"]
                item["capture_float_ops_match_production"] = ops_equal
                ok = not worse and consistency and same_top and finite and layout_equal and ops_equal
                item["pass"] = ok
                failures += not ok
                report["cases"].append(item)
                print(f"{kind} seed={seed} fp8={fp8} rows={rows}: {'PASS' if ok else 'FAIL'} "
                      f"worse_columns={len(worse)} max_abs={errors[0]['max_abs']:.9g}/{errors[1]['max_abs']:.9g}",
                      flush=True)
    report.update(passed=len(report["cases"]) - failures, failed=failures, skipped=0)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['passed']} passed, {failures} failed, 0 skipped")
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
