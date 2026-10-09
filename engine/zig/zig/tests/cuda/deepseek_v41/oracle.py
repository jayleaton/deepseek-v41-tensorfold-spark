"""Dev-time oracle for DeepSeek-V4.1's kernels (runs in the PyTorch image on the GPU, never at runtime): fixtures for
`tf-dsv41-test all <out>/fixtures` from the Python engine's own extensions (a compatible Python TensorFold checkout) on
seeded inputs. Each case is a directory: manifest.json (params, arrays) plus one raw little-endian file per array.

usage: PYTHONPATH=<python-checkout>/src python -B oracle.py <out dir>
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import torch

from tensorfold.cuda.exl3 import experts as x3
from tensorfold.families.deepseek_v41.cuda import expert_loads, expert_prefill
from tensorfold.families.deepseek_v41.cuda import x3gm as gm

DEV = torch.device("cuda:0")
CB = 2  # mul1


def save(out: Path, params: dict, arrays: dict[str, torch.Tensor]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"params": params, "arrays": {}}
    for name, t in arrays.items():
        t = t.detach().contiguous().cpu()
        (out / f"{name}.bin").write_bytes(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
        manifest["arrays"][name] = {"file": f"{name}.bin", "dtype": str(t.dtype).replace("torch.", ""),
                                    "shape": list(t.shape)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def trellis(g: torch.Generator, k: int, n: int, k2: int) -> torch.Tensor:
    """A random stored trellis int16 [K/16, N/16, 16 K2 / 2 ... ] (every word value is a valid mul1 state)."""
    return torch.randint(-32768, 32768, (k // 16, n // 16, 8 * k2), generator=g, dtype=torch.int32).to(torch.int16)


def scales(g: torch.Generator, e: int, n: int, mag: float) -> torch.Tensor:
    sign = torch.randint(0, 2, (e, n), generator=g) * 2 - 1
    return (sign * (0.5 + torch.rand(e, n, generator=g)) * mag).to(torch.float16)


class Layer:
    """E experts' gate / up / down trellises back to back in one int16 buffer (ragged: a width per expert)."""

    def __init__(self, g, E, D, I, k2g, k2d, shx=False):
        self.E, self.D, self.I = E, D, I
        parts, offs = [], {"g": [], "u": [], "d": []}
        at = 0
        for kind, K, N, ws in (("g", D, I, k2g), ("u", D, I, k2g), ("d", I, D, k2d)):
            for e in range(E):
                t = trellis(g, K, N, ws[e])
                offs[kind].append(at * 2)
                parts.append(t.reshape(-1))
                at += t.numel()
        self.words = torch.cat(parts).to(DEV)
        self.off = {k: torch.tensor(v, dtype=torch.int64) for k, v in offs.items()}
        self.k2 = {"g": list(k2g), "u": list(k2g), "d": list(k2d)}
        base = self.words.data_ptr()
        self.tp = {k: (v + base).to(DEV) for k, v in self.off.items()}
        self.k2t = {k: torch.tensor(v, dtype=torch.int32, device=DEV) for k, v in self.k2.items()}
        self.suh_g = scales(g, E, D, 1.0).to(DEV)
        self.suh_u = self.suh_g.clone() if shx else scales(g, E, D, 1.0).to(DEV)
        self.svh_g = scales(g, E, I, 0.02).to(DEV)
        self.svh_u = scales(g, E, I, 0.02).to(DEV)
        self.suh_d = scales(g, E, I, 1.0).to(DEV)
        self.svh_d = scales(g, E, D, 0.02).to(DEV)

    def arrays(self) -> dict:
        return {"words": self.words, "off_g": self.off["g"], "off_u": self.off["u"], "off_d": self.off["d"],
                "k2_g": self.k2t["g"], "k2_u": self.k2t["u"], "k2_d": self.k2t["d"], "suh_g": self.suh_g,
                "suh_u": self.suh_u, "svh_g": self.svh_g, "svh_u": self.svh_u, "suh_d": self.suh_d,
                "svh_d": self.svh_d}


def experts_case(out: Path, g, *, R, slots, E, D, I, k2g, k2d, input_dtype, limit, skip_every=0, prefill=False):
    """The decode chain (exl3.experts.routed's launches) with every x3ld / x3pf setting's Z beside upstream's."""
    L = Layer(g, E, D, I, k2g, k2d)
    up, ld, pf = x3._ext(), expert_loads._ext(), expert_prefill._ext()
    P = R * slots
    maxu = min(P, E)
    x = torch.randn(R, D, generator=g).to(input_dtype).to(DEV)
    pick = torch.stack([torch.randperm(E - 1, generator=g)[:slots - 1] for _ in range(R)])
    pick = torch.cat([pick, torch.full((R, 1), E - 1)], 1).to(torch.int32)        # the last slot: the shared expert
    if skip_every:
        pick[::skip_every, 0] = E                                                    # a slot past the table: skipped
    pick = pick.to(DEV)
    wts = torch.rand(R, slots, generator=g).to(DEV)
    ids = torch.zeros(maxu, dtype=torch.int32, device=DEV)
    count = torch.zeros(1, dtype=torch.int32, device=DEV)
    members = torch.full((maxu, R), -1, dtype=torch.int32, device=DEV)
    up.group(pick, ids, count, members, R, slots, E)
    xg = torch.zeros(P, D, dtype=torch.float16, device=DEV)
    xu = torch.zeros_like(xg)
    up.rot_in(x, x.stride(0), pick, L.suh_g, L.suh_u, xg, xu, R, D, slots, E)
    ntg, wg, skg, pfg = x3.default_config(D, I, True)
    ntd, wd, skd, pfd = x3.default_config(I, D, False)
    z = torch.zeros(max(2 * skg * I, skd * D) * P, dtype=torch.float32, device=DEV)
    k2gu = L.k2["g"] + L.k2["u"]
    lo_gu, hi_gu, lo_d, hi_d = min(k2gu), max(k2gu), min(L.k2["d"]), max(L.k2["d"])
    arrays = {"x": x, "pick": pick, "wts": wts, **L.arrays(), "uids": ids, "ucount": count, "members": members,
              "xg": xg, "xu": xu}

    def zrun(tag, mats, x0, x1, tp, k2t, K, N, sk, lo, hi, nt, w, pfn):
        z.zero_()
        up.grouped(x0, x1, tp[0], tp[1], k2t[0], k2t[1], ids, count, members, z, mats, K, N, P, sk, slots, CB, nt, w,
                   pfn, lo, hi)
        arrays[f"z_{tag}"] = z.clone()
        for nt_, pd in expert_loads.CFGS:
            if expert_loads.k2_range(lo, hi) and expert_loads.fits(K, N, sk, w, (nt_, pd)):
                z.zero_()
                ld.grouped(x0, x1, tp[0], tp[1], k2t[0], k2t[1], ids, count, members, z, mats, K, N, P, sk, slots,
                           CB, nt_, pd, 0, *expert_loads.k2_range(lo, hi), False)
                arrays[f"z_{tag}_ld_{nt_}_{pd}"] = z.clone()
        if prefill:
            for nt_, mtl in ((4, 4), (8, 2)):
                if expert_prefill.k2_range(lo, hi) and expert_prefill.fits(K, N, sk, w, (nt_, mtl)):
                    z.zero_()
                    pf.grouped(x0, x1, tp[0], tp[1], k2t[0], k2t[1], ids, count, members, z, mats, K, N, P, sk,
                               slots, CB, nt_, mtl, *expert_prefill.k2_range(lo, hi))
                    arrays[f"z_{tag}_pf_{nt_}_{mtl}"] = z.clone()
        z.zero_()
        up.grouped(x0, x1, tp[0], tp[1], k2t[0], k2t[1], ids, count, members, z, mats, K, N, P, sk, slots, CB, nt, w,
                   pfn, lo, hi)

    zrun("gu", 2, xg, xu, (L.tp["g"], L.tp["u"]), (L.k2t["g"], L.k2t["u"]), D, I, skg, lo_gu, hi_gu, ntg, wg, pfg)
    xd = torch.zeros(P, I, dtype=torch.float16, device=DEV)
    up.gateup_epilogue(z, pick, L.svh_g, L.svh_u, L.suh_d, xd, R, P, I, skg, slots, E, float(limit), x3.ACT_F32)
    zrun("d", 1, xd, xd, (L.tp["d"], L.tp["d"]), (L.k2t["d"], L.k2t["d"]), I, D, skd, lo_d, hi_d, ntd, wd, pfd)
    y = torch.zeros(P, D, dtype=torch.float32, device=DEV)
    up.down_epilogue(z, pick, L.svh_d, y, R, P, D, skd, slots, E)
    outc = torch.zeros(R, D, dtype=torch.float32, device=DEV)
    up.combine(y, wts, outc, R, D, slots)
    y2 = torch.zeros_like(y)
    outd = torch.zeros_like(outc)
    up.down_combine(z, pick, L.svh_d, y2, wts, outd, R, P, D, skd, slots, E)
    torch.cuda.synchronize()
    assert torch.equal(outc, outd), "down_combine != down_epilogue + combine"
    for k, v in list(arrays.items()):
        if k.startswith("z_") and "_ld_" in k or "_pf_" in k:
            base = "z_gu" if k.startswith("z_gu") else "z_d"
            assert torch.equal(v, arrays[base]), f"{k} != upstream"           # the Python engine's own claim
    arrays.update(xd=xd, y=y, out=outc)
    save(out, {"kind": "experts", "R": R, "slots": slots, "E": E, "D": D, "I": I, "input": "bf16"
               if input_dtype == torch.bfloat16 else "f16", "limit": limit if math.isfinite(limit) else 0.0,
               "limit_inf": int(not math.isfinite(limit)),
               "act_mode": x3.ACT_F32, "lo_gu": lo_gu, "hi_gu": hi_gu, "lo_d": lo_d, "hi_d": hi_d, "sk_gu": skg,
               "sk_d": skd, "nt_gu": ntg, "w_gu": wg, "pf_gu": pfg, "nt_d": ntd, "w_d": wd, "pf_d": pfd}, arrays)
    return {"ld_settings": sum("_ld_" in k for k in arrays), "pf_settings": sum("_pf_" in k for k in arrays)}


def x3gm_case(out: Path, g, *, R, slots, E, D, I, k2g, k2d, shx, input_dtype, limit, runs, dequant=None):
    """x3gm.run / _run_ragged's launches for each (gu cfg, dn cfg, ticket) in ``runs`` ("tuned": the table's)."""
    ragged = len(set(k2g)) > 1 or len(set(k2d)) > 1
    L = Layer(g, E, D, I, k2g, k2d, shx=shx)
    ext, up = gm._ext(), x3._ext()
    P = R * slots
    mats_x = 1 if shx else 2
    x = torch.randn(R, D, generator=g).to(input_dtype).to(DEV)
    pick = torch.stack([torch.randperm(E, generator=g)[:slots] for _ in range(R)]).to(torch.int32).to(DEV)
    wts = torch.rand(R, slots, generator=g).to(DEV)
    xs = torch.zeros(mats_x, P, D, dtype=torch.float16, device=DEV)
    xg, xu = xs[0], xs[mats_x - 1]
    ext.rot(x, x.stride(0), pick, L.suh_g, L.suh_u, xg, xu, D, slots, P, mats_x)
    arrays = {"x": x, "pick": pick, "wts": wts, **L.arrays(), "xs": xs}
    params = {"kind": "x3gm", "R": R, "slots": slots, "E": E, "D": D, "I": I, "shx": int(shx), "ragged": int(ragged),
              "input": "bf16" if input_dtype == torch.bfloat16 else "f16", "limit": limit}
    if ragged:
        tg, tu, td = L.tp["g"], L.tp["u"], L.tp["d"]
        k2w = torch.tensor(L.k2["g"] + [0], dtype=torch.int32, device=DEV)
        k2dw = torch.tensor(L.k2["d"] + [0], dtype=torch.int32, device=DEV)
    else:
        n_gate = E * D * I // 256 * 16 * k2g[0] // 2      # int16 elements of the gate stack
        tg = L.words[:n_gate].view(torch.int32).view(E, D // 16, I // 16, 4 * k2g[0])
        tu = L.words[n_gate:2 * n_gate].view(torch.int32).view(E, D // 16, I // 16, 4 * k2g[0])
        td = L.words[2 * n_gate:].view(torch.int32).view(E, I // 16, D // 16, 4 * k2d[0])
        params.update(base_u=n_gate * 2, base_d=4 * n_gate)
    ticket = torch.zeros(1, dtype=torch.int32, device=DEV)
    base_cfg = gm.config({})
    for r, run in enumerate(runs):
        xd = torch.zeros(P, I, dtype=torch.float16, device=DEV)
        y = torch.zeros(P, D, dtype=torch.float32, device=DEV)
        tk = True
        cg_all = cd_all = None
        if run != "tuned":
            cg_all, cd_all, tk = run
        for kind, widths in (("gu", sorted(set(k2g))), ("dn", sorted(set(k2d)))):
            for k2 in widths:
                if cg_all is None:
                    cg, cd = gm.tuned(base_cfg, R, slots, E, shx, k2gu=k2, k2d=k2)
                else:
                    cg, cd = cg_all, cd_all
                c = cg if kind == "gu" else cd
                if not gm.fits(kind, c, k2, 1 if shx else 2):
                    raise SystemExit(f"run {run}: {kind} cfg {c} does not fit K2 {k2}")
                pk = pick
                if ragged:
                    w = (k2dw if kind == "dn" else k2w)[pick.long()]
                    pk = torch.where(w == k2, pick, torch.full_like(pick, E))
                order, pe, poff, pcnt, npass = gm.plan(pk, E, gm.bm_of(kind, c))
                arrays[f"run{r}_{kind}_{k2}_pk"] = pk.contiguous()          # x3gm_plan.cu's input
                for part, t in zip(("order", "pe", "poff", "pcnt", "npass"), (order, pe, poff, pcnt, npass)):
                    arrays[f"run{r}_{kind}_{k2}_{part}"] = t
                ticket.zero_()
                if kind == "gu":
                    ext.gateup(xg, xu, tg, tu, order, pe, poff, pcnt, npass, ticket, L.svh_g, L.svh_u, L.suh_d, xd,
                               D, I, k2, shx, c, tk, float(limit), ragged)
                else:
                    ext.down(xd, td, order, pe, poff, pcnt, npass, ticket, L.svh_d, y, I, D, k2, c, tk, ragged)
                params[f"run{r}_{kind}_{k2}"] = c
        params[f"run{r}_ticket"] = int(tk)
        o = torch.zeros(R, D, dtype=torch.float32, device=DEV)
        up.combine(y, wts, o, R, D, slots)
        arrays.update({f"run{r}_xd": xd, f"run{r}_y": y, f"run{r}_out": o})
    params["runs"] = len(runs)
    if hasattr(ext, "gateup2"):                        # x3gm v2 (gm2_kernel): _run_v2's "one" mode, both tickets
        k2g_t = torch.tensor(L.k2["g"] + [k2g[0] if not ragged else 0], dtype=torch.int32, device=DEV)
        k2d_t = torch.tensor(L.k2["d"] + [k2d[0] if not ragged else 0], dtype=torch.int32, device=DEV)
        cg, cd = gm.tuned2({"gu": None, "dn": None}, shx)
        order, pe, poff, pcnt, npass = gm.plan(pick, E, gm.bm_of("gu", cg))
        arrays.update(v2_k2g=k2g_t, v2_k2d=k2d_t, v2_order=order, v2_pe=pe, v2_poff=poff, v2_pcnt=pcnt, v2_npass=npass)
        for tk in (True, False):
            xd = torch.zeros(P, I, dtype=torch.float16, device=DEV)
            y = torch.zeros(P, D, dtype=torch.float32, device=DEV)
            ticket.zero_()
            ext.gateup2(xg, xu, tg, tu, k2g_t, order, pe, poff, pcnt, npass, ticket, L.svh_g, L.svh_u, L.suh_d, xd, D,
                        I, shx, cg, tk, float(limit), ragged)
            ticket.zero_()
            ext.down2(xd, td, k2d_t, order, pe, poff, pcnt, npass, ticket, L.svh_d, y, I, D, cd, tk, ragged)
            o = torch.zeros(R, D, dtype=torch.float32, device=DEV)
            up.combine(y, wts, o, R, D, slots)
            torch.cuda.synchronize()
            if tk:                                     # v2 == gm_kernel, bit for bit
                assert torch.equal(xd, arrays["run0_xd"]) and torch.equal(y, arrays["run0_y"]), "gm2 != gm"
            arrays.update({f"v2_t{int(tk)}_xd": xd, f"v2_t{int(tk)}_y": y, f"v2_t{int(tk)}_out": o})
        params.update(v2=1, v2_gu=cg, v2_dn=cd)
    if dequant:
        K, N, k2 = dequant
        t = trellis(g, K, N, k2).to(DEV)
        a = torch.empty(K, N, dtype=torch.float16, device=DEV)
        ext.dequant(t.view(torch.int32), a, k2)
        b = torch.empty_like(a)
        up.dequant(t, b, k2, CB)
        arrays.update(dequant_in=t, dequant_gm=a, dequant_up=b)
        params.update(dequant_k=K, dequant_n=N, dequant_k2=k2)
    torch.cuda.synchronize()
    save(out, params, arrays)
    return {"runs": len(runs), "ragged": ragged}


def plan_case(out: Path, g) -> dict:
    """x3gm.plan at prod sizes (2,048 / 4,096-row blocks over 384 experts, DSpark's 128) with pairs sent past the
    table (a ragged launch's other widths), for x3gm_plan.cu."""
    cases = [(2048, 6, 384, 64, 0.0), (4096, 6, 384, 128, 0.3), (1, 6, 384, 64, 0.0), (2048, 6, 128, 64, 0.5),
             (3, 7, 13, 64, 0.0), (700, 6, 384, 128, 0.9)]
    arrays, params = {}, {"kind": "plan", "cases": len(cases)}
    for i, (R, S, E, bm, past) in enumerate(cases):
        pk = torch.randint(0, E, (R, S), generator=g, dtype=torch.int32)
        pk[torch.rand(R, S, generator=g) < past] = E
        pk = pk.to(DEV)
        order, pe, poff, pcnt, npass = gm.plan(pk, E, bm)
        params.update({f"c{i}_E": E, f"c{i}_bm": bm, f"c{i}_P": R * S})
        arrays.update({f"c{i}_pk": pk, f"c{i}_order": order, f"c{i}_pe": pe, f"c{i}_poff": poff, f"c{i}_pcnt": pcnt,
                       f"c{i}_npass": npass})
    torch.cuda.synchronize()
    save(out, params, arrays)
    return {"cases": len(cases)}


def topk_case(out: Path, g) -> dict:
    """pick.top on a rank's vocabulary half (64,640 columns) of bf16-valued logits with many exact ties."""
    from tensorfold.families.deepseek_v41.cuda import pick
    R, C = 6, 64640
    lg = (torch.randint(-96, 97, (R, C), generator=g).float() / 16.0)          # bf16-exact, heavy ties
    lg[1] = torch.randn(C, generator=g).to(torch.bfloat16).float()               # a realistic row
    lg[2, ::7] = 6.0                                                              # one value at many columns
    lg[3] = -0.0
    lg = lg.to(DEV)
    arrays, params = {"lg": lg}, {"kind": "topk", "R": R, "C": C, "ks": 4}
    for i, k in enumerate((1, 5, 64, 1024)):
        vals, cols = pick.top(lg, k)
        arrays.update({f"vals{i}": vals, f"cols{i}": cols})
        params[f"k{i}"] = k
    torch.cuda.synchronize()
    save(out, params, arrays)
    return {"rows": R}


def pointwise_case(out: Path, g) -> dict:
    """prefill_moe's torch pointwise ops (shared_expert's clamps / silu / mul, forward's adds) on values that cross
    the limit, with NaN / inf / -0.0 and silu's extremes."""
    R, n, limit = 7, 2304, 10.0
    gg = torch.randn(R, n, generator=g) * 8
    uu = torch.randn(R, n, generator=g) * 8
    gg[0, :6] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0, 100.0, -100.0])
    uu[0, :6] = torch.tensor([1.0, float("nan"), 2.0, float("inf"), -float("inf"), -0.0])
    gg, uu = gg.to(DEV), uu.to(DEV)
    act = torch.nn.functional.silu(gg.clone().clamp_(max=limit)).mul_(uu.clone().clamp_(min=-limit, max=limit))
    a = (torch.randn(R, 5120, generator=g) * 3).to(DEV)
    b = (torch.randn(R, 5120, generator=g) * 3).to(DEV)
    s32 = a.clone()
    s32 += b
    y16 = torch.empty((R, 5120), dtype=torch.bfloat16, device=DEV)
    torch.add(a, b, out=y16)
    torch.cuda.synchronize()
    save(out, {"kind": "pointwise", "R": R, "n": n, "D": 5120, "limit": limit},
         {"g": gg, "u": uu, "act": act, "a": a, "b": b, "sum32": s32, "sum16": y16})
    return {"rows": R}

KEY_NONE = -(2 ** 63)


def top_positions(k: torch.Tensor, count: int) -> torch.Tensor:
    """index.top_positions at 8474f31, verbatim."""
    R, n = k.shape
    top = torch.topk(k, min(count, n), dim=1, sorted=False).values
    pos = torch.where(top == KEY_NONE, torch.full_like(top, -1), 0x7FFFFFFF - (top & 0xFFFFFFFF))
    pos = torch.where(pos < 0, torch.full_like(pos, 2 ** 31 - 1), pos)
    pos = torch.sort(pos, dim=1).values
    pos = torch.where(pos == 2 ** 31 - 1, torch.full_like(pos, -1), pos)
    if pos.shape[1] < count:
        pos = torch.cat([pos, torch.full((R, count - pos.shape[1]), -1, dtype=pos.dtype, device=pos.device)], 1)
    return pos.to(torch.int32).contiguous()


def pfglue_case(out: Path, g) -> dict:
    """prefill_glue.cu's references: block_prefill.zig's glue steps as the Python engine runs them (8474f31:
    index.top_positions / candidate_keys, backend.visible_counts, prefill_moe's kit picks, x3gm._run_ragged's width
    mask, the indexer head weights' casts, blocks._projection, blocks.stage_rows), on edge inputs."""
    arrays, params = {}, {"kind": "pfglue"}
    # top_positions: unique keys (score bits above, 0x7FFFFFFF - position below, as sort_keys builds them), KEY_NONE
    # pads, low halves past 0x7FFFFFFF (a negative position: dropped), rows shorter and longer than count
    tp = [(4, 3000, 512), (3, 300, 512), (2, 20000, 2048), (2, 1500, 2048)]
    for i, (R, n, count) in enumerate(tp):
        hi = torch.randint(-(2 ** 31), 2 ** 31, (R, n), generator=g, dtype=torch.int64)
        hi[:, : n // 7] = hi[0, 0]                                    # many equal scores: the positions decide
        lo = torch.stack([torch.randperm(n, generator=g) for _ in range(R)]).to(torch.int64)
        k = (hi << 32) | (0x7FFFFFFF - lo)
        k[:, 1::13] = KEY_NONE
        k[0, 2::29] = (hi[0, 2::29] << 32) | 0xFFFFFFF0               # low half past 0x7FFFFFFF
        k = k.to(DEV)
        arrays[f"tp{i}_keys"] = k
        arrays[f"tp{i}_want"] = top_positions(k, count)
        params.update({f"tp{i}_R": R, f"tp{i}_n": n, f"tp{i}_count": count})
    params["tp"] = len(tp)
    # candidate_keys: ascending blocks, -1 padded
    R, nb, bs = 3, 2048, 8
    blocks = torch.sort(torch.randint(0, 50000, (R, nb), generator=g), dim=1).values.to(torch.int32)
    blocks[:, 1900:] = -1
    blocks = blocks.to(DEV)
    j = blocks.to(torch.int64)[:, :, None] * bs + torch.arange(bs, device=DEV)
    j = torch.where(blocks[:, :, None] >= 0, j, torch.full_like(j, -1))
    arrays.update(ck_blocks=blocks, ck_want=j.reshape(R, -1).to(torch.int32).contiguous())
    params.update(ck_R=R, ck_nb=nb, ck_bs=bs)
    # visible_counts over top_positions' selections, ratios 4 and 128
    sel = arrays["tp0_want"]
    pos = torch.tensor([0, 1500, 9000, 2 ** 20], dtype=torch.int64, device=DEV)
    for ratio in (4, 128):
        nvis = (pos + 1) // ratio
        arrays[f"vc{ratio}_want"] = ((sel >= 0) & (sel.to(torch.long) < nvis[:, None])).sum(1).to(torch.int32)
    arrays["vc_pos"] = pos
    # the prefill MoE's picks: 6 routed slots + the shared one, kit weights through fp16 (edges included)
    n, slots, topk = 37, 7, 6
    pick = torch.randint(0, 384, (n, slots), generator=g, dtype=torch.int32)
    pick[:, topk:] = 384
    wts = torch.rand(n, slots, generator=g) * 2
    wts[0, :6] = torch.tensor([65520.0, 1e-8, 6.1e-5, float("nan"), -0.0, 1.0 + 2 ** -11])
    pick, wts = pick.to(DEV), wts.to(DEV)
    arrays.update(gp_pick=pick, gp_wts=wts, gp_pk=pick[:, :topk].contiguous(),
                  gp_w6=wts.to(torch.float16).to(torch.float32)[:, :topk].contiguous())
    params.update(gp_n=n, gp_slots=slots, gp_topk=topk)
    # x3gm._run_ragged's width mask (picks of E: the masked shared slot)
    E, P = 384, 999
    tab = torch.tensor([[4, 6, 8, 10][i % 4] for i in range(E)], dtype=torch.int32)
    pk = torch.randint(0, E + 1, (P,), generator=g, dtype=torch.int32)
    tab, pk = tab.to(DEV), pk.to(DEV)
    k2w = torch.cat([tab, torch.zeros(1, dtype=torch.int32, device=DEV)])
    for k2 in (4, 10):
        arrays[f"wm{k2}_want"] = torch.where(k2w[pk.long()] == k2, pk, torch.full_like(pk, E))
    arrays.update(wm_pick=pk, wm_tab=tab)
    params.update(wm_E=E, wm_P=P)
    # the indexer head weights: fp64 -> bf16 -> fp32 (double rounding cases, NaN, inf, overflow, denormals)
    w = torch.randn(4096, generator=g, dtype=torch.float64) * 3
    w[:12] = torch.tensor([float("nan"), float("inf"), -float("inf"), 1e300, -1e300, 1e-300, 4e-39,
                           1.0 + 2 ** -8 + 2 ** -30, 1.0 + 2 ** -8 - 2 ** -30, 1.0 + 2 ** -8, 3.4e38, -0.0],
                          dtype=torch.float64)
    w = w.to(DEV)
    arrays.update(f64_in=w, f64_want=w.to(torch.bfloat16).to(torch.float32))
    # blocks._projection: [kv | gate] fp32 and kv alone
    n2, hd = 33, 512
    kv = torch.randn(n2, hd, generator=g).to(torch.bfloat16).to(DEV)
    gate = torch.randn(n2, hd, generator=g).to(torch.bfloat16).to(DEV)
    arrays.update(pj_kv=kv, pj_gate=gate, pj_two=torch.cat([kv.float(), gate.float()], 1).contiguous(),
                  pj_one=kv.float().contiguous())
    params.update(pj_n=n2, pj_hd=hd)
    # blocks.stage_rows: a window of rows between rings of different sizes (4-byte and byte rows)
    for i, (sr, dr, rb, lo, hi) in enumerate([(4096, 256, 512, 1500, 1627), (256, 4096, 6, 9000, 9128)]):
        src = torch.randint(0, 256, (sr, rb), generator=g, dtype=torch.uint8).to(DEV)
        dst = torch.randint(0, 256, (dr, rb), generator=g, dtype=torch.uint8).to(DEV)
        want = dst.clone()
        p = torch.arange(lo, hi, dtype=torch.long, device=DEV)
        want[p % dr] = src[p % sr]
        arrays.update({f"rc{i}_src": src, f"rc{i}_dst": dst, f"rc{i}_want": want})
        params.update({f"rc{i}_sr": sr, f"rc{i}_dr": dr, f"rc{i}_rb": rb, f"rc{i}_lo": lo, f"rc{i}_hi": hi})
    torch.cuda.synchronize()
    save(out, params, arrays)
    return {"cases": 8}


def glue_case(out: Path, g) -> dict:
    """The window's torch glue (forward.py / pick.py) on edge values: ids out of the rank's range, -0 / +0, NaN,
    inf, halfway and denormal values."""
    V, D, lo, W, n = 200, 512, 1000, 3, 7
    embed = torch.randn(V, D, generator=g).to(torch.bfloat16)
    embed[3, :4] = torch.tensor([-0.0, float("nan"), float("inf"), 1e-40]).to(torch.bfloat16)
    embed = embed.to(DEV)
    ids = torch.tensor([1003, 999, 1199, 1200, 1000, 5, 1003], dtype=torch.long, device=DEV)
    local = ids - lo                                                       # forward.embed_rows, then the bf16 send
    ok = (local >= 0) & (local < V)
    rows = embed.index_select(0, local.clamp(0, V - 1)).to(torch.float32)
    rows = torch.where(ok[:, None], rows, torch.zeros_like(rows))
    send = rows.to(torch.bfloat16)
    recv = torch.randn(W, n, D, generator=g).to(torch.bfloat16)
    recv[0, :, :8] = -0.0
    recv[1, :, :4] = 0.0
    recv[2, 0, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")]).to(torch.bfloat16)
    recv = recv.to(DEV)
    parts = recv.float()                                                   # forward.allsum (parts fp32, rank order)
    acc = parts[0].clone()
    for r in range(1, W):
        acc += parts[r]
    summed = acc.to(torch.bfloat16).repeat(1, 4).contiguous()
    x = torch.randn(n, D, generator=g) * 3
    x[0, :8] = torch.tensor([float("nan"), float("inf"), -0.0, 1e-41, 70000.0, -70000.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -9])
    x[1, :4] = torch.tensor([2 ** -20, 6.1e-5, 5.9e-8, 1.0 + 2 ** -11])
    x = x.to(DEV)
    cast = x.to(torch.bfloat16)
    kw = x.to(torch.float16).to(torch.float32)                             # pick.kit_weights
    kl = x.to(torch.bfloat16).to(x.dtype)                                  # pick.kit_logits
    gc = torch.randn(W, n, 24, generator=g).to(torch.bfloat16).to(DEV)
    cols = gc.view(W, n, 24).permute(1, 0, 2).reshape(n, -1).contiguous()  # forward.gather_cols
    src = torch.randn(5, 1100, generator=g).to(torch.bfloat16).to(DEV)[:, :1024]
    carry = src[2].to(torch.float32).contiguous()
    torch.cuda.synchronize()
    save(out, {"kind": "glue", "V": V, "D": D, "lo": lo, "W": W, "n": n, "c": 24, "ld": 1100, "row": 2, "start": 3000},
         {"embed": embed, "ids": ids, "send": send, "recv": recv, "summed": summed, "x": x, "cast": cast, "kw": kw,
          "kl": kl, "gc": gc, "cols": cols, "src": src.as_strided((5, 1100), (1100, 1)),   # the whole [5, 1100] rows (row stride 1,100)
          "carry": carry})
    return {"rows": n}


def mhcdec_case(out: Path, g, *, R: int, mode: str, tap: bool = False, in_place: bool = False) -> dict:
    """An mHC site of a decode window on Python's decode path with mhc_cuda and mhc_pf off (Triton: `_site_dec` at
    <= 16 rows split, `_site` above, then `_finish_k`): TF_DSV41_MHC_PFDEC's reference (mhc_pf's site_kernel + the
    normed input + mhc_cuda's coef_kernel must give every output's bits). D = 5,120 (both kernels' constant), bf16
    `fn` and partials (prod), 2 ranks; streams with -0.0 and large values."""
    from tensorfold.families.deepseek_v41.cuda import mhc
    D, W = 5120, 2
    bf, f32 = torch.bfloat16, torch.float32
    x = torch.randn(R, 4 * D, generator=g) * 2
    x[0, :4] = torch.tensor([-0.0, 0.0, 3.0e4, -2.5e-30])
    x = x.to(bf).to(DEV)
    gathered = (torch.randn(W, R, D, generator=g) * 0.5).to(bf).to(DEV)
    sig = lambda t: torch.sigmoid(t)
    ppre = (sig(torch.randn(R, 4, generator=g)) + 1e-6).to(f32).to(DEV)
    ppost = (2 * sig(torch.randn(R, 4, generator=g))).to(f32).to(DEV)
    pcomb = torch.softmax(torch.randn(R, 4, 4, generator=g), -1).reshape(R, 16).to(f32).to(DEV)
    fn = (torch.randn(24, 4 * D, generator=g) * 0.02).to(bf).to(DEV)
    base = (torch.randn(24, generator=g) * 0.5).to(f32).to(DEV)
    scale = (torch.rand(3, generator=g) + 0.05).to(f32).to(DEV)
    nw = (1 + 0.1 * torch.randn(D, generator=g)).to(f32).to(DEV)
    xout0 = torch.full((R, 4 * D), -7.0, dtype=bf, device=DEV)
    taps0 = torch.full((R, 3 * D), 5.0, dtype=bf, device=DEV)       # DSpark's tap buffer: column block 1 is this site's
    out_ = torch.full((R, D), -3.0, dtype=bf, device=DEV)
    scratch = mhc.Scratch(R, DEV)
    scratch.part.fill_(0.0)
    coefs = mhc.Coefs(R, DEV)
    prev = mhc.Coefs(R, DEV, pre=ppre.clone(), post=ppost.clone(), comb=pcomb.clone())
    hc = mhc.Hc(fn, base, scale)
    xs = x.clone()
    xout = xs if in_place else xout0.clone()
    taps = taps0.clone()
    saved = {k: os.environ.get(k) for k in ("TF_DSV41_MHC_CUDA", "TF_DSV41_MHC_PF")}
    os.environ["TF_DSV41_MHC_CUDA"] = "0"
    os.environ["TF_DSV41_MHC_PF"] = "0"
    try:
        if mode == "boundary":
            mhc.boundary(xs, xout, gathered, prev, hc, nw, out_, coefs, scratch, taps[:, D:2 * D] if tap else None)
        else:
            mhc.site(xs, hc, nw, out_, coefs, scratch, ppre if mode == "site2" else None)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    torch.cuda.synchronize()
    arrays = {"x": x, "xout0": xout0, "g": gathered, "ppre": ppre, "ppost": ppost, "pcomb": pcomb, "fn": fn,
              "base": base, "scale": scale, "nw": nw, "taps0": taps0, "out0": torch.full((R, D), -3.0, dtype=bf),
              "xs": xs, "xout": xout, "taps": taps, "part": scratch.part, "c": scratch.c, "out": out_,
              "npre": coefs.pre, "npost": coefs.post, "ncomb": coefs.comb}
    save(out, {"kind": "mhcdec", "R": R, "D": D, "W": W, "mode": mhc_mode[mode], "tap": int(tap),
               "in_place": int(in_place), "eps": mhc.EPS, "hc_eps": mhc.HC_EPS, "post_alpha": mhc.POST_ALPHA,
               "iters": mhc.ITERS}, arrays)
    return {"rows": R, "mode": mode, "tap": tap, "in_place": in_place}


mhc_mode = {"boundary": 0, "site1": 1, "site2": 2}

def kv_norm_store_case(out: Path, rows: bool) -> dict:
    from tensorfold.families.deepseek_v41.cuda import rmsnorm
    from tensorfold.families.deepseek_v41.cuda.csa2 import compress
    g = torch.Generator().manual_seed(4143)
    n, ring = 24, 256
    x = torch.randn(n, 512, generator=g).to(torch.bfloat16).to(DEV)
    x[0].zero_()
    w = (torch.randn(512, generator=g) * 0.3 + 1).to(DEV)
    angles = torch.randn(1024, 32, generator=g).to(DEV)
    cs = torch.cat((angles.cos(), angles.sin()), dim=1).contiguous()
    pos = ((torch.arange(n, device=DEV) * 7 + 251).to(torch.int64) if rows
           else torch.tensor([251], dtype=torch.int32, device=DEV))
    sl = (torch.arange(n, device=DEV) % 3).to(torch.int64)
    sl[::5] = -1
    v = torch.full((3 * ring + 1, 576), 0xa5, dtype=torch.uint8, device=DEV)
    scales = torch.full((3 * ring + 1, 8), 0xa5, dtype=torch.uint8, device=DEV)
    norm = torch.empty_like(x)
    rmsnorm._rms[(n,)](x, 512, w, norm, 512, 512, 1.0 / 512.0, 1e-20,
                       BK=512, HAS_W=True, NARROW=True, PDL=False, num_warps=4, **rmsnorm.LAUNCH)
    compress._kv_store[(n,)](norm, 512, cs, 64, v, scales, 576, 8, pos,
                            RATIO=0, RING=ring, PT=None, PSH=0, SL=sl if rows else None,
                            PTS=0, ROWS=rows, PDL=False, num_warps=4)
    torch.cuda.synchronize()
    save(out, {"kind": "kvglue", "n": n, "ring": ring, "rows": int(rows), "eps": 1e-20},
         {"x": x, "w": w, "cs": cs, "pos": pos, "sl": sl, "norm": norm, "v": v, "s": scales})
    return {"rows": n, "row_mode": rows}


def want(name: str) -> bool:
    """TF_DSV41_ORACLE_ONLY: a regex of the case names to write (unset: every case)."""
    pat = os.environ.get("TF_DSV41_ORACLE_ONLY")
    return pat is None or re.match(pat, name) is not None


def main() -> None:
    out = Path(sys.argv[1]) / "fixtures"
    g = torch.Generator().manual_seed(4141)
    # Independent generators leave every existing fixture's random sequence unchanged.
    report = {}
    for rows in (False, True):
        name = "kv_norm_store_rows" if rows else "kv_norm_store_scalar"
        if want(name):
            report[name] = kv_norm_store_case(out / name, rows)
    D, I, E = 1024, 512, 13
    # decode / verify (x3ld): routed 2-3.5 bits + a 6-bit shared expert (E1 (2, 12)), 5-bit shared (2, 10), DSpark (8, 8)
    mixed_g = [4, 5, 6, 7, 4, 5, 6, 7, 4, 6, 5, 7, 12]
    mixed_d = [5, 4, 7, 6, 5, 4, 6, 7, 5, 4, 6, 7, 12]
    # Wide decode glue: runs the actual Zig groupRotIn ABI at its production input width.
    if want("experts_group_rot_r24"):
        report["experts_group_rot_r24"] = experts_case(
            out / "experts_group_rot_r24", torch.Generator().manual_seed(4142), R=24, slots=7, E=E, D=5120, I=I,
            k2g=mixed_g, k2d=mixed_d, input_dtype=torch.bfloat16, limit=10.0, skip_every=2)
    for R in (1, 3, 16):
        report[f"experts_e1_r{R}"] = experts_case(out / f"experts_e1_r{R}", g, R=R, slots=7, E=E, D=D, I=I,
                                                  k2g=mixed_g, k2d=mixed_d, input_dtype=torch.bfloat16, limit=10.0,
                                                  skip_every=2 if R > 1 else 0)
    if want("experts_k210_r8"): report["experts_k210_r8"] = experts_case(out / "experts_k210_r8", g, R=8, slots=7, E=E, D=D, I=I,
                                             k2g=[4] * 12 + [10], k2d=[6] * 12 + [8], input_dtype=torch.float16,
                                             limit=math.inf)
    if want("experts_dspark_r4"): report["experts_dspark_r4"] = experts_case(out / "experts_dspark_r4", g, R=4, slots=7, E=E, D=D, I=I,
                                               k2g=[8] * E, k2d=[8] * E, input_dtype=torch.bfloat16, limit=10.0)
    # prefill windows (x3pf beside x3ld / upstream)
    if want("experts_pf_r64"): report["experts_pf_r64"] = experts_case(out / "experts_pf_r64", g, R=64, slots=7, E=E, D=D, I=I, k2g=mixed_g,
                                            k2d=mixed_d, input_dtype=torch.bfloat16, limit=10.0, prefill=True)
    # x3gm: uniform (one launch a projection) and ragged (one a width), every configuration that fits
    E2 = 16
    cfgs = ["tuned", (0, 0, True), (1, 1, False), (2, 2, True), (3, 3, True), (4, 4, True), (5, 0, False)]
    if want("x3gm_k6_shx"): report["x3gm_k6_shx"] = x3gm_case(out / "x3gm_k6_shx", g, R=128, slots=6, E=E2, D=D, I=I, k2g=[6] * E2,
                                      k2d=[6] * E2, shx=True, input_dtype=torch.bfloat16, limit=10.0, runs=cfgs,
                                      dequant=(256, 256, 6))
    if want("x3gm_k4_two"): report["x3gm_k4_two"] = x3gm_case(out / "x3gm_k4_two", g, R=128, slots=6, E=E2, D=D, I=I, k2g=[4] * E2,
                                      k2d=[4] * E2, shx=False, input_dtype=torch.bfloat16, limit=10.0,
                                      runs=["tuned", (0, 0, True), (1, 1, True), (2, 2, False), (4, 4, True),
                                            (5, 3, True)], dequant=(128, 256, 4))
    widths_g = [3, 4, 5, 6, 7, 8, 10, 4, 3, 5, 6, 7, 8, 10, 4, 6]
    widths_d = [10, 8, 7, 6, 5, 4, 3, 6, 4, 4, 3, 5, 7, 8, 10, 6]
    if want("x3gm_ragged_shx"): report["x3gm_ragged_shx"] = x3gm_case(out / "x3gm_ragged_shx", g, R=256, slots=6, E=E2, D=D, I=I, k2g=widths_g,
                                          k2d=widths_d, shx=True, input_dtype=torch.bfloat16, limit=10.0,
                                          runs=["tuned", (0, 0, True), (2, 2, True), (5, 1, False)],
                                          dequant=(256, 128, 3))
    if want("x3gm_ragged_two"): report["x3gm_ragged_two"] = x3gm_case(out / "x3gm_ragged_two", g, R=200, slots=6, E=E2, D=D, I=I,
                                          k2g=widths_g, k2d=widths_d, shx=False, input_dtype=torch.float16,
                                          limit=10.0, runs=["tuned", (1, 0, True), (2, 4, True)],
                                          dequant=(128, 128, 10))
    # prefill chunks (TF_DSV41_PREFILL_CHUNK / ROWS 2,048, adaptive 1,024 / 512, a short tail): the tuned v1 run (v2's
    # reference) and v2 at both tickets; 2,048 rows over prod's 384 experts (~32 passes an expert at bm 64)
    widths_g384 = [widths_g[i % len(widths_g)] for i in range(384)]
    widths_d384 = [widths_d[(i * 7) % len(widths_d)] for i in range(384)]
    if want("x3gm_pf2048_ragged"): report["x3gm_pf2048_ragged"] = x3gm_case(out / "x3gm_pf2048_ragged", g, R=2048, slots=6, E=384, D=D, I=I,
                                             k2g=widths_g384, k2d=widths_d384, shx=True, input_dtype=torch.bfloat16,
                                             limit=10.0, runs=["tuned"])
    # production geometry (2,048-row capture: D 5,120, I 1,152, gate / up widths 4 / 6 / 8 / 10), over 16 experts
    # (~768 rows an expert: many passes each) so the fixture stays near 1 GB
    if want("x3gm_pf2048_prod"): report["x3gm_pf2048_prod"] = x3gm_case(out / "x3gm_pf2048_prod", g, R=2048, slots=6, E=16, D=5120, I=1152,
                                           k2g=[4, 6, 8, 10] * 4, k2d=[6, 4, 10, 8] * 4, shx=False,
                                           input_dtype=torch.bfloat16, limit=10.0, runs=["tuned"])
    if want("x3gm_pf1024_k4"): report["x3gm_pf1024_k4"] = x3gm_case(out / "x3gm_pf1024_k4", g, R=1024, slots=6, E=64, D=D, I=I, k2g=[4] * 64,
                                         k2d=[4] * 64, shx=False, input_dtype=torch.bfloat16, limit=10.0, runs=["tuned"])
    if want("x3gm_pf512_shx"): report["x3gm_pf512_shx"] = x3gm_case(out / "x3gm_pf512_shx", g, R=512, slots=6, E=64, D=D, I=I, k2g=[6] * 64,
                                         k2d=[6] * 64, shx=True, input_dtype=torch.bfloat16, limit=10.0, runs=["tuned"])
    if want("x3gm_pf_tail1337"): report["x3gm_pf_tail1337"] = x3gm_case(out / "x3gm_pf_tail1337", g, R=1337, slots=6, E=64, D=D, I=I,
                                           k2g=widths_g384[:64], k2d=widths_d384[:64], shx=False,
                                           input_dtype=torch.bfloat16, limit=10.0, runs=["tuned"])
    if want("x3gm_plan"): report["x3gm_plan"] = plan_case(out / "x3gm_plan", g)
    if want("topk_keys"): report["topk_keys"] = topk_case(out / "topk_keys", g)
    if want("pointwise"): report["pointwise"] = pointwise_case(out / "pointwise", g)
    if want("glue"): report["glue"] = glue_case(out / "glue", g)
    if want("pfglue"): report["pfglue"] = pfglue_case(out / "pfglue", g)
    # TF_DSV41_MHC_PFDEC: mHC sites of decode windows (1-64 rows) on Python's Triton decode path
    for R, mode, tap, ip in ((1, "boundary", True, False), (5, "boundary", True, False), (16, "boundary", True, False),
                             (17, "boundary", True, False), (20, "boundary", False, False), (24, "boundary", True, False),
                             (32, "boundary", True, False), (33, "boundary", True, True), (48, "boundary", False, True),
                             (64, "boundary", True, True), (5, "boundary", False, True), (20, "boundary", False, True),
                             (5, "site1", False, False), (24, "site1", False, False), (5, "site2", False, False),
                             (24, "site2", False, False)):
        name = f"mhcdec_{mode}_r{R}{'_tap' if tap else ''}{'_inplace' if ip else ''}"
        if want(name): report[name] = mhcdec_case(out / name, g, R=R, mode=mode, tap=tap, in_place=ip)
    (out.parent / "oracle.json").write_text(json.dumps({"device": torch.cuda.get_device_name(0), **report}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
