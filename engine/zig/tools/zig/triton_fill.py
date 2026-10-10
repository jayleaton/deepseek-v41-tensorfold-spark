#!/usr/bin/env python3
"""Triton variants compiled ahead of time from Python's own Triton source, for every specialization a served Zig config
can launch (``tf-dsv41-m1 aot-needs``), so serving no longer depends on which shapes a reference run happened to launch.

Each variant goes through the JIT's own steps, without a GPU: the function's binder over stand-in arguments of the
recorded specialization (CPU tensors of the dtype, 16-byte aligned or not; ints of the value class; the constexprs as
given), ``JITFunction._pack_args``, ``ASTSource`` and ``triton.compile`` for the served target (``create_binder`` /
``_do_compile`` with ``driver.active`` replaced by the target). So the signature, constexprs, attrs and options are the
ones Python's launch would make; ``verify`` checks it against captured variants (signature and attrs equal, then the
cubin's SASS, and its bytes where the reference cubins are at hand).

Byte identity with the served image's JIT needs its toolchain and its files:
- Triton 3.7.1 and ptxas V13.3.73 (the image's /usr/local/cuda/bin/ptxas: TRITON_PTXAS_PATH / TRITON_PTXAS_BLACKWELL_PATH;
  PyPI nvidia-cuda-nvcc==13.3.73 ships it);
- the line info's paths: the kernels' files as the image imported them (``--served-py`` /dsv41-tf/src, ``--served-triton``
  /usr/local/lib/python3.12/dist-packages/triton; get_jit_fn_file_line is mapped);
- ptxas writes each file's mtime and size into the DWARF line table, from the file at that path when it compiles:
  ``--stamps`` (``stamps`` reads them back from any cubin set of the image, or stats the image's tree) compiles inside a
  bubblewrap root where those paths exist with those stamps. Without stamps the code (SASS) is the same and only the
  debug line table's file mtimes differ.

Options (num_warps, num_stages, maxnreg, ...) are a call site's, not the arguments': ``options`` derives each function's
rule from captured manifests (one set a function, or a rule of its constexprs: mhc_dec.launch's maxnreg) and checks it
against every captured variant; ``compile`` refuses a function without one.

    triton_fill.py sigs    --py SRC --out sigs.json
    triton_fill.py options --manifests M.json... --out options.json
    triton_fill.py stamps  --cubins DIR... | --root DIR --served ROOT  --out stamps.json
    triton_fill.py verify  --py SRC --manifest M.json [--cubins DIR] [--stamps S] [--only NAME]
    triton_fill.py compile --py SRC --needs needs.json --options options.json [--have AOTDIR...] [--stamps S] --out AOTDIR
    triton_fill.py sass    --fill AOTDIR --served AOTDIR      (the same specializations' SASS and bytes; no Triton needed)
    triton_fill.py merge   SERVED_AOTDIR FILL_AOTDIR          (the fill's variants the served set lacks, by specialization)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVED_PY = "/dsv41-tf/src"
SERVED_TRITON = "/usr/local/lib/python3.12/dist-packages/triton"
# where the served family's kernels live (the EXL3 prefill GEMM is shared code): other families reuse kernel names
KERNEL_DIRS = ("tensorfold/families/deepseek_v41", "tensorfold/cuda")
# the Zig engine's own Triton kernels (no Python engine twin, e.g. the row-blocked stream top-k): a package beside this
# script, imported after the served tree (they reuse its helpers)
OWN_DIR = HERE / "dsv41_triton"
OWN_PKG = "dsv41_zig_triton"
# call-site options a launch passes (the rest are the backend's defaults, the same for every launch)
OPTION_KEYS = ("num_warps", "num_stages", "num_ctas", "maxnreg", "enable_fp_fusion", "launch_pdl",
               "launch_cooperative_grid")


def jsonable(v):
    """A constexpr as triton_aot_manifest records it (floats with their fp32 / fp64 bits)."""

    if isinstance(v, float):
        return {"float": v, "fp32_bits": "0x%08x" % struct.unpack("<I", struct.pack("<f", v))[0],
                "fp64_bits": "0x%016x" % struct.unpack("<Q", struct.pack("<d", v))[0]}
    if hasattr(v, "value") and type(v).__name__ == "constexpr":
        return jsonable(v.value)
    return v


def unjson(v):
    """A recorded constexpr back to its Python value (a float from its fp64 bits: exact)."""

    if isinstance(v, dict) and "fp64_bits" in v:
        return struct.unpack("<d", struct.pack("<Q", int(v["fp64_bits"], 16)))[0]
    if isinstance(v, dict) and "float" in v:
        return float(v["float"])
    return v


# ---- the Python tree's kernels ----------------------------------------------------------------------------------

def load_tree(py: Path, served_py: str, served_triton: str) -> dict:
    """Imports every module of the family that defines a Triton kernel; maps the line info's paths to the served
    image's. Returns {kernel name: [JITFunction, ...]}."""

    sys.path.insert(0, str(py))
    import triton
    import triton.compiler.code_generator as cg
    from triton.runtime.jit import JITFunction

    pairs = [(str(py), served_py), (os.path.dirname(triton.__file__), served_triton)]
    orig = cg.get_jit_fn_file_line

    def mapped(fn):
        name, line = orig(fn)
        for a, b in pairs:
            if name.startswith(a):
                return b + name[len(a):], line
        return name, line

    cg.get_jit_fn_file_line = mapped
    sys.path.append(str(OWN_DIR))
    out: dict = {}
    files = [(f, py) for d in KERNEL_DIRS for f in sorted((py / d).rglob("*.py"))]
    files += [(f, OWN_DIR) for f in sorted((OWN_DIR / OWN_PKG).rglob("*.py"))]
    for f, root in files:
        text = f.read_text(errors="replace")
        if "triton.jit" not in text:
            continue
        mod = ".".join(f.relative_to(root).with_suffix("").parts)
        try:
            m = importlib.import_module(mod)
        except Exception as e:                      # a module that needs a GPU extension at import
            print(f"[fill] skip {mod}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        for v in vars(m).values():
            if isinstance(v, JITFunction) and v.fn.__module__ == mod:
                out.setdefault(v.fn.__name__, []).append(v)
    return out


def full_name(fn) -> str:
    return f"{fn.fn.__module__}.{fn.fn.__qualname__}"


def sigs(kernels: dict) -> dict:
    """{name: [{function, params: [{name, constexpr, nospec, noalign}]}]}: what the Zig side reduces calls with."""

    out = {}
    for name, fns in sorted(kernels.items()):
        out[name] = [{"function": full_name(fn), "params": [
            {"name": p.name, "constexpr": bool(p.is_constexpr), "nospec": bool(p.do_not_specialize),
             "noalign": bool(p.do_not_specialize_on_alignment)} for p in fn.params]} for fn in fns]
    return out


def resolve(kernels: dict, name: str, function: str | None, arg_names) -> object:
    """The JITFunction a launch names: by its module path when known, else the one whose parameters hold the call's."""

    fns = kernels.get(name) or []
    if function:
        for fn in fns:
            if full_name(fn) == function:
                return fn
        raise KeyError(f"{function}: not in the tree")
    fit = [fn for fn in fns if set(arg_names) <= set(fn.arg_names)]
    if len(fit) != 1:
        raise KeyError(f"{name}: {len(fit)} kernels take {sorted(arg_names)}")
    return fit[0]


# ---- one variant through the JIT's steps ------------------------------------------------------------------------

_DT = None


def torch_dtype(ptr_type: str):
    global _DT
    import torch

    if _DT is None:
        _DT = {"*bf16": torch.bfloat16, "*fp16": torch.float16, "*fp32": torch.float32, "*fp64": torch.float64,
               "*i8": torch.int8, "*u8": torch.uint8, "*i16": torch.int16, "*i32": torch.int32, "*i64": torch.int64,
               "*i1": torch.bool, "*fp8e4nv": torch.float8_e4m3fn, "*fp8e5": torch.float8_e5m2}
    return _DT[ptr_type]


def stand_in(spec: dict):
    """A runtime argument of the recorded specialization: a CPU tensor (16-aligned or one element past), an int of the
    value class, a float."""

    import torch

    ty = spec["type"]
    if ty.startswith("*"):
        t = torch.zeros(64, dtype=torch_dtype(ty))
        return t if spec.get("div16", False) else t[1:]
    if ty in ("fp32", "fp64", "fp16", "bf16"):
        return float(spec.get("value", 0.5))
    if "value" in spec:
        return int(spec["value"])
    big = ty in ("i64", "u64")
    return (1 << 36 if big else 0) + (16 if spec.get("div16") else 17)


class Target:
    def __init__(self, arch: int) -> None:
        from triton.backends.compiler import GPUTarget
        from triton.compiler.compiler import make_backend

        self.target = GPUTarget("cuda", arch, 32)
        self.backend = make_backend(self.target)


def specialize(fn, tgt: Target, consts: dict, runtime: dict, options: dict):
    """(options, signature, constexprs, attrs) as JITFunction.run's binder + _pack_args make them."""

    from triton.runtime.jit import create_function_from_signature

    binder = create_function_from_signature(fn.signature, fn.params, tgt.backend)
    kwargs = {}
    for p in fn.params:
        if p.name in consts:
            kwargs[p.name] = consts[p.name]
        elif p.name in runtime:
            kwargs[p.name] = stand_in(runtime[p.name])
    kwargs.update(options)
    bound, spec, opts = binder(**kwargs)
    return fn._pack_args(tgt.backend, kwargs, bound, spec, opts)


def compile_one(fn, tgt: Target, consts: dict, runtime: dict, options: dict):
    import triton
    from triton.compiler import ASTSource

    opts, signature, constexprs, attrs = specialize(fn, tgt, consts, runtime, options)
    k = triton.compile(ASTSource(fn, signature, constexprs, attrs), target=tgt.target, options=opts.__dict__)
    return k, signature, constexprs, attrs


def entry_of(fn, k, signature: dict, constexprs: dict, attrs: dict) -> dict:
    """The compiled variant as a manifest entry (dsv41_m1_capture.pack_aot's input)."""

    names = fn.arg_names
    md = k.metadata._asdict()
    named_c = {names[p[0]]: jsonable(v) for p, v in constexprs.items()}
    div = {names[p[0]]: [list(a) for a in v] == [["tt.divisibility", 16]] for p, v in attrs.items()}
    runtime = [n for n, t in signature.items() if t != "constexpr"]
    return {"name": k.name, "function": full_name(fn), "hash": k.hash, "signature": dict(signature),
            "constexprs": named_c, "abi": [{"name": n, "divisibility_16": div.get(n, False)} for n in runtime],
            "metadata": {x: md.get(x) for x in ("name", "num_warps", "num_ctas", "num_stages", "maxnreg", "shared",
                                                 "global_scratch_size", "global_scratch_align", "profile_scratch_size",
                                                 "launch_pdl", "arch", "triton_version")}}


def need_of_manifest(k: dict, fn) -> tuple[dict, dict]:
    """A captured kernel as (constexprs, runtime arguments): the inverse of what Triton recorded."""

    consts = {n: unjson(v) for n, v in k["constexprs"].items()}
    attrs = k.get("attrs", {})
    runtime = {}
    for n, ty in k["signature"].items():
        if ty == "constexpr":
            continue
        runtime[n] = {"type": ty, "div16": [["tt.divisibility", 16]] == attrs.get(n)}
    return consts, runtime


def options_of(k: dict) -> dict:
    md = k["metadata"]
    return {x: md[x] for x in OPTION_KEYS if x in md and md[x] is not None}


# ---- options rules ----------------------------------------------------------------------------------------------

# call sites whose options follow their constexprs (Python's own code, mirrored; `options` checks them on every capture)
RULES = {
    # mhc_dec.launch: maxnreg = TF_DSV41_MHC_DEC_MAXNREG (128) when RB <= 4 and num_warps == 4
    "tensorfold.families.deepseek_v41.cuda.mhc_dec._site_dec":
        lambda c, o: {**o, **({"maxnreg": 128} if c.get("RB", 99) <= 4 and o.get("num_warps") == 4 else {})},
}


# call sites of kernels no captured run launched: their launch's options as Python passes them (the backend fills the
# rest, as JITFunction.run's parse_options does)
SOURCE = {
    "tensorfold.families.deepseek_v41.cuda.csa2.dtopk._dtopk": ({"num_warps": 8}, "csa2/dtopk.py:167 num_warps=WARPS"),
    "tensorfold.families.deepseek_v41.cuda.prune._prune": ({"num_warps": 1}, "prune.py:296 num_warps=1"),
}


def rule_options(rules: dict, function: str, consts: dict) -> dict:
    r = rules.get(function)
    if r is None:
        raise KeyError(f"{function}: no options rule (no captured variant; add one to options.json)")
    base = {k: v for k, v in r.items() if k != "source"}
    fx = RULES.get(function)
    return fx(consts, base) if fx else base


def derive_options(manifests: list[Path]) -> tuple[dict, list]:
    """{function: options} (the captured variants' options without RULES' parts), and every capture they disagree on."""

    seen: dict = {}
    for m in manifests:
        for k in json.loads(m.read_text()).get("kernels", []):
            o = options_of(k)
            f = k["function"]
            if f in RULES:
                o.pop("maxnreg", None)
            seen.setdefault(f, {}).setdefault(json.dumps(o, sort_keys=True), []).append((str(m), k["hash"]))
    rules, bad = {}, []
    for f, opts in seen.items():
        if len(opts) > 1:
            bad.append({"function": f, "options": list(opts)})
            continue
        rules[f] = json.loads(next(iter(opts)))
    for m in manifests:
        for k in json.loads(m.read_text()).get("kernels", []):
            f = k["function"]
            if f in rules and rule_options(rules, f, {n: unjson(v) for n, v in k["constexprs"].items()}) != options_of(k):
                bad.append({"function": f, "hash": k["hash"], "manifest": str(m), "rule": "differs"})
    return rules, bad


# ---- stamps: ptxas's DWARF file entries -------------------------------------------------------------------------

def elf_sections(b: bytes) -> dict:
    """{name: bytes} of an ELF64 little-endian file."""

    shoff = struct.unpack_from("<Q", b, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x3A)
    hdrs = [struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * shentsize) for i in range(shnum)]
    stro = hdrs[shstrndx][4]
    out = {}
    for h in hdrs:
        name = b[stro + h[0]:b.index(b"\0", stro + h[0])].decode()
        out[name] = b[h[4]:h[4] + h[5]]
    return out


def uleb(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        x = b[i]
        r |= (x & 0x7F) << s
        s += 7
        i += 1
        if x < 0x80:
            return r, i


def line_files(sec: bytes) -> dict:
    """{path: (mtime, size)} of a DWARF 2-4 .debug_line section's file tables."""

    out, off = {}, 0
    while off + 10 < len(sec):
        unit = struct.unpack_from("<I", sec, off)[0]
        ver = struct.unpack_from("<H", sec, off + 4)[0]
        p = off + 10
        p += 1 + (1 if ver >= 4 else 0) + 3
        opbase = sec[p]
        p += 1 + opbase - 1
        dirs = []
        while sec[p] != 0:
            e = sec.index(b"\0", p)
            dirs.append(sec[p:e].decode())
            p = e + 1
        p += 1
        while sec[p] != 0:
            e = sec.index(b"\0", p)
            name = sec[p:e].decode()
            d, p = uleb(sec, e + 1)
            mt, p = uleb(sec, p)
            sz, p = uleb(sec, p)
            path = name if name.startswith("/") or d == 0 else f"{dirs[d - 1]}/{name}"
            if mt or sz:
                out[path] = (mt, sz)
        off += 4 + unit
    return out


def stamps_of_cubins(dirs: list[Path]) -> dict:
    out = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*.cubin")):
            secs = elf_sections(f.read_bytes())
            for name in (".nv.merc.debug_line", ".debug_line"):
                if name in secs:
                    for path, st in line_files(secs[name]).items():
                        if out.setdefault(path, st) != st:
                            raise SystemExit(f"{path}: stamps {out[path]} and {st} in one set")
    return out


def stamped_root(stamps: dict, py: Path, served_py: str, served_triton: str) -> Path:
    """A scratch tree holding each stamped served path: the local file it maps to, with the image's mtime (its size
    must be the image's: the same source)."""

    import triton

    pairs = [(served_py, str(py)), (served_triton, os.path.dirname(triton.__file__))]
    root = Path(tempfile.mkdtemp(prefix="triton-fill-root-"))
    for path, (mt, sz) in stamps.items():
        src = next((b + path[len(a):] for a, b in pairs if path.startswith(a)), None)
        if src is None or not os.path.exists(src):
            raise SystemExit(f"stamp {path}: no local file maps to it")
        if os.path.getsize(src) != sz:
            raise SystemExit(f"stamp {path}: local {src} is {os.path.getsize(src)} bytes, the image's {sz}: not the "
                             "served source")
        dst = root / path.lstrip("/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        os.utime(dst, (mt, mt))
    return root


def in_bwrap(root: Path) -> list[str]:
    """bwrap arguments: the host's top-level entries, then the stamped tree's over them (new tops created)."""

    args = ["bwrap", "--die-with-parent", "--dev", "/dev", "--proc", "/proc"]
    for e in sorted(os.listdir("/")):
        p = "/" + e
        if e in ("dev", "proc"):
            continue
        if os.path.islink(p):
            args += ["--symlink", os.readlink(p), p]
        elif os.path.isdir(p):
            args += ["--bind", p, p]
    files = [Path(d) / f for top in sorted(os.listdir(root)) for d, _, fs in os.walk(root / top) for f in fs]
    # a served path whose directory the host lacks: a tmpfs on its deepest existing ancestor (below /, which is
    # bwrap's own tmpfs) lets bwrap create it; that ancestor's host files are hidden in the sandbox
    tmp = set()
    for full in files:
        parts = Path("/" + str(full.relative_to(root))).parts
        for i in range(2, len(parts)):
            if not os.path.isdir(os.path.join(*parts[:i])):
                if i > 2:
                    tmp.add(os.path.join(*parts[:i - 1]))
                break
    for t in sorted(tmp):
        print(f"[fill] sandbox: tmpfs over {t}", file=sys.stderr)
        args += ["--tmpfs", t]
    for full in files:
        args += ["--ro-bind", str(full), "/" + str(full.relative_to(root))]
    return args


def reexec_stamped(a) -> None:
    """Runs this command again inside bwrap with the stamped files at the served paths (once)."""

    if os.environ.get("TRITON_FILL_STAMPED") or not a.stamps:
        return
    stamps = {k: tuple(v) for k, v in json.loads(Path(a.stamps).read_text()).items()}
    import triton  # noqa: F401  (the local Triton's path for the map)

    root = stamped_root(stamps, Path(a.py).resolve(), a.served_py, a.served_triton)
    env = dict(os.environ, TRITON_FILL_STAMPED="1", TRITON_CACHE_DIR=tempfile.mkdtemp(prefix="triton-fill-cache-"))
    cmd = in_bwrap(root) + ["--", sys.executable, *sys.argv]
    try:
        raise SystemExit(subprocess.run(cmd, env=env).returncode)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---- SASS -------------------------------------------------------------------------------------------------------

def sass(cubin: bytes) -> str:
    tool = os.environ.get("TRITON_NVDISASM_PATH")
    if not tool:
        import triton

        tool = os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin/nvdisasm")
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(cubin)
        f.flush()
        r = subprocess.run([tool, "-c", f.name], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvdisasm: {r.stderr.strip()}")
    return r.stdout


# ---- commands ---------------------------------------------------------------------------------------------------

def fresh_cache() -> None:
    """A fresh Triton cache for the run: its key does not cover the line-info map or the stamps."""

    if not os.environ.get("TRITON_FILL_STAMPED"):
        os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton-fill-cache-")
    import atexit

    atexit.register(shutil.rmtree, os.environ["TRITON_CACHE_DIR"], True)


def cmd_verify(a) -> int:
    reexec_stamped(a)
    fresh_cache()
    kernels = load_tree(Path(a.py).resolve(), a.served_py, a.served_triton)
    man = json.loads(Path(a.manifest).read_text())
    cub = Path(a.cubins) if a.cubins else None
    tgts: dict = {}
    res = {"kernels": 0, "spec_equal": 0, "sass_equal": 0, "bytes_equal": 0, "sha_equal": 0, "no_reference": 0,
           "failed": []}
    for k in man["kernels"]:
        if a.only and k["name"] not in a.only:
            continue
        res["kernels"] += 1
        arch = k["metadata"]["target"]["arch"]
        tgt = tgts.setdefault(arch, Target(arch))
        try:
            fn = resolve(kernels, k["name"], k["function"], [])
            consts, runtime = need_of_manifest(k, fn)
            got, signature, constexprs, attrs = compile_one(fn, tgt, consts, runtime, options_of(k))
        except Exception as e:
            res["failed"].append({"hash": k["hash"], "name": k["name"], "error": f"{type(e).__name__}: {e}"})
            continue
        e = entry_of(fn, got, signature, constexprs, attrs)
        want_attrs = {n: v for n, v in k.get("attrs", {}).items() if v}
        have_attrs = {fn.arg_names[p[0]]: [list(x) for x in v] for p, v in attrs.items() if v}
        same_spec = e["signature"] == k["signature"] and have_attrs == want_attrs and \
            json.dumps(e["constexprs"], sort_keys=True) == json.dumps(k["constexprs"], sort_keys=True)
        res["spec_equal"] += same_spec
        mine = got.asm["cubin"]
        res["sha_equal"] += hashlib.sha256(mine).hexdigest() == k.get("cubin_sha256")
        ref = (cub / f"{k['hash']}.cubin") if cub else None
        if ref is None or not ref.exists():
            res["no_reference"] += 1
            if not same_spec:
                res["failed"].append({"hash": k["hash"], "name": k["name"], "error": "specialization differs"})
            continue
        theirs = ref.read_bytes()
        same_sass = sass(mine) == sass(theirs)
        res["sass_equal"] += same_sass
        res["bytes_equal"] += mine == theirs
        if not (same_spec and same_sass):
            res["failed"].append({"hash": k["hash"], "name": k["name"], "spec": same_spec, "sass": same_sass})
    print(json.dumps(res, indent=1))
    return 0 if not res["failed"] else 1


def have_specs(dirs: list[str]) -> set:
    """Variant keys of existing aot sets (aot.json): what the engine would already find."""

    out = set()
    for d in dirs:
        p = Path(d) / "aot.json"
        if p.exists():
            for k in json.loads(p.read_text())["kernels"]:
                out.add(spec_key(k))
    return out


def spec_key(k: dict) -> str:
    """aot.zig's identity of a variant: fn, constexprs (ints, fp32 bits), runtime params (type, div16)."""

    params = [(p["name"], p["type"], bool(p["div16"])) for p in k["params"]]
    return json.dumps([k["fn"], sorted((n, v.get("int"), v.get("f32")) for n, v in k["consts"].items()), params])


def cmd_compile(a) -> int:
    reexec_stamped(a)
    fresh_cache()
    sys.path.insert(0, str(HERE))
    import dsv41_m1_capture as cap

    kernels = load_tree(Path(a.py).resolve(), a.served_py, a.served_triton)
    rules = json.loads(Path(a.options).read_text())["functions"]
    needs = json.loads(Path(a.needs).read_text())["kernels"]
    have = have_specs(a.have or [])
    jit = sigs_jit(kernels)
    out = Path(a.out)
    (out / "cubins").mkdir(parents=True, exist_ok=True)
    aot_path = out / "aot.json"
    packed = json.loads(aot_path.read_text())["kernels"] if aot_path.exists() else []
    have |= {spec_key(k) for k in packed}
    tgt = Target(a.arch)
    res = {"needs": len(needs), "had": 0, "compiled": 0, "failed": []}
    for n in needs:
        try:
            fn = resolve(kernels, n["fn"], n.get("function"), list(n["consts"]) + list(n["runtime"]))
            consts = {k: unjson(v) for k, v in n["consts"].items()}
            opts = rule_options(rules, full_name(fn), consts)
            sp = specialize(fn, tgt, consts, n["runtime"], opts)
        except Exception as e:
            res["failed"].append({"fn": n["fn"], "consts": n["consts"], "error": f"{type(e).__name__}: {e}"})
            continue
        # the variant's identity before compiling it: an existing set's variant is not compiled again
        key = spec_key(packed_of(fn, cap, jit, *sp[1:], None))
        if key in have:
            res["had"] += 1
            continue
        try:
            import triton
            from triton.compiler import ASTSource

            k = triton.compile(ASTSource(fn, sp[1], sp[2], sp[3]), target=tgt.target, options=sp[0].__dict__)
        except Exception as e:
            res["failed"].append({"fn": n["fn"], "consts": n["consts"], "error": f"{type(e).__name__}: {e}"})
            continue
        packed_k = packed_of(fn, cap, jit, sp[1], sp[2], sp[3], k)
        if packed_k["global_scratch"] or packed_k["profile_scratch"]:
            res["failed"].append({"fn": n["fn"], "error": "global / profile scratch (aot.zig refuses it)"})
            continue
        (out / "cubins" / f"{k.hash}.cubin").write_bytes(k.asm["cubin"])
        packed.append(packed_k)
        have.add(key)
        res["compiled"] += 1
    aot_path.write_text(json.dumps({"generator": "tools/zig/triton_fill.py", "kernels": packed}, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0 if not res["failed"] else 1


def packed_of(fn, cap, jit: dict, signature: dict, constexprs: dict, attrs: dict, k) -> dict:
    """The variant in aot.json's form (dsv41_m1_capture.pack_aot's fields); `k` None: its identity only."""

    names = fn.arg_names
    div = {names[p[0]]: [list(x) for x in v] == [["tt.divisibility", 16]] for p, v in attrs.items()}
    runtime = [n for n, t in signature.items() if t != "constexpr"]
    e = {"function": full_name(fn), "signature": dict(signature),
         "constexprs": {names[p[0]]: jsonable(v) for p, v in constexprs.items()},
         "abi": [{"name": n, "divisibility_16": div.get(n, False)} for n in runtime]}
    out = {"fn": fn.fn.__name__, "params": cap.aot_params(e, runtime, jit), "consts": cap.aot_consts(e)}
    if k is None:
        return out
    md = k.metadata._asdict()
    return {"fn": k.name, "hash": k.hash, "name": md["name"], "num_warps": md["num_warps"],
            "num_ctas": md.get("num_ctas", 1), "shared": md.get("shared", 0),
            "global_scratch": md.get("global_scratch_size", 0) or 0,
            "global_align": md.get("global_scratch_align", 1) or 1,
            "profile_scratch": md.get("profile_scratch_size", 0) or 0, "pdl": bool(md.get("launch_pdl", False)),
            "params": out["params"], "consts": out["consts"], "filled": True}


def sigs_jit(kernels: dict) -> dict:
    """dsv41_m1_capture.specialization()'s form: {function: {do_not_specialize: [...]}}."""

    return {full_name(fn): {"params": [p.name for p in fn.params],
                            "do_not_specialize": [p.name for p in fn.params if p.do_not_specialize]}
            for fns in kernels.values() for fn in fns}


def cmd_sass(a) -> int:
    """Every filled variant whose specialization a served set also holds: SASS (nvdisasm -c) and bytes compared."""

    def load(d):
        return {spec_key(k): k for k in json.loads((Path(d) / "aot.json").read_text())["kernels"]}

    fill, served = load(a.fill), load(a.served)
    res = {"filled": len(fill), "served": len(served), "both": 0, "sass_equal": 0, "bytes_equal": 0, "only_filled": 0,
           "differ": []}
    for key, k in fill.items():
        s = served.get(key)
        if s is None:
            res["only_filled"] += 1
            continue
        res["both"] += 1
        mine = (Path(a.fill) / "cubins" / f"{k['hash']}.cubin").read_bytes()
        theirs = (Path(a.served) / "cubins" / f"{s['hash']}.cubin").read_bytes()
        same = sass(mine) == sass(theirs)
        res["sass_equal"] += same
        res["bytes_equal"] += mine == theirs
        if not same:
            res["differ"].append({"fn": k["fn"], "filled": k["hash"], "served": s["hash"]})
    print(json.dumps(res, indent=1))
    return 0 if not res["differ"] else 1


def cmd_merge(a) -> int:
    """The filled variants a served set lacks (by specialization, not hash: the fill's hashes are another host's)."""

    dst, src = Path(a.dst), Path(a.src)
    d = json.loads((dst / "aot.json").read_text())
    have = {spec_key(k) for k in d["kernels"]}
    added = 0
    for k in json.loads((src / "aot.json").read_text())["kernels"]:
        if spec_key(k) in have:
            continue
        shutil.copyfile(src / "cubins" / f"{k['hash']}.cubin", dst / "cubins" / f"{k['hash']}.cubin")
        d["kernels"].append(k)
        have.add(spec_key(k))
        added += 1
    (dst / "aot.json").write_text(json.dumps(d, indent=1) + "\n")
    print(json.dumps({"added": added, "kernels": len(d["kernels"])}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def tree(p):
        p.add_argument("--py", required=True, help="the served Python tree's src/ (prod's TF_COMMIT)")
        p.add_argument("--served-py", default=SERVED_PY)
        p.add_argument("--served-triton", default=SERVED_TRITON)
        p.add_argument("--stamps", help="stamps.json: compile where the served paths hold the image's files")

    p = sub.add_parser("sigs")
    tree(p)
    p.add_argument("--out", required=True)
    p = sub.add_parser("options")
    p.add_argument("--manifests", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("stamps")
    p.add_argument("--cubins", nargs="*", default=[])
    p.add_argument("--root", nargs="*", default=[], help="DIR[=SERVED]: stat the .py files under DIR, served at "
                   "SERVED (in the served image: /dsv41-tf/src and its Triton)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("verify")
    tree(p)
    p.add_argument("--manifest", required=True)
    p.add_argument("--cubins", help="the captured cubins (<hash>.cubin): SASS and bytes")
    p.add_argument("--only", nargs="*")
    p = sub.add_parser("compile")
    tree(p)
    p.add_argument("--needs", required=True)
    p.add_argument("--options", required=True)
    p.add_argument("--have", nargs="*", help="aot sets whose variants are not compiled again")
    p.add_argument("--arch", type=int, default=121)
    p.add_argument("--out", required=True)
    p = sub.add_parser("sass")
    p.add_argument("--fill", required=True, help="the filled aot dir")
    p.add_argument("--served", required=True, help="a served aot dir (its captured cubins)")
    p = sub.add_parser("merge")
    p.add_argument("dst")
    p.add_argument("src")
    a = ap.parse_args()
    if a.cmd == "sass":
        return cmd_sass(a)
    if a.cmd == "merge":
        return cmd_merge(a)
    if a.cmd == "sigs":
        kernels = load_tree(Path(a.py).resolve(), a.served_py, a.served_triton)
        Path(a.out).write_text(json.dumps({"generator": "tools/zig/triton_fill.py", "kernels": sigs(kernels)},
                                          indent=1) + "\n")
        print(json.dumps({"kernels": len(kernels), "out": a.out}))
        return 0
    if a.cmd == "options":
        rules, bad = derive_options([Path(m) for m in a.manifests])
        for f, (kw, site) in SOURCE.items():
            rules.setdefault(f, dict(kw, source=site))
        Path(a.out).write_text(json.dumps({"generator": "tools/zig/triton_fill.py", "functions": rules,
                                           "rules": sorted(RULES)}, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"functions": len(rules), "disagree": bad}, indent=1))
        return 0 if not bad else 1
    if a.cmd == "stamps":
        st = stamps_of_cubins([Path(d) for d in a.cubins])
        for r in a.root:
            d, _, served = r.partition("=")
            for f in Path(d).rglob("*.py"):
                s = f.stat()
                st[(served or d).rstrip("/") + "/" + str(f.relative_to(d))] = (int(s.st_mtime), s.st_size)
        Path(a.out).write_text(json.dumps(st, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"files": len(st), "out": a.out}))
        return 0
    if a.cmd == "verify":
        return cmd_verify(a)
    return cmd_compile(a)


if __name__ == "__main__":
    raise SystemExit(main())
