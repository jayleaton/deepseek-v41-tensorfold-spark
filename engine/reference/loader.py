"""Checkpoint access for the reference: the MiaAI / dealignai EXL3 2.9 bpw V4.1-Flash pack, read lazily.

- ``SafetensorsDir``: a local checkpoint folder (``model.safetensors.index.json`` + shards), read with ``pread``; whole
  tensors, or rows of a 2-D tensor (the embedding, the Engram tables) without reading the rest.
- ``RemoteSafetensorsDir``: the same over ``ssh`` (read-only byte ranges served by a ``python3`` one-shot on the
  host), with whole tensors cached on local disk. Used to run single layers on real tensors from the workstation
  without copying the 197 GB checkpoint.
- ``CheckpointLoader``: turns names into the reference's weight objects. Every EXL3 group (``P.trellis``, ``P.suh``,
  ``P.svh``, marker ``P.mul1``) becomes an ``Exl3Linear`` whose bits come from its trellis shape (K map not needed;
  ``exl3_k_map.json`` only cross-checks); dequantization happens on first use and is dropped by ``release()``.

Checkpoint names (DeepSeek's native namespace; ``wo_a`` is stored as one EXL3 group per output group,
``attn.wo_a.slice.{g}``):

    embed.weight, norm.weight, head.{trellis,suh,svh,mul1}
    layers.L.{attn_norm,ffn_norm}.weight, layers.L.hc_{attn,ffn}_{fn,base,scale}
    layers.L.attn.{wq_a,wkv,wq_b,wo_b,wo_a.slice.g}.*, attn.{q_norm,kv_norm}.weight, attn.attn_sink
    layers.L.attn.compressor.{wkv,wgate}.*, compressor.norm.weight            (KV sources 2, 8, 14, 20; no wgate on 20)
    layers.L.attn.indexer.{wq_b,wk}.*, indexer.{weights_proj,k_norm}.weight    (wk / k_norm on KV sources only)
    layers.L.engram.wkv.*, engram.{q_weight,k_weight}                          (L = 1, 14; the tables live in the
                                                                                original FP8 shards 47-48)
    layers.L.ffn.gate.{weight,bias}, ffn.experts.E.w{1,2,3}.*, ffn.shared_experts.w{1,2,3}.*
    mtp.i.* (DSpark block i, same block layout) + mtp.0.main_proj.*, mtp.0.main_norm.weight,
    mtp.2.{norm.weight, markov_head.embed.weight, markov_head.head.weight, confidence_head.proj.weight}
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .attention import AttnWeights, CompressorWeights, IndexerWeights
from .config import Config
from .engram import EngramWeights, NgramHasher, build_compressed_token_map
from .exl3 import Exl3Weight, codebook_from_marker
from .hc import HcParams
from .model import DSparkWeights, LayerWeights, ModelWeights
from .moe import ExpertWeights, MoEWeights
from .ops import F32, DenseLinear, Exl3Linear, Linear, Numerics, bf16, fp8_e4m3_dequant

# Reference CLIs put the recipe root on sys.path, while the shared adapters use
# sibling imports from scripts just like the maintained native entry points.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from safe_exec import run as supervised_run  # noqa: E402
from safe_quote import join_checked  # noqa: E402

DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.uint8,
}
SIZES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
         "F8_E4M3": 1, "F8_E8M0": 1}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    file: str
    dtype: str
    shape: tuple[int, ...]
    start: int          # absolute byte offset in the file
    nbytes: int

    @property
    def row_bytes(self) -> int:
        return self.nbytes // self.shape[0] if self.shape else self.nbytes


def _to_tensor(buf: bytes | bytearray | memoryview, dtype: str, shape: tuple[int, ...]) -> torch.Tensor:
    t = torch.frombuffer(bytearray(buf), dtype=torch.uint8) if len(buf) else torch.empty(0, dtype=torch.uint8)
    return t.view(DTYPES[dtype]).reshape(shape)


def parse_header(raw: dict, file: str, header_len: int) -> dict[str, TensorInfo]:
    out = {}
    base = 8 + header_len
    for name, e in raw.items():
        if name == "__metadata__":
            continue
        a, b = e["data_offsets"]
        out[name] = TensorInfo(name, file, e["dtype"], tuple(e["shape"]), base + a, b - a)
    return out


class SafetensorsDir:
    """A local safetensors checkpoint folder."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._headers: dict[str, dict[str, TensorInfo]] = {}
        idx = self.root / "model.safetensors.index.json"
        if idx.exists():
            self.weight_map: dict[str, str] = json.loads(idx.read_text())["weight_map"]
        else:
            self.weight_map = {}
            for p in sorted(self.root.glob("*.safetensors")):
                for n in self._header(p.name):
                    self.weight_map[n] = p.name

    def _header(self, file: str) -> dict[str, TensorInfo]:
        if file not in self._headers:
            with open(self.root / file, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._headers[file] = parse_header(json.loads(f.read(n)), file, n)
        return self._headers[file]

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def names(self) -> Iterable[str]:
        return self.weight_map.keys()

    def info(self, name: str) -> TensorInfo:
        return self._header(self.weight_map[name])[name]

    def read_ranges(self, file: str, ranges: list[tuple[int, int]]) -> list[bytes]:
        fd = os.open(self.root / file, os.O_RDONLY)
        try:
            return [os.pread(fd, n, off) for off, n in ranges]
        finally:
            os.close(fd)

    def read_file(self, rel: str) -> bytes:
        return (self.root / rel).read_bytes()

    def tensor(self, name: str) -> torch.Tensor:
        i = self.info(name)
        (buf,) = self.read_ranges(i.file, [(i.start, i.nbytes)])
        return _to_tensor(buf, i.dtype, i.shape)

    def rows(self, name: str, row_ids: torch.Tensor) -> torch.Tensor:
        """Rows ``row_ids`` (any shape) of a tensor whose first dim indexes rows -> [*row_ids.shape, *row]."""

        i = self.info(name)
        flat = row_ids.reshape(-1).to(torch.int64)
        uniq, inv = torch.unique(flat, return_inverse=True)
        rb = i.row_bytes
        bufs = self.read_ranges(i.file, [(i.start + int(r) * rb, rb) for r in uniq.tolist()])
        rows = _to_tensor(b"".join(bufs), i.dtype, (uniq.numel(), *i.shape[1:]))
        return rows[inv].reshape(*row_ids.shape, *i.shape[1:])


_REMOTE_READER = r"""
import json, os, sys
req = json.loads(sys.stdin.readline())
out = sys.stdout.buffer
for item in req:
    fd = os.open(sys.argv[1], os.O_RDONLY)
    try:
        for off, n in item["ranges"]:
            done = 0
            while done < n:
                chunk = os.pread(fd, min(n - done, 1 << 24), off + done)
                if not chunk:
                    raise SystemExit("short read")
                out.write(chunk)
                done += len(chunk)
    finally:
        os.close(fd)
out.flush()
"""

_REMOTE_HEADERS = r"""
import json, os, struct, sys, glob
root = sys.argv[1]
res = {"index": None, "headers": {}}
p = os.path.join(root, "model.safetensors.index.json")
if os.path.exists(p):
    res["index"] = json.load(open(p))["weight_map"]
for f in sorted(glob.glob(os.path.join(root, "*.safetensors"))):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        res["headers"][os.path.basename(f)] = {"header_len": n, "header": json.loads(fh.read(n))}
sys.stdout.write(json.dumps(res))
"""


class RemoteSafetensorsDir(SafetensorsDir):
    """A checkpoint folder on another host, read over ssh (read-only). Whole tensors are cached under
    ``cache_dir`` so a test that runs twice fetches once."""

    SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ControlMaster=auto", "-o", "ControlPath=/tmp/dsv41-ref-ssh-%r@%h:%p",
                "-o", "ControlPersist=120")
    SSH_TIMEOUT = 300

    def __init__(self, host: str, root: str, cache_dir: str | Path, ssh_opts: tuple[str, ...] = SSH_OPTS,
                 cache_experts: bool = False):
        # Validate raw settings even when cached metadata avoids the SSH call.
        join_checked([host, root, *ssh_opts])
        self.host, self.remote_root, self.ssh_opts = host, root.rstrip("/"), ssh_opts
        self.cache_experts = cache_experts
        self._mem: dict[str, torch.Tensor] = {}
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        meta = self.root / "headers.json"
        fetch = not meta.exists()
        raw = self._ssh(["python3", "-c", _REMOTE_HEADERS, self.remote_root]) if fetch else meta.read_bytes()
        # Only cache a parsed metadata document, never a diagnostic dump.
        try:
            m = json.loads(raw)
            headers = {f: parse_header(v["header"], f, v["header_len"]) for f, v in m["headers"].items()}
            index = m.get("index")
            if index is not None and (not isinstance(index, dict)
                                      or any(not isinstance(n, str) or not isinstance(f, str)
                                             for n, f in index.items())):
                raise ValueError
            weight_map = index or {n: f for f, h in headers.items() for n in h}
        except (ValueError, TypeError, KeyError, AttributeError):
            raise RuntimeError("invalid reference metadata") from None
        if fetch:
            meta.write_text(json.dumps({"headers": m["headers"], "index": index}))
        self._headers, self.weight_map = headers, weight_map

    def _ssh(self, argv: list[str], stdin: bytes | None = None) -> bytes:
        cmd = ["ssh", *self.ssh_opts, self.host, join_checked(argv)]
        r = supervised_run(cmd, input=stdin, capture_output=True, text=False, check=False, timeout=self.SSH_TIMEOUT)
        if r.returncode != 0:
            raise RuntimeError(f"reference ssh failed (exit {r.returncode})")
        return r.stdout

    def _header(self, file: str) -> dict[str, TensorInfo]:
        return self._headers[file]

    def read_ranges(self, file: str, ranges: list[tuple[int, int]]) -> list[bytes]:
        if not ranges:
            return []
        req = json.dumps([{"ranges": [list(r) for r in ranges]}]) + "\n"
        raw = self._ssh(["python3", "-c", _REMOTE_READER, f"{self.remote_root}/{file}"], req.encode())
        out, pos = [], 0
        for _, n in ranges:
            out.append(raw[pos:pos + n])
            pos += n
        if pos != len(raw):
            raise RuntimeError(f"remote read returned {len(raw)} bytes, expected {pos}")
        return out

    def read_file(self, rel: str) -> bytes:
        local = self.root / "files" / rel
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(self._ssh(["cat", f"{self.remote_root}/{rel}"]))
        return local.read_bytes()

    def _cache_path(self, name: str) -> Path:
        i = self.info(name)
        key = hashlib.sha1(f"{i.file}:{name}".encode()).hexdigest()[:16]
        return self.root / "tensors" / f"{key}.bin"

    def _cacheable(self, name: str) -> bool:
        return self.cache_experts or ".experts." not in name

    def prefetch(self, names: Iterable[str]) -> None:
        """Fetch several tensors in one ssh round trip a file. Routed experts stay in memory only (never written to
        disk unless ``cache_experts``): a forward fetches the experts it routes to, not the checkpoint."""

        todo: dict[str, list[str]] = {}
        for n in names:
            if n in self._mem or (self._cacheable(n) and self._cache_path(n).exists()):
                continue
            todo.setdefault(self.info(n).file, []).append(n)
        for file, group in todo.items():
            infos = [self.info(n) for n in group]
            bufs = self.read_ranges(file, [(i.start, i.nbytes) for i in infos])
            for n, i, buf in zip(group, infos, bufs):
                if self._cacheable(n):
                    path = self._cache_path(n)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".part")
                    tmp.write_bytes(buf)
                    tmp.rename(path)
                else:
                    self._mem[n] = _to_tensor(buf, i.dtype, i.shape)

    def drop(self, names: Iterable[str] | None = None) -> None:
        """Forget in-memory (expert) tensors."""

        if names is None:
            self._mem.clear()
        else:
            for n in names:
                self._mem.pop(n, None)

    def tensor(self, name: str) -> torch.Tensor:
        if name in self._mem:
            return self._mem[name]
        i = self.info(name)
        path = self._cache_path(name)
        if self._cacheable(name) and path.exists() and path.stat().st_size == i.nbytes:
            return _to_tensor(path.read_bytes(), i.dtype, i.shape)
        self.prefetch([name])
        return self.tensor(name)


class FileRows:
    """An Engram table (FP8 rows + UE8M0 scales) read row by row from a safetensors source."""

    def __init__(self, src: SafetensorsDir, weight: str, scale: str, block: int = 32):
        self.src, self.weight, self.scale, self.block = src, weight, scale, block

    def rows(self, index: torch.Tensor) -> torch.Tensor:
        w = self.src.rows(self.weight, index)
        s = self.src.rows(self.scale, index)
        return fp8_e4m3_dequant(w, s, self.block)


class CheckpointLoader:
    """Builds ``ModelWeights`` / ``DSparkWeights`` from a V4.1 EXL3 checkpoint source."""

    def __init__(self, cfg: Config, src: SafetensorsDir, numerics: Numerics = Numerics(),
                 engram_src: SafetensorsDir | None = None, device: str = "cpu"):
        self.cfg, self.src, self.num, self.engram_src, self.device = cfg, src, numerics, engram_src, device

    # -- primitives ------------------------------------------------------------------------------------------

    def has(self, name: str) -> bool:
        return name in self.src

    def native(self, name: str, as_bf16_param: bool = False) -> torch.Tensor:
        """A plain tensor in fp32. ``as_bf16_param``: vLLM keeps this one as a bf16 parameter (fp16 / fp32 values
        in the checkpoint are rounded once at load)."""

        t = self.src.tensor(name).to(F32)
        return bf16(t).to(self.device) if as_bf16_param else t.to(self.device)

    def exl3_weight(self, prefix: str) -> Exl3Weight:
        if f"{prefix}.mul1" in self.src:
            cb = codebook_from_marker(int(self.src.tensor(f"{prefix}.mul1").reshape(-1)[0]))
        elif f"{prefix}.mcg" in self.src:
            cb = codebook_from_marker(int(self.src.tensor(f"{prefix}.mcg").reshape(-1)[0]))
        else:
            cb = "3inst"
        w = Exl3Weight(self.src.tensor(f"{prefix}.trellis"), self.src.tensor(f"{prefix}.suh"),
                       self.src.tensor(f"{prefix}.svh"), cb)
        return w.to(self.device) if self.device != "cpu" else w

    def is_exl3(self, prefix: str) -> bool:
        return f"{prefix}.trellis" in self.src

    def linear(self, prefix: str, registry: list | None = None) -> Linear:
        """An EXL3 group (lazy) or a native ``prefix.weight`` [out, in]."""

        if self.is_exl3(prefix):
            info = self.src.info(f"{prefix}.trellis")
            lin = Exl3Linear(lambda p=prefix: self.exl3_weight(p), self.num, k=16 * info.shape[0],
                             n=16 * info.shape[1], name=prefix)
            if registry is not None:
                registry.append(lin)
            return lin
        return DenseLinear(self.native(f"{prefix}.weight", as_bf16_param=True))

    @staticmethod
    def prefix_of(cfg: Config, layer: int) -> str:
        return f"layers.{layer}" if layer < cfg.num_hidden_layers else f"mtp.{layer - cfg.num_hidden_layers}"

    # -- the model -------------------------------------------------------------------------------------------

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.src.rows("embed.weight", ids.to(torch.int64)).to(F32).to(self.device)

    def hc(self, p: str, which: str) -> HcParams:
        return HcParams(self.native(f"{p}.hc_{which}_fn"), self.native(f"{p}.hc_{which}_base"),
                        self.native(f"{p}.hc_{which}_scale"))

    def attn(self, layer: int, reg: list) -> AttnWeights:
        cfg = self.cfg
        p = f"{self.prefix_of(cfg, layer)}.attn"
        comp = ix = None
        if cfg.is_kv_source(layer):
            comp = CompressorWeights(
                self.linear(f"{p}.compressor.wkv", reg),
                self.linear(f"{p}.compressor.wgate", reg) if self.has(f"{p}.compressor.wgate.trellis") else None,
                self.native(f"{p}.compressor.norm.weight"))
        if cfg.is_index_source(layer):
            own = cfg.is_kv_source(layer)
            ix = IndexerWeights(
                self.linear(f"{p}.indexer.wq_b", reg),
                DenseLinear(self.native(f"{p}.indexer.weights_proj.weight", as_bf16_param=True)),
                self.linear(f"{p}.indexer.wk", reg) if own else None,
                self.native(f"{p}.indexer.k_norm.weight") if own else None)
        return AttnWeights(
            wq_a=self.linear(f"{p}.wq_a", reg), wkv=self.linear(f"{p}.wkv", reg), wq_b=self.linear(f"{p}.wq_b", reg),
            wo_a=[self.linear(f"{p}.wo_a.slice.{g}", reg) for g in range(cfg.o_groups)],
            wo_b=self.linear(f"{p}.wo_b", reg), q_norm=self.native(f"{p}.q_norm.weight"),
            kv_norm=self.native(f"{p}.kv_norm.weight"), attn_sink=self.native(f"{p}.attn_sink"),
            compressor=comp, indexer=ix)

    def moe(self, layer: int, reg: list) -> MoEWeights:
        p = f"{self.prefix_of(self.cfg, layer)}.ffn"

        def expert(e: int) -> ExpertWeights:      # not registered: dropped after each use
            return ExpertWeights(self.linear(f"{p}.experts.{e}.w1"), self.linear(f"{p}.experts.{e}.w2"),
                                 self.linear(f"{p}.experts.{e}.w3"))

        shared = None
        if self.has(f"{p}.shared_experts.w1.trellis") or self.has(f"{p}.shared_experts.w1.weight"):
            shared = ExpertWeights(self.linear(f"{p}.shared_experts.w1", reg),
                                   self.linear(f"{p}.shared_experts.w2", reg),
                                   self.linear(f"{p}.shared_experts.w3", reg))
        def prefetch(ids: list[int]) -> None:
            if hasattr(self.src, "prefetch"):
                self.src.prefetch([f"{p}.experts.{e}.w{j}.{part}" for e in ids for j in (1, 2, 3)
                                   for part in ("trellis", "suh", "svh", "mul1") if self.has(f"{p}.experts.{e}.w{j}.{part}")])

        def done(ids: list[int]) -> None:
            if hasattr(self.src, "drop"):
                self.src.drop([f"{p}.experts.{e}.w{j}.{part}" for e in ids for j in (1, 2, 3)
                               for part in ("trellis", "suh", "svh", "mul1")])

        return MoEWeights(gate=self.native(f"{p}.gate.weight", as_bf16_param=True),
                          bias=self.native(f"{p}.gate.bias"), experts=expert, shared=shared,
                          prefetch=prefetch, done=done)

    def engram(self, layer: int, reg: list) -> EngramWeights | None:
        if layer not in self.cfg.engram_layer_ids:
            return None
        p = f"layers.{layer}.engram"
        if self.engram_src is None:
            raise ValueError(f"layer {layer} has an Engram module: pass engram_src (the original FP8 shards 47-48)")
        table = FileRows(self.engram_src, f"{p}.embed.weight", f"{p}.embed.scale")
        return EngramWeights(self.linear(f"{p}.wkv", reg), self.native(f"{p}.q_weight"),
                             self.native(f"{p}.k_weight"), table)

    def layer(self, layer: int, with_engram: bool = True) -> LayerWeights:
        p = self.prefix_of(self.cfg, layer)
        if hasattr(self.src, "prefetch"):          # one round trip for the layer's non-expert tensors
            self.src.prefetch([n for n in self.src.names()
                               if n.startswith(p + ".") and ".experts." not in n
                               and (with_engram or ".engram." not in n)])
        reg: list[Exl3Linear] = []
        lw = LayerWeights(
            attn_norm=self.native(f"{p}.attn_norm.weight"), ffn_norm=self.native(f"{p}.ffn_norm.weight"),
            hc_attn=self.hc(p, "attn"), hc_ffn=self.hc(p, "ffn"), attn=self.attn(layer, reg),
            moe=self.moe(layer, reg), engram=self.engram(layer, reg) if with_engram else None)
        lw.release = lambda: [r.release(packed=True) for r in reg] and None
        return lw

    def norm(self) -> torch.Tensor:
        return self.native("norm.weight")

    def head(self) -> Linear:
        return self.linear("head")

    def model_weights(self, with_head: bool = True, with_engram: bool = True) -> ModelWeights:
        return ModelWeights(embed=self.embed, layer=lambda L: self.layer(L, with_engram),
                            norm=self.norm() if with_head else None, head=self.head() if with_head else None)

    def _mtp_find(self, suffix: str) -> str | None:
        for i in reversed(range(self.cfg.num_nextn_predict_layers)):
            n = f"mtp.{i}.{suffix}"
            if self.has(n) or self.has(f"{n}.trellis") or self.has(f"{n}.weight"):
                return n
        return None

    def dspark(self) -> DSparkWeights:
        cfg = self.cfg
        blocks = [self.layer(cfg.num_hidden_layers + i) for i in range(cfg.num_nextn_predict_layers)]
        mp = self._mtp_find("main_proj")
        conf = self._mtp_find("confidence_head.proj.weight")
        return DSparkWeights(
            blocks=blocks, main_proj=self.linear(mp), main_norm=self.native(f"{self._mtp_find('main_norm')}.weight"),
            norm=self.native(f"{self._mtp_find('norm')}.weight"),
            markov_w1=self.native(f"{self._mtp_find('markov_head.embed')}.weight", as_bf16_param=True),
            markov_w2=self.native(f"{self._mtp_find('markov_head.head')}.weight", as_bf16_param=True),
            confidence=self.native(conf) if conf else None)


def token_map_from(src: SafetensorsDir, cfg: Config, tokenizer_json: str | Path | None = None) -> list[int]:
    """The Engram compressed-vocabulary map (checks the size against ``engram_compressed_vocab_size``)."""

    if tokenizer_json is None:
        path = src.root / "files" / "tokenizer.json" if isinstance(src, RemoteSafetensorsDir) else \
            src.root / "tokenizer.json"
        if isinstance(src, RemoteSafetensorsDir):
            src.read_file("tokenizer.json")
        tokenizer_json = path
    lookup, n = build_compressed_token_map(str(tokenizer_json))
    if n != cfg.engram_compressed_vocab_size:
        raise ValueError(f"compressed vocab {n} != config {cfg.engram_compressed_vocab_size}: the Engram hashes "
                         "would not match the tables")
    return lookup


def hasher_from(src: SafetensorsDir, cfg: Config, tokenizer_json: str | Path | None = None) -> NgramHasher:
    return NgramHasher(cfg, token_map_from(src, cfg, tokenizer_json))


def config_from(src: SafetensorsDir) -> Config:
    return Config.from_dict(json.loads(src.read_file("config.json")))


__all__ = ["SafetensorsDir", "RemoteSafetensorsDir", "CheckpointLoader", "FileRows", "TensorInfo",
           "token_map_from", "hasher_from", "config_from", "np"]
