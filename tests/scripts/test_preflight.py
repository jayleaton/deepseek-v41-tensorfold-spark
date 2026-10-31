"""scripts/preflight.py with a fake /proc/meminfo, a fake mountinfo and small Engram shards in a temp folder."""

from __future__ import annotations

import importlib.util
import json
import os
import struct
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "preflight", ROOT / "scripts" / "preflight.py"
)
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)

GIB_KB = 1024 * 1024
CFG = {
    "engram_layer_ids": [1, 14],
    "engram_max_ngram_size": 3,
    "engram_n_heads": 4,
    "engram_vocab_size": 100,
}
BOOT_LOG = """\
info(kv): kv: before the pool: rank 0 pool 2,100,000 tokens = 2.07 GiB (1060 B a token, split KV); predicted worst \
MemAvailable 5.41 GiB (hard floor 4.0)
info(kv): kv: rank 0: boot growth 1.20 GiB measured (MemAvailable 11.60 at the check - pool 2.07 - 8.33 now); worst \
serving point 5.33 GiB (post-boot reserve 0.00, TF_DSV41_POST_BOOT_GIB; dip 3.00, TF_DSV41_SERVE_DIP_GIB; hard floor 4.0)
"""


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in pf.SETTINGS:
        monkeypatch.delenv(k, raising=False)


def write_meminfo(path: Path, avail_gib: float, total_gib: float = 119.7) -> Path:
    path.write_text(
        f"MemTotal:       {int(total_gib * GIB_KB)} kB\nMemFree:        1000 kB\n"
        f"MemAvailable:   {int(avail_gib * GIB_KB)} kB\nShmem:          0 kB\n"
    )
    return path


def write_mountinfo(path: Path, mountpoint: Path, fstype: str, source: str) -> Path:
    mp = str(mountpoint).replace(" ", "\\040")
    path.write_text(
        "22 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n"
        f"40 22 0:50 / {mp} rw,relatime shared:9 - {fstype} {source} rw\n"
    )
    return path


def write_shards(
    d: Path, rank: int, truncate: bool = False, layout_cfg: dict = CFG
) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for layer, (lo, hi, total) in pf.engram_layout(layout_cfg, rank, 2).items():
        head = bytearray(pf.ENGRAM_HEADER)
        struct.pack_into(
            "<6Q", head, 0, pf.ENGRAM_MAGIC, layer, lo, hi, total, pf.ENGRAM_ROW
        )
        rows = (hi - lo) * pf.ENGRAM_ROW - (100 if truncate else 0)
        (d / f"engram-l{layer}-r{rank}of2.bin").write_bytes(
            bytes(head) + os.urandom(rows)
        )


def setup(
    tmp_path: Path,
    fstype: str = "ext4",
    source: str = "/dev/nvme1n1",
    avail: float = 120.0,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    engram = tmp_path / "engram dir"
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(CFG))
    write_shards(engram, 0)
    mi = write_mountinfo(tmp_path / "mountinfo", tmp_path, fstype, source)
    mem = write_meminfo(tmp_path / "meminfo", avail)
    cfg = tmp_path / "prod-zig.env"
    cfg.write_text(
        f'MODEL_DIR="{model}"\nENGRAM_DIR="{engram}"\nTF_DSV41_POOL_TOKENS=2100000\nTF_DSV41_KV_SPLIT=1\n'
        "export TF_DSV41_FLOOR_HARD_GIB=4  # comment\nTF_DSV41_PF_4K=1\n"
    )
    return (
        [
            "--config",
            str(cfg),
            "--mountinfo",
            str(mi),
            "--meminfo",
            str(mem),
            "--shm",
            str(tmp_path / "noshm"),
            "--json",
        ],
        engram,
        model,
    )


def checks(capsys) -> dict:
    out = json.loads(capsys.readouterr().out)
    return {c["check"]: c for c in out["checks"]}, out["exit"]


def test_all_ok(tmp_path, capsys, monkeypatch):
    args, engram, _ = setup(tmp_path)
    # Any accidental real probe sees a deterministic 1 ms read, regardless of host load.
    monkeypatch.setattr(
        pf, "time", SimpleNamespace(perf_counter_ns=count(0, 1_000_000).__next__)
    )
    calls = []

    def probe(path, reads, seed):
        calls.append((path, reads, seed))
        return {
            "reads": reads,
            "direct": True,
            "p50_us": 90,
            "p99_us": 120,
            "max_us": 150,
        }

    assert pf.main(args + ["--rank", "0", "--probe", "32"], latency_probe=probe) == 0
    assert len(calls) == 1 and calls[0][0].parent == engram
    assert calls[0][1:] == (32, 0)
    c, code = checks(capsys)
    assert code == 0
    assert (
        c["engram-fs"]["status"] == "ok" and c["engram-fs"]["data"]["fstype"] == "ext4"
    )
    assert (
        c["engram-files"]["status"] == "ok"
        and "Engram layout" in c["engram-files"]["detail"]
    )
    assert c["engram-latency"]["data"]["reads"] == 32
    assert c["settings"]["data"]["TF_DSV41_FLOOR_HARD_GIB"] == "4"


@pytest.mark.parametrize(
    ("latency_us", "code"), [(299, 0), (300, 0), (301, 1), (1000, 1)]
)
def test_latency_threshold_with_measured_clock(
    tmp_path, capsys, monkeypatch, latency_us, code
):
    args, _, _ = setup(tmp_path)
    monkeypatch.setattr(
        pf,
        "time",
        SimpleNamespace(perf_counter_ns=count(0, latency_us * 1000).__next__),
    )
    # Exercise the measurement and threshold path without device I/O or actual sleeps.
    monkeypatch.setattr(pf.os, "preadv", lambda fd, buffers, offset: 4096)
    assert pf.main(args + ["--rank", "0", "--probe", "32"]) == code
    c, _ = checks(capsys)
    assert c["engram-latency"]["data"]["p50_us"] == latency_us
    assert c["engram-latency"]["status"] == ("warn" if code else "ok")


def test_rank_inferred_and_wrong_rank_layout(tmp_path, capsys):
    args, engram, _ = setup(tmp_path)
    assert pf.main(args + ["--no-probe"]) == 0  # only rank 0's shards: inferred
    capsys.readouterr()
    # rank 1's file names carrying rank 0's rows: the layout check catches it
    for p in engram.glob("*-r0of2.bin"):
        p.rename(p.with_name(p.name.replace("-r0of2", "-r1of2")))
    assert pf.main(args + ["--no-probe", "--rank", "1"]) == 2
    c, _ = checks(capsys)
    assert "the pack's layout gives rank 1" in c["engram-files"]["detail"]


@pytest.mark.parametrize(
    ("fstype", "source", "status"),
    [
        ("nfs4", "nas:/export", "fail"),
        ("fuse.sshfs", "host:/data", "fail"),
        ("tmpfs", "tmpfs", "fail"),
        ("fuse.mergerfs", "pool", "warn"),
        ("xfs", "/dev/sdb1", "ok"),
    ],
)
def test_engram_filesystem(tmp_path, capsys, fstype, source, status):
    args, _, _ = setup(tmp_path, fstype, source)
    code = pf.main(args + ["--rank", "0", "--no-probe"])
    c, _ = checks(capsys)
    assert c["engram-fs"]["status"] == status
    assert code == pf.RANK[status]


def test_mount_longest_prefix_and_escapes(tmp_path):
    mi = write_mountinfo(tmp_path / "mi", Path("/srv/a b"), "nfs", "nas:/x")
    assert pf.mount_of("/srv/a b/engram", str(mi))["fstype"] == "nfs"
    assert pf.mount_of("/srv/a bc", str(mi))["fstype"] == "ext4"
    assert pf.mount_of("/srv/a b", str(mi))["mountpoint"] == "/srv/a b"


def test_truncated_missing_and_tmp(tmp_path, capsys):
    args, engram, _ = setup(tmp_path)
    write_shards(engram, 0, truncate=True)
    assert pf.main(args + ["--rank", "0", "--no-probe"]) == 2
    c, _ = checks(capsys)
    assert "truncated" in c["engram-files"]["detail"]
    write_shards(engram, 0)
    (engram / "engram-l14-r0of2.bin").unlink()
    assert pf.main(args + ["--rank", "0", "--no-probe"]) == 2
    c, _ = checks(capsys)
    assert "engram-l14-r0of2.bin is missing" in c["engram-files"]["detail"]
    write_shards(engram, 0)
    (engram / "engram-l1-r0of2.bin.tmp").write_bytes(b"x")
    assert pf.main(args + ["--rank", "0", "--no-probe"]) == 1
    c, _ = checks(capsys)
    assert c["engram-files"]["status"] == "warn"


def test_bad_magic(tmp_path, capsys):
    args, engram, _ = setup(tmp_path)
    (engram / "engram-l1-r0of2.bin").write_bytes(b"\0" * 8192)
    assert pf.main(args + ["--rank", "0", "--no-probe"]) == 2
    c, _ = checks(capsys)
    assert "bad magic" in c["engram-files"]["detail"]


def test_memory_estimate(tmp_path, capsys):
    args, _, model = setup(tmp_path, avail=55.0)
    with open(
        model / "model-00001.safetensors", "wb"
    ) as f:  # 100 GiB sparse: 50 GiB a rank
        f.truncate(100 << 30)
    assert pf.main(args + ["--rank", "0", "--no-probe"]) == 1
    c, _ = checks(capsys)
    d = c["memory"]["data"]
    assert c["memory"]["status"] == "warn"
    assert d["weights_gib_estimate"] == pytest.approx(50.0)
    assert d["pool_gib_estimate"] == pytest.approx(2100000 * 1060 / 2**30, abs=0.01)
    assert d["need_gib_estimate"] == pytest.approx(
        50 + d["pool_gib_estimate"] + 3 + 4, abs=0.01
    )
    assert "estimate" in c["memory"]["detail"]
    args2, _, _ = setup(tmp_path / "b", avail=100.0)
    assert pf.main(args2 + ["--rank", "0", "--no-probe", "--min-avail-gib", "112"]) == 2


def test_shm_warns(tmp_path, capsys, monkeypatch):
    args, _, _ = setup(tmp_path)

    class St:
        f_blocks, f_bfree, f_frsize = 3 << 20, 1 << 20, 4096  # 8 GiB used

    real = os.statvfs
    monkeypatch.setattr(
        pf.os, "statvfs", lambda p: St() if p == "/fake/shm" else real(p)
    )
    args[args.index("--shm") + 1] = "/fake/shm"
    assert pf.main(args + ["--rank", "0", "--no-probe"]) == 1
    c, _ = checks(capsys)
    assert c["memory"]["data"]["shm_gib"] == pytest.approx(8.0)


def test_after_boot_from_log(tmp_path, capsys):
    args, _, _ = setup(tmp_path, avail=8.3)
    log = tmp_path / "r0.log"
    log.write_text("unrelated line\n" + BOOT_LOG)
    assert (
        pf.main(
            args + ["--rank", "0", "--no-probe", "--after-boot", "--boot-log", str(log)]
        )
        == 0
    )
    c, _ = checks(capsys)
    p = c["pool"]["data"]
    assert (
        p["pool_tokens"] == 2_100_000
        and p["bytes_a_token"] == 1060
        and p["worst_gib"] == 5.33
    )
    # 2.1M + (5.33 - 4.0 - 0.25) GiB / 1060 B, rounded down to 10,000
    assert (
        p["suggested_max_pool_tokens"]
        == (2_100_000 + int(1.08 * 2**30 / 1060)) // 10_000 * 10_000
    )
    assert "4K prefill workspace" in c["pool"]["detail"]
    assert pf.suggest_pool(2_100_000, 4.0, 4.0, 0.25, 1060) < 2_100_000


def test_after_boot_refused_and_fallback(tmp_path, capsys):
    args, _, _ = setup(tmp_path, avail=9.0)
    log = tmp_path / "r0.log"
    log.write_text(
        "error(kv): kv: rank 0: ...: the pool does not fit; the largest that does: "
        "TF_DSV41_POOL_TOKENS=1834000\n"
    )
    assert (
        pf.main(
            args + ["--rank", "0", "--no-probe", "--after-boot", "--boot-log", str(log)]
        )
        == 2
    )
    c, _ = checks(capsys)
    assert c["pool"]["data"]["refused_largest"] == 1_834_000
    assert (
        pf.main(
            args
            + [
                "--rank",
                "0",
                "--no-probe",
                "--after-boot",
                "--boot-log",
                str(tmp_path / "none"),
            ]
        )
        == 1
    )
    c, _ = checks(capsys)
    assert (
        c["pool"]["data"]["measured"] is False
        and "unmeasured" in c["pool"]["detail"]
        and "suggested_max_pool_tokens" not in c["pool"]["data"]
        and "worst_gib" not in c["pool"]["data"]
    )


def test_env_file_parsing(tmp_path):
    f = tmp_path / "e.env"
    f.write_text("# c\nA=\"x y\"\nexport B=2 # note\nC='q'\nbad line\n")
    assert pf.read_env_file(f) == {"A": "x y", "B": "2", "C": "q"}


def test_text_report(tmp_path, capsys):
    args, _, _ = setup(tmp_path)
    args.remove("--json")
    assert pf.main(args + ["--rank", "0", "--no-probe"]) == 0
    out = capsys.readouterr().out
    assert "[OK  ] engram-fs" in out and "preflight: ok (exit 0)" in out


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({}, "warn"),
        (None, "warn"),
        ([], "warn"),
        ("ok", "warn"),
        ({"unrecognized": True}, "warn"),
        ({"ok": "true"}, "warn"),
        ({"status": "ok"}, "warn"),
        ({"ok": True, "graphs": []}, "warn"),
        ({"ok": True, "graphs": {"decode": None}}, "warn"),
        ({"ok": True, "graphs": {"decode": {}}}, "warn"),
        ({"status": "draining"}, "fail"),
        ({"ok": False}, "fail"),
        ({"ok": True, "fatal": "failed"}, "fail"),
        ({"status": "ok", "warming": True}, "warn"),
        ({"ok": True, "graphs": {"decode": {"under_floor": True}}}, "warn"),
        ({"ok": True, "graphs": {"decode": {"under_floor": False}}}, "ok"),
    ],
)
def test_health_shape(monkeypatch, body, status):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(body).encode()

    class Opener:
        def open(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(pf.urllib.request, "build_opener", lambda *args: Opener())
    report = pf.Report()
    pf.check_health(report, "http://localhost")
    assert report.checks[0]["status"] == status
    if report.checks[0].get("data", {}).get("measured") is False:
        assert "unmeasured" in report.checks[0]["detail"]


@pytest.mark.parametrize("as_json", [True, False])
def test_preflight_redacts_errors_and_details(tmp_path, capsys, monkeypatch, as_json):
    args, _, _ = setup(tmp_path)
    if not as_json:
        args.remove("--json")
    key = "preflight-" + 'arbitrary"value'
    shaped = "sk-" + "synthetic" * 4
    config = Path(args[args.index("--config") + 1])
    with config.open("a") as f:
        f.write("API_KEY='" + key + "'\n")

    def failed_health(rep, url):
        rep.add("health", pf.FAIL, f"rejected {key}; {shaped}", response=key)

    monkeypatch.setattr(pf, "check_health", failed_health)
    log = tmp_path / "boot.log"
    log.write_text(BOOT_LOG)
    assert (
        pf.main(
            args
            + [
                "--rank",
                "0",
                "--no-probe",
                "--after-boot",
                "--boot-log",
                str(log),
                "--url",
                "http://localhost",
            ]
        )
        == 2
    )
    printed = capsys.readouterr()
    assert "<redacted>" in printed.out
    assert (
        key not in printed.out + printed.err and shaped not in printed.out + printed.err
    )
    if as_json:
        assert json.loads(printed.out)["exit"] == 2


def test_health_http_error_is_failure(monkeypatch):
    class Opener:
        def open(self, url, **kwargs):
            raise pf.urllib.error.HTTPError(url, 503, "Unavailable", {}, None)

    monkeypatch.setattr(pf.urllib.request, "build_opener", lambda *args: Opener())
    report = pf.Report()
    pf.check_health(report, "http://localhost")
    assert report.code == 2 and "HTTP 503" in report.checks[0]["detail"]
