"""The settings manifest tracks the pin, and the example profiles use only engine names."""

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import engine_settings  # noqa: E402

PROFILES = ["prod-zig.env.example", "prod-zig-flash17.env.example"]


def test_manifest_is_for_the_pinned_engine():
    revision, settings = engine_settings.manifest()
    assert revision == (ROOT / "config/tensorfold.rev").read_text().strip()
    assert len(settings) > 100 and all(engine_settings.ENGINE_NAME.fullmatch(n) for n in settings)
    assert not set(engine_settings.LEGACY) & settings


@pytest.mark.skipif(not os.environ.get("TENSORFOLD_SOURCE", "").startswith("/"),
                    reason="TENSORFOLD_SOURCE (an engine git checkout) not provided")
def test_manifest_matches_the_engine_source_at_the_pin():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/engine_settings.py"), "check",
                             os.environ["TENSORFOLD_SOURCE"]], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr


def test_prefix_parsed_knob_groups_list_their_composed_names():
    lines = [
        'r:zig/src/core/compute_lane.zig:        c.workers = try knob(u8, get, prefix ++ "_WORKERS", c.workers, 1, 64);',
        'r:zig/src/core/compute_lane.zig:        if (get(prefix ++ "_PYTHON")) |p| if (p.len > 0) {',
        'r:zig/src/families/x/serve.zig:    c = try api.compute_lane.Config.parse(&@import("knobs.zig").env, "TF_X_LANE");',
        # A suffix in another module, and a parse call naming no prefix-parsed module, compose nothing.
        'r:zig/src/core/other.zig:    get(prefix ++ "_DEPTH")',
        'r:zig/src/families/x/serve.zig:    v = try std.json.parse(bytes, "TF_X_JSON")',
    ]
    assert engine_settings.composed_names(lines) == {"TF_X_LANE_WORKERS", "TF_X_LANE_PYTHON"}


def test_manifest_lists_the_compute_lane_knob_group():
    _, settings = engine_settings.manifest()
    assert {"TF_DSV41_COMPUTE_LANE"} | {"TF_DSV41_COMPUTE_LANE_" + s for s in (
        "WORKERS", "MAX_MS", "MAX_RESULT_TOKENS", "MAX_CALLS", "MEM_MB", "PYTHON", "CPUS")} <= settings


@pytest.mark.parametrize("profile", PROFILES)
def test_example_profiles_set_only_settings_the_pinned_engine_reads(profile):
    _, settings = engine_settings.manifest()
    text = (ROOT / "config" / profile).read_text()
    names = set(re.findall(r"^((?:TF|TENSORFOLD|GLM53)_[A-Z0-9_]+)=", text, re.MULTILINE))
    assert names <= settings | engine_settings.RECIPE_NAMES, sorted(names - settings - engine_settings.RECIPE_NAMES)
    assert not engine_settings.legacy_messages(names)
    # The transport and serving settings arm E lost under their old names.
    for name in ("TF_TP_ROCE_FAST", "TF_TP_MAILBOX_MAX_KB", "TF_TP_PLAN_LINK", "TF_TP_PLAN_PIN",
                 "TF_TP_PLAN_TIMEOUT", "TF_DRAIN_S", "TF_SSE_KEEPALIVE_S", "TENSORFOLD_REQUEST_LOG"):
        assert name in names


TRANSPORT = re.compile(r"TF_(?:TP_|ROCE|NCCL_LIB$|GRAPH_CACHE$|COMM_BACKEND$)")


def test_every_transport_setting_is_classified_common_or_rank_local():
    _, settings = engine_settings.manifest()
    common, local = engine_settings.TRANSPORT_COMMON, engine_settings.RANK_LOCAL
    assert not common & local
    unclassified = sorted(n for n in settings if TRANSPORT.match(n)
                          and n not in common | local | engine_settings.LAUNCHER_OWNED)
    assert not unclassified, f"classify as common unless the engine shows it is per-rank: {unclassified}"


@pytest.mark.skipif(not os.environ.get("TENSORFOLD_SOURCE", "").startswith("/"),
                    reason="TENSORFOLD_SOURCE (an engine git checkout) not provided")
def test_rank_local_split_follows_the_engine_transport_digest():
    revision, _ = engine_settings.manifest()
    source = subprocess.run(["git", "-C", os.environ["TENSORFOLD_SOURCE"], "show",
                             f"{revision}:zig/src/fabric/tp/config.zig"],
                            capture_output=True, text=True, timeout=60, check=True).stdout
    digest = source.split("pub fn digest", 1)[1].split("return out;", 1)[0]
    assert set(re.findall(r"\bc\.(\w+)", digest)) == engine_settings.DIGEST_FIELDS
    body = source.split("\ntest ", 1)[0]
    names = set(re.findall(r'"(TF_[A-Z0-9_]+)"', body))
    classified = engine_settings.TRANSPORT_COMMON | engine_settings.RANK_LOCAL | engine_settings.LAUNCHER_OWNED
    assert names <= classified, sorted(names - classified)


def engine_file(path):
    revision, _ = engine_settings.manifest()
    return subprocess.run(["git", "-C", os.environ["TENSORFOLD_SOURCE"], "show", f"{revision}:{path}"],
                          capture_output=True, text=True, timeout=60, check=True).stdout


@pytest.mark.skipif(not os.environ.get("TENSORFOLD_SOURCE", "").startswith("/"),
                    reason="TENSORFOLD_SOURCE (an engine git checkout) not provided")
def test_bootstrap_settings_before_the_digest_compare_are_common():
    # RR10b-1: Session.open runs only Bootstrap.open before openBootstrapped agrees on Config.digest.
    tp = engine_file("zig/src/fabric/tp/tp.zig")
    session = tp.split("pub fn open(s: *Session", 1)[1].split("pub fn openBootstrapped(", 1)[0]
    calls = set(re.findall(r"\b([A-Za-z_][\w.]*)\(", session)) - {"open"}
    assert "defer b.close();" in session  # runs at scope exit, not before the compare
    assert calls == {"requested.resolved", "Bootstrap.open", "b.close", "s.openBootstrapped"}, sorted(calls)
    agreed = tp.split("pub fn openBootstrapped(", 1)[1].split("b.agree(&cfg.digest()", 1)[0]
    assert set(re.findall(r"\b([A-Za-z_][\w.]*)\(", agreed)) == {"requested.resolved"}
    # The DeepSeek-V4.1 Engram preflight opens the bootstrap earlier; what it reads is shared by the pair identity.
    preflight = engine_file("zig/src/families/deepseek_v41/engram_preflight.zig").split("\ntest ", 1)[0]
    read = set(re.findall(r'"(TF_[A-Z0-9_]+)"', preflight))
    assert read == {"TF_DSV41_ENGRAM_VALIDATE", "TF_DSV41_ENGRAM_DIR"} and not read & engine_settings.RANK_LOCAL
    bootstrap = engine_file("zig/src/fabric/tp/bootstrap.zig")
    body = bootstrap.split("pub fn open(cfg: Config)", 1)[1].split("\n    }\n", 1)[0]
    assert set(re.findall(r"\bcfg\.(\w+)", body)) == engine_settings.BOOTSTRAP_FIELDS
    # world, rank, master and port are launcher-owned; the plan port must match on both ranks
    config = engine_file("zig/src/fabric/tp/config.zig").split("\ntest ", 1)[0]
    assert re.search(r'"TF_TP_PLAN_PORT"', config) and "link_port" in config
    assert "TF_TP_PLAN_PORT" in engine_settings.TRANSPORT_COMMON
    assert {"TF_TP_MASTER", "TF_TP_PORT", "TF_TP_WORLD", "TF_TP_RANK"} <= engine_settings.LAUNCHER_OWNED
