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
