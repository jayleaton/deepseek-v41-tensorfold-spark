from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "profile", ["prod-zig.env.example", "prod-zig-flash17.env.example"]
)
def test_example_profile_describes_known_pin_state(profile):
    pin = (ROOT / "config/tensorfold.rev").read_text().strip()
    assert len(pin) == 40 and all(c in "0123456789abcdef" for c in pin)
    engine = (ROOT / "docs/ENGINE.md").read_text()
    assert pin[:9] in engine
    assert "flash17 `dsq-1` checkpoint admission from `98f2a3896`" in engine

    comments = "\n".join(
        line.removeprefix("# ")
        for line in (ROOT / "config" / profile).read_text().splitlines()
        if line.startswith("#")
    )
    assert "current pin includes flash17 dsq-1 admission" in comments
    assert "pack/device/API qualification remains pending" in comments
    assert "docs/ENGINE.md" in comments
    assert "unresolved at the initial pin" not in comments
    assert "historical measurements do not qualify this pin" in comments


@pytest.mark.parametrize(
    "profile", ["prod-zig.env.example", "prod-zig-flash17.env.example"]
)
def test_example_profile_leaves_the_compute_lane_off(profile):
    lines = (ROOT / "config" / profile).read_text().splitlines()
    assert not [line for line in lines if line.startswith("TF_DSV41_COMPUTE_LANE")]


def test_flash17_profile_sets_the_release_candidate_settings():
    lines = (ROOT / "config/prod-zig-flash17.env.example").read_text().splitlines()
    settings = dict(line.split("=", 1) for line in lines if line.startswith("TF_DSV41_"))
    assert {name: settings.get(name) for name in ("TF_DSV41_KV_MIRROR", "TF_DSV41_ROT_FOLD")} == {
        "TF_DSV41_KV_MIRROR": "4096", "TF_DSV41_ROT_FOLD": "3"}
    for name in ("IDX_SPLIT", "NO_COPIES", "TOKENIZER_MESSAGE_MEMO", "PF_SCRATCH_ALIAS", "ENGRAM_VALIDATE", "IDX_IOTA"):
        assert settings.get("TF_DSV41_" + name) == "1"
    assert settings.get("TF_DSV41_DEPTH_OVERHEAD_MS") == "0"
