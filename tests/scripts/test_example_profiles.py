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
