import os
from pathlib import Path
import re
import subprocess

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


TOOL_FULL = {
    "TF_DSV41_COMPUTE_LANE": "1", "TF_DSV41_RUNNER": "1",
    "TF_DSV41_RUNNER_CIRCUIT": "/opt/tensorfold/share/runner/calc.tfpc",
    "TF_DSV41_RUNNER_FORMS": "fence,repl,inline,trace", "TF_DSV41_RUNNER_BACKEND": "graph",
    "TF_DSV41_RUNNER_GRAPH_BANK": "/opt/tensorfold/share/runner/graph-v7-powmod/bank/bank.json",
    "TF_DSV41_RUNNER_STATED": "0", "TF_DSV41_RUNNER_INLINE_CHAIN": "0", "TF_DSV41_RUNNER_INLINE_MARK": "0",
    "TF_DSV41_RUNNER_SYNC_GRAPH": "0", "TF_DSV41_RUNNER_COMPILE_STEPS": "10522", "TF_DSV41_RUNNER_CELLS": "1",
    "TF_DSV41_COMPUTE_LANE_RUNNER": "1",
    "TF_DSV41_COMPUTE_LANE_RUNNER_CIRCUIT": "/opt/tensorfold/share/runner/calc.tfpc",
    "TF_DSV41_COMPUTE_LANE_RUNNER_BACKEND": "graph",
    "TF_DSV41_COMPUTE_LANE_RUNNER_GRAPH_BANK": "/opt/tensorfold/share/runner/graph-v7-powmod/bank/bank.json",
    "TF_DSV41_COMPUTE_LANE_RUNNER_SETTLE_US": "3000", "TF_DSV41_RUNNER_COMMIT": "0",
}
# Unread while the runner switches are 0; empty is unset to the engine.
UNSET = dict.fromkeys(("TF_DSV41_RUNNER_CIRCUIT", "TF_DSV41_RUNNER_BACKEND", "TF_DSV41_RUNNER_GRAPH_BANK",
                       "TF_DSV41_RUNNER_COMPILE_STEPS", "TF_DSV41_RUNNER_CELLS", "TF_DSV41_COMPUTE_LANE_RUNNER_CIRCUIT",
                       "TF_DSV41_COMPUTE_LANE_RUNNER_BACKEND", "TF_DSV41_COMPUTE_LANE_RUNNER_GRAPH_BANK",
                       "TF_DSV41_COMPUTE_LANE_RUNNER_SETTLE_US"), "")
LANE = {
    "TF_DSV41_COMPUTE_LANE": "1", "TF_DSV41_COMPUTE_LANE_DEFAULT": "0", "TF_DSV41_COMPUTE_LANE_RUNNER": "0",
    "TF_DSV41_RUNNER": "0", "TF_DSV41_RUNNER_COMMIT": "0", "TF_DSV41_RUNNER_STATED": "0",
    "TF_DSV41_RUNNER_INLINE_CHAIN": "0", "TF_DSV41_RUNNER_INLINE_MARK": "0", "TF_DSV41_RUNNER_SYNC_GRAPH": "0",
    "TF_DSV41_RUNNER_FORMS": "fence,repl,inline", **UNSET,
}
OFF = {**LANE, "TF_DSV41_COMPUTE_LANE": "0", "TF_DSV41_RUNNER_FORMS": ""}
COMPUTE = ("TF_DSV41_COMPUTE_LANE", "TF_DSV41_RUNNER")


def flash17():
    return (ROOT / "config/prod-zig-flash17.env.example").read_text().splitlines()


def test_flash17_default_is_the_tool_full_profile_for_every_chat():
    settings = [line.split("=", 1) for line in flash17() if line.startswith(COMPUTE)]
    assert len(settings) == len(TOOL_FULL) + 1
    assert dict(settings) == {**TOOL_FULL, "TF_DSV41_COMPUTE_LANE_DEFAULT": "1"}
    assert not [line for line in flash17() if line.startswith("TF_CUDA_SCHED")]


def test_flash17_lane_arm_is_a_commented_opt_in_that_replaces_the_default():
    lines = flash17()
    start = next(i for i, line in enumerate(lines) if line.startswith("# Opt-in: arm L"))
    assert "Replace the nineteen lines above" in " ".join(lines[start:start + 4])
    block = [line[2:].split("=", 1) for line in lines[start:] if line.startswith("# TF_DSV41_")]
    assert dict(block[:len(LANE)]) == LANE and len(block[:len(LANE)]) == len(LANE)
    assert set(LANE) == set(TOOL_FULL) | {"TF_DSV41_COMPUTE_LANE_DEFAULT"}  # every name the default sets


def test_rollback_profile_sets_every_compute_name_off():
    lines = (ROOT / "config/prod-zig.env.example").read_text().splitlines()
    assert dict(line.split("=", 1) for line in lines if line.startswith(COMPUTE)) == OFF


def arm_l(text):
    """The flash17 profile with its default block replaced by the arm-L block, as its comment says."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("# Opt-in: arm L"))
    lane = [line[2:] for line in lines[start:] if line.startswith("# TF_DSV41_")][:len(LANE)]
    kept = [line for line in lines[:start] if not line.startswith(COMPUTE)]
    return "\n".join(kept + lane + lines[start:]) + "\n"


def runnable(text):
    """A profile with its required placeholders filled, as a user's copy would be."""
    filled = {"IMAGE": "fixture", "TF_TP_MASTER": "rank0.example", "NCCL_SOCKET_IFNAME": "eth0", "NCCL_IB_HCA": "mlx5_0"}
    out = []
    for line in text.splitlines():
        name = line.split("=", 1)[0]
        if "=" in line and not line.startswith("#") and "<" in line:
            line = f"{name}={filled.get(name, '/srv/' + name.lower())}"
        out.append(line)
    return "\n".join(out) + "\n"


def effective(tmp_path, target, exported=None):
    """run-zig.sh's --effective-profile for CONFIG=target, from a shell that first ran `set -a; source exported`."""
    script = 'set -a; source "$1"; set +a; ' if exported else ""
    script += 'export CONFIG="$2"; bash scripts/run-zig.sh 0 --effective-profile'
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TF_", "TENSORFOLD_", "NCCL_"))}
    result = subprocess.run(["bash", "-c", script, "_", str(exported or ""), str(target)], cwd=ROOT, env=env,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    found = {}
    for line in result.stdout.splitlines():
        match = re.fullmatch(r"([A-Z0-9_]+)=(.*)  \[(\w+)\]", line)
        if match:
            found[match[1]] = (match[2], match[3])
    return found, result.stderr


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def compute(found):
    return {name: value for name, (value, _) in found.items() if name.startswith(COMPUTE)}


def test_switching_from_the_exported_default_to_arm_l_or_the_rollback_keeps_no_default_value(tmp_path):
    default = write(tmp_path, "default.env", runnable("\n".join(flash17())))
    lane = write(tmp_path, "arm-l.env", runnable(arm_l("\n".join(flash17()))))
    rollback = write(tmp_path, "rollback.env", runnable((ROOT / "config/prod-zig.env.example").read_text()))

    clean_l, _ = effective(tmp_path, lane)
    after_l, warning = effective(tmp_path, lane, exported=default)
    assert compute(clean_l) == compute(after_l) == LANE
    assert "calling shell" not in warning and not [n for n, (_, src) in after_l.items() if src == "caller"]

    clean_rollback, _ = effective(tmp_path, rollback)
    after_rollback, warning = effective(tmp_path, rollback, exported=default)
    assert compute(clean_rollback) == compute(after_rollback) == OFF
    # what the rollback does not set keeps the exported default's value: named, never silent
    assert after_rollback["TF_DSV41_KV_MIRROR"] == ("4096", "caller") and "TF_DSV41_KV_MIRROR" not in clean_rollback
    assert "from the calling shell, not CONFIG" in warning and "TF_DSV41_KV_MIRROR" in warning
    assert not [n for n in compute(after_rollback) if after_rollback[n][1] == "caller"]


def test_flash17_profile_sets_the_release_candidate_settings():
    lines = (ROOT / "config/prod-zig-flash17.env.example").read_text().splitlines()
    settings = dict(line.split("=", 1) for line in lines if line.startswith("TF_DSV41_"))
    assert {name: settings.get(name) for name in ("TF_DSV41_KV_MIRROR", "TF_DSV41_ROT_FOLD")} == {
        "TF_DSV41_KV_MIRROR": "4096", "TF_DSV41_ROT_FOLD": "3"}
    for name in ("IDX_SPLIT", "NO_COPIES", "TOKENIZER_MESSAGE_MEMO", "PF_SCRATCH_ALIAS", "ENGRAM_VALIDATE", "IDX_IOTA"):
        assert settings.get("TF_DSV41_" + name) == "1"
    assert settings.get("TF_DSV41_DEPTH_OVERHEAD_MS") == "0"
