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


def test_rollback_profile_leaves_the_compute_lane_off():
    lines = (ROOT / "config/prod-zig.env.example").read_text().splitlines()
    assert not [line for line in lines if line.startswith("TF_DSV41_COMPUTE_LANE")]


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
LANE = {
    "TF_DSV41_COMPUTE_LANE": "1", "TF_DSV41_COMPUTE_LANE_RUNNER": "0", "TF_DSV41_RUNNER": "0",
    "TF_DSV41_RUNNER_COMMIT": "0", "TF_DSV41_RUNNER_STATED": "0", "TF_DSV41_RUNNER_INLINE_CHAIN": "0",
    "TF_DSV41_RUNNER_INLINE_MARK": "0", "TF_DSV41_RUNNER_SYNC_GRAPH": "0", "TF_DSV41_RUNNER_FORMS": "fence,repl,inline",
}
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


def test_flash17_profile_sets_the_release_candidate_settings():
    lines = (ROOT / "config/prod-zig-flash17.env.example").read_text().splitlines()
    settings = dict(line.split("=", 1) for line in lines if line.startswith("TF_DSV41_"))
    assert {name: settings.get(name) for name in ("TF_DSV41_KV_MIRROR", "TF_DSV41_ROT_FOLD")} == {
        "TF_DSV41_KV_MIRROR": "4096", "TF_DSV41_ROT_FOLD": "3"}
    for name in ("IDX_SPLIT", "NO_COPIES", "TOKENIZER_MESSAGE_MEMO", "PF_SCRATCH_ALIAS", "ENGRAM_VALIDATE", "IDX_IOTA"):
        assert settings.get("TF_DSV41_" + name) == "1"
    assert settings.get("TF_DSV41_DEPTH_OVERHEAD_MS") == "0"
