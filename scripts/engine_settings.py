#!/usr/bin/env python3
"""The engine settings the launcher forwards, from the pinned TensorFold source.

config/tensorfold-settings.txt lists every TF_* / TENSORFOLD_* name the pinned engine
reads (string literals in its non-test zig/src). The launcher forwards exactly those
names that are set, plus NCCL_*. It never forwards credential names, or names it sets
itself. It refuses renamed or removed settings, so an old profile cannot silently fall
back to an engine default.

Commands (names arrive on stdin, one per line; values are never read from argv):
  forward               print the names to forward; refuse legacy names (exit 2)
  profile ARGV...       print the effective container environment of a docker run argv
  generate SRC [OUT]    write the manifest from an engine checkout at config/tensorfold.rev
  check SRC             exit 1 when the manifest differs from the engine checkout
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config/tensorfold-settings.txt"
REV_FILE = ROOT / "config/tensorfold.rev"
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
ENGINE_NAME = re.compile(r"(?:TF|TENSORFOLD)_[A-Z0-9_]+")
# Names the launcher sets itself with an explicit value (scripts/run-zig.sh).
LAUNCHER_OWNED = frozenset({
    "TF_TP_WORLD", "TF_TP_RANK", "TF_TP_DEVICE", "TF_TP_MASTER", "TF_TP_PORT", "TF_COMM_BACKEND",
    "TF_DSV41_ASSETS", "TF_DSV41_ENGRAM_DIR", "TF_TP_KERNEL_IMAGE",
})
# The pair identity follows the engine's own split. fabric/tp/config.zig Config.digest
# hashes only these fields; every other transport setting may differ per rank.
DIGEST_FIELDS = frozenset({"world", "backend", "mailbox_max_bytes", "roce_chunked_bytes", "failfast",
                           "roce_wire", "roce_fallback", "graph_cache", "graph_cache_env"})
# Transport settings feeding those fields (world and backend are launcher-owned), plus
# the older cuda/roce.zig protocol switches. Both ranks must agree on these.
TRANSPORT_COMMON = frozenset({"TF_TP_MAILBOX_MAX_KB", "TF_TP_ROCE_CHUNKED_KB", "TF_TP_FAILFAST",
                              "TF_TP_ROCE_WIRE", "TF_TP_ROCE_FALLBACK", "TF_GRAPH_CACHE",
                              "TF_ROCE", "TF_ROCE_MAX_KB"})
# Per-rank settings, forwarded but outside the identity both ranks must share: placement
# (CPUs, devices, ports), timeouts, the fast-path tuning the engine marks local, and diagnostics.
RANK_LOCAL = frozenset({
    "TF_TP_PLAN_PORT", "TF_TP_PLAN_TIMEOUT", "TF_TP_PLAN_LINK", "TF_TP_PLAN_SPIN_US", "TF_TP_PLAN_PIN",
    "TF_TP_MAILBOX_TIMEOUT_S", "TF_TP_FAILFAST_GRACE_S", "TF_TP_PEER_MEM_S",
    "TF_TP_ROCE_HCA", "TF_TP_ROCE_HCAS", "TF_TP_ROCE_TC", "TF_TP_ROCE_FAST", "TF_TP_ROCE_POLLERS",
    "TF_TP_ROCE_STAGGER_NS", "TF_TP_ROCE_BLOCKS", "TF_TP_ROCE_CPU", "TF_NCCL_LIB",
    "TF_TP_ALLOW_HOST_ONLY", "TF_TP_LOCAL", "TF_TP_STATS", "TF_TP_TRACE",  # host-only tests; flashnext only
    "TF_ROCE_HCA", "TF_ROCE_GID_INDEX",
})
# Recipe settings that become engine flags or host behaviour, never container environment.
RECIPE_NAMES = frozenset({"TF_DSV41_THINKING", "TF_DSV41_DEFAULT_EFFORT", "TF_DSV41_LAUNCH_T0",
                          "TF_NATIVE_REQUIRE_FILE", "TENSORFOLD_SOURCE", "TENSORFOLD_REV", "TF_SRC"})
RECIPE_PREFIXES = ("TF_SAFE_",)
# Profile names the pinned engine no longer reads: the replacement, or why the line goes.
LEGACY = {
    "GLM53_TF_ROCE_FAST": "TF_TP_ROCE_FAST",
    "GLM53_TF_ROCE_POLLERS": "TF_TP_ROCE_POLLERS",
    "GLM53_TF_ROCE_STAGGER_NS": "TF_TP_ROCE_STAGGER_NS",
    "GLM53_TF_ROCE_MAX_KB": "TF_TP_MAILBOX_MAX_KB",
    "TF_DSV41_PLAN_LINK": "TF_TP_PLAN_LINK",
    "TF_DSV41_PLAN_PIN": "TF_TP_PLAN_PIN",
    "TF_DSV41_PLAN_TIMEOUT": "TF_TP_PLAN_TIMEOUT",
    "TF_DSV41_DRAIN_S": "TF_DRAIN_S",
    "TF_DSV41_SSE_KEEPALIVE_S": "TF_SSE_KEEPALIVE_S",
    "TF_DSV41_REQUEST_LOG": "TENSORFOLD_REQUEST_LOG",
    "TF_DSV41_GRAMMAR": None,  # response_format output is always grammar-held (tools: TENSORFOLD_TOOL_GRAMMAR)
    "TF_DSV41_INDEX_KV": None,  # the CUDA index KV is always FP8
    "TF_DSV41_DISCONNECT": None,  # a client disconnect always cancels its request
}


def manifest(path=MANIFEST):
    """(revision, names) from the checked-in manifest."""
    revision, names = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith("rev "):
            revision = line[4:].strip()
        else:
            names.append(line.strip())
    return revision, frozenset(names)


def legacy_messages(names):
    from redact import secret_name
    out = []
    for name in sorted(set(names)):
        if name in LEGACY:
            new = LEGACY[name]
            out.append(f"{name} is not read by the pinned engine; rename it to {new}" if new
                       else f"{name} is not read by the pinned engine; remove it (fixed engine behaviour)")
        elif name.startswith("GLM53_TF_") and not secret_name(name):
            out.append(f"{name} is not read by the pinned engine; see config/tensorfold-settings.txt")
    return out


def forwarded(names, settings):
    from redact import secret_name
    return sorted(n for n in set(names)
                  if (n in settings or n.startswith("NCCL_"))
                  and n not in LAUNCHER_OWNED and n not in RECIPE_NAMES and not secret_name(n))


def ignored(names, settings):
    """Engine-looking names that the pinned engine does not read (typos, removed settings)."""
    from redact import secret_name
    return sorted(n for n in set(names)
                  if ENGINE_NAME.fullmatch(n) and n not in settings and n not in RECIPE_NAMES
                  and not n.startswith(RECIPE_PREFIXES) and not secret_name(n))


def _stdin_names():
    names = [line.strip() for line in sys.stdin if line.strip()]
    return [n for n in names if NAME.fullmatch(n)]


def forward_command():
    names = _stdin_names()
    problems = legacy_messages(names)
    for message in problems:
        print(message, file=sys.stderr)
    if problems:
        return 2
    revision, settings = manifest()
    for name in ignored(names, settings):
        print(f"warning: {name} is not read by TensorFold {revision[:9]}; it is not forwarded", file=sys.stderr)
    for name in forwarded(names, settings):
        print(name)
    return 0


def profile_command(argv):
    """The environment a docker run argv gives its container, and names left out."""
    from redact import secret_name
    names = _stdin_names()
    revision, settings = manifest()
    env, i = [], 0
    while i < len(argv):
        if argv[i] == "-e" and i + 1 < len(argv):
            name, eq, value = argv[i + 1].partition("=")
            env.append((name, value if eq else os.environ.get(name, ""), "launcher" if eq else "profile"))
            i += 2
            continue
        i += 1
    print(f"# effective engine environment (TensorFold {revision[:9]})")
    for name, value, source in sorted(env):
        print(f"{name}={value}  [{source}]")
    secrets = sorted(n for n in set(names) if secret_name(n))
    if secrets:
        print("# credential names present, never forwarded (values not shown)")
        for name in secrets:
            print(name)
    left = ignored(names, settings)
    if left:
        print("# set but not read by the pinned engine, not forwarded")
        for name in left:
            print(name)
    return 0


def engine_names(source, revision):
    sys.path.insert(0, str(ROOT / "scripts"))
    from safe_exec import run
    result = run(["git", "-C", str(source), "grep", "-h", "-o", "-E", r'"(TF|TENSORFOLD)_[A-Z0-9_]+"',
                  revision, "--", "zig/src/*.zig", ":!*_test.zig"], timeout=120)
    if result.returncode != 0:
        raise SystemExit("cannot read the engine source at " + revision)
    return sorted({line.strip().strip('"') for line in result.stdout.splitlines() if line.strip()})


def render(revision, names):
    return ("# Every TF_* / TENSORFOLD_* name the pinned TensorFold reads: string literals in its\n"
            "# non-test zig/src. Regenerate on every pin bump:\n"
            "#   python3 scripts/engine_settings.py generate /path/to/tensorfold-checkout\n"
            f"rev {revision}\n" + "".join(name + "\n" for name in names))


def main(argv):
    sys.path.insert(0, str(ROOT / "scripts"))
    command = argv[0] if argv else ""
    if command == "forward" and len(argv) == 1:
        return forward_command()
    if command == "profile":
        return profile_command(argv[1:])
    if command in ("generate", "check") and len(argv) in (2, 3):
        revision = REV_FILE.read_text().strip()
        names = engine_names(argv[1], revision)
        if command == "generate":
            Path(argv[2] if len(argv) == 3 else MANIFEST).write_text(render(revision, names))
            return 0
        listed_revision, listed = manifest()
        if listed_revision != revision or listed != set(names):
            print("config/tensorfold-settings.txt differs from the engine source; regenerate it", file=sys.stderr)
            print("missing: " + " ".join(sorted(set(names) - listed)), file=sys.stderr)
            print("extra: " + " ".join(sorted(listed - set(names))), file=sys.stderr)
            return 1
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    from safe_output import cli
    raise SystemExit(cli(lambda: main(sys.argv[1:])))
