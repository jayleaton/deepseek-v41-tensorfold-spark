"""The build-only reference tree for rope.bin and sigs.json/AOT is pinned and rebuildable from a clean checkout."""

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/fetch-reference.sh"
TOOLS = ROOT / "engine/zig/tools/zig"
CONFIG = ROOT / "engine/zig/zig/src/families/deepseek_v41/fixtures/config.json"


def pins():
    text = SCRIPT.read_text()
    recipe = re.search(r"^recipe_rev=([0-9a-f]{40})$", text, re.M).group(1)
    sums = re.findall(r"([0-9a-f]{64})  (patches/[\w.-]+\.patch)", text)
    return recipe, sums


def reference_imports(path):
    """The tensorfold modules a generator imports; the deleted delivery path supplied them."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tensorfold"):
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names if alias.name.startswith("tensorfold"))
    return names


def resolves(src, name):
    parts = name.split(".")
    for end in range(len(parts), 0, -1):
        base = src.joinpath(*parts[:end])
        if base.with_suffix(".py").is_file() or (base / "__init__.py").is_file():
            return True
    return False


def test_generators_import_only_the_pinned_patch_modules():
    imports = reference_imports(TOOLS / "dsv41_rope_tables.py")
    package = "tensorfold.families.deepseek_v41.cuda"
    assert imports == {f"{package}.rope", f"{package}.config.Config"}
    assert not (ROOT / "vendor/TensorFold").exists() and not (ROOT / "patches").exists()
    recipe, sums = pins()
    assert [name for _, name in sums] == [f"patches/{n}.patch" for n in (
        "0001-spark-stack-060", "0002-deepseek-v41-family", "0003-dsv41-literal-added-tokens")]
    try:
        blobs = [subprocess.run(["git", "-C", str(ROOT), "show", f"{recipe}:{name}"],
                                capture_output=True, check=True).stdout for _, name in sums]
    except subprocess.CalledProcessError:
        pytest.skip("shallow checkout: the pinned recipe commit is fetched by the script instead")
    assert [hashlib.sha256(blob).hexdigest() for blob in blobs] == [digest for digest, _ in sums]
    family = blobs[1]
    added = set(re.findall(rb"^\+\+\+ b/(\S+)$", family, re.M))
    for module in ("rope", "config"):
        assert f"src/tensorfold/families/deepseek_v41/cuda/{module}.py".encode() in added


@pytest.mark.skipif(os.environ.get("TF_REFERENCE_FETCH") != "1",
                    reason="TF_REFERENCE_FETCH=1 fetches the pinned upstream revision over the network")
def test_clean_checkout_generates_assets(tmp_path):
    destination = tmp_path / "reference"
    result = subprocess.run(["bash", str(SCRIPT), str(destination)], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    src = destination / "src"
    assert all(resolves(src, name) for name in reference_imports(TOOLS / "dsv41_rope_tables.py"))
    for kernels in ("tensorfold/families/deepseek_v41", "tensorfold/cuda"):
        assert any((src / kernels).rglob("*.py"))
    assert subprocess.run(["bash", str(SCRIPT), str(destination)], capture_output=True).returncode == 2
    pytest.importorskip("torch")
    rope = tmp_path / "rope.bin"
    subprocess.run([sys.executable, str(TOOLS / "dsv41_rope_tables.py"), "--config", str(CONFIG), "--rows", "64",
                    "--out", str(rope)], env=dict(os.environ, PYTHONPATH=str(src)), check=True, timeout=300)
    assert rope.read_bytes()[:16] == b"DSV41RP1" + (64).to_bytes(4, "little") + (64).to_bytes(4, "little")
    assert rope.stat().st_size == 16 + 2 * 64 * 64 * 4
    pytest.importorskip("triton")
    sigs = tmp_path / "sigs.json"
    subprocess.run([sys.executable, str(TOOLS / "triton_fill.py"), "sigs", "--py", str(src), "--out", str(sigs)],
                   check=True, timeout=600, capture_output=True)
    generated = json.loads(sigs.read_text())["kernels"]
    engine = os.environ.get("TENSORFOLD_SOURCE", "")
    pinned = Path(engine) / "zig/src/families/deepseek_v41/fixtures/aot-sigs.json"
    if engine.startswith("/") and pinned.is_file():
        assert generated == json.loads(pinned.read_text())["kernels"]
    else:
        assert len(generated) == 88
