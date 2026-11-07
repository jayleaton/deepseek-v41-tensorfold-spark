import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_runner_assets  # noqa: E402


def write(root, files):
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    lines = [f"{hashlib.sha256(data).hexdigest()}  {name}" for name, data in sorted(files.items())]
    manifest = root.parent / "assets.sha256"
    manifest.write_text("# pins\n" + "\n".join(lines) + "\n")
    return manifest


def test_exact_files_pass_and_every_difference_is_named(tmp_path):
    root = tmp_path / "runner"
    manifest = write(root, {"calc.tfpc": b"circuit", "graph-v7-powmod/bank/bank.json": b"{}"})
    assert check_runner_assets.check(manifest, root) == []
    (root / "calc.tfpc").write_bytes(b"other")
    (root / "extra").write_bytes(b"")
    (root / "graph-v7-powmod/bank/bank.json").unlink()
    assert check_runner_assets.check(manifest, root) == [
        "missing graph-v7-powmod/bank/bank.json", "unlisted extra", "sha256 differs calc.tfpc"]


def test_symlinks_and_escaping_names_are_refused(tmp_path):
    root = tmp_path / "runner"
    manifest = write(root, {"calc.tfpc": b"circuit"})
    (root / "link").symlink_to(root / "calc.tfpc")
    result = subprocess.run([sys.executable, "scripts/check_runner_assets.py", str(manifest), str(root)], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 2 and "not a regular file: link" in result.stderr
    manifest.write_text(f"{'0' * 64}  ../calc.tfpc\n")
    (root / "link").unlink()
    try:
        check_runner_assets.check(manifest, root)
    except ValueError as error:
        assert "bad manifest line" in str(error)
    else:
        raise AssertionError("an escaping name was accepted")


def test_the_pinned_assets_are_the_ones_the_default_profile_names():
    pins = check_runner_assets.listed(ROOT / "config/runner-assets.sha256")
    assert pins["calc.tfpc"] == "3a421a2b8ba7c138c502ac205a8139ea05eeb2f7086715fe939b492ebbc18cf3"
    assert pins["graph-v7-powmod/bank/bank.json"] == "26862f0a758af4e01a1d8c69d501ad49ee1657162b36d057b8a3c312fcf12512"
    assert "graph-v7-powmod/bank/Qwenputer-MIT.txt" in pins
    profile = (ROOT / "config/prod-zig-flash17.env.example").read_text()
    for name in ("calc.tfpc", "graph-v7-powmod/bank/bank.json"):
        assert f"/opt/tensorfold/share/runner/{name}\n" in profile
