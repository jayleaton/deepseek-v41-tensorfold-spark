import gzip
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.error

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_runner_assets  # noqa: E402
import fetch_runner_assets  # noqa: E402


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


URL = "https://example.invalid/releases/download/runner-assets-x/runner-assets-x.tar.gz"


def archive_bytes(files, extra=None):
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if extra:
            tar.addfile(extra)
    return raw.getvalue()


def pinned(tmp_path, files, archive=None, extra=None):
    """A manifest for `files`, a release file pinning the archive of `archive` (default: `files`) and that archive."""
    manifest = tmp_path / "assets.sha256"
    manifest.write_text("".join(f"{hashlib.sha256(d).hexdigest()}  {n}\n" for n, d in sorted(files.items())))
    data = archive_bytes(files if archive is None else archive, extra)
    rel = tmp_path / "runner-assets.release"
    rel.write_text(f"# test\nurl {URL}\nsha256 {hashlib.sha256(data).hexdigest()}\n")
    return manifest, rel, data


class Served:
    def __init__(self, data):
        self.stream = io.BytesIO(data)

    def __enter__(self):
        return self.stream

    def __exit__(self, *exc):
        return False


FILES = {"calc.tfpc": b"circuit", "graph-v7-powmod/bank/bank.json": b"{}"}


def test_download_checks_the_archive_and_every_file(tmp_path, monkeypatch):
    manifest, rel, data = pinned(tmp_path, FILES)
    monkeypatch.setattr(fetch_runner_assets.urllib.request, "urlopen", lambda url, timeout: Served(data))
    fetch_runner_assets.fetch(tmp_path / "runner", rel, manifest)
    assert (tmp_path / "runner/calc.tfpc").read_bytes() == b"circuit"
    assert not list(tmp_path.glob(".runner-*"))  # the downloaded archive does not stay


def test_unpacked_modes_come_from_the_archive_not_the_umask(tmp_path):
    files = {"graph-v7-powmod/bank/bank.json": b"{}"}
    manifest, rel, _ = pinned(tmp_path, files)
    archive = tmp_path / "local.tar.gz"
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for name in ("graph-v7-powmod", "graph-v7-powmod/bank"):
            info = tarfile.TarInfo(name)
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            tar.addfile(info)
        info = tarfile.TarInfo("graph-v7-powmod/bank/bank.json")
        info.size, info.mode = 2, 0o644
        tar.addfile(info, io.BytesIO(b"{}"))
    archive.write_bytes(raw.getvalue())
    rel.write_text(f"url {URL}\nsha256 {hashlib.sha256(archive.read_bytes()).hexdigest()}\n")
    old = os.umask(0o002)
    try:
        fetch_runner_assets.fetch(tmp_path / "runner", rel, manifest, archive=archive)
    finally:
        os.umask(old)
    modes = {p.relative_to(tmp_path / "runner").as_posix(): p.stat().st_mode & 0o777
             for p in (tmp_path / "runner").rglob("*")}
    assert modes == {"graph-v7-powmod": 0o755, "graph-v7-powmod/bank": 0o755, "graph-v7-powmod/bank/bank.json": 0o644}


def test_a_failed_download_names_the_file_and_the_url(tmp_path, monkeypatch):
    manifest, rel, _ = pinned(tmp_path, FILES)

    def refuse(url, timeout):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(fetch_runner_assets.urllib.request, "urlopen", refuse)
    monkeypatch.setattr(fetch_runner_assets.time, "sleep", lambda s: None)
    with pytest.raises(fetch_runner_assets.AssetError) as error:
        fetch_runner_assets.fetch(tmp_path / "runner", rel, manifest)
    assert "cannot download runner-assets-x.tar.gz from " + URL in str(error.value)


def test_a_wrong_archive_hash_stops_before_unpacking(tmp_path, monkeypatch):
    manifest, rel, data = pinned(tmp_path, FILES)
    monkeypatch.setattr(fetch_runner_assets.urllib.request, "urlopen", lambda url, timeout: Served(data + b"x"))
    with pytest.raises(fetch_runner_assets.AssetError) as error:
        fetch_runner_assets.fetch(tmp_path / "runner", rel, manifest)
    assert f"sha256 mismatch for runner-assets-x.tar.gz from {URL}" in str(error.value)
    assert not any((tmp_path / "runner").iterdir())


def test_a_missing_or_changed_file_is_named(tmp_path):
    manifest, rel, _ = pinned(tmp_path, FILES, archive={"calc.tfpc": b"other"})
    archive = tmp_path / "local.tar.gz"
    archive.write_bytes(archive_bytes({"calc.tfpc": b"other"}))
    with pytest.raises(fetch_runner_assets.AssetError) as error:
        fetch_runner_assets.fetch(tmp_path / "runner", rel, manifest, archive=archive)
    message = str(error.value)
    assert "missing graph-v7-powmod/bank/bank.json" in message and "sha256 differs calc.tfpc" in message
    assert URL in message


def test_unsafe_archive_entries_are_refused(tmp_path):
    link = tarfile.TarInfo("calc.tfpc.link")
    link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
    manifest, rel, data = pinned(tmp_path, FILES, extra=link)
    archive = tmp_path / "local.tar.gz"
    archive.write_bytes(data)
    with pytest.raises(fetch_runner_assets.AssetError) as error:
        fetch_runner_assets.fetch(tmp_path / "runner", rel, manifest, archive=archive)
    assert "unsafe entry: calc.tfpc.link" in str(error.value)


def test_the_cli_reports_on_stderr_and_fails(tmp_path):
    result = subprocess.run([sys.executable, "scripts/fetch_runner_assets.py", "--archive", str(tmp_path / "none.tgz"),
                             str(tmp_path / "runner")], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 1 and result.stderr.startswith("runner assets: ")


def test_the_pinned_release_is_this_repository_s_asset():
    url, digest = fetch_runner_assets.release()
    assert url.startswith("https://github.com/jayleaton/deepseek-v41-tensorfold-spark/releases/download/")
    assert url.endswith(".tar.gz") and len(digest) == 64
