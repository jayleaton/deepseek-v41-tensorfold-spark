"""The CUDA forward-compatibility layer: its pinned driver files, the build context and the entry's choice."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import cuda_compat  # noqa: E402
import engine_settings  # noqa: E402

FILES = {"libcuda.so.610.43.02": b"driver", "libnvidia-nvvm70.so.4": b"nvvm"}
LINKS = {"libcuda.so.1": "libcuda.so.610.43.02", "libcuda.so": "libcuda.so.1"}
IMAGE = "mia@sha256:" + "a" * 64
MIRROR = "registry.example/base@sha256:" + "b" * 64


def manifests(tmp_path):
    files, links = tmp_path / "cuda-compat.sha256", tmp_path / "cuda-compat.links"
    files.write_text("# test\n" + "".join(f"{hashlib.sha256(d).hexdigest()}  {n}\n" for n, d in sorted(FILES.items())))
    links.write_text("# test\n" + "".join(f"{n} -> {t}\n" for n, t in sorted(LINKS.items())))
    rel = tmp_path / "cuda-compat.release"
    rel.write_text(f"# test\nimage {IMAGE}\nmirror {MIRROR}\npath /usr/local/cuda/compat/lib.real\n")
    return files, links, rel


def driver_dir(root, files=FILES, links=LINKS):
    root.mkdir()
    for name, data in files.items():
        (root / name).write_bytes(data)
    for name, target in links.items():
        (root / name).symlink_to(target)
    return root


def test_the_exact_driver_passes_and_a_tampered_file_fails(tmp_path):
    files, links, _ = manifests(tmp_path)
    root = driver_dir(tmp_path / "compat")
    assert cuda_compat.check(root, files, links) == []
    (root / "libcuda.so.610.43.02").write_bytes(b"drivex")
    assert cuda_compat.check(root, files, links) == ["sha256 differs libcuda.so.610.43.02"]
    assert cuda_compat.main(["check", str(root)]) == 1


def test_every_other_difference_is_named(tmp_path):
    files, links, _ = manifests(tmp_path)
    root = driver_dir(tmp_path / "compat")
    (root / "libnvidia-nvvm70.so.4").unlink()
    (root / "extra.so").write_bytes(b"")
    (root / "libcuda.so").unlink()
    (root / "libcuda.so").symlink_to("libcuda.so.610.43.02")
    (root / "libcuda.so.1").unlink()
    (root / "libcuda.so.1").write_bytes(b"driver")
    assert cuda_compat.check(root, files, links) == [
        "missing libnvidia-nvvm70.so.4", "unlisted extra.so",
        "link differs libcuda.so: libcuda.so.610.43.02, expected libcuda.so.1",
        "libcuda.so.1 is a file, expected a link"]
    (root / "sub").mkdir()
    with pytest.raises(ValueError, match="not a plain file or link: sub"):
        cuda_compat.check(root, files, links)


def test_manifests_and_release_refuse_loose_pins(tmp_path):
    files, links, rel = manifests(tmp_path)
    files.write_text(f"{'0' * 64}  ../libcuda.so\n")
    with pytest.raises(ValueError, match="bad manifest line"):
        cuda_compat.pins(files, links)
    rel.write_text("image mia:latest\nmirror registry.example/base:26.07\npath /usr/local/cuda/compat/lib.real\n")
    with pytest.raises(cuda_compat.CompatError, match="name@sha256"):
        cuda_compat.release(rel)


def test_the_committed_pins_parse():
    image, mirror, where = cuda_compat.release()
    pins = cuda_compat.pins()
    kinds = [kind for kind, _ in pins.values()]
    assert (kinds.count("file"), kinds.count("link")) == (8, 8)
    assert image == "tensorfold-mia-dualtf@sha256:678f4d2cac0301f5632df4adca84026df94cb560eb99ac7617488bcaa93aa751"
    assert mirror == "nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c"
    assert where == "/usr/local/cuda-13.3/compat/lib.real"
    driver = "568153835562ea89c0c1669635e5631ada5f0862689025e3ed4a383086413e3a"
    assert pins["libcuda.so.610.43.02"] == ("file", driver)
    assert pins["libcuda.so.1"] == ("link", "libcuda.so.610.43.02")


class FakeDocker:
    """docker for extract(): `local` digests are present; cp writes `files` and `links` into the destination."""

    def __init__(self, local, files=FILES, links=LINKS):
        self.local, self.files, self.links, self.calls = set(local), files, links, []

    def __call__(self, *args, check=True, timeout=600):
        self.calls.append(args[0])
        out = ""
        if args[0] == "image":
            ref = args[-1]
            if ref not in self.local:
                raise cuda_compat.CompatError("no such image")
            out = f"sha256:{'c' * 64} {ref}\n"
        if args[0] == "pull":
            self.local.add(args[-1])
        if args[0] == "create":
            out = "container1\n"
        if args[0] == "cp":
            driver_dir(Path(args[2]), self.files, self.links)
        return SimpleNamespace(stdout=out, stderr="", returncode=0)


def test_extract_takes_the_pinned_image_else_pulls_the_mirror(tmp_path, monkeypatch):
    files, links, rel = manifests(tmp_path)
    fake = FakeDocker({IMAGE})
    monkeypatch.setattr(cuda_compat, "docker", fake)
    assert cuda_compat.extract(tmp_path / "a", rel, files, links) == IMAGE
    assert "pull" not in fake.calls
    fake = FakeDocker(set())
    monkeypatch.setattr(cuda_compat, "docker", fake)
    assert cuda_compat.extract(tmp_path / "b", rel, files, links) == MIRROR
    assert fake.calls[:3] == ["image", "pull", "image"]


def test_extract_fails_closed_on_a_tampered_image_and_still_removes_its_container(tmp_path, monkeypatch):
    files, links, rel = manifests(tmp_path)
    fake = FakeDocker({IMAGE}, files={**FILES, "libcuda.so.610.43.02": b"patched"})
    monkeypatch.setattr(cuda_compat, "docker", fake)
    with pytest.raises(cuda_compat.CompatError, match=f"sha256 differs libcuda.so.610.43.02 .*of {IMAGE}"):
        cuda_compat.extract(tmp_path / "c", rel, files, links)
    assert fake.calls[-1] == "rm"


def test_the_entry_goes_ahead_of_the_base_entrypoint_once():
    base = {"Config": {"Entrypoint": ["/opt/nvidia/nvidia_entrypoint.sh"], "Cmd": ["bash"]}}
    assert cuda_compat.entry_lines(base) == (
        'ENTRYPOINT ["/opt/tensorfold/bin/tf-compat-entry", "/opt/nvidia/nvidia_entrypoint.sh"]\nCMD ["bash"]\n')
    assert cuda_compat.entry_lines({"Config": {"Entrypoint": None, "Cmd": None}}) == (
        'ENTRYPOINT ["/opt/tensorfold/bin/tf-compat-entry"]\nCMD []\n')
    with pytest.raises(cuda_compat.CompatError, match="already runs tf-compat-entry"):
        cuda_compat.entry_lines({"Config": {"Entrypoint": [cuda_compat.ENTRY_PATH], "Cmd": []}})


def test_the_receipt_needs_every_check():
    base = {"Id": "sha256:b", "Size": 10, "Config": {"Entrypoint": None, "Cmd": ["x"]}, "RootFS": {"Layers": ["l1"]}}
    image = {"Id": "sha256:i", "Size": 15, "Config": {"Entrypoint": [cuda_compat.ENTRY_PATH], "Cmd": ["x"]},
             "RootFS": {"Layers": ["l1", "l2"]}}
    host = "tf-compat-entry: host CUDA driver (kernel module unknown; TF_CUDA_COMPAT=1)\n"
    forced = "tf-compat-entry: host CUDA driver (kernel module unknown; TF_CUDA_COMPAT=0)\n"
    done = cuda_compat.receipt(base, image, "t", host, forced, True)
    assert (done["delta_bytes"], done["cuda_compat"], done["base_id"]) == (5, "610.43.02", "sha256:b")
    for args, message in [((host, host, True), "must log the host driver"), ((host, forced, False), "arguments"),
                          (("compatibility on", forced, True), "must log the host driver")]:
        with pytest.raises(cuda_compat.CompatError, match=message):
            cuda_compat.receipt(base, image, "t", *args)
    with pytest.raises(cuda_compat.CompatError, match="base layers"):
        cuda_compat.receipt(base, {**image, "RootFS": {"Layers": ["other", "l2"]}}, "t", host, forced, True)


def test_the_build_context_files_agree():
    dockerfile = (ROOT / "docker/zig/compat.Dockerfile").read_text()
    assert "COPY compat/ /opt/tensorfold/compat/" in dockerfile
    assert f"COPY tf-compat-entry {cuda_compat.ENTRY_PATH}" in dockerfile
    assert "ENTRYPOINT" not in dockerfile and "CMD" not in dockerfile  # cuda_compat.py appends both from the base
    assert os.access(ROOT / "docker/zig/tf-compat-entry", os.X_OK)


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX sh")
def test_the_entry_logs_its_choice_and_passes_arguments_through():
    env = {**os.environ, "TF_CUDA_COMPAT": "0"}
    entry = str(ROOT / "docker/zig/tf-compat-entry")
    done = subprocess.run(["sh", entry, "sh", "-c", 'test "$1" = "a b"', "sh", "a b"], env=env, capture_output=True,
                          text=True)
    assert done.returncode == 0
    assert done.stderr.startswith("tf-compat-entry: host CUDA driver") and "TF_CUDA_COMPAT=0" in done.stderr


def test_the_launcher_forwards_the_entry_setting():
    _, settings = engine_settings.manifest()
    assert "TF_CUDA_COMPAT" not in settings  # the image's entry reads it, not the engine
    assert engine_settings.forwarded(["TF_CUDA_COMPAT"], settings) == ["TF_CUDA_COMPAT"]
    assert not engine_settings.ignored(["TF_CUDA_COMPAT"], settings)
