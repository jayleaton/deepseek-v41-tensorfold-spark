"""Inspect the real launcher's Docker boundary without starting a container."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def launcher(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    docker = bindir / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1] == 'image':\n"
        "    print(os.environ['LAUNCHER_TEST_REV'] if 'org.tensorfold.revision' in sys.argv[4] else 'sha256:fixture')\n"
        "else:\n"
        "    record = {'argv': sys.argv[1:], 'forwarded': {name: os.environ.get(name) "
        "for name in ('TF_EXPERIMENTAL', 'TF_TP_ROCE_HCA')}}\n"
        "    Path(os.environ['LAUNCHER_TEST_ARGV']).write_text(json.dumps(record))\n"
        "    sys.exit(44)\n"
    )
    docker.chmod(0o700)
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "sigs.json").write_text("{}")
    values = {
        "IMAGE": "fixture", "TF_TP_MASTER": "rank0.example", "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_IB_HCA": "mlx5_0", "TF_TP_PORT": "29571", "TF_COMM_BACKEND": "roce",
        "MODEL_DIR": tmp_path / "model", "ENGRAM_DIR": tmp_path / "engram",
        "ASSETS_DIR": assets, "PREPARED_DIR": tmp_path / "prepared", "STATE_DIR": tmp_path / "state",
        "CACHE_DIR": tmp_path / "cache",
    }
    config = tmp_path / "config.env"
    config.write_text("".join(f"{key}={shlex.quote(str(value))}\n" for key, value in values.items()))
    argv_file = tmp_path / "argv.json"
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"], CONFIG=str(config),
               LAUNCHER_TEST_REV=(ROOT / "config/tensorfold.rev").read_text().strip(),
               LAUNCHER_TEST_ARGV=str(argv_file), TF_TP_ROCE_HCA="fixture_hca")
    return env, argv_file, values


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("contradictory_rank", [False, True])
def test_check_aot_effective_rank(launcher, rank, contradictory_rank):
    env, argv_file, values = launcher
    if contradictory_rank:
        env["TF_TP_RANK"] = str(1 - rank)
    else:
        env.pop("TF_TP_RANK", None)
    result = subprocess.run(["bash", "scripts/run-zig.sh", str(rank), "--check-aot"],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 44, result.stdout + result.stderr
    record = json.loads(argv_file.read_text())
    argv = record["argv"]
    effective = {}
    for i, arg in enumerate(argv):
        if arg == "-e":
            name, sep, value = argv[i + 1].partition("=")
            effective[name] = value if sep else record["forwarded"].get(name, env.get(name, str(values.get(name, ""))))
    assert effective["TF_TP_RANK"] == str(rank)
    assert effective["TF_TP_WORLD"] == "2" and effective["TF_TP_DEVICE"] == "0"
    assert effective["TF_EXPERIMENTAL"] == "deepseek"
    assert effective["TF_DSV41_ASSETS"] == "/assets"
    assert effective["TF_DSV41_ENGRAM_DIR"] == "/engram"
    assert effective["TF_TP_MASTER"] == "rank0.example"
    assert effective["TF_TP_PORT"] == "29571"
    assert effective["TF_COMM_BACKEND"] == "roce"
    assert effective["TF_TP_KERNEL_IMAGE"] == "/opt/tensorfold/share/tp_mailbox.fatbin"
    assert effective["TF_TP_ROCE_HCA"] == "fixture_hca"
    assert "TF_TP_ROCE_HCA" in argv and "fixture_hca" not in argv
    assert "--runtime=runc" in argv and "--gpus" not in argv
    assert argv[-3:] == ["aot-needs", "/assets/sigs.json", f"/state/aot-needs-r{rank}.json"]


@pytest.mark.parametrize("mode", ["--check-aot", "--dry-run"])
def test_launcher_excludes_credential_names(launcher, mode):
    env, argv_file, _ = launcher
    credential_names = ("TF_DSV41_API_KEY", "TF_DSV41_AUTH_TOKEN", "TF_CUDA_PASSWORD",
                        "GLM53_TF_SECRET", "NCCL_CREDENTIAL")
    secret = "rr1" + "LauncherCredentialOnly"
    env.update(dict.fromkeys(credential_names, secret))
    env.update(TF_DSV41_POOL_TOKENS="2100000", TF_DSV41_MAX_TOKENS="32768")
    result = subprocess.run(["bash", "scripts/run-zig.sh", "1", mode],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == (44 if mode == "--check-aot" else 0)
    printed = result.stdout + result.stderr
    if mode == "--check-aot":
        argv = json.loads(argv_file.read_text())["argv"]
        exposed_names = any(name in argv for name in credential_names)
        exposed_value = any(secret in arg for arg in argv)
    else:
        argv = shlex.split(result.stdout)
        exposed_names = any(name in printed for name in credential_names)
        exposed_value = secret in printed
    # Boolean assertions avoid echoing the synthetic credential in pytest diagnostics.
    assert not exposed_names, "credential-named variables reached the Docker boundary"
    printed_value = secret in printed
    assert not exposed_value and not printed_value, "credential value reached launcher output"
    for name in ("TF_DSV41_POOL_TOKENS", "TF_DSV41_MAX_TOKENS", "TF_TP_ROCE_HCA"):
        assert name in argv
        assert argv[argv.index(name) - 1] == "-e"


def test_credential_in_a_native_control_is_refused_without_dropping_the_setting(launcher):
    env, argv_file, _ = launcher
    secret = "rr1" + "NativeControlCredential"
    env.update(API_KEY=secret, TF_DSV41_NATIVE_CONTROL=secret)
    result = subprocess.run(["bash", "scripts/run-zig.sh", "1", "--dry-run"],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 2
    leaked = secret in result.stdout + result.stderr
    assert not leaked, "credential reached native-environment refusal"
    assert not argv_file.exists()
