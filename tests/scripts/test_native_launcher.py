"""Inspect the real launcher's Docker boundary without starting a container."""

import json
import os
from pathlib import Path
import re
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
        "    with open(os.environ['LAUNCHER_TEST_ARGV'] + '.inspect', 'a') as log:\n"
        "        log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "    if 'org.tensorfold.revision' in sys.argv[4]:\n"
        "        print(os.environ['LAUNCHER_TEST_REV'])\n"
        "    elif sys.argv[4].startswith('{\"config\":'):\n"
        "        print(json.dumps({'config': {'Env': ['PATH=/usr/bin'], 'Cmd': ['bash']}, 'layers': ['sha256:' + '1' * 64,\n"
        "                          'sha256:' + '2' * 64], 'os': 'linux', 'architecture': 'arm64', 'variant': ''}))\n"
        "    else:\n"
        "        print('sha256:fixture')\n"
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
    env.update(TF_DSV41_POOL_TOKENS="2100000", TF_TP_MAILBOX_MAX_KB="1024")
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
    for name in ("TF_DSV41_POOL_TOKENS", "TF_TP_MAILBOX_MAX_KB", "TF_TP_ROCE_HCA"):
        assert name in argv
        assert argv[argv.index(name) - 1] == "-e"


def test_credential_in_a_native_control_is_refused_without_dropping_the_setting(launcher):
    env, argv_file, _ = launcher
    secret = "rr1" + "NativeControlCredential"
    env.update(API_KEY=secret, TF_DSV41_ROWS_CAP=secret)
    result = subprocess.run(["bash", "scripts/run-zig.sh", "1", "--dry-run"],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 78
    leaked = secret in result.stdout + result.stderr
    assert not leaked, "credential reached native-environment refusal"
    assert not argv_file.exists()


@pytest.mark.parametrize("rank", [0, 1])
def test_auth_file_is_mounted_only_for_head_serving(launcher, tmp_path, rank):
    env, _, _ = launcher
    key_file = tmp_path / "keys with spaces"
    secret = "rr6" + "FileOnlyServingCredential"
    key_file.write_text(secret + "\n")
    key_file.chmod(0o600)
    env["NATIVE_AUTH_FILE"] = str(key_file)
    result = subprocess.run(["bash", "scripts/run-zig.sh", str(rank), "--dry-run"],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    argv = shlex.split(result.stdout)
    mount = str(key_file) + ":/run/tensorfold/auth-keys:ro"
    assert (mount in argv) == (rank == 0)
    assert ("--api-key-file" in argv) == (rank == 0)
    if rank == 0:
        assert argv[argv.index("--api-key-file") + 1] == "/run/tensorfold/auth-keys"
    leaked = secret in result.stdout + result.stderr
    assert not leaked, "file credential reached launcher output"
    assert "NATIVE_AUTH_FILE" not in argv


@pytest.mark.parametrize("rank", [0, 1])
def test_auth_file_is_not_mounted_for_aot_diagnostics(launcher, tmp_path, rank):
    env, argv_file, _ = launcher
    key_file = tmp_path / "keys"
    # Diagnostics do not even validate/open the serving-only key file.
    env["NATIVE_AUTH_FILE"] = str(key_file)
    result = subprocess.run(["bash", "scripts/run-zig.sh", str(rank), "--check-aot"],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 44, result.stdout + result.stderr
    argv = json.loads(argv_file.read_text())["argv"]
    assert not any("auth-keys" in value or str(key_file) in value for value in argv)
    assert "--api-key-file" not in argv


@pytest.mark.parametrize("kind", ["missing", "empty", "public", "symlink", "directory", "relative"])
def test_head_refuses_unsafe_auth_file_before_docker(launcher, tmp_path, kind):
    env, argv_file, _ = launcher
    key_file = tmp_path / "keys"
    if kind == "directory":
        key_file.mkdir(mode=0o600)
    elif kind != "missing":
        key_file.write_text("" if kind == "empty" else "rr6FileCredential\n")
        key_file.chmod(0o644 if kind == "public" else 0o600)
    if kind == "symlink":
        link = tmp_path / "link"
        link.symlink_to(key_file)
        key_file = link
    env["NATIVE_AUTH_FILE"] = "keys" if kind == "relative" else str(key_file)
    result = subprocess.run(["bash", "scripts/run-zig.sh", "0", "--dry-run"],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "docker run" not in result.stdout
    assert not argv_file.exists()


@pytest.mark.parametrize("name", ["api-key", "secret", "auth-keys"])
def test_head_names_a_credential_like_auth_file_name_before_docker(launcher, tmp_path, name):
    # The mount argument "<path>:/run/...:ro" reads as "name: value" to the argument check for these names.
    env, argv_file, _ = launcher
    key_file = tmp_path / name
    key_file.write_text("rr6FileCredential\n")
    key_file.chmod(0o600)
    env["NATIVE_AUTH_FILE"] = str(key_file)
    result = subprocess.run(["bash", "scripts/run-zig.sh", "0", "--dry-run"],
                            cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "reads as a credential label" in result.stderr
    assert "rr6FileCredential" not in result.stdout + result.stderr
    assert not argv_file.exists()


@pytest.mark.parametrize("shared", ["state", "cache", "model", "checkout"])
def test_head_refuses_auth_file_inside_a_shared_directory(launcher, tmp_path, shared):
    # Every other container mounts these directories, and diagnostics read them.
    env, argv_file, _ = launcher
    parent = ROOT / "tests" if shared == "checkout" else tmp_path / shared
    if shared == "checkout" and not os.access(parent, os.W_OK):
        pytest.skip("the checkout case needs a writable checkout")
    parent.mkdir(parents=True, exist_ok=True)
    key_file = parent / f"rr8-auth-keys-{tmp_path.name}"
    try:
        key_file.write_text("rr8SharedDirectoryCredential\n")
        key_file.chmod(0o600)
        env["NATIVE_AUTH_FILE"] = str(key_file)
        result = subprocess.run(["bash", "scripts/run-zig.sh", "0", "--dry-run"],
                                cwd=ROOT, env=env, text=True, capture_output=True)
    finally:
        key_file.unlink()
    assert result.returncode == 2
    assert "outside the checkout" in result.stderr
    assert "docker run" not in result.stdout
    assert not argv_file.exists()


@pytest.mark.skipif(os.geteuid() != 0, reason="creating a root-owned file needs root")
def test_head_accepts_a_root_owned_auth_file(launcher, tmp_path):
    env, _, _ = launcher
    key_file = tmp_path / "root-keys"
    key_file.write_text("rr8RootOwnedCredential\n")
    key_file.chmod(0o600)
    assert key_file.stat().st_uid == 0
    env["NATIVE_AUTH_FILE"] = str(key_file)
    result = subprocess.run(["bash", "scripts/run-zig.sh", "0", "--dry-run"],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert str(key_file) + ":/run/tensorfold/auth-keys:ro" in shlex.split(result.stdout)


@pytest.mark.parametrize("source", ["config", "caller"])
@pytest.mark.parametrize("entry", ["rank", "pair-wrapper"])
def test_native_auth_environment_requires_file_migration(launcher, source, entry):
    env, argv_file, _ = launcher
    secret = "rr6" + "NativeEnvironmentCredential"
    if source == "config":
        with Path(env["CONFIG"]).open("a") as stream:
            stream.write("TENSORFOLD_API_KEY=" + secret + "\n")
    else:
        env["TENSORFOLD_API_KEY"] = secret
    argv = (["bash", "scripts/run-zig.sh", "0", "--dry-run"] if entry == "rank" else
            ["bash", "scripts/serve.sh", "rank", "0", "--dry-run"])
    result = subprocess.run(argv,
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "NATIVE_AUTH_FILE" in result.stderr
    leaked = secret in result.stdout + result.stderr
    assert not leaked, "legacy authentication value reached migration diagnostic"
    assert not argv_file.exists()


def test_caller_native_key_with_config_file_uses_file_transport(launcher, tmp_path):
    env, _, _ = launcher
    secret = "rr6" + "CallerNativeCredential"
    env["TENSORFOLD_API_KEY"] = secret
    key_file = tmp_path / "keys"
    key_file.write_text(secret + "\n")
    key_file.chmod(0o600)
    with Path(env["CONFIG"]).open("a") as stream:
        stream.write("NATIVE_AUTH_FILE=" + str(key_file) + "\n")
    result = subprocess.run(["bash", "scripts/serve.sh", "rank", "0", "--dry-run"],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0
    argv = shlex.split(result.stdout)
    assert "--api-key-file" in argv
    assert "TENSORFOLD_API_KEY" not in argv
    leaked = secret in result.stdout + result.stderr
    assert not leaked, "caller credential reached file-auth launch output"


def test_explicit_runner_environment_keeps_registry_for_nested_validation(launcher):
    sys.path.insert(0, str(ROOT / "scripts"))
    from safe_exec import run as checked_run
    env, marker, _ = launcher
    value = "rr6" + "ExplicitEnvironmentCredentialOnly"
    env["API_KEY"] = value
    with Path(env["CONFIG"]).open("a") as config:
        config.write("HOST=" + value + "\n")
    result = checked_run(["bash", "scripts/run-zig.sh", "0", "--dry-run"],
                         cwd=ROOT, env=env, timeout=20)
    code = result.returncode
    refused = "credential value refused" in result.stderr
    spawned = marker.exists()
    leaked = value in result.stdout + result.stderr
    assert not leaked, "explicit registry value reached output"
    assert code == 78
    assert refused
    assert not spawned


sys.path.insert(0, str(ROOT / "scripts"))
import engine_settings  # noqa: E402
from redact import secret_name  # noqa: E402


def forwarded_names(argv):
    return [argv[i + 1].partition("=")[0] for i, arg in enumerate(argv) if arg == "-e"]


@pytest.mark.parametrize("rank", [0, 1])
def test_launcher_forwards_every_setting_the_pinned_engine_reads(launcher, rank):
    # The arm E defect: profiles set GLM53_TF_ROCE_* / TF_DSV41_PLAN_*, the engine read
    # TF_TP_*, and the launcher forwarded no TF_TP_* at all, so the RoCE fast path, the
    # 1 MiB mailbox and the RDMA plan link silently fell back to engine defaults.
    env, _, _ = launcher
    _, settings = engine_settings.manifest()
    expected = sorted(n for n in settings if n not in engine_settings.LAUNCHER_OWNED
                      and n not in engine_settings.RECIPE_NAMES and not secret_name(n))
    assert {"TF_TP_ROCE_FAST", "TF_TP_MAILBOX_MAX_KB", "TF_TP_PLAN_LINK", "TF_DRAIN_S",
            "TENSORFOLD_REQUEST_LOG"} <= set(expected)
    with Path(env["CONFIG"]).open("a") as stream:
        stream.write("".join(f"{name}=1\n" for name in expected))
        stream.write("TENSORFOLD_BENCHMARK_TOKEN=rr10BenchmarkTokenOnly\n")
    result = subprocess.run(["bash", "scripts/run-zig.sh", str(rank), "--dry-run"],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    names = forwarded_names(shlex.split(result.stdout))
    missing = sorted(set(expected) - set(names))
    assert not missing, f"engine settings not forwarded: {missing}"
    for owned in ("TF_TP_RANK", "TF_TP_WORLD", "TF_TP_MASTER", "TF_TP_PORT", "TF_COMM_BACKEND"):
        assert names.count(owned) == 1
    assert "TENSORFOLD_BENCHMARK_TOKEN" not in names
    assert "rr10BenchmarkTokenOnly" not in result.stdout + result.stderr


@pytest.mark.parametrize("legacy", sorted(engine_settings.LEGACY))
def test_launcher_refuses_a_setting_the_pinned_engine_no_longer_reads(launcher, legacy):
    env, argv_file, _ = launcher
    value = "rr10LegacyValue"
    with Path(env["CONFIG"]).open("a") as stream:
        stream.write(f"{legacy}={value}\n")
    result = subprocess.run(["bash", "scripts/run-zig.sh", "0", "--dry-run"],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert legacy in result.stderr
    replacement = engine_settings.LEGACY[legacy]
    assert (replacement or "remove it") in result.stderr
    assert value not in result.stdout + result.stderr
    assert "docker run" not in result.stdout and not argv_file.exists()


def test_launcher_warns_about_an_engine_name_the_pin_does_not_read(launcher):
    env, _, _ = launcher
    with Path(env["CONFIG"]).open("a") as stream:
        stream.write("TF_DSV41_ROWS_CAPP=64\nTF_TP_ROCE_FAST=1\n")
    result = subprocess.run(["bash", "scripts/run-zig.sh", "1", "--dry-run"],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "TF_DSV41_ROWS_CAPP is not read by TensorFold" in result.stderr
    names = forwarded_names(shlex.split(result.stdout))
    assert "TF_DSV41_ROWS_CAPP" not in names and "TF_TP_ROCE_FAST" in names


def test_effective_profile_prints_settings_and_only_credential_names(launcher):
    env, argv_file, _ = launcher
    secret = "rr10" + "EffectiveProfileCredential"
    with Path(env["CONFIG"]).open("a") as stream:
        stream.write(f"TF_TP_MAILBOX_MAX_KB=1024\nTF_TP_PLAN_LINK=rdma\nHF_TOKEN={secret}\n"
                     "TF_DSV41_DEFAULT_EFFORT=high\nTF_DSV41_ROWS_CAPP=64\n")
    result = subprocess.run(["bash", "scripts/run-zig.sh", "0", "--effective-profile"],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "TF_TP_MAILBOX_MAX_KB=1024  [profile]" in lines
    assert "TF_TP_PLAN_LINK=rdma  [profile]" in lines
    assert "TF_TP_RANK=0  [launcher]" in lines
    assert "HF_TOKEN" in lines and "TF_DSV41_ROWS_CAPP" in lines
    leaked = secret in result.stdout + result.stderr
    assert not leaked, "credential value reached the effective profile"
    assert not argv_file.exists()


@pytest.mark.parametrize("effort, code", [("high", 0), ("extreme", 2)])
def test_default_effort_becomes_the_head_flag(launcher, effort, code):
    env, _, _ = launcher
    with Path(env["CONFIG"]).open("a") as stream:
        stream.write(f"TF_DSV41_DEFAULT_EFFORT={effort}\n")
    result = subprocess.run(["bash", "scripts/run-zig.sh", "0", "--dry-run"],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == code
    if code == 0:
        argv = shlex.split(result.stdout)
        assert argv[argv.index("--reasoning-effort") + 1] == "high"
        assert "TF_DSV41_DEFAULT_EFFORT" not in forwarded_names(argv)


def test_pair_identity_uses_the_image_content_not_the_store_specific_id(launcher):
    # A loaded image keeps its config and layers on every node, but its .Id is the manifest digest under Docker's
    # containerd store and the config digest under overlay2, so identical images on a mixed pair would refuse each other.
    env, argv_file, values = launcher
    for key in ("MODEL_DIR", "ENGRAM_DIR", "CACHE_DIR"):
        Path(values[key]).mkdir()
    result = subprocess.run(["bash", "scripts/run-zig.sh", "0", "--identity"], cwd=ROOT, env=env, text=True,
                            capture_output=True, check=False)
    # The fixture has no pack, so the asset check is the first refusal after the image's identity was captured.
    assert "control parse failed" not in result.stderr
    assert result.stderr.startswith("native asset check:"), result.stderr
    formats = [call[call.index("-f") + 1] for call in map(json.loads, Path(str(argv_file) + ".inspect").read_text().splitlines())]
    identity = [f for f in formats if f.startswith('{"config":')]
    assert identity and all(part in identity[0] for part in ("{{json .Config}}", "{{json .RootFS.Layers}}", "{{json .Architecture}}"))
    assert "{{.Id}}" not in formats, formats


LAYERS = ["sha256:" + "1" * 64, "sha256:" + "2" * 64]


def image(config, layers=LAYERS, **platform):
    return json.dumps({"config": config, "layers": layers, "os": "linux", "architecture": "arm64", "variant": "",
                       **platform})


def identity(raw):
    sys.path.insert(0, str(ROOT / "scripts"))
    from watchdog_io import filter_output
    return filter_output("image_identity", raw + "\n")


RUNTIME = {"Env": ["PATH=/usr/bin", "TF_IMAGE_DEFAULT=1"], "Cmd": ["bash"], "WorkingDir": "/opt/tensorfold",
           "Labels": {"org.tensorfold.revision": "a" * 40}}


def test_image_identity_agrees_across_the_two_store_representations():
    # overlay2 renders the legacy container fields and unset values as null or empty; the containerd store omits them.
    legacy = {"Hostname": "", "Domainname": "", "User": "", "AttachStdin": False, "Tty": False, "Image": "sha256:" + "9" * 64,
              "Entrypoint": None, "Volumes": None, "OnBuild": None, "ExposedPorts": {}, **RUNTIME}
    assert identity(image(legacy)) == identity(image(RUNTIME, variant=None))
    assert re.fullmatch(r"sha256:[0-9a-f]{64}\n", identity(image(RUNTIME)))


@pytest.mark.parametrize("change", [
    {"Env": RUNTIME["Env"] + ["TF_DSV41_BAKED=1"]},
    {"Entrypoint": ["/bin/sh", "-c"]},
    {"User": "nobody"},
    {"WorkingDir": "/"},
])
def test_image_identity_disagrees_on_equal_layers_with_a_different_runtime_config(change):
    assert identity(image({**RUNTIME, **change})) != identity(image(RUNTIME))


def test_image_identity_disagrees_on_layer_order_and_platform():
    assert identity(image(RUNTIME, layers=LAYERS[::-1])) != identity(image(RUNTIME))
    assert identity(image(RUNTIME, architecture="amd64")) != identity(image(RUNTIME))


def test_image_identity_prints_only_the_digest():
    env_value = "invented-image-env-value-7f3a"
    out = identity(image({**RUNTIME, "Env": ["TF_PROBE=" + env_value]}))
    assert env_value not in out and re.fullmatch(r"sha256:[0-9a-f]{64}\n", out)


@pytest.mark.parametrize("raw", ["", "sha256:fixture", "[]", image(RUNTIME, layers=[]), image(RUNTIME, layers=["sha256:fixture"]),
                                 image({"Env": "PATH=/usr/bin"}), image({"Entrypoint": [1]}), json.dumps({"layers": LAYERS})])
def test_image_identity_control_refuses_anything_else(raw):
    with pytest.raises(ValueError):
        identity(raw)
