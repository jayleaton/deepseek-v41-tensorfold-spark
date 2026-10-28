"""Publication scan includes tracked/untracked files and rejects planted private data."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def checkout(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("check-public.sh", "check_secrets.py", "check_public_history.py", "redact.py", "check-secrets-allow.txt"):
        shutil.copyfile(ROOT / "scripts" / name, scripts / name)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def scan(root, shell=None):
    return subprocess.run([shell or os.environ.get("TEST_BASH", "bash"), "scripts/check-public.sh"],
                          cwd=root, text=True, capture_output=True)


def test_tracks_spaces_and_untracked_scope(tmp_path):
    root = checkout(tmp_path)
    f = root / "file with spaces.txt"
    f.write_text("ordinary content\n")
    subprocess.run(["git", "add", f.name], cwd=root, check=True)
    assert scan(root).returncode == 0
    f.write_text("private=" + ".".join(("192", "168", "42", "3")) + "\n")
    assert scan(root).returncode == 1
    f.write_text("ordinary content\n")
    untracked = root / "new file.txt"
    untracked.write_text("private=" + ".".join(("192", "168", "42", "4")) + "\n")
    assert scan(root).returncode == 1


def test_ignored_local_excluded_but_tracked_stays_in_scope(tmp_path):
    root = checkout(tmp_path)
    f = root / "local.txt"
    f.write_text("private=" + ".".join(("192", "168", "42", "5")) + "\n")
    (root / ".gitignore").write_text("local.txt\n")
    assert scan(root).returncode == 0
    subprocess.run(["git", "add", "-f", f.name], cwd=root, check=True)
    assert scan(root).returncode == 1



def test_history_keeps_all_alias_paths_and_removed_data(tmp_path):
    root = checkout(tmp_path)
    private = '.'.join(('192', '168', '42', '6'))
    (root / 'fixture.txt').write_text(private + '\n')
    (root / 'leak.txt').write_text(private + '\n')
    subprocess.run(['git', 'add', '.'], cwd=root, check=True)
    subprocess.run(['git', '-c', 'user.name=Example', '-c', 'user.email=user@example.com',
                    'commit', '-qm', 'fixture'], cwd=root, check=True)
    result = subprocess.run(['python3', 'scripts/check_public_history.py',
                             'private|192[.]168[.]|fixture.txt'], cwd=root, text=True, capture_output=True)
    assert result.returncode == 1
    assert 'leak.txt:1:' in result.stdout and 'fixture.txt:1:' not in result.stdout


@pytest.fixture
def bash32():
    shell = os.environ.get("BASH32")
    if not shell:
        pytest.skip("set BASH32 to the real Bash 3.2 executable")
    version = subprocess.run([shell, "-c", 'printf "%s.%s" "${BASH_VERSINFO[0]}" "${BASH_VERSINFO[1]}"'],
                             text=True, capture_output=True, check=True)
    assert version.stdout == "3.2", "BASH32 must run GNU Bash 3.2"
    return shell


def test_bash32_scan_clean_and_findings(tmp_path, bash32):
    root = checkout(tmp_path)
    f = root / "file with spaces\nand newline.txt"
    f.write_text("ordinary content\n")
    result = scan(root, bash32)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "check-public: clean" in result.stdout
    f.write_text("private=" + ".".join(("192", "168", "42", "7")) + "\n")
    result = scan(root, bash32)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "private IPv4" in result.stdout and "check-public: FINDINGS" in result.stdout
    assert "unbound variable" not in result.stderr


def test_bash32_empty_scan_list(tmp_path, bash32):
    root = checkout(tmp_path)
    # This additional pattern excludes every file, exercising the empty scan array.
    (root / ".check-public.local").write_text("never-match\t*\n")
    result = scan(root, bash32)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "check-public: clean" in result.stdout


def test_bash32_empty_file_list(tmp_path, bash32):
    root = checkout(tmp_path)
    (root / ".gitignore").write_text("scripts/\n.gitignore\n")
    result = scan(root, bash32)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "no files to scan" in result.stderr
    assert "unbound variable" not in result.stderr
