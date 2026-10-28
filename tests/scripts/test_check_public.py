"""Publication scan includes tracked/untracked files and rejects planted private data."""
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def checkout(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("check-public.sh", "check_secrets.py", "check_public_history.py", "redact.py", "check-secrets-allow.txt"):
        shutil.copyfile(ROOT / "scripts" / name, scripts / name)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def scan(root):
    return subprocess.run([os.environ.get("TEST_BASH", "bash"), "scripts/check-public.sh"],
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
