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
    for name in ("check-public.sh", "check_secrets.py", "check_public_history.py", "redact.py", "check-secrets-allow.txt",
                 "safe_exec.py", "safe_output.py", "safe_quote.py", "watchdog_io.py"):
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


def test_local_agent_tooling_is_ignored_and_refused_if_forced(tmp_path):
    root = checkout(tmp_path)
    shutil.copyfile(ROOT / ".gitignore", root / ".gitignore")
    skill = root / ".claude" / "skills" / "verify" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("ordinary content\n")
    assert scan(root).returncode == 0
    subprocess.run(["git", "add", "-f", str(skill.relative_to(root))], cwd=root, check=True)
    result = scan(root)
    assert result.returncode == 1 and ".claude/skills/verify/SKILL.md" in result.stdout



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


def test_patch_series_internal_names_fail_in_strings_too(tmp_path):
    # The line patches/0074 carried before the fix: the test host's name inside a report string, not a comment.
    root = checkout(tmp_path)
    (root / "patches").mkdir()
    patch = root / "patches" / "0001-fixture.patch"
    host = "node" + "01"
    patch.write_text(f'+    "timing_scope": "{host} nice19; CPU clock excludes QUIET pauses"\n')
    result = scan(root)
    assert result.returncode == 1 and "internal names in the engine patch series" in result.stdout, result.stdout
    patch.write_text('+    "timing_scope": "test host, nice 19; CPU clock excludes QUIET pauses"\n')
    assert scan(root).returncode == 0
    # Outside patches/ the series rule does not apply (docs may describe a two-node pair in their own words).
    patch.unlink()
    (root / "notes.txt").write_text(f"rank 1 runs on {host}\n")
    assert scan(root).returncode == 0


def test_history_fails_on_a_removed_pre_scrub_patch(tmp_path):
    # A patch committed with the test host's name and scrubbed in a later commit: the current tree is clean, but the
    # history still carries the old blob, and --history must name it with the series rule (case-sensitive).
    root = checkout(tmp_path)
    author = "user@example.com"
    env = {**os.environ, "ALLOWED_AUTHOR": author}

    def commit(message):
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "-c", "user.name=Example", "-c", f"user.email={author}", "commit", "-qm", message],
                       cwd=root, check=True)

    def history():
        return subprocess.run([os.environ.get("TEST_BASH", "bash"), "scripts/check-public.sh", "--history"],
                              cwd=root, text=True, capture_output=True, env=env, check=False)

    (root / "patches").mkdir()
    patch = root / "patches" / "0001-fixture.patch"
    patch.write_text('+    "timing_scope": "test host, nice 19"\n+# Node one and node two are prose, not names\n')
    commit("clean series")
    result = history()
    assert result.returncode == 0, result.stdout
    host = "node" + "01"
    patch.write_text(f'+    "timing_scope": "{host} nice19"\n')
    commit("pre-scrub patch")
    patch.write_text('+    "timing_scope": "test host, nice 19"\n')
    commit("scrub")
    assert scan(root).returncode == 0
    result = history()
    assert result.returncode == 1, result.stdout
    assert "== history: internal names in the engine patch series" in result.stdout
    assert "patches/0001-fixture.patch:1:" in result.stdout and host in result.stdout


def test_bench_host_names_fail_but_benchmark_words_pass(tmp_path):
    # bench/ carries study evidence: a development host's name fails there; GSM8K's names do not.
    root = checkout(tmp_path)
    note = root / "bench" / "study" / "README.md"
    note.parent.mkdir(parents=True)
    note.write_text('Cody eats three times as many cookies; run it on the operator workstation.\n')
    assert scan(root).returncode == 0
    for host in ("node" + "01", "Node" + "02", "dev" + "-box", "dgx" + "02"):
        note.write_text(f"never run this on {host}\n")
        result = scan(root)
        assert result.returncode == 1 and "development host names in bench/" in result.stdout, (host, result.stdout)


JAY = "58495321+jayleaton@users.noreply.github.com"


def history_repo(tmp_path):
    root = checkout(tmp_path)
    env = {k: v for k, v in os.environ.items() if k not in ("ALLOWED_AUTHOR", "ALLOWED_AUTHORS")}

    def commit(message, email=JAY):
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "-c", "user.name=Jay Leaton", "-c", f"user.email={email}", "commit", "-qm", message],
                       cwd=root, check=True, env=env)
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True,
                              capture_output=True).stdout.strip()

    def history(*rng):
        return subprocess.run([os.environ.get("TEST_BASH", "bash"), "scripts/check-public.sh", "--history", *rng],
                              cwd=root, text=True, capture_output=True, env=env, check=False)

    return root, commit, history


def test_history_range_scans_only_the_outgoing_commits_and_their_bench_blobs(tmp_path):
    # Jay Leaton's noreply address is a public identity by default; a bench blob that named a host fails history
    # wherever it is reachable, and a range limits the scan to the commits a branch would publish.
    root, commit, history = history_repo(tmp_path)
    note = root / "bench" / "README.md"
    note.parent.mkdir()
    note.write_text("run it on " + "node" + "01\n")
    base = commit("inherited")
    note.write_text("run it on the serving node\n")
    commit("scrub")
    result = history()
    assert result.returncode == 1 and "development host names in bench/" in result.stdout, result.stdout
    assert f"{base}:bench/README.md:1:" in result.stdout
    result = history(f"{base}..HEAD")
    assert result.returncode == 0, result.stdout
    assert "not authored / committed" not in result.stdout
    # an outgoing commit that adds a host name and a later one that rewords it: the range still fails
    note.write_text("mirror on " + "dev" + "-box\n")
    commit("leak")
    note.write_text("mirror on the operator workstation\n")
    commit("reword")
    assert scan(root).returncode == 0
    result = history(f"{base}..HEAD")
    assert result.returncode == 1 and "development host names in bench/" in result.stdout, result.stdout


def test_history_range_refuses_identities_outside_the_allowlist(tmp_path):
    root, commit, history = history_repo(tmp_path)
    (root / "a.txt").write_text("one\n")
    base = commit("base")
    (root / "a.txt").write_text("two\n")
    commit("other identity", email="someone@example.com")
    result = history(f"{base}..HEAD")
    assert result.returncode == 1 and "someone@example.com" in result.stdout, result.stdout
    assert history("--bogus").returncode == 2
