"""RR1's source-PR format and new core-lint regressions are executable gates."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def ruff(*args):
    binary = os.environ.get('RUFF') or shutil.which('ruff')
    if not binary:
        pytest.fail('Ruff 0.17.3 is required for the tooling check gate')
    return subprocess.run([binary, args[0], "--no-cache", *args[1:]], cwd=ROOT, capture_output=True, text=True)


def test_folded_source_pr_format():
    result = ruff('format', '--isolated', '--check', 'bench/speed.py',
                  'scripts/preflight.py', 'tests/bench/test_speed.py',
                  'tests/scripts/test_preflight.py')
    assert result.returncode == 0, result.stdout + result.stderr


def test_reviewed_new_core_lint_findings():
    result = ruff('check', '--isolated', '--select', 'E4,E7,E9,F',
                  'scripts/guard/backend.py', 'scripts/guard/profile.py',
                  'tests/scripts/test_native_assets.py')
    assert result.returncode == 0, result.stdout + result.stderr
