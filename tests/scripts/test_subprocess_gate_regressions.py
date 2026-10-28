"""Black-box gate mutations; transplant these tests alone to exercise the old gate."""
import subprocess
import sys

import pytest

import test_subprocess_boundary as audit


@pytest.mark.parametrize('source', [
    'import subprocess; spawn = subprocess.Popen; spawn(["true"])',
    'import subprocess; module = subprocess; module.Popen(["true"])',
    'import os; getattr(os, "system")("true")',
    'import os; os.posix_spawn("/bin/true", ["true"], {})',
    'import os; os.spawnlp(0, "true", "true")',
    'import os; os.execvp("true", ["true"])',
    'import pty; pty.spawn(["true"])',
    'import asyncio; loop=asyncio.get_running_loop(); loop.subprocess_shell(factory,"true")',
])
def test_actual_python_gate_rejects_mutated_source(tmp_path, monkeypatch, source):
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/probe.py').write_text(source)
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    with pytest.raises(AssertionError):
        audit.test_python_orchestrators_have_no_direct_subprocess_or_exec()


@pytest.mark.parametrize('source', [
    b'trap \'eval "$command" &\' EXIT',
    b'source /dev/stdin <<\'EOF\'\ndocker inspect x\nEOF\n',
    b'read -r -d "" command <<\'EOF\'\ndocker inspect x\nEOF\ntrap "$command" EXIT',
    b'safe_command python3 -c "import os; os.system(\'true\')"',
])
def test_actual_shell_gate_rejects_mutated_source(source):
    assert audit.bypasses(source), 'gate accepted an executable escape'


def test_assignment_alias_mutation_can_really_spawn(tmp_path, monkeypatch):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    marker = tmp_path / 'controlled-child'
    payload = 'from pathlib import Path; Path(' + repr(str(marker)) + ').touch()'
    source = 'import subprocess,sys\nlaunch=subprocess.Popen\nlaunch([sys.executable,"-c",' + repr(payload) + ']).wait()\n'
    path = scripts / 'probe.py'
    path.write_text(source)
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    try:
        audit.test_python_orchestrators_have_no_direct_subprocess_or_exec()
    except AssertionError:
        rejected = True
    else:
        rejected = False
        subprocess.run([sys.executable, str(path)], check=True, capture_output=True, timeout=10)
        assert marker.exists(), 'accepted mutation did not exercise actual child creation'
    assert rejected, 'gate admitted a mutation that spawned a real controlled child'
