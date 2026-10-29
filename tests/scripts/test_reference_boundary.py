"""Exercise the real reference constructor and byte transport through an SSH observer."""

import json
import os
from pathlib import Path
import struct
import sys

import pytest

pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.reference.loader import RemoteSafetensorsDir  # noqa: E402


# Execute the requested read-only Python/cat operation locally. Persist only
# disclosure booleans; never retain raw SSH argv or credential-bearing output.
OBSERVER = r'''import json
import os
from pathlib import Path
import shlex
import sys

words = shlex.split(sys.argv[-1])
key = os.environ['API_KEY']
Path(os.environ['REFERENCE_OBSERVER']).write_text(json.dumps({
    'started': True,
    'raw_key': any(key in arg for arg in sys.argv[1:]),
    'decoded_key': any(key in arg for arg in words),
    'model_preserved': words[-1] == os.environ['REFERENCE_MODEL'],
}))
mode = os.environ.get('REFERENCE_RESPONSE', '')
if mode == 'failure':
    print(key)
    print(key, file=sys.stderr)
    raise SystemExit(47)
if mode == 'invalid':
    print(key)
    raise SystemExit(0)
if mode in {'extra', 'index'}:
    index = json.loads(os.environ.get('REFERENCE_INDEX', 'null'))
    print(json.dumps({'headers': {}, 'index': index, 'environment_dump': key}))
    raise SystemExit(0)
if words[:2] == ['python3', '-c']:
    os.execv(sys.executable, [sys.executable, *words[1:]])
if words[0] == 'cat':
    os.execv('/bin/cat', words)
raise SystemExit(78)
'''


@pytest.fixture
def reference(tmp_path, monkeypatch):
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    ssh = bindir / 'ssh'
    ssh.write_text(f'#!{sys.executable}\n' + OBSERVER)
    ssh.chmod(0o700)
    observer = tmp_path / 'observed.json'
    model = tmp_path / 'model'
    model.mkdir()
    monkeypatch.setenv('PATH', str(bindir) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('API_KEY', 'rr5ReferenceSyntheticKey')
    monkeypatch.setenv('REFERENCE_OBSERVER', str(observer))
    monkeypatch.setenv('REFERENCE_MODEL', str(model))
    monkeypatch.delenv('REFERENCE_RESPONSE', raising=False)
    return model, tmp_path / 'cache', observer


@pytest.mark.parametrize('shape', ['', ' ', "'", "'\""])
@pytest.mark.parametrize('setting', ['model', 'host', 'ssh_option'])
@pytest.mark.parametrize('cached', [False, True])
def test_constructor_refuses_raw_credentials_before_real_ssh(reference, monkeypatch, shape, setting, cached):
    model, cache, observer = reference
    key = 'rr5Reference' + shape + 'SyntheticCredential'
    monkeypatch.setenv('API_KEY', key)
    if cached:
        cache.mkdir()
        (cache / 'headers.json').write_text('{"index":null,"headers":{}}')
    root = str(model / key) if setting == 'model' else str(model)
    host = key if setting == 'host' else 'worker.example.invalid'
    opts = ('-o', 'ControlPath=' + key) if setting == 'ssh_option' else ()
    with pytest.raises(ValueError):
        RemoteSafetensorsDir(host, root, cache, ssh_opts=opts)
    assert not observer.exists(), 'credential-bearing SSH process started'


@pytest.mark.parametrize('suffix', ['plain', 'with spaces', "with'quote", "with'and\"quotes"])
def test_constructor_preserves_benign_paths_and_binary_stdin_reads(reference, monkeypatch, suffix):
    model, cache, observer = reference
    model = model / suffix
    model.mkdir()
    monkeypatch.setenv('REFERENCE_MODEL', str(model))
    # Include invalid UTF-8, NULs and a literal known credential: tensor bytes
    # must not pass through a diagnostic text decoder or redactor.
    payload = bytes(range(256)) + os.environ['API_KEY'].encode() + b'\0\xff'
    (model / 'tensor.bin').write_bytes(payload)
    header = json.dumps({'weight': {'dtype': 'U8', 'shape': [len(payload)],
                                    'data_offsets': [0, len(payload)]}}).encode()
    (model / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + payload)
    remote = RemoteSafetensorsDir('worker.example.invalid', str(model), cache, ssh_opts=())
    record = json.loads(observer.read_text())
    assert record == {'started': True, 'raw_key': False, 'decoded_key': False, 'model_preserved': True}
    assert remote.read_ranges('tensor.bin', [(0, 256), (256, len(payload) - 256)]) == [payload[:256], payload[256:]]
    assert remote.read_file('tensor.bin') == payload
    assert remote.info('weight').nbytes == len(payload)
    assert remote.tensor('weight').numpy().tobytes() == payload
    assert remote._cache_path('weight').read_bytes() == payload


@pytest.mark.parametrize('mode', ['failure', 'invalid'])
def test_constructor_failure_withholds_diagnostics_and_metadata_cache(reference, monkeypatch, capsys, mode):
    model, cache, observer = reference
    monkeypatch.setenv('REFERENCE_RESPONSE', mode)
    expected = 'reference ssh failed (exit 47)' if mode == 'failure' else 'invalid reference metadata'
    with pytest.raises(RuntimeError) as raised:
        RemoteSafetensorsDir('worker.example.invalid', str(model), cache, ssh_opts=())
    assert str(raised.value) == expected
    assert observer.exists()
    assert not (cache / 'headers.json').exists()
    output = capsys.readouterr()
    assert not output.out and not output.err


@pytest.mark.parametrize('operation', ['read_ranges', 'read_file'])
def test_later_file_paths_are_checked_before_ssh(reference, operation):
    model, cache, observer = reference
    remote = RemoteSafetensorsDir('worker.example.invalid', str(model), cache, ssh_opts=())
    observer.unlink()
    filename = os.environ['API_KEY'] + '.bin'
    with pytest.raises(ValueError):
        if operation == 'read_ranges':
            remote.read_ranges(filename, [(0, 1)])
        else:
            remote.read_file(filename)
    assert not observer.exists()


@pytest.mark.parametrize('index', [None, {}, {'weight': 'model.safetensors'}])
def test_metadata_cache_discards_extra_response_fields(reference, monkeypatch, index):
    model, cache, observer = reference
    monkeypatch.setenv('REFERENCE_RESPONSE', 'extra')
    monkeypatch.setenv('REFERENCE_INDEX', json.dumps(index))
    remote = RemoteSafetensorsDir('worker.example.invalid', str(model), cache, ssh_opts=())
    assert observer.exists()
    cached = (cache / 'headers.json').read_text()
    assert json.loads(cached) == {'headers': {}, 'index': index}
    assert os.environ['API_KEY'] not in cached
    assert remote.weight_map == (index or {})


@pytest.mark.parametrize('index', [[], ['weight'], 'diagnostic', 1, False, {'weight': 1}, {'weight': None}])
def test_metadata_rejects_non_string_index_mappings(reference, monkeypatch, index):
    model, cache, observer = reference
    monkeypatch.setenv('REFERENCE_RESPONSE', 'index')
    monkeypatch.setenv('REFERENCE_INDEX', json.dumps(index))
    with pytest.raises(RuntimeError, match='^invalid reference metadata$'):
        RemoteSafetensorsDir('worker.example.invalid', str(model), cache, ssh_opts=())
    assert observer.exists()
    assert not (cache / 'headers.json').exists()
