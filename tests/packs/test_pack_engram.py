"""Engram discovery preserves release-index paths and emits actionable setup errors."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/pack_engram.py'
spec = importlib.util.spec_from_file_location('pack_engram', SCRIPT)
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


def run(src):
    return subprocess.run([sys.executable, '-I', str(SCRIPT), '--src', str(src), '--list'],
                          capture_output=True, text=True)


def test_list_without_index_has_download_instruction(tmp_path):
    r = run(tmp_path)
    assert r.returncode == 2
    assert 'download model.safetensors.index.json into --src' in r.stderr
    assert 'Traceback' not in r.stderr


def test_list_preserves_bare_and_prefixed_index_paths(tmp_path):
    for path in ('model-00047.safetensors', 'shards/model-00047.safetensors'):
        (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
            'layers.1.engram.embed.weight': path, 'layers.1.engram.embed.scale': path}}))
        r = run(tmp_path)
        assert r.returncode == 0 and r.stdout.strip() == path


def test_discovery_rejects_unsafe_index_paths(tmp_path):
    for path in ('../outside.safetensors', '/outside.safetensors'):
        (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
            'layers.1.engram.embed.weight': path}}))
        assert run(tmp_path).returncode == 2


def test_pack_and_check_both_ranks_from_raw_source(tmp_path):
    import numpy as np
    cfg = {'engram_layer_ids': [1], 'engram_max_ngram_size': 4, 'engram_n_heads': 8, 'engram_vocab_size': 3}
    primes = pack.layout(cfg)[1]
    n = sum(primes)
    weight = np.arange(n * 256, dtype=np.uint8).reshape(n, 256)
    scales = np.arange(n * 8, dtype=np.uint8).reshape(n, 8)
    meta = {}
    data = b''
    for kind, array in [('weight', weight), ('scale', scales)]:
        raw = array.tobytes()
        meta[f'layers.1.engram.embed.{kind}'] = {
            'dtype': 'U8', 'shape': list(array.shape), 'data_offsets': [len(data), len(data) + len(raw)]}
        data += raw
    header = json.dumps(meta).encode()
    (tmp_path / 'source.safetensors').write_bytes(len(header).to_bytes(8, 'little') + header + data)
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
        k: 'source.safetensors' for k in meta}}))
    output = tmp_path / 'packed'
    pack.pack(tmp_path, cfg, output, [0, 1], 2)
    assert pack.check(tmp_path, cfg, output, [0, 1], 2, 20) == 0
    path = output / 'engram-l1-r1of2.bin'
    with path.open('r+b') as f:
        f.seek(4096)
        f.write(b'broken')
    assert pack.check(tmp_path, cfg, output, [1], 2, 20) == 1
