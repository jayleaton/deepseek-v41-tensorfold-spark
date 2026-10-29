"""Native rank checks bind pack metadata, role, profile and exact AOT specializations."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('check_native_assets', ROOT / 'scripts/check_native_assets.py')
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


@pytest.fixture
def rank(tmp_path, monkeypatch):
    monkeypatch.setenv('TF_DSV41_CONTEXT', '4')
    model, engram, out = (tmp_path / p for p in ('model', 'engram', 'assets'))
    for p in (model, engram, out / 'aot/cubins'):
        p.mkdir(parents=True)
    config = {'engram_layer_ids': [1], 'vocab_size': 4}
    (model / 'config.json').write_text(json.dumps(config))
    (model / 'tokenizer.json').write_text('{}')
    (model / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': 'weights.safetensors'}}))
    (model / 'weights.safetensors').write_bytes(b'fixture weight bytes')
    (out / 'rope.bin').write_bytes(b'DSV41RP1' + struct.pack('<II', 2052, 64) + bytes(2 * 2052 * 64 * 4))
    def sha(name):
        return hashlib.sha256((model / name).read_bytes()).digest()
    token_map = b'DSENGR01' + struct.pack('<6I', 4, 3, 2, 4, 8, 1) + sha('tokenizer.json') + sha('config.json')
    token_map += struct.pack('<2I4Q24I24I', 1, 48, 1, 3, 5, 7, *([2] * 24), *range(0, 48, 2))
    token_map += struct.pack('<4I', 0, 1, 2, 2)
    (out / 'engram-token-map.bin').write_bytes(token_map)
    (out / 'sigs.json').write_text('{}')
    need = {'fn': 'kernel', 'consts': {'flag': True, 'optional': None, 'scale': {'fp64_bits': '0x3fe0000000000000'}},
            'runtime': {'X': {'type': '*fp32', 'div16': True}}}
    (out / 'needs.json').write_text(json.dumps({'kernels': [need]}))
    kernel = {'fn': 'kernel', 'hash': 'a' * 64, 'consts': {'flag': {'int': 1}, 'scale': {'f32': 0x3f000000}},
              'params': [{'name': 'X', 'type': '*fp32', 'div16': True}]}
    (out / 'aot/aot.json').write_text(json.dumps({'kernels': [kernel]}))
    (out / 'aot/cubins' / ('a' * 64 + '.cubin')).write_bytes(b'\x7fELFfixture cubin')
    for r in range(2):
        (engram / f'engram-l1-r{r}of2.bin').write_bytes(
            struct.pack('<6Q', 0x31344E4531565344, 1, r * 24, (r+1) * 24, 48, 264) + bytes(4096-48+24*264))
    return argparse.Namespace(rank=0, model=model, engram=engram, assets=out, context=4, revision='a'*40, image_id='sha256:fixture')


def test_pair_identity_agrees_with_correct_rank_shards(rank):
    before = assets.identity(rank)
    rank.rank = 1
    assert before == assets.identity(rank)


def test_wrong_rank_shard_is_refused(rank):
    (rank.engram / 'engram-l1-r0of2.bin').write_bytes((rank.engram / 'engram-l1-r1of2.bin').read_bytes())
    with pytest.raises(ValueError, match='wrong rank'):
        assets.identity(rank)


def test_common_profile_and_transport_change_identity(rank, monkeypatch):
    before = assets.identity(rank)
    monkeypatch.setenv('TF_DSV41_SLOTS', '4')
    changed = assets.identity(rank)
    assert before != changed
    monkeypatch.setenv('TF_TP_PORT', '29572')
    assert changed != assets.identity(rank)


RANK_LOCAL = sorted(assets.engine_settings.RANK_LOCAL)
COMMON = sorted(assets.engine_settings.TRANSPORT_COMMON)


def pair(rank, monkeypatch, first, second):
    identities = []
    for r, env in enumerate((first, second)):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        rank.rank = r
        identities.append(assets.identity(rank))
    return identities


@pytest.mark.parametrize('name', RANK_LOCAL)
def test_rank_local_transport_settings_keep_the_pair_identity(rank, monkeypatch, name):
    # RR10-1: per-node placement (TF_TP_ROCE_CPU=2 on one node, 5 on the other) must not
    # make preflight refuse the pair; the engine leaves these out of its own digest.
    first, second = pair(rank, monkeypatch, {name: '2'}, {name: '5'})
    assert first == second


@pytest.mark.parametrize('name', COMMON)
def test_common_transport_settings_change_the_pair_identity(rank, monkeypatch, name):
    first, second = pair(rank, monkeypatch, {name: '1024'}, {name: '512'})
    assert first != second


def test_proxy_cpus_differ_but_mailbox_sizes_must_match(rank, monkeypatch):
    first, second = pair(rank, monkeypatch, {'TF_TP_MAILBOX_MAX_KB': '1024', 'TF_TP_ROCE_CPU': '3'},
                         {'TF_TP_MAILBOX_MAX_KB': '1024', 'TF_TP_ROCE_CPU': '7'})
    assert first == second
    first, second = pair(rank, monkeypatch, {'TF_TP_MAILBOX_MAX_KB': '1024', 'TF_TP_ROCE_CPU': '3'},
                         {'TF_TP_MAILBOX_MAX_KB': '256', 'TF_TP_ROCE_CPU': '3'})
    assert first != second


def test_token_map_provenance_and_rope_geometry_are_checked(rank):
    (rank.model / 'tokenizer.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='token-map geometry/provenance'):
        assets.identity(rank)
    (rank.assets / 'rope.bin').write_bytes(b'DSV41RP1')
    with pytest.raises(struct.error):
        assets.identity(rank)


def test_aot_exact_specialization_and_cubin_are_required(rank):
    assets.coverage(rank.assets / 'needs.json', rank.assets / 'aot')
    path = rank.assets / 'aot/aot.json'
    obj = json.loads(path.read_text())
    obj['kernels'][0]['params'][0]['div16'] = False
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='missing variants'):
        assets.coverage(rank.assets / 'needs.json', rank.assets / 'aot')
    (rank.assets / 'aot/cubins' / ('a'*64+'.cubin')).unlink()
    with pytest.raises(FileNotFoundError):
        assets.identity(rank)


def test_empty_aot_manifest_is_refused(rank):
    (rank.assets / 'aot/aot.json').write_text('{"kernels":[]}')
    with pytest.raises(ValueError, match='manifest is empty'):
        assets.identity(rank)
