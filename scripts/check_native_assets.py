#!/usr/bin/env python3
"""Check native rank prerequisites and compare common pack/image/profile identities without loading a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))
from safe_output import cli, safe_print as print  # noqa: E402
import engine_settings  # noqa: E402


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def specialization(kernel, compiled):
    if compiled:
        consts = kernel['consts']
        params = [(p['name'], p['type'], p.get('div16', False)) for p in kernel['params']]
    else:
        consts = {}
        for key, value in kernel['consts'].items():
            if isinstance(value, (int, bool)):
                consts[key] = {'int': int(value)}
            elif isinstance(value, dict) and 'fp64_bits' in value:
                number = struct.unpack('<d', int(value['fp64_bits'], 16).to_bytes(8, 'little'))[0]
                consts[key] = {'f32': struct.unpack('<I', struct.pack('<f', number))[0]}
            elif value is not None:
                raise ValueError('unsupported AOT constant')
        params = [(key, value['type'], value.get('div16', False)) for key, value in kernel['runtime'].items()]
    return json.dumps([kernel['fn'], consts, sorted(params)], sort_keys=True)


def coverage(needs, aot):
    kernels = json.loads((aot / 'aot.json').read_text())['kernels']
    if not kernels:
        raise ValueError('AOT manifest is empty')
    for kernel in kernels:
        key = kernel['hash']
        if not re.fullmatch('[A-Za-z0-9_-]+', key):
            raise ValueError('unsafe AOT cubin key')
        with (aot / 'cubins' / (key + '.cubin')).open('rb') as stream:
            if stream.read(4) != b'\x7fELF':
                raise ValueError('missing/invalid ELF cubin: ' + key)
    if needs:
        wanted = json.loads(needs.read_text())['kernels']
        have = {specialization(k, True) for k in kernels}
        missing = [k['fn'] for k in wanted if specialization(k, False) not in have]
        if not wanted or missing:
            raise ValueError(f'AOT plan empty or missing variants: {sorted(set(missing))}')


def identity(args):
    files = {name: sha(args.model / name) for name in ('config.json', 'tokenizer.json', 'model.safetensors.index.json')}
    cfg = json.loads((args.model / 'config.json').read_text())
    cfg = {**cfg, **(cfg.get('text_config') or {})}
    with (args.assets / 'rope.bin').open('rb') as stream:
        magic, rows, cols = struct.unpack('<8sII', stream.read(16))
    if magic != b'DSV41RP1' or cols != 64 or rows < args.context + 2048:
        raise ValueError('rope.bin must hold context + 2048 rows of 64 fp32 columns')
    if (args.assets / 'rope.bin').stat().st_size != 16 + 2 * rows * cols * 4:
        raise ValueError('rope.bin has an incorrect size')
    data = (args.assets / 'engram-token-map.bin').read_bytes()
    if data[:8] != b'DSENGR01' or len(data) < 96:
        raise ValueError('missing/invalid native Engram token map')
    vocab, compressed, pad, order, heads, layers = struct.unpack_from('<6I', data, 8)
    if (order != 4 or heads != 8 or not 0 <= pad < compressed <= vocab or vocab != cfg['vocab_size']
            or layers != len(cfg['engram_layer_ids']) or len(data) != 96 + layers * 232 + vocab * 4
            or data[32:64].hex() != files['tokenizer.json'] or data[64:96].hex() != files['config.json']):
        raise ValueError('Engram token-map geometry/provenance differs from this pack')
    for i, layer in enumerate(cfg['engram_layer_ids']):
        got_layer, total = struct.unpack_from('<2I', data, 96 + i * 232)
        primes = struct.unpack_from('<24I', data, 136 + i * 232)
        lo, hi = sum(primes[:args.rank * 12]), sum(primes[:(args.rank + 1) * 12])
        shard = args.engram / f'engram-l{layer}-r{args.rank}of2.bin'
        with shard.open('rb') as stream:
            header = struct.unpack('<6Q', stream.read(48))
        if (got_layer != layer or total != sum(primes) or header != (0x31344E4531565344, layer, lo, hi, total, 264)
                or shard.stat().st_size != 4096 + (hi - lo) * 264):
            raise ValueError('wrong rank or invalid Engram shard: ' + shard.name)
    coverage(None, args.assets / 'aot')
    for name in ('rope.bin', 'engram-token-map.bin', 'sigs.json', 'aot/aot.json'):
        files['assets/' + name] = sha(args.assets / name)
    # The forwarded engine settings both ranks must share; rank-local ones are left out.
    _, settings = engine_settings.manifest()
    profile = {k: os.environ[k] for k in engine_settings.forwarded(os.environ, settings)
               if not k.startswith('NCCL_') and k not in engine_settings.RANK_LOCAL}
    transport = {key: os.environ.get(key, default) for key, default in
                 [('TF_COMM_BACKEND', 'roce'), ('TF_TP_MASTER', ''), ('TF_TP_PORT', '29571')]}
    value = {'engine': args.revision, 'image': args.image_id, 'files': files,
             'profile': profile, 'transport': transport}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--needs', type=Path)
    parser.add_argument('--aot', type=Path)
    parser.add_argument('--rank', type=int, choices=(0, 1))
    for name in ('model', 'engram', 'assets'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--context', type=int)
    parser.add_argument('--revision')
    parser.add_argument('--image-id')
    args = parser.parse_args()
    try:
        if args.needs and args.aot:
            coverage(args.needs, args.aot)
        elif any(x is None for x in (args.rank, args.model, args.engram, args.assets, args.context, args.revision, args.image_id)):
            parser.error('provide needs/aot, or all native rank identity inputs')
        else:
            print(identity(args))
        return 0
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        print(f'native asset check: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(cli(main))
