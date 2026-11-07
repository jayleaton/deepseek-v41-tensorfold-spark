"""Token alignment of replies: per reply the reasoning token boundaries (char offset of each token's end).
native     the reply carries the engine's own ids (return_token_ids) and they hash to its token_sha256: boundaries
           come from decoding those ids (byte-level pieces), and the decoded reasoning must equal the receipt's
retokenized  no native ids: completion ids rebuilt from text with the model tokenizer, splitting at native cell writes
           (each engine write tokenized on its own), accepted only when sha256(','.join(ids)) equals token_sha256
--check-native decodes every hash-matched retokenized reply's ids the native way and asserts the same boundaries.
Usage: python3 -I tokalign.py TOKENIZER_JSON RUN OUT.jsonl [--glob '*/*/*.json'] [--check-native]
       (a python with `tokenizers`: rank0-host ~/venvs/exl3_v41)"""
import argparse
import codecs
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument('tokenizer')
ap.add_argument('run')
ap.add_argument('out')
ap.add_argument('--glob', default='*/*/*.json')
ap.add_argument('--check-native', action='store_true')
args = ap.parse_args()
tok = Tokenizer.from_file(args.tokenizer)
THINK, EOS = tok.token_to_id('</think>'), 1
ADDED = {t['id']: t['content'].encode() for t in json.loads(Path(args.tokenizer).read_text()).get('added_tokens', [])}


def byte_decoder():
    """GPT-2's byte-level alphabet, inverted: the printable char each byte maps to -> that byte."""
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTES = byte_decoder()


def piece(i):
    return ADDED[i] if i in ADDED else bytes(BYTES[ch] for ch in tok.id_to_token(i))


def native_ends(ids):
    """(reasoning text, its token end offsets) decoded from completion ids up to </think>, or all of them when a
    reply ended inside its reasoning (a server deadline: no </think>)."""
    dec, raw, n, ends = codecs.getincrementaldecoder('utf-8')(), b'', 0, []
    for i in ids[:reasoning_len(ids)]:
        b = piece(i)
        raw += b
        n += len(dec.decode(b))
        ends.append(n)
    return raw.decode('utf-8', errors='replace'), ends


def reasoning_len(ids):
    return ids.index(THINK) if THINK in ids else len(ids)


def sha(ids):
    return hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()


def pieces(text, cells, field):
    """Split text at this field's owned writes: [(segment, is_write)], writes located by native byte offset."""
    raw, out, pos = text.encode(), [], 0
    for c in sorted((c for c in cells if c.get('field') == field and c.get('status') == 'ok' and c.get('at') is not None
                     and c.get('value')), key=lambda c: c['at']):
        v = c['value'].encode()
        at = c['at']
        if raw[at:at + len(v)] != v or at < pos:
            return None
        out += [(raw[pos:at].decode(), False), (raw[at:at + len(v)].decode(), True)]
        pos = at + len(v)
    return out + [(raw[pos:].decode(), False)]


def encode(segs):
    ids, ends = [], []  # ends: char offset (into the field) where each token ends
    off = 0
    for s, _ in segs:
        if not s:
            continue
        e = tok.encode(s, add_special_tokens=False)
        ids += e.ids
        ends += [off + b for _, b in e.offsets]
        off += len(s)
    return ids, ends


def variants(rs, ct, cells):
    whole = [[(rs, False)], [(ct, False)]]
    split = [pieces(rs, cells, 'reasoning'), pieces(ct, cells, 'content')]
    yield 'plain', whole
    if None not in split:
        yield 'split', split


out = open(args.out, 'w')
stat = {}
for f in sorted(Path(args.run).glob(args.glob)):
    if f.name == 'arm.json':
        continue
    r = json.loads(f.read_text())
    if not isinstance(r, dict) or 'reasoning' not in r or 'id' not in r:
        continue
    rs, ct = r.get('reasoning') or '', r.get('content') or ''
    cells = ((r.get('extra') or {}).get('tensorfold') or {}).get('runner', {}).get('cells') or []
    cells = cells if isinstance(cells, list) else []
    want = ((r.get('extra') or {}).get('tensorfold') or {}).get('token_sha256')
    nat = ((r.get('extra') or {}).get('tensorfold') or {}).get('token_ids')
    best = None
    if isinstance(nat, list) and want and sha(nat) == want:
        got = native_ends(nat)
        if got and got[0] == rs:
            best = ('native', nat[:reasoning_len(nat)], got[1])
        else:
            stat['native-text-mismatch'] = stat.get('native-text-mismatch', 0) + 1
    for name, (sr, sc) in [] if best else variants(rs, ct, cells):
        ir, er = encode(sr)
        ic, _ = encode(sc)
        for tail in ([EOS], []):
            ids = ir + [THINK] + ic + tail
            if want and sha(ids) == want:
                best = (name + ('+eos' if tail else ''), ir, er)
                break
        if best:
            break
    if args.check_native and best and best[0] != 'native':
        got = native_ends(best[1] + [THINK])
        assert got and got[0] == rs and got[1] == best[2], f'native decode differs: {f}'
        stat['native-checked'] = stat.get('native-checked', 0) + 1
    label = f'{f.parent.parent.name}/{f.parent.name}'
    key = best[0] if best else ('no-sha' if not want else 'unmatched')
    stat[key] = stat.get(key, 0) + 1
    ir, er = (best[1], best[2]) if best else encode([(rs, False)])
    out.write(json.dumps({'arm': label, 'id': r['id'], 'aligned': key, 'reasoning_ids': len(ir),
                          'usage_reasoning': (r['usage'].get('completion_tokens_details') or {}).get(
                              'reasoning_tokens'),
                          'completion': r['usage'].get('completion_tokens'), 'finish': r.get('finish'),
                          'reasoning_token_ends': er}) + '\n')
print(json.dumps(stat))
