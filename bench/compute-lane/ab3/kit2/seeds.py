"""Seeded arms (`X:sN`, directory X-sN) pooled per boot and arm: mean, spread and per-seed values, paired by seed."""
import re
import statistics

SEED = re.compile(r'^(.*)-s(\d+)$')


def split(label):
    """'boot/rt-s2' -> ('boot/rt', 2); an unseeded label -> (label, None)."""
    m = SEED.match(label)
    return (m.group(1), int(m.group(2))) if m else (label, None)


def spread(xs):
    return {'mean': round(statistics.mean(xs), 1), 'sd': round(statistics.stdev(xs), 1) if len(xs) > 1 else None,
            'min': min(xs), 'max': max(xs), 'n': len(xs)}


def pooled(report, detail):
    """Per seeded arm: solved (raw, sign-aware) and reason+content tokens over its seeds, with and without mul30-14."""
    groups = {}
    for label in report['arms']:
        name, seed = split(label)
        if seed is not None:
            groups.setdefault(name, []).append((seed, label))
    out = {}
    for name, runs in sorted(groups.items()):
        per = {}
        for seed, label in sorted(runs):
            rows = [r for r in detail if r['arm'] == label]
            tok = [r['completion_tokens'] or 0 for r in rows]
            per[seed] = {'solved': sum(r['solved'] for r in rows), 'solved_signed': sum(r['solved_signed'] for r in rows),
                         'tokens': sum(tok), 'tokens_no_mul30_14': sum(t for r, t in zip(rows, tok) if r['id'] != 'mul30-14'),
                         'n': len(rows)}
        out[name] = {'seeds': per, **{k: spread([v[k] for v in per.values()])
                                      for k in ('solved', 'solved_signed', 'tokens', 'tokens_no_mul30_14')}}
    return out


def markdown(pool):
    if not pool:
        return []
    out = ['', '| seeded arm | seeds | sign-aware solved mean (min..max) | tokens mean ± sd (min..max) | '
           'without mul30-14 mean ± sd | per seed tokens |', '|---|---|---|---|---|---|']
    for name, p in pool.items():
        s, t, u = p['solved_signed'], p['tokens'], p['tokens_no_mul30_14']
        sd = lambda x: '' if x['sd'] is None else f' ± {x["sd"]:.0f}'  # noqa: E731
        out.append(f'| {name} | {s["n"]} | {s["mean"]} ({s["min"]}..{s["max"]}) | {t["mean"]:.0f}{sd(t)} ({t["min"]}..{t["max"]}) | '
                   f'{u["mean"]:.0f}{sd(u)} | ' + ', '.join(f's{k} {v["tokens"]}' for k, v in p['seeds'].items()) + ' |')
    return out
