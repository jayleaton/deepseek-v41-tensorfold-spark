#!/usr/bin/env python3
"""Inputs T5-W8-DECISION.md needs beyond w8_gate: the primary token change at the rule's 20,000 paired hierarchical
draws (tasks, then seeds within a task; RNG 20261103), censoring per arm and seed with paired transitions, cap
sensitivity (complete-in-both pairs, and the share of the difference from censored pairs), the completed-request wall
ratio, and the re-check share (recheck.py spans counted in token-aligned reasoning ids, the same draws).
ALIGN is tokalign's jsonl: per reply the reasoning token end offsets and whether the ids reproduce token_sha256.
Usage: python3 -B w8_rule_inputs.py --run RUN --manifest M.json --align ALIGN.jsonl --out DIR [--draws 20000]
Writes DIR/rule-inputs.json, DIR/spans.jsonl (every flagged span with its text) and DIR/review.jsonl (blind packet:
every flagged span plus the first 10 unflagged owned replies per arm by task id and seed, arms hidden)."""
import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recheck  # noqa: E402

RNG_SEED, OUTLIER, CENSORED = 20261103, 'mul30-14', ('timeout', 'deadline', 'length', 'cut')


def pct(xs, q):
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1))] if xs else None


def ratio(rows, key):
    off = sum(r['off'][key] for r in rows)
    return sum(r['on'][key] for r in rows) / off if off else None


def boot(rows, draws, fns):
    """Every fn over the same draws: sample tasks with replacement, then that task's pairs with replacement."""
    by = {}
    for r in rows:
        by.setdefault(r['id'], []).append(r)
    keys, rng, out = sorted(by), random.Random(RNG_SEED), {k: [] for k in fns}
    for _ in range(draws):
        sample = [x for k in rng.choices(keys, k=len(keys)) for x in rng.choices(by[k], k=len(by[k]))]
        for name, fn in fns.items():
            out[name].append(fn(sample))
    return {k: [round(pct([v for v in vs if v is not None], 0.025), 4),
                round(pct([v for v in vs if v is not None], 0.975), 4)] for k, vs in out.items()}


def change(rows):
    r = ratio(rows, 'tokens')
    return None if r is None else 100 * (r - 1)


def median_task(rows):
    by = {}
    for r in rows:
        by.setdefault(r['id'], []).append(r['on']['tokens'] - r['off']['tokens'])
    return statistics.median(statistics.mean(v) for v in by.values()) if by else None


def share(rows, side, owned_only=False, key='recheck'):
    use = [r[side] for r in rows if not owned_only or r[side]['owned']]
    tot = sum(x['reasoning_ids'] for x in use)
    return 100 * sum(x[key] for x in use) / tot if tot else None


def share_delta(rows, key='recheck'):
    a, b = share(rows, 'on', key=key), share(rows, 'off', key=key)
    return None if a is None or b is None else a - b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--align', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--draws', type=int, default=20000)
    ap.add_argument('--review', help='review-judgments.jsonl: kept ranges replace the flagged spans')
    ap.add_argument('--outlier-ids', help="JSON list of task ids for the outlier sensitivity (W9: T5's predeclared "
                                          'mul30 lineage); default W8\'s mul30-14')
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads(Path(a.manifest).read_text())
    align = {(x['arm'], x['id']): x for x in map(json.loads, open(a.align))}
    review = {x['key']: x for x in map(json.loads, open(a.review))} if a.review else None
    rows, spans_out = [], []
    for p in man['pairs']:
        row = {'id': p['id'], 'seed': p['seed']}
        for side in ('off', 'on'):
            rec = json.loads((run / p[side] / f'{p["id"]}.json').read_text())
            cells = (((rec.get('extra') or {}).get('tensorfold') or {}).get('runner') or {}).get('cells')
            cells = cells if isinstance(cells, list) else []
            al = align[(p[side], p['id'])]
            sp = recheck.spans(rec.get('reasoning') or '', cells)
            n = recheck.tokens_in(sp, al['reasoning_token_ends'])
            kept = []  # the reviewer's ranges, absolute in the reasoning
            for x in sp if review is not None else []:
                j = review[hashlib.sha256(f'{p[side]}/{p["id"]}/{x["start"]}'.encode()).hexdigest()[:12]]
                kept += [{'start': x['start'] + b0, 'end': x['start'] + b1} for b0, b1 in j['keep']]
            missed = None
            if review is not None:
                j = review.get(hashlib.sha256(f'{p[side]}/{p["id"]}/unflagged'.encode()).hexdigest()[:12])
                missed = None if j is None else recheck.tokens_in(
                    [{'start': b0, 'end': b1} for b0, b1 in j['missed']], al['reasoning_token_ends'])
            text = rec.get('reasoning') or ''
            for s in sp:
                spans_out.append({'arm': p[side], 'side': side, 'id': p['id'], 'seed': p['seed'], **s,
                                  'tokens': recheck.tokens_in([s], al['reasoning_token_ends']),
                                  'before': text[max(0, s['start'] - 400):s['start']],
                                  'text': text[s['start']:s['end']],
                                  'after': text[s['end']:s['end'] + 200]})
            row[side] = {'reasoning': text, 'writes': [(w['start'], w['end'], w['display']) for w in
                                                         recheck.landed(text, cells)],
                         'label': p[side], 'finish': rec.get('finish'), 'elapsed_s': rec.get('elapsed_s'),
                         'tokens': (rec.get('usage') or {}).get('completion_tokens'),
                         'tokens_counted': bool((rec.get('usage') or {}).get('counted')),
                         'owned': any(c.get('status') == 'ok' for c in cells), 'aligned': al['aligned'],
                         'reasoning_ids': al['reasoning_ids'], 'recheck': n, 'spans': len(sp),
                         'reviewed': recheck.tokens_in(kept, al['reasoning_token_ends']), 'missed_in_sample': missed}
        rows.append(row)
    res = {'pairs': len(rows), 'draws': a.draws, 'rng_seed': RNG_SEED, 'manifest_sha256':
           hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()}
    # primary token change, its CI and the rule's companion views
    outliers = set(json.loads(Path(a.outlier_ids).read_text())) if a.outlier_ids else {OUTLIER}
    seeds = sorted({r['seed'] for r in rows})
    no_out = [r for r in rows if r['id'] not in outliers]
    ci = boot(rows, a.draws, {'change': change, 'median_task': median_task, 'recheck_delta_pp': share_delta,
                              'reviewed_delta_pp': lambda rs: share_delta(rs, 'reviewed')})
    ci_no = boot(no_out, a.draws, {'change': change})
    res['tokens'] = {
        'off': sum(r['off']['tokens'] for r in rows), 'on': sum(r['on']['tokens'] for r in rows),
        'change_pct': round(change(rows), 3), 'change_ci95': ci['change'],
        'without_outlier': {'change_pct': round(change(no_out), 3), 'change_ci95': ci_no['change'],
                            'removed_tasks': sorted(outliers & {r['id'] for r in rows}),
                            'pairs': len(no_out)},
        'per_seed': {s: round(change([r for r in rows if r['seed'] == s]), 3) for s in seeds},
        'median_task_change': round(median_task(rows), 3), 'median_task_change_ci95': ci['median_task'],
        'counted_not_native': sorted(f'{r[s]["label"]}/{r["id"]}' for r in rows for s in ('off', 'on')
                                     if r[s]['tokens_counted'])}
    # per family (same draws within the family) and leave-one-task-out (a secondary diagnostic, not a gate)
    fam = {}
    for r in rows:
        fam.setdefault(r['id'].split('-')[-2], []).append(r)
    res['tokens']['per_family'] = {f: {'pairs': len(rs), 'change_pct': round(change(rs), 3),
                                       'change_ci95': boot(rs, a.draws, {'change': change})['change']}
                                   for f, rs in sorted(fam.items())}
    by_task = {}
    for r in rows:
        by_task.setdefault(r['id'], []).append(r)
    loto = sorted((round(change([r for r in rows if r['id'] != t]), 3), t) for t in by_task)
    res['tokens']['leave_one_task_out'] = {'min': loto[0], 'max': loto[-1], 'tasks': len(loto)}
    # censoring
    cens = {}
    for side in ('off', 'on'):
        for s in ('all', *seeds):
            use = [r[side] for r in rows if s == 'all' or r['seed'] == s]
            fin = {}
            for x in use:
                fin[x['finish']] = fin.get(x['finish'], 0) + 1
            cens[f'{side} seed {s}'] = {'n': len(use), 'finish': fin,
                                        'cap_or_timeout': sum(x['finish'] in CENSORED for x in use)}
    trans = {}
    for r in rows:
        k = f'{r["off"]["finish"]} -> {r["on"]["finish"]}'
        trans[k] = trans.get(k, 0) + 1
    res['censoring'] = {'by_arm_seed': cens, 'paired_transitions_off_to_on': trans,
                        'censored_pairs': sorted(
                            f'{r["id"]} s{r["seed"]} ({r["off"]["finish"]}/{r["on"]["finish"]})'
                            for r in rows if {r['off']['finish'], r['on']['finish']} & set(CENSORED))}
    # cap sensitivity
    both = [r for r in rows if r['off']['finish'] not in CENSORED and r['on']['finish'] not in CENSORED]
    cen = [r for r in rows if r not in both]
    diff_all = sum(r['on']['tokens'] - r['off']['tokens'] for r in rows)
    diff_cen = sum(r['on']['tokens'] - r['off']['tokens'] for r in cen)
    res['cap_sensitivity'] = {
        'complete_in_both': {'pairs': len(both), 'tasks': len({r['id'] for r in both}),
                             'off': sum(r['off']['tokens'] for r in both), 'on': sum(r['on']['tokens'] for r in both),
                             'change_pct': round(change(both), 3),
                             'change_ci95': boot(both, a.draws, {'change': change})['change']},
        'censored_pairs': {'pairs': len(cen), 'token_difference': diff_cen},
        'primary_token_difference': diff_all,
        'share_of_primary_difference_from_censored_pct': round(100 * diff_cen / diff_all, 2) if diff_all else None,
        'note': 'removing every pair cut in either arm leaves exactly the complete-in-both set'}
    # wall time of completed requests (stop), timeouts kept apart; PROFILE boot is outside the manifest
    wall = {}
    for name, use in (('completed in each arm', None), ('complete in both pairs', both)):
        sel = {side: [r[side] for r in (use or rows) if r[side]['finish'] not in CENSORED] for side in ('off', 'on')}
        w = {side: sum(x['elapsed_s'] for x in xs) for side, xs in sel.items()}
        wall[name] = {'off_s': round(w['off'], 1), 'on_s': round(w['on'], 1), 'n_off': len(sel['off']),
                      'n_on': len(sel['on']), 'ratio_on_off': round(w['on'] / w['off'], 4)}
    wall['censored requests (kept apart)'] = {side: [(r['id'], r['seed'], r[side]['finish'], r[side]['elapsed_s'])
                                                     for r in rows if r[side]['finish'] in CENSORED]
                                              for side in ('off', 'on')}
    res['wall'] = wall
    # re-check share
    rc = {}
    for side in ('off', 'on'):
        owned = [r for r in rows if r[side]['owned']]
        rc[side] = {'reasoning_ids': sum(r[side]['reasoning_ids'] for r in rows),
                    'recheck_tokens': sum(r[side]['recheck'] for r in rows),
                    'share_pct': round(share(rows, side), 3),
                    'owned_replies': len(owned),
                    'owned_reasoning_ids': sum(r[side]['reasoning_ids'] for r in owned),
                    'owned_recheck_tokens': sum(r[side]['recheck'] for r in owned),
                    'owned_share_pct': round(share(rows, side, True), 3),
                    'replies_with_spans': sum(r[side]['spans'] > 0 for r in rows),
                    'spans': sum(r[side]['spans'] for r in rows)}
    al_count = {}
    for r in rows:
        for side in ('off', 'on'):
            al_count[r[side]['aligned']] = al_count.get(r[side]['aligned'], 0) + 1
    res['recheck'] = {'by_arm': rc, 'delta_pp': round(share_delta(rows), 3), 'delta_pp_ci95': ci['recheck_delta_pp'],
                      'token_alignment': al_count, 'detector': 'recheck.py (sha256 ' + hashlib.sha256(
                          Path(recheck.__file__).read_bytes()).hexdigest() + ')',
                      'status': 'detector output before independent review'}
    if review is not None:
        rv = {}
        for side in ('off', 'on'):
            owned = [r for r in rows if r[side]['owned']]
            sample = [r[side] for r in rows if r[side]['missed_in_sample'] is not None]
            rv[side] = {'recheck_tokens': sum(r[side]['reviewed'] for r in rows),
                        'share_pct': round(share(rows, side, key='reviewed'), 3),
                        'owned_recheck_tokens': sum(r[side]['reviewed'] for r in owned),
                        'owned_share_pct': round(share(rows, side, True, 'reviewed'), 3),
                        'unflagged_sample': {'replies': len(sample),
                                             'with_missed': sum(x['missed_in_sample'] > 0 for x in sample),
                                             'missed_tokens': sum(x['missed_in_sample'] for x in sample),
                                             'reasoning_ids': sum(x['reasoning_ids'] for x in sample)}}
        res['recheck']['reviewed'] = {
            'by_arm': rv, 'delta_pp': round(share_delta(rows, 'reviewed'), 3),
            'delta_pp_ci95': ci['reviewed_delta_pp'], 'judgments_sha256': hashlib.sha256(
                Path(a.review).read_bytes()).hexdigest(),
            'note': 'flagged spans cut to the blind reviewer\'s kept ranges; misses in the unflagged sample are '
                    'reported, not extrapolated'}
        res['recheck']['status'] = 'detector output, then the independent blind review applied'
    (out / 'rule-inputs.json').write_text(json.dumps(res, indent=1) + '\n')
    with (out / 'spans.jsonl').open('w') as f:
        for s in spans_out:
            f.write(json.dumps(s) + '\n')
    # blind review packet: arm and side replaced by an opaque key; unflagged owned replies, first 10 per arm by id/seed
    review = []
    for s in spans_out:
        key = hashlib.sha256(f'{s["arm"]}/{s["id"]}/{s["start"]}'.encode()).hexdigest()[:12]
        review.append({'key': key, 'kind': 'flagged', 'id': s['id'], 'start': s['start'], 'end': s['end'],
                       'tokens': s['tokens'], 'landed': s['display'], 'before': s['before'], 'text': s['text'],
                       'after': s['after']})
    for side in ('off', 'on'):
        pick = sorted(((r['id'], r['seed']) for r in rows if r[side]['owned'] and not r[side]['spans']))[:10]
        for tid, seed in pick:
            x = next(r[side] for r in rows if r['id'] == tid and r['seed'] == seed)
            key = hashlib.sha256(f'{x["label"]}/{tid}/unflagged'.encode()).hexdigest()[:12]
            review.append({'key': key, 'kind': 'unflagged', 'id': tid, 'writes': x['writes'], 'text': x['reasoning'],
                           'path': f'{x["label"]}/{tid}.json'})
    review.sort(key=lambda x: x['key'])
    arm_of = {hashlib.sha256(f'{s["arm"]}/{s["id"]}/{s["start"]}'.encode()).hexdigest()[:12]: s['arm']
              for s in spans_out}
    (out / 'review-key.json').write_text(json.dumps({x['key']: arm_of.get(x['key'], x.get('path'))
                                                     for x in review}, indent=1) + '\n')
    with (out / 'review.jsonl').open('w') as f:
        for x in review:
            f.write(json.dumps({k: v for k, v in x.items() if k != 'path'}) + '\n')
    print(json.dumps({k: res[k] for k in ('tokens', 'recheck')}, indent=1))


if __name__ == '__main__':
    main()
