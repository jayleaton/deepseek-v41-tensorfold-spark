#!/usr/bin/env python3
"""Scores a re-check detector against a blind review (w8_rule_inputs.py's packet plus the reviewer's judgments), in
characters of reasoning. Ground truth = the reviewer's kept ranges of every flagged span plus the misses it found in
the unflagged sample; the reviewed region = those flagged spans plus the sampled unflagged replies whole.
  precision    detector characters inside the reviewed region that are ground truth
  recall       ground-truth characters the detector covers
  unreviewed   detector characters outside the reviewed region (no judgment either way)
Development data only: a detector tuned on a review is scored on it optimistically; a new cohort's own blind review
is the real measure.
Usage: python3 -B recheck_eval.py --run RUN --review-dir DIR --detector recheck.py [--detector old.py ...]
DIR holds review.jsonl, review-judgments.jsonl and review-key.json."""
import argparse
import importlib.util
import json
from pathlib import Path


def load(path, n):
    spec = importlib.util.spec_from_file_location(f'detector{n}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def chars(ranges):
    out = set()
    for a, b in ranges:
        out.update(range(a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--review-dir', required=True)
    ap.add_argument('--detector', action='append', required=True)
    a = ap.parse_args()
    run, d = Path(a.run), Path(a.review_dir)
    items = {json.loads(l)['key']: json.loads(l) for l in open(d / 'review.jsonl')}
    judged = {json.loads(l)['key']: json.loads(l) for l in open(d / 'review-judgments.jsonl')}
    where = json.loads((d / 'review-key.json').read_text())
    truth, region, sampled = {}, {}, set()  # reply path -> sets of character offsets; whole-reply samples
    for k, x in items.items():
        j = judged[k]
        if x['kind'] == 'flagged':
            path = f'{where[k]}/{x["id"]}.json'
            region.setdefault(path, set()).update(range(x['start'], x['end']))
            truth.setdefault(path, set()).update(chars((x['start'] + p, x['start'] + q) for p, q in j['keep']))
        else:
            path = where[k]
            sampled.add(path)
            region.setdefault(path, set()).update(range(len(x['text'])))
            truth.setdefault(path, set()).update(chars(j['missed']))
    replies = {}
    for f in sorted(run.glob('*/*/*.json')):
        if f.name == 'arm.json':
            continue
        r = json.loads(f.read_text())
        if isinstance(r, dict) and 'reasoning' in r:
            cells = (((r.get('extra') or {}).get('tensorfold') or {}).get('runner') or {}).get('cells')
            replies[f'{f.parent.parent.name}/{f.parent.name}/{f.name}'] = (r['reasoning'] or '',
                                                                          cells if isinstance(cells, list) else [])
    gt = sum(len(v) for v in truth.values())
    for n, path in enumerate(a.detector):
        mod = load(path, n)
        tp = fp = out_reviewed = out_other = 0
        part = {'flagged': [0, 0, 0], 'sample': [0, 0, 0]}  # tp, fp, truth
        for p, (text, cells) in replies.items():
            got = chars((s['start'], s['end']) for s in mod.spans(text, cells))
            reg, t = region.get(p, set()), truth.get(p, set())
            tp += len(got & t)
            fp += len(got & reg - t)
            if p in region:
                q = part['sample' if p in sampled else 'flagged']
                q[0] += len(got & t)
                q[1] += len(got & reg - t)
                q[2] += len(t)
            if p in region:
                out_reviewed += len(got - reg)
            else:
                out_other += len(got)
        print(json.dumps({'detector': path, 'version': getattr(mod, 'VERSION', 'v1'), 'truth_chars': gt,
                          'precision': round(tp / (tp + fp), 4) if tp + fp else None,
                          'recall': round(tp / gt, 4) if gt else None, 'tp': tp, 'fp': fp,
                          'unreviewed_in_reviewed_replies': out_reviewed, 'unreviewed_elsewhere': out_other,
                          **{f'{k}_region': {'tp': v[0], 'fp': v[1], 'truth': v[2]} for k, v in part.items()}}))


if __name__ == '__main__':
    main()
