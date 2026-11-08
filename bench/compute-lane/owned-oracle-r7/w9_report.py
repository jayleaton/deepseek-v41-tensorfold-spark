#!/usr/bin/env python3
"""W9 owned-oracle report (T5-W9-SPEC.md, rule fields from T5-W8-DECISION.md): one verdict per 45-task block, one pooled.
  w9_report.py --run RUN --blocks DIR --out OUT [--oracle r2|r5|r6|r7] [--only 1,2]
DIR holds T4's w9-block-NN-manifest.json (w9_kit.py: pairs with their off/on arm labels). A block's receipts are
RUN/run-NN/<label>/<id>.json, or RUN/<label>/<id>.json when there is no run-NN. Per block the pinned oracle's gate.sh
judges exactly the manifest's receipts (OUT/block-NN/oracle/gate.json is the file w8_gate.py --oracle reads); this
script adds, per arm, what the rule asks of engine writes beyond the oracle's verdicts:
  engine_bytes      rule 1 / BLOCK: wrong landed bytes (any FAIL verdict, partial writes included), unknown, unplaced
  cut_gate          rule 5: cut owned records <= 1% of owned records (never rounded), each at a native end (finish
                    length/cap/deadline in a final receipt); whitespace-only cuts are counted and verify nothing
  owned writes      runner counters per form above that form's records: an owned write with no record (inconclusive)
  provenance        records: final receipt vs the stream's live set (05aa3debe): every live record must be its final
                    record's prefix (same cell, field, offset, span; value and landed never ahead); a reply holding only
                    live records (a client close) has unauthenticated coverage: inconclusive
  imitations        model-written `[exact` labels with no trace record, per reply and per emitted label (values are
                    not checked here)
Verdict: FAIL (BLOCK / investigate) on a wrong engine write, a text-mode owned mismatch, an unreconciled live record or
a cut at no native end; INCONCLUSIVE on missing replies, live-only replies, unknown/inconclusive records or an oracle
that did not run; else PASS. The cut share is a recommendation condition (keep opt-in), reported beside the verdict.
Exit 0 PASS, 1 FAIL, 3 INCONCLUSIVE, 2 error. Node CPU only (2 GiB scope, nice 19); pauses under QUIET via gate.sh's caller."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
# T5 agreed to r2's exact bytes. r5 (T5's r3 OBJECT fixed, deadline cuts) is OBJECTed (divmod cuts), r6 too (span/display); r7
# fixes both and moves in only when T5 AGREEs: set agreed.
ORACLES = {
    'r2': {'dir': HOME / 'work/owned-oracle/tool', 'agreed': True,
           'sha256': '5aeed889ade60282ce3ddc34e2b24e038ddf893a9abfce39715d748f8ce11751'},
    'r5': {'dir': HOME / 'work/owned-oracle/tool-r5', 'agreed': False,
           'sha256': '82a9e4ec894cfa43c42680c01b0e64ea05be17e18bf0733b78caf4a1b7ca05b3'},
    'r6': {'dir': HOME / 'work/owned-oracle/tool-r6', 'agreed': False,
           'sha256': '327e35f007ceb2f8e544675636b2050987cda7df678ae110975c51ce07fac4f5'},
    'r7': {'dir': HOME / 'work/owned-oracle/tool-r7', 'agreed': False,
           'sha256': '35f2bdf0243df7917b2b61098a1ff4a91bf1044854634912ab4800495db853fc'},
}
FAIL = ('wrong-value', 'wrong-span', 'not-in-text', 'unparsed', 'wrong-check')
OPEN = ('unknown', 'inconclusive', 'short')
NATIVE_END = ('length', 'cap', 'deadline')
CUT_LIMIT = 0.01
EXIT = {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 3}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def worst(vs):
    return 'FAIL' if 'FAIL' in vs else 'INCONCLUSIVE' if 'INCONCLUSIVE' in vs else 'PASS'


def runner_of(d):
    """The receipt's final runner block (measure receipts: extra.tensorfold; OpenAI form: response.tensorfold)."""
    ex = d.get('extra') if isinstance(d.get('extra'), dict) else {}
    tf = d.get('tensorfold') or ex.get('tensorfold') or {}
    if isinstance(d.get('response'), dict):
        tf = d['response'].get('tensorfold') or tf
    return tf.get('runner') if isinstance(tf, dict) and isinstance(tf.get('runner'), dict) else {}


def finish_of(d):
    if isinstance(d.get('response'), dict) and d['response'].get('choices'):
        return d['response']['choices'][0].get('finish_reason')
    return d.get('finish')


def reconcile(d):
    """(state, problems): state final | live-only | none; problems: live records their final record contradicts."""
    run = runner_of(d)
    cells = run.get('cells') if isinstance(run.get('cells'), list) else None
    if cells is None:
        return 'none', []
    if run.get('live') is True:
        return 'live-only', []
    live = d.get('live_cells')
    bad = []
    if isinstance(live, list):
        final = {(c.get('cell'), c.get('field'), c.get('at')): c for c in cells if isinstance(c, dict)}
        for c in live:
            f = final.get((c.get('cell'), c.get('field'), c.get('at')))
            v, fv = str(c.get('value', '')).encode(), str((f or {}).get('value', '')).encode()
            if f is None:
                bad.append(f"live cell {c.get('cell')} at {c.get('at')}: no final record")
            elif c.get('span') != f.get('span') or c.get('form') != f.get('form') or not fv.startswith(v) \
                    or (c.get('landed') or 0) > (f.get('landed') or 0):
                bad.append(f"live cell {c.get('cell')} at {c.get('at')}: {c.get('value')!r} is not a prefix of final "
                           f"{f.get('value')!r} (span {c.get('span')!r} / {f.get('span')!r})")
    return 'final', bad


def imitations(d, cells):
    """Model-written `[exact` labels outside every trace record, and all emitted labels, over both fields."""
    n = labels = 0
    for field in ('reasoning', 'content'):
        t = d.get(field) or ''
        if isinstance(d.get('response'), dict) and d['response'].get('choices'):
            m = d['response']['choices'][0].get('message') or {}
            t = m.get('reasoning_content' if field == 'reasoning' else 'content') or ''
        tb = t.encode()
        spans = [(c.get('at') or 0, (c.get('at') or 0) + len(str(c.get('value', '')).encode()))
                 for c in cells if c.get('form') == 'trace' and c.get('field') == field]
        for m in re.finditer(rb'\[exact', tb):
            labels += 1
            n += not any(a <= m.start() < b for a, b in spans)
    return n, labels


def blank():
    return {'replies': 0, 'replies_expected': 0, 'replies_missing': 0, 'replies_with_records': 0,
            'owned_records': 0, 'ok': 0, **{k: 0 for k in FAIL + OPEN}, 'cut': 0, 'cut_whitespace_only': 0,
            'cut_not_native_end': 0, 'receipts_final': 0, 'receipts_live_only': 0, 'receipts_live_reconciled': 0,
            'live_contradictions': 0, 'records_short': 0, 'replies_with_imitation': 0, 'imitation_labels': 0, 'labels_emitted': 0,
            'finish': {}}


def add(a, b):
    for k, v in b.items():
        if k == 'finish':
            for f, n in v.items():
                a['finish'][f] = a['finish'].get(f, 0) + n
        elif isinstance(v, int):
            a[k] += v
    return a


def rule_fields(arms, oracle_gate, text_mismatches, problems, oracle_ok, missing):
    """T5-W8-DECISION.md's engine-write fields from per-arm counts, and the verdict."""
    tot = add(add(blank(), arms['off']), arms['on'])
    wrong = sum(tot[k] for k in FAIL)
    cut_gate = {}
    for side, x in arms.items():
        share = x['cut'] / x['owned_records'] if x['owned_records'] else 0.0
        cut_gate[side] = {'owned_records': x['owned_records'], 'cut_records': x['cut'], 'cut_share': share,
                          'cut_share_le_1pct': share <= CUT_LIMIT, 'cuts_at_native_end': x['cut_not_native_end'] == 0,
                          'whitespace_only_cuts': x['cut_whitespace_only']}
    fail = []
    if wrong:
        fail.append(f'wrong engine writes {wrong} (' + ', '.join(f'{k} {tot[k]}' for k in FAIL if tot[k]) + ')')
    if text_mismatches:
        fail.append(f'text-mode owned mismatches {text_mismatches}')
    if tot['live_contradictions']:
        fail.append(f'unreconciled live records {tot["live_contradictions"]}')
    if tot['cut_not_native_end']:
        fail.append(f'cuts at no native end {tot["cut_not_native_end"]}')
    open_ = []
    if not oracle_ok:
        open_.append('oracle did not run')
    if missing:
        open_.append(f'replies missing {missing}')
    if tot['records_short']:
        open_.append(f'owned writes without a record {tot["records_short"]} (runner counters above records, per form)')
    if tot['receipts_live_only']:
        open_.append(f'live-only replies {tot["receipts_live_only"]} (client close: coverage unauthenticated)')
    for k in OPEN:
        if tot[k]:
            open_.append(f'{k} records {tot[k]}')
    v = 'FAIL' if fail else 'INCONCLUSIVE' if open_ else 'PASS'
    return {
        'verdict': v,
        'decision': {'FAIL': 'BLOCK promotion / investigate', 'INCONCLUSIVE': 'inconclusive (missing evidence)',
                     'PASS': 'engine writes pass'}[v]
        + ('' if v == 'FAIL' or all(c['cut_share_le_1pct'] for c in cut_gate.values()) else '; cut gate fails: keep opt-in'),
        'why': fail + open_,
        'engine_bytes': {'wrong_landed_bytes': wrong, **{k: tot[k] for k in FAIL}, 'unknown': tot['unknown'],
                         'unplaced': tot['not-in-text'], 'inconclusive': tot['inconclusive'], 'short': tot['short'],
                         'complete_verified_values': tot['ok'], 'cut': tot['cut'],
                         'text_owned_mismatches': text_mismatches, 'owned_without_record': tot['records_short'],
                         'all_landed_bytes_authenticated_and_correct': not wrong and not text_mismatches
                         and not tot['records_short']
                         and not tot['unknown'] and not tot['inconclusive'] and not tot['receipts_live_only']},
        'cut_gate': cut_gate,
        'provenance': {'receipts_final': tot['receipts_final'], 'receipts_live_only': tot['receipts_live_only'],
                       'live_sets_reconciled': tot['receipts_live_reconciled'],
                       'live_contradictions': tot['live_contradictions'], 'problems': problems[:50]},
        'imitations': {s: {'replies': x['replies'], 'replies_with_imitation': x['replies_with_imitation'],
                           'per_reply': x['replies_with_imitation'] / x['replies'] if x['replies'] else None,
                           'labels_emitted': x['labels_emitted'], 'imitation_labels': x['imitation_labels'],
                           'per_label': x['imitation_labels'] / x['labels_emitted'] if x['labels_emitted'] else None}
                       for s, x in arms.items()},
        'arms': arms,
        'oracle': oracle_gate,
    }


def block(n, man, run, out, tool):
    root = run / f'run-{n:02d}' if (run / f'run-{n:02d}').is_dir() else run
    stage, od = out / f'block-{n:02d}' / 'in', out / f'block-{n:02d}' / 'oracle'
    side_of, problems, missing = {}, [], 0
    arms = {'off': blank(), 'on': blank()}
    for p in man['pairs']:
        for side in ('off', 'on'):
            arms[side]['replies_expected'] += 1
            src = root / p[side] / f"{p['id']}.json"
            dst = stage / p[side] / f"{p['id']}.json"
            if not src.is_file():
                missing += 1
                arms[side]['replies_missing'] += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.is_symlink():
                dst.symlink_to(src.resolve())
            side_of[str(dst)] = side
    for path, side in side_of.items():
        a, d = arms[side], json.load(open(path))
        a['replies'] += 1
        f = finish_of(d)
        a['finish'][str(f)] = a['finish'].get(str(f), 0) + 1
        state, bad = reconcile(d)
        run = runner_of(d)
        cells = [c for c in (run.get('cells') or []) if isinstance(c, dict)]
        for form in ('inline', 'trace', 'fence', 'repl'):  # every owned write needs its record (w8_gate's check)
            a['records_short'] += max(0, ((run.get(form) or {}).get('owned') or 0) - sum(c.get('form') == form for c in cells))
        if state != 'none':
            a['replies_with_records'] += 1
        if state == 'final':
            a['receipts_final'] += 1
            a['receipts_live_reconciled'] += isinstance(d.get('live_cells'), list) and not bad
        elif state == 'live-only':
            a['receipts_live_only'] += 1
        a['live_contradictions'] += len(bad)
        problems += [f'{Path(path).relative_to(stage)}: {b}' for b in bad]
        im, labels = imitations(d, cells)
        a['replies_with_imitation'] += im > 0
        a['imitation_labels'] += im
        a['labels_emitted'] += labels
    od.mkdir(parents=True, exist_ok=True)
    log = od.parent / 'gate.log'
    with open(log, 'w') as fh:
        rc = subprocess.run(['bash', str(tool / 'gate.sh'), str(od), str(stage)], stdout=fh, stderr=subprocess.STDOUT).returncode
    gate_path = od / 'gate.json'
    oracle_ok = rc in (0, 1, 3) and gate_path.exists()
    gate = json.load(open(gate_path)) if oracle_ok else None
    text = (gate or {}).get('text_owned_mismatches', 0)
    if oracle_ok:
        for line in open(od / 'evals.jsonl'):
            c = json.loads(line)
            if c.get('source') != 'record':
                continue
            for path in dict.fromkeys(c.get('paths') or []):
                side = side_of.get(path)
                if side is None:
                    continue
                a = arms[side]
                a['owned_records'] += 1
                v = c.get('verdict')
                if v in a and v != 'finish':
                    a[v] += 1
                if v == 'cut':
                    a['cut_whitespace_only'] += not str(c.get('written', '')).strip()
                    a['cut_not_native_end'] += finish_of(json.load(open(path))) not in NATIVE_END
    res = rule_fields(arms, gate, text, problems, oracle_ok, missing)
    res.update(block=n, oracle_rc=rc, receipts=str(root), gate_json=str(gate_path))
    return res


def markdown(rep):
    L = [f"# W9 owned-oracle report: pooled **{rep['pooled']['verdict']}** ({rep['pooled']['decision']})", '',
         f"Oracle {rep['oracle']['name']} sha256 {rep['oracle']['sha256']} ({rep['oracle']['dir']}). "
         f"Blocks {len(rep['blocks'])} of {rep['blocks_expected']}.", '',
         '| block | verdict | replies | missing | owned records | ok | wrong | unknown | cut (off/on share) | '
         'live-only | live contradictions | imitations off/on | why |',
         '|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|']
    for b in rep['blocks'] + [dict(rep['pooled'], block='pooled')]:
        e, c, p, i = b['engine_bytes'], b['cut_gate'], b['provenance'], b['imitations']
        arms = b['arms']
        L.append(f"| {b['block']} | {b['verdict']} | {arms['off']['replies'] + arms['on']['replies']} | "
                 f"{arms['off']['replies_missing'] + arms['on']['replies_missing']} | "
                 f"{c['off']['owned_records'] + c['on']['owned_records']} | {e['complete_verified_values']} | "
                 f"{e['wrong_landed_bytes']} | {e['unknown']} | {e['cut']} ({c['off']['cut_share']:.2%}/{c['on']['cut_share']:.2%}) | "
                 f"{p['receipts_live_only']} | {p['live_contradictions']} | "
                 f"{i['off']['imitation_labels']}/{i['on']['imitation_labels']} | {'; '.join(b['why']) or '-'} |")
    L += ['', 'Fields follow T5-W8-DECISION.md (carried by T5-W9-SPEC.md): engine_bytes = rule 1 and the BLOCK list '
          '(wrong landed bytes incl. partial writes; unknown/unparsed/unplaced never verified), cut_gate = rule 5 '
          '(<= 1% of owned records per arm, never rounded, at a native end), provenance = records reconciled against '
          'the stream\'s live sets, imitations = model-written labels per reply and per label. Full fields: w9-report.json.']
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--blocks', required=True, help="directory of T4's w9-block-NN-manifest.json")
    ap.add_argument('--out', required=True)
    ap.add_argument('--oracle', default='r2', choices=sorted(ORACLES))
    ap.add_argument('--only', default='', help='comma list of block numbers (default every manifest found)')
    ap.add_argument('--blocks-expected', type=int, default=10)
    a = ap.parse_args()
    o = ORACLES[a.oracle]
    if not o['agreed']:
        print(f"w9-report: oracle {a.oracle} is not AGREED by T5; use r2", file=sys.stderr)
        return 2
    tool = Path(os.environ.get('W9_ORACLE_DIR') or o['dir'])
    got = sha256(tool / 'oracle.py') if (tool / 'oracle.py').exists() else None
    if got != o['sha256']:
        print(f"w9-report: {tool}/oracle.py sha256 {got} is not {a.oracle}'s pinned {o['sha256']}", file=sys.stderr)
        return 2
    run, out = Path(a.run).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mans = sorted(Path(a.blocks).glob('w9-block-*-manifest.json'))
    only = {int(x) for x in a.only.split(',') if x}
    blocks = []
    for m in mans:
        man = json.load(open(m))
        n = int(man.get('block') or re.search(r'block-(\d+)', m.name).group(1))
        if only and n not in only:
            continue
        b = block(n, man, run, out, tool)
        b['manifest'], b['manifest_sha256'] = str(m), sha256(m)
        blocks.append(b)
        print(f"block {n}: {b['verdict']} (oracle rc {b['oracle_rc']}): {'; '.join(b['why']) or 'clean'}", flush=True)
    if not blocks:
        print('w9-report: no block manifests', file=sys.stderr)
        return 2
    arms = {s: add(blank(), {}) for s in ('off', 'on')}
    for b in blocks:
        for s in arms:
            add(arms[s], b['arms'][s])
    problems = [f"block {b['block']}: {p}" for b in blocks for p in b['provenance']['problems']]
    missing = sum(b['arms'][s]['replies_missing'] for b in blocks for s in arms)
    pooled = rule_fields(arms, None, sum(b['engine_bytes']['text_owned_mismatches'] for b in blocks), problems,
                         all(b['oracle_rc'] in (0, 1, 3) for b in blocks), missing)
    expect = len(only) if only else a.blocks_expected
    absent = expect - len(blocks)
    if absent > 0:
        pooled['why'].append(f'blocks missing {absent}')
    pooled['verdict'] = worst([pooled['verdict']] + [b['verdict'] for b in blocks] + (['INCONCLUSIVE'] if absent > 0 else []))
    if pooled['verdict'] != 'PASS' and pooled['decision'].startswith('engine writes pass'):
        pooled['decision'] = {'FAIL': 'BLOCK promotion / investigate', 'INCONCLUSIVE': 'inconclusive (missing evidence)'}[pooled['verdict']]
    pooled['blocks'] = {b['block']: b['verdict'] for b in blocks}
    rep = {'oracle': {'name': a.oracle, 'sha256': o['sha256'], 'dir': str(tool)}, 'blocks_expected': expect,
           'blocks': blocks, 'pooled': pooled}
    (out / 'w9-report.json').write_text(json.dumps(rep, indent=1))
    (out / 'w9-report.md').write_text(markdown(rep))
    print(f"w9-report {pooled['verdict']}: blocks " + ', '.join(f"{k} {v}" for k, v in pooled['blocks'].items())
          + f"; pooled owned records {arms['off']['owned_records'] + arms['on']['owned_records']}, "
          f"ok {pooled['engine_bytes']['complete_verified_values']}, wrong {pooled['engine_bytes']['wrong_landed_bytes']}, "
          f"cut {pooled['engine_bytes']['cut']}")
    return EXIT[pooled['verdict']]


if __name__ == '__main__':
    sys.exit(main())
