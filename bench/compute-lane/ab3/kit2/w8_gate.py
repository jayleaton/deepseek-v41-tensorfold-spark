#!/usr/bin/env python3
"""W8 trace-default hard acceptance gates (T5-W8-TRACE-SPEC.md) over a finished window: integrity, solve loss, owned
values, staleness, and the paired token report.
Usage: python3 -B w8_gate.py --run RUN --manifest MANIFEST.json --tasks tasks.jsonl --oracle gate.json
           --stale stale.json [--json OUT.json] [--expect-pairs 135] [--arm-key KEY ...] > w8-gate.md
       python3 -B w8_gate.py --selftest
Exit 0 when every gate passes, 1 when any fails, 3 when one is inconclusive and none fails.
Offsets: records place writes by UTF-8 byte; prefix agreement compares reasoning+content bytes, and the line before a
claim is found after converting its byte offset to characters."""
import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tasks as T  # noqa: E402
from analyze import EXACT_LINE  # noqa: E402

FORMS = ('fence', 'repl', 'inline', 'trace')
SETTINGS = ('temperature', 'top_p', 'top_k')
FAIL_VERDICTS = ('wrong-value', 'wrong-span', 'not-in-text')
OPEN_VERDICTS = ('unknown', 'unparsed', 'cut')
OUTLIER = 'mul30-14'
DRAWS, RNG_SEED = 2000, 20261103
LITERAL = re.compile(r'[A-Za-z_]\w*|\d+')
EXIT = {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 3}


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def norm(label):
    """'<boot>/<arm>' as stored on disk: an arm's ':' is '-'."""
    boot, arm = label.split('/', 1)
    return f'{boot}/{arm.replace(":", "-")}'


def runner(rec):
    return ((rec.get('extra') or {}).get('tensorfold') or {}).get('runner') or {}


def owned_cells(rec):
    return [c for c in runner(rec).get('cells') or [] if c.get('status') == 'ok']


def char_of(text, at):
    """A byte offset into text as a character offset."""
    return len(text.encode()[:at].decode('utf-8', 'ignore'))


def verdict(fail, open_):
    return 'FAIL' if fail else 'INCONCLUSIVE' if open_ else 'PASS'


def load(a):
    run = Path(a.run)
    man = json.loads(Path(a.manifest).read_text())
    for p in man['pairs']:
        p['off'], p['on'] = norm(p['off']), norm(p['on'])
    tasks = {t['id']: t for t in jsonl(a.tasks)}
    replies = {}
    for p in man['pairs']:
        for side in ('off', 'on'):
            f = run / p[side] / f'{p["id"]}.json'
            replies[(p[side], p['id'])] = json.loads(f.read_text()) if f.exists() else None
    gp = run / 'grades.jsonl'
    grades = {(g['arm'], g['id']): g for g in jsonl(gp)} if gp.exists() else {}
    return run, man, tasks, replies, grades


def cut_ids(run, label):
    """Ids a cut.jsonl (arm or boot directory) lists for this arm."""
    out = set()
    for p in (run / label / 'cut.jsonl', run / label.split('/')[0] / 'cut.jsonl'):
        if p.exists():
            out.update(r.get('id') for r in jsonl(p))
    return out


def prefix(off, on):
    """Common byte prefix of reasoning+content, the first owned write in either reply (same bytes), and whether the
    texts part before any owned write."""
    x, y = [((r.get('reasoning') or '') + (r.get('content') or '')).encode() for r in (off, on)]
    lo, hi = 0, min(len(x), len(y))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if x[:mid] == y[:mid] else (lo, mid - 1)
    starts = [c['at'] + (len((r.get('reasoning') or '').encode()) if c.get('field') == 'content' else 0)
              for r in (off, on) for c in owned_cells(r) if c.get('at') is not None]
    first = min(starts, default=None)
    same = x == y
    return {'common_bytes': lo, 'first_owned_byte': first, 'identical': same,
            'before_owned': not same and (first is None or lo < first)}


def integrity(run, man, tasks, replies, grades, a):
    pairs, fail, open_, missing, rows, flagged = man['pairs'], [], [], [], [], []
    if len(pairs) != a.expect_pairs:
        fail.append(f'manifest has {len(pairs)} pairs, expected {a.expect_pairs}')
    sha = hashlib.sha256(Path(a.tasks).read_bytes()).hexdigest()
    if man.get('tasks_sha256') != sha:
        fail.append(f'tasks file sha256 {sha[:12]} is not the manifest\'s {str(man.get("tasks_sha256"))[:12]}')
    seen, seeds = {}, set()
    settings = man.get('settings') or {}
    for p in pairs:
        pid = f'{p["id"]} s{p["seed"]}'
        if (p['id'], p['seed']) in seeds:
            fail.append(f'{pid} is in the manifest twice')
        seeds.add((p['id'], p['seed']))
        if p['id'] not in tasks:
            fail.append(f'{pid}: task not in the tasks file')
        for side in ('off', 'on'):
            key = (p[side], p['id'])
            if key in seen:
                fail.append(f'{key[0]}/{key[1]} is used by {seen[key]} and {pid}')
            seen[key] = pid
            rec = replies[key]
            if rec is None:
                cut = ' (listed in cut.jsonl, no reply file)' if p['id'] in cut_ids(run, p[side]) else ''
                missing.append(f'{p[side]}/{p["id"]}{cut}')
            elif rec.get('id') != p['id']:
                fail.append(f'{p[side]}/{p["id"]}: the reply says id {rec.get("id")}')
        off, on = replies[(p['off'], p['id'])], replies[(p['on'], p['id'])]
        if off is None or on is None:
            continue
        ro, rn = off.get('request'), on.get('request')
        if not isinstance(ro, dict) or not isinstance(rn, dict):
            open_.append(f'{pid}: a reply has no request')
            continue
        diff = sorted(k for k in set(ro) | set(rn) if k not in a.arm_key and ro.get(k) != rn.get(k))
        if diff:
            fail.append(f'{pid}: off and on requests differ in {", ".join(diff)}')
        if ro.get('seed') != p['seed'] or rn.get('seed') != p['seed']:
            fail.append(f'{pid}: request seeds {ro.get("seed")}/{rn.get("seed")}, manifest {p["seed"]}')
        bad = [k for k in SETTINGS if k in settings and ro.get(k) != settings[k]]
        if bad:
            fail.append(f'{pid}: request {", ".join(bad)} not the manifest settings')
        pr = dict(prefix(off, on), id=p['id'], seed=p['seed'])
        rows.append(pr)
        if pr['before_owned']:
            flagged.append(pr)
    if missing:
        open_.append(f'{len(missing)} manifest replies missing')
    refs = {f'{label}/{tid}' for label, tid in replies}
    outside = sorted(f'{f.parent.parent.name}/{f.parent.name}/{f.stem}' for f in run.glob('*/*/*.json')
                     if f.name != 'arm.json' and f'{f.parent.parent.name}/{f.parent.name}/{f.stem}' not in refs)
    grades_outside = sorted(f'{k[0]}/{k[1]}' for k in grades if k not in replies)
    return {'verdict': verdict(fail, open_), 'fail': fail, 'inconclusive': open_, 'missing': missing,
            'pairs': len(pairs), 'not_counted_replies': outside, 'not_counted_grades': grades_outside,
            'prefix': rows, 'prefix_before_owned': flagged, 'prefix_unit': 'UTF-8 bytes of reasoning+content'}


def correct(rec, g, task):
    """(sign-aware, raw): the grade, a stop finish and non-empty content. None when the reply or grade is missing."""
    if rec is None or g is None:
        return None
    done = rec.get('finish') == 'stop' and bool((rec.get('content') or '').strip())
    signed = g.get('ok_signed')
    if signed is None:
        c = task['check']
        signed = T.same_number(T.final_number(rec.get('content')), c['truth']) if c['kind'] == 'number' else g['ok']
    return done and bool(signed), done and bool(g.get('ok'))


def solve_loss(man, tasks, replies, grades):
    lost, lost_raw, open_, totals, per = [], [], [], {}, {}
    for p in man['pairs']:
        task = tasks.get(p['id'])
        side_ok = {}
        for side in ('off', 'on'):
            key = (p[side], p['id'])
            side_ok[side] = None if task is None else correct(replies[key], grades.get(key), task)
        per[(p['id'], p['seed'])] = side_ok
        if side_ok['off'] is None or side_ok['on'] is None:
            open_.append(f'{p["id"]} s{p["seed"]}: reply or grade missing')
            continue
        on, g = replies[(p['on'], p['id'])], grades.get((p['on'], p['id'])) or {}
        row = {'id': p['id'], 'seed': p['seed'], 'on': p['on'], 'finish': on.get('finish'),
               'got_signed': g.get('got_signed'), 'got': g.get('got'), 'why': g.get('why'),
               'empty': not (on.get('content') or '').strip()}
        if side_ok['off'][0] and not side_ok['on'][0]:
            lost.append(row)
        if side_ok['off'][1] and not side_ok['on'][1]:
            lost_raw.append(row)
        for side in ('off', 'on'):
            for key in ('all', f'seed {p["seed"]}', f'family {task["set"]}'):
                t = totals.setdefault(side, {}).setdefault(key, {'signed': 0, 'raw': 0, 'n': 0})
                t['signed'] += side_ok[side][0]
                t['raw'] += side_ok[side][1]
                t['n'] += 1
    fail = [f'{len(lost)} on-only sign-aware losses'] if lost else []
    return {'verdict': verdict(fail, open_), 'fail': fail, 'inconclusive': open_, 'lost': lost,
            'lost_raw': lost_raw, 'solved': totals}, per


def oracle_records(replies):
    """Records the oracle should judge: one reply text per field, deduplicated as its extract step does."""
    seen = {}
    for rec in replies.values():
        if rec is None:
            continue
        run = runner(rec)
        counts = {k: (run.get(k) or {}).get('owned', 0) for k in ('inline', 'trace', 'fence', 'repl')}
        recs = run.get('cells') if isinstance(run.get('cells'), list) else None
        for field in ('reasoning', 'content'):
            text = rec.get(field)
            if isinstance(text, str) and text and recs is not None:
                mine = [r for r in recs if r.get('field') == field]
                seen[hashlib.sha1((text + json.dumps([counts, mine], sort_keys=True)).encode()).hexdigest()] = len(mine)
    return sum(seen.values())


def owned(replies, on_labels, oracle_path):
    fail, open_, short, imitations, counts = [], [], [], [], {}
    for (label, tid), rec in sorted(replies.items()):
        if rec is None:
            continue
        run, side = runner(rec), 'on' if label in on_labels else 'off'
        cells = run.get('cells')
        if not isinstance(cells, list):
            if (rec.get('reasoning') or '') + (rec.get('content') or ''):
                short.append(f'{label}/{tid}: no native records')
            cells = []
        for c in cells:
            if c.get('status') == 'ok':
                k = f'{side} {c.get("form")} {c.get("route") or "-"}'
                counts[k] = counts.get(k, 0) + 1
        for form in FORMS:
            want = (run.get(form) or {}).get('owned') or 0
            have = sum(c.get('form') == form and c.get('status') == 'ok' for c in cells)
            if want > have:
                short.append(f'{label}/{tid}: {form} counters owned {want}, {have} ok records')
        for field in ('reasoning', 'content'):
            text = rec.get(field) or ''
            ats = [c['at'] for c in cells if c.get('form') == 'trace' and c.get('field') == field
                   and c.get('at') is not None]
            for m in EXACT_LINE.finditer(text):
                s = len(text[:m.start()].encode())
                e = s + len(m.group(0).encode())
                if not any(s - 1 <= x < e for x in ats):
                    imitations.append({'reply': f'{label}/{tid}', 'side': side, 'field': field,
                                       'line': m.group(0)[:160]})
    if short:
        open_.append(f'{len(short)} replies with missing record evidence')
    gate, want = None, oracle_records(replies)
    if not oracle_path or not Path(oracle_path).exists():
        open_.append('oracle gate.json missing')
    else:
        gate = json.loads(Path(oracle_path).read_text())
        bad = {k: gate.get(k) or 0 for k in FAIL_VERDICTS + ('text_owned_mismatches',)}
        wrong = [w for w in gate.get('wrong') or [] if w.get('verdict') != 'unparsed']
        if sum(bad.values()) or wrong:
            fail.append('oracle: ' + ', '.join(f'{k} {v}' for k, v in bad.items()) + f'; {len(wrong)} wrong listed')
        for k in OPEN_VERDICTS:
            if gate.get(k):
                open_.append(f'oracle {k} {gate[k]}')
        if (gate.get('records') or 0) < want:
            open_.append(f'oracle judged {gate.get("records")} records; the manifest replies hold {want}')
    return {'verdict': verdict(fail, open_), 'fail': fail, 'inconclusive': open_, 'missing_records': short,
            'owned_by_side_form_route': dict(sorted(counts.items())), 'imitations': imitations,
            'oracle': gate, 'records_expected': want}


def stale_gate(replies, on_labels, stale_path):
    fail, open_, named, ours = [], [], [], {}
    for (label, tid), rec in sorted(replies.items()):
        if rec is None or label not in on_labels:
            continue
        for c in owned_cells(rec):
            if c.get('form') != 'trace':
                continue
            ours[label] = ours.get(label, 0) + 1
            text = rec.get(c.get('field')) or ''
            line = text[:char_of(text, c.get('at') or 0)].rstrip('\n').split('\n')[-1]
            display = c.get('display') or c.get('span') or ''
            have = set(LITERAL.findall(line))
            absent = sorted({x for x in LITERAL.findall(display) if x not in have})
            if absent:
                named.append({'reply': f'{label}/{tid}', 'display': display, 'value': c.get('value'),
                              'line_before': line[-160:], 'absent': absent})
    arms = {}
    if not stale_path or not Path(stale_path).exists():
        open_.append('stale JSON missing')
    else:
        s = json.loads(Path(stale_path).read_text())
        arms = s.get('arms') or {}
        for label in sorted(on_labels):
            arm = arms.get(label)
            if arm is None:
                open_.append(f'{label} not in the stale JSON')
                continue
            hits = [t for t in arm.get('replies') or [] if (label, t) in replies]
            if arm.get('stale') and (hits or not arm.get('replies')):
                fail.append(f'{label}: {arm["stale"]} stale claims in {", ".join(hits) or "?"}')
            if (arm.get('claims') or 0) < ours.get(label, 0):
                open_.append(f'{label}: stale scanned {arm.get("claims")} claims, receipts hold {ours[label]}')
        if s.get('claims_not_placed', 1):
            open_.append(f'claims_not_placed {s.get("claims_not_placed")}')
    return {'verdict': verdict(fail, open_), 'fail': fail, 'inconclusive': open_, 'arms': arms,
            'trace_claims': ours, 'named_claims': named}


def tokens(rec):
    u = (rec or {}).get('usage') or {}
    tot, rs = u.get('completion_tokens'), (u.get('completion_tokens_details') or {}).get('reasoning_tokens')
    return {'reasoning': rs, 'content': None if tot is None or rs is None else tot - rs, 'combined': tot}


def summary(groups):
    """(on/off ratio of summed tokens, median over tasks of the task's mean on-off delta) for [[(off, on)...]...]."""
    off = sum(x for g in groups for x, _ in g)
    on = sum(y for g in groups for _, y in g)
    med = statistics.median(statistics.mean(y - x for x, y in g) for g in groups) if groups else None
    return (on / off if off else None), med


def interval(xs):
    xs = sorted(x for x in xs if x is not None)
    return None if not xs else [round(xs[int(0.025 * (len(xs) - 1))], 4), round(xs[int(0.975 * (len(xs) - 1))], 4)]


def stats(rows):
    use = [r for r in rows if r['off']['combined'] is not None and r['on']['combined'] is not None]
    by = {}
    for r in use:
        by.setdefault(r['id'], []).append((r['off']['combined'], r['on']['combined']))
    ratio, med = summary(list(by.values()))
    rng, keys, draws = random.Random(RNG_SEED), sorted(by), []
    for _ in range(DRAWS if by else 0):
        draws.append(summary([rng.choices(by[k], k=len(by[k])) for k in rng.choices(keys, k=len(keys))]))
    part = {}
    for k in ('reasoning', 'content'):
        both = [r for r in rows if r['off'][k] is not None and r['on'][k] is not None]
        off = sum(r['off'][k] for r in both)
        part[f'ratio_{k}'] = round(sum(r['on'][k] for r in both) / off, 4) if off else None
    return {'pairs': len(rows), 'pairs_with_usage': len(use), 'tasks': len(by),
            'off_tokens': sum(x for g in by.values() for x, _ in g),
            'on_tokens': sum(y for g in by.values() for _, y in g),
            'ratio': None if ratio is None else round(ratio, 4), 'ratio_ci95': interval(d[0] for d in draws),
            'median_task_delta': med, 'median_task_delta_ci95': interval(d[1] for d in draws), **part}


def numbers(man, tasks, replies, grades, per):
    rows = []
    for p in man['pairs']:
        row = {'id': p['id'], 'seed': p['seed'], 'group': p.get('group'),
               'family': (tasks.get(p['id']) or {}).get('set')}
        for side in ('off', 'on'):
            rec, g = replies[(p[side], p['id'])], grades.get((p[side], p['id'])) or {}
            ok = per[(p['id'], p['seed'])][side]
            row[side] = {'label': p[side], 'finish': None if rec is None else rec.get('finish'), **tokens(rec),
                         'ok_signed': None if ok is None else ok[0], 'ok_raw': None if ok is None else ok[1],
                         'grade_ok': g.get('ok'), 'grade_ok_signed': g.get('ok_signed')}
        rows.append(row)
    groups = {'all': rows, f'without {OUTLIER}': [r for r in rows if r['id'] != OUTLIER]}
    for fam in sorted({r['family'] for r in rows if r['family']}):
        for tag, keep in (('', rows), (f' without {OUTLIER}', groups[f'without {OUTLIER}'])):
            sub = [r for r in keep if r['family'] == fam]
            if sub and (not tag or len(sub) != sum(r['family'] == fam for r in rows)):
                groups[f'family {fam}{tag}'] = sub
    return {'pairs': rows, 'paired': {k: stats(v) for k, v in groups.items()},
            'bootstrap': {'draws': DRAWS, 'rng_seed': RNG_SEED, 'levels': 'tasks, then seeds within a task'}}


def evaluate(a):
    run, man, tasks, replies, grades = load(a)
    on_labels = {p['on'] for p in man['pairs']}
    solve, per = solve_loss(man, tasks, replies, grades)
    gates = {'integrity': integrity(run, man, tasks, replies, grades, a), 'solve_loss': solve,
             'owned': owned(replies, on_labels, a.oracle), 'stale': stale_gate(replies, on_labels, a.stale)}
    vs = [g['verdict'] for g in gates.values()]
    overall = 'FAIL' if 'FAIL' in vs else 'INCONCLUSIVE' if 'INCONCLUSIVE' in vs else 'PASS'
    return {'overall': overall, 'exit': EXIT[overall], 'gates': gates,
            'numbers': numbers(man, tasks, replies, grades, per),
            'inputs': {k: str(getattr(a, k)) for k in ('run', 'manifest', 'tasks', 'oracle', 'stale')}}


def tok(t):
    return '/'.join('-' if t[k] is None else str(t[k]) for k in ('reasoning', 'content', 'combined'))


def markdown(res):
    g = res['gates']
    out = ['# W8 trace-default acceptance gates', '', f'**Overall: {res["overall"]}**', '',
           '| gate | verdict | fail | inconclusive |', '|---|---|---|---|']
    for name, x in g.items():
        out.append(f'| {name} | {x["verdict"]} | {"; ".join(x["fail"][:6]) or "-"} | '
                   f'{"; ".join(x["inconclusive"][:6]) or "-"} |')
    i = g['integrity']
    out += ['', f'## Integrity: {i["pairs"]} pairs, {len(i["missing"])} replies missing, '
            f'{len(i["not_counted_replies"])} receipt files and {len(i["not_counted_grades"])} grade rows outside '
            'the manifest (not counted)']
    out += [f'- missing {m}' for m in i['missing']]
    out += ['', f'Prefix agreement ({i["prefix_unit"]}): {len(i["prefix_before_owned"])} of {len(i["prefix"])} '
            'pairs part before any owned write (investigate; not a failure).']
    out += [f'- {r["id"]} s{r["seed"]}: common {r["common_bytes"]} bytes, first owned at '
            f'{r["first_owned_byte"]}' for r in i['prefix_before_owned']]
    s = g['solve_loss']
    out += ['', f'## Solve loss: {len(s["lost"])} sign-aware, {len(s["lost_raw"])} raw', '',
            '| task | seed | on arm | finish | empty | got (sign-aware / raw) | why |', '|---|---|---|---|---|---|---|']
    out += [f'| {r["id"]} | {r["seed"]} | {r["on"]} | {r["finish"]} | {r["empty"]} | {r["got_signed"]} / {r["got"]} | '
            f'{(r["why"] or "-")[:80]} |' for r in s['lost']]
    out += ['', '| totals | off sign-aware (raw) / n | on sign-aware (raw) / n |', '|---|---|---|']
    for key in sorted(s['solved'].get('off') or {}, key=lambda k: (k != 'all', not k.startswith('seed'), k)):
        o, n = s['solved']['off'][key], s['solved'].get('on', {}).get(key, {'signed': 0, 'raw': 0, 'n': 0})
        out.append(f'| {key} | {o["signed"]} ({o["raw"]}) / {o["n"]} | {n["signed"]} ({n["raw"]}) / {n["n"]} |')
    o = g['owned']
    out += ['', f'## Owned values: records expected by the oracle {o["records_expected"]}; owned by side/form/route '
            + (', '.join(f'{k} {v}' for k, v in o['owned_by_side_form_route'].items()) or 'none')]
    if o['oracle']:
        out.append('Oracle: ' + ', '.join(f'{k} {o["oracle"].get(k)}' for k in
                                          ('records', 'ok') + FAIL_VERDICTS + OPEN_VERDICTS
                                          + ('text_owned_mismatches',)))
    out += [f'- missing records: {m}' for m in o['missing_records']]
    out += ['', f'Imitations (model-written `[exact...]` lines with no native record; never owned): '
            f'{len(o["imitations"])}']
    out += [f'- {m["reply"]} {m["field"]}: `{m["line"]}`' for m in o['imitations']]
    st = g['stale']
    out += ['', '## Stale: ' + (', '.join(f'{k} claims {v.get("claims")} stale {v.get("stale")}'
                                          for k, v in st['arms'].items()) or 'no stale JSON'), '',
            f'Named live trace claims for hand review ({len(st["named_claims"])}):']
    out += [f'- {m["reply"]}: `{m["display"]}` = {m["value"]}; absent {m["absent"]} from `{m["line_before"]}`'
            for m in st['named_claims']]
    n = res['numbers']
    out += ['', f'## Tokens (actual usage; hierarchical bootstrap {DRAWS} draws, seed {RNG_SEED})', '',
            '| group | pairs (with usage) | tasks | off | on | on/off | 95% | reason on/off | content on/off | '
            'median task delta | 95% |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for k, v in n['paired'].items():
        out.append(f'| {k} | {v["pairs"]} ({v["pairs_with_usage"]}) | {v["tasks"]} | {v["off_tokens"]} | '
                   f'{v["on_tokens"]} | {v["ratio"]} | {v["ratio_ci95"]} | {v["ratio_reasoning"]} | '
                   f'{v["ratio_content"]} | {v["median_task_delta"]} | {v["median_task_delta_ci95"]} |')
    out += ['', '| task | seed | off finish | on finish | off reason/content/total | on reason/content/total | '
            'off ok (raw) | on ok (raw) |', '|---|---|---|---|---|---|---|---|']
    for r in n['pairs']:
        out.append(f'| {r["id"]} | {r["seed"]} | {r["off"]["finish"]} | {r["on"]["finish"]} | {tok(r["off"])} | '
                   f'{tok(r["on"])} | {r["off"]["ok_signed"]} ({r["off"]["ok_raw"]}) | '
                   f'{r["on"]["ok_signed"]} ({r["on"]["ok_raw"]}) |')
    return '\n'.join(out) + '\n'


# ---- self-test ----------------------------------------------------------------------------------------------------
MINI = {'add30-01': ('add30', 'Café step: 123+456.', '123+456', '579'),
        'mul30-14': ('mul30', 'Next 12*34.', '12*34', '408'),
        'bignum-03': ('bignum', 'Let N = 3; take N minus twenty.', 'N-20', '-17')}


def mini_reply(tid, seed, label, trace):
    fam, line, disp, val = MINI[tid]
    head = f'{line}\n'
    reasoning = head + (f'[exact, computed by the engine: {disp} = {val}]\nGood.' if trace else f'{disp} is {val}.')
    counters = {f: {'closed': 0, 'owned': 0, 'abstained': 0, 'owned_tokens': 0} for f in FORMS}
    cells = []
    if trace:
        counters['trace'] = {'closed': 1, 'owned': 1, 'abstained': 0, 'owned_tokens': 9}
        cells = [{'form': 'trace', 'route': 'graph', 'status': 'ok', 'span': disp, 'display': disp, 'value': val,
                  'field': 'reasoning', 'at': len(head.encode())}]
    total = 1000 + 100 * seed + len(tid) * 7 - (60 if trace else 0)
    return {'id': tid, 'set': fam, 'arm': label, 'reasoning': reasoning, 'content': f'The answer is {val}.',
            'request': {'seed': seed, 'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'max_tokens': 32768,
                        'runner_guide': {'sha256': 'g0'}},
            'usage': {'completion_tokens': total, 'completion_tokens_details': {'reasoning_tokens': total - 20}},
            'finish': 'stop', 'elapsed_s': 1.0, 'extra': {'tensorfold': {'runner': dict(counters, cells=cells)}}}


def mini(root, case):
    """A 3-task x 2-seed run in root, changed by `case`; returns the arguments evaluate() takes."""
    run = root / 'run'
    tl = [{'id': t, 'set': v[0], 'prompt': t, 'check': {'kind': 'number', 'truth': v[3]}} for t, v in MINI.items()]
    (root / 'tasks.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tl))
    pairs, recs = [], {}
    for seed in (1, 2):
        off, on = f'w8off/a-s{seed}', f'w8on/t-s{seed}'
        for tid in MINI:
            pairs.append({'id': tid, 'seed': seed, 'group': f'G{seed}', 'off': off,
                          'on': on.replace('-s2', ':s2')})
            recs[(off, tid)], recs[(on, tid)] = mini_reply(tid, seed, off, False), mini_reply(tid, seed, on, True)
    grades = {k: {'arm': k[0], 'id': k[1], 'ok': True, 'got': MINI[k[1]][3],
                  **({'ok_signed': True} if 'w8on' in k[0] else {})} for k in recs}
    oracle = {'pass': True, 'records': 3, 'ok': 3, 'wrong-value': 0, 'wrong-span': 0, 'not-in-text': 0,
              'unparsed': 0, 'cut': 0, 'unknown': 0, 'text_owned_mismatches': 0, 'marks': {}, 'wrong': []}
    stale = {'arms': {f'w8{x}/{y}-s{s}': {'claims': 3 if x == 'on' else 0, 'stale': 0, 'replies': []}
                      for x, y in (('off', 'a'), ('on', 't')) for s in (1, 2)},
             'claims_placed_on_their_line': 6, 'claims_not_placed': 0}
    if case == 'oracle-wrong':
        oracle.update({'pass': False, 'ok': 2, 'wrong-value': 1,
                       'wrong': [{'verdict': 'wrong-value', 'form': 'trace'}]})
    elif case == 'stale':
        stale['arms']['w8on/t-s1'].update(stale=1, replies=['add30-01'])
    elif case in ('shared-miss', 'on-loss'):
        keys = [('w8off/a-s1', 'add30-01'), ('w8on/t-s1', 'add30-01')] if case == 'shared-miss' else [('w8on/t-s2',
                                                                                                         'mul30-14')]
        for k in keys:
            recs[k]['content'] = 'The answer is 578.'
            grades[k].update(ok=False, ok_signed=False, got='578')
    elif case == 'missing':
        del recs[('w8on/t-s1', 'bignum-03')], grades[('w8on/t-s1', 'bignum-03')]
    elif case == 'no-records':
        recs[('w8on/t-s2', 'add30-01')]['extra']['tensorfold']['runner']['cells'] = []
    elif case == 'imitation':
        recs[('w8on/t-s1', 'mul30-14')]['content'] = '[exact, computed by the engine: 2*3 = 6]\nThe answer is 408.'
    elif case == 'timeout':
        recs[('w8on/t-s2', 'bignum-03')].update(finish='timeout', content='')
        grades[('w8on/t-s2', 'bignum-03')] = {'arm': 'w8on/t-s2', 'id': 'bignum-03', 'ok': False,
                                             'why': 'finish timeout'}
    for (label, tid), r in recs.items():
        d = run / label
        d.mkdir(parents=True, exist_ok=True)
        (d / 'arm.json').write_text('{}')
        (d / f'{tid}.json').write_text(json.dumps(r))
    (run / 'w8on' / 't-s1' / 'extra-01.json').write_text(json.dumps(mini_reply('add30-01', 1, 'w8on/t-s1', True)))
    (run / 'grades.jsonl').write_text(''.join(json.dumps(g) + '\n' for g in grades.values()))
    sha = hashlib.sha256((root / 'tasks.jsonl').read_bytes()).hexdigest()
    (root / 'manifest.json').write_text(json.dumps({'pairs': pairs, 'tasks_sha256': sha, 'settings': {
        'temperature': 1.0, 'top_p': 0.95, 'top_k': 20}}))
    (root / 'gate.json').write_text(json.dumps(oracle))
    (root / 'stale.json').write_text(json.dumps(stale))
    return argparse.Namespace(run=str(run), manifest=str(root / 'manifest.json'), tasks=str(root / 'tasks.jsonl'),
                              oracle=str(root / 'gate.json'), stale=str(root / 'stale.json'), expect_pairs=6,
                              arm_key=['arm'])


def selftest():
    res = {}
    for case in ('clean', 'oracle-wrong', 'stale', 'shared-miss', 'on-loss', 'missing', 'no-records', 'imitation',
                 'timeout'):
        with tempfile.TemporaryDirectory() as tmp:
            res[case] = evaluate(mini(Path(tmp), case))
            markdown(res[case])
    v = {c: {k: g['verdict'] for k, g in r['gates'].items()} for c, r in res.items()}
    lost = {c: [(x['id'], x['seed'], x['finish']) for x in r['gates']['solve_loss']['lost']] for c, r in res.items()}
    own = {c: r['gates']['owned'] for c, r in res.items()}
    on_n = res['timeout']['gates']['solve_loss']['solved']['on']['all']['n']
    checks = [
        ('(a) clean run passes every gate', res['clean']['overall'] == 'PASS' and res['clean']['exit'] == 0
         and not res['clean']['gates']['integrity']['prefix_before_owned']
         and len(res['clean']['gates']['stale']['named_claims']) == 2),
        ('(b) wrong owned value fails owned', v['oracle-wrong']['owned'] == 'FAIL'
         and res['oracle-wrong']['exit'] == 1),
        ('(c) stale old-binding claim fails stale', v['stale']['stale'] == 'FAIL' and res['stale']['exit'] == 1),
        ('(d) shared miss does not fail solve loss', v['shared-miss']['solve_loss'] == 'PASS'
         and res['shared-miss']['overall'] == 'PASS'),
        ('(d) on-only loss fails solve loss', v['on-loss']['solve_loss'] == 'FAIL'
         and lost['on-loss'] == [('mul30-14', 2, 'stop')]),
        ('(e) missing pair member is inconclusive', v['missing']['integrity'] == 'INCONCLUSIVE'
         and res['missing']['overall'] == 'INCONCLUSIVE' and res['missing']['exit'] == 3),
        ('(f) counters owned>0 without records is inconclusive', v['no-records']['owned'] == 'INCONCLUSIVE'
         and res['no-records']['exit'] == 3),
        ('(g) model-written trace line is an imitation, owned unchanged', len(own['imitation']['imitations']) == 1
         and own['imitation']['verdict'] == 'PASS' and len(own['clean']['imitations']) == 0
         and own['imitation']['owned_by_side_form_route'] == own['clean']['owned_by_side_form_route']),
        ('(h) timeout stays in the denominator and is a loss', v['timeout']['solve_loss'] == 'FAIL'
         and v['timeout']['integrity'] == 'PASS' and lost['timeout'] == [('bignum-03', 2, 'timeout')] and on_n == 6
         and res['timeout']['numbers']['paired']['all']['pairs'] == 6),
    ]
    for name, ok in checks:
        print(f'{"ok" if ok else "FAIL"} {name}')
    good = all(ok for _, ok in checks)
    print(f'W8GATE-SELFTEST {"PASS" if good else "FAIL"}')
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run')
    ap.add_argument('--manifest')
    ap.add_argument('--tasks')
    ap.add_argument('--oracle', help='owned-oracle gate.json')
    ap.add_argument('--stale', help='stale_live.py output JSON')
    ap.add_argument('--json', help='write the full result here')
    ap.add_argument('--expect-pairs', type=int, default=135)
    ap.add_argument('--arm-key', action='append', default=['arm'], help='request key allowed to differ by arm')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not (a.run and a.manifest and a.tasks):
        ap.error('--run, --manifest and --tasks are required')
    res = evaluate(a)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(markdown(res))
    sys.exit(res['exit'])


if __name__ == '__main__':
    main()
