"""window_report.py: one report per experiment window (rank0-host/rank1-host only; window-report.sh runs it after the oracle gate).

  window_report.py OUT ROOT MAP [DETECTOR]
    OUT       the window's report directory; OUT/oracle must hold gate.sh's output for ROOT (evals, mismatches, gate.json)
    ROOT      the receipts directory
    MAP       arm=guide pairs, comma separated (`r=none,rg=@d_mirror,rs=@trust`), a JSON file {arm: guide}, or `auto`
              (labels from each arm's runner_* request fields; the guide text by its sha256 prefix)
    DETECTOR  T5's frozen doubt patterns (default: detector.json beside this tool, a byte copy of
              in-model-runner/t5-doubt-evidence/20261103/detector.json)

Per arm (a receipt's `arm`, else `variant`, else its directory), it reports:
  - the owned-oracle gate on records: ok / wrong-value / wrong-span / not-in-text / cut / short / unknown / unparsed, plus
    text-mode owned mismatches for receipts without records;
  - recheck.py's re-verification cost;
  - T5's doubt detector: `denial` (any pattern but only_mental) and `mental_only`, over reasoning + content with fences
    masked, as in T5's doubt-analyze.py;
  - replies, reasoning and completion tokens (mean, total), solve and cap hits (finish `length`).
Solve comes from, in order: the window's grades.jsonl (T4's analyze.py: {arm: '<run>/<arm>', id, ok}), the receipt's own
grade (S1: grade.correct), T5's reference answers for the t6 smoke prompts (content holds every reference value and
finish is not `length`), else it is not graded. Writes OUT/window-report.md and OUT/window-report.json."""
import json, os, re, sys, math, statistics, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recheck  # noqa: E402

# T5's doubt-analyze.py reference answers for the t6 smoke prompts (S2 trace, TE smoke)
REF = {'t6-run-check-001': [str(314159 * 271828)], 't6-run-check-009': [str(2**100 - 3**50)],
       't6-arithmetic-008': [str(2**37 - 3**19)],
       't6-big-number-001': [str(12345678901234567890 * 9876543210987654321)],
       't6-big-number-007': [str(987654321987654321**2)],
       't6-big-number-019': ['1155727349790921717910215407028', '23663814570294126502'],
       't6-big-number-015': [str(int('FEDCBA98765432100123456789ABCDEF', 16))],
       't6-big-number-008': [str(pow(7, 123, 1000000007))], 't6-big-number-011': [str(pow(3, 400, 10**18)).zfill(18)],
       't6-big-number-002': [str(x) for x in divmod(10**30 - 1, 97)],
       't6-big-number-010': [str(math.gcd(12345678901234567890, 98765432109876543210))],
       't6-big-number-017': [str(math.prod(range(41, 61)))]}
VERDICTS = ('ok', 'wrong-value', 'wrong-span', 'not-in-text', 'wrong-check', 'cut', 'inconclusive', 'short', 'unknown', 'unparsed')


def ref_norm(s):
    return s.replace('{,}', '').replace(',', '').replace('_', '').replace('\\,', '').replace('\\!', '').lower()


def detector(path):
    return {k: re.compile(v, re.I) for k, v in json.load(open(path)).items()}


def detect(rx, s):
    """T5's detect(): fenced blocks masked, every pattern's matches."""
    masked = list(s)
    for m in re.finditer(r'```[\s\S]*?```', s):
        masked[m.start():m.end()] = ' ' * (m.end() - m.start())
    t = ''.join(masked)
    return [k for k, r in rx.items() for _ in r.finditer(t)]


def receipt(p):
    """(id, arm, reasoning, content, usage, finish, own grade, rel dir) for a reply receipt, else None."""
    try:
        x = json.load(open(p))
    except Exception:
        return None
    if not isinstance(x, dict):
        return None
    resp = x['response'] if isinstance(x.get('response'), dict) else x
    ch = (resp.get('choices') or [{}])
    msg = ch[0].get('message', {}) if isinstance(ch, list) and ch else {}
    s = x.get('reasoning', msg.get('reasoning_content')) or ''
    c = x.get('content', msg.get('content')) or ''
    u = resp.get('usage') or x.get('usage')
    if not isinstance(u, dict) or not (s or c):
        return None
    g = x.get('grade')
    grade = g.get('correct') if isinstance(g, dict) else g if isinstance(g, bool) else None
    return dict(id=x.get('id'), arm=recheck.arm_of(p, x, None), reasoning=s, content=c, usage=u, request=x.get('request') or {},
                finish=x.get('finish', ch[0].get('finish_reason') if isinstance(ch, list) and ch else None), grade=grade)


def auto_label(req):
    """An arm's guide label from its request: runner_guide by the sha256 prefix of its text, other runner_* fields as k=v."""
    parts = []
    for k in sorted(req):
        if k.startswith('runner_'):
            v = req[k]
            parts.append(f"guide#{hashlib.sha256(str(v).encode()).hexdigest()[:8]}" if k == 'runner_guide' else f'{k}={str(v)[:40]}')
    return ', '.join(parts) or 'no runner_* request fields'


def main(out, root, amap, det):
    root = os.path.realpath(root)
    # `arm=guide` pairs; a comma starts a new pair only before `<arm>=`, so a guide label may hold commas.
    # `auto`: each arm's first receipt's runner_* request fields (the guide text by its sha256 prefix)
    guides = {} if amap == 'auto' else json.load(open(amap)) if os.path.isfile(amap) else dict(
        kv.strip().split('=', 1) for kv in re.split(r',(?=\s*[\w.-]+=)', amap) if kv.strip())
    rx = detector(det)
    grades = {}
    for gp in [os.path.join(root, 'grades.jsonl')]:
        if os.path.exists(gp):
            for line in open(gp):
                g = json.loads(line)
                grades[(g['arm'], g['id'])] = g.get('ok')
    A = {}
    for dp, ds, fs in os.walk(root):
        ds[:] = [d for d in ds if d != 'gates']  # a window's health and smoke checks, not experiment replies
        for f in sorted(fs):
            if not f.endswith('.json'):
                continue
            p = os.path.join(dp, f)
            r = receipt(p)
            if not r:
                continue
            if amap == 'auto' and r['arm'] not in guides:
                guides[r['arm']] = auto_label(r['request'])
            a = A.setdefault(r['arm'], {'replies': 0, 'rt': [], 'ct': [], 'solve': 0, 'graded': 0, 'cap': 0, 'denial': 0,
                                        'mental_only': 0, 'grade_source': set()})
            rt = (r['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens') or 0
            a['replies'] += 1; a['rt'].append(rt); a['ct'].append(r['usage'].get('completion_tokens') or 0)
            a['cap'] += r['finish'] == 'length'
            rel = os.path.relpath(dp, root)
            ok, src = None, None
            if (rel, r['id']) in grades:
                ok, src = grades[(rel, r['id'])], 'grades.jsonl'
            elif r['grade'] is not None:
                ok, src = r['grade'], 'receipt'
            elif r['id'] in REF:
                ok = r['finish'] != 'length' and all(ref_norm(v) in ref_norm(r['content']) for v in REF[r['id']])
                src = 'T5 reference answers'
            if src:
                a['graded'] += 1; a['solve'] += bool(ok); a['grade_source'].add(src)
            hits = detect(rx, r['reasoning'] + '\n' + r['content'])
            a['denial'] += any(k != 'only_mental' for k in hits)
            a['mental_only'] += bool(hits) and all(k == 'only_mental' for k in hits)
    # owned-oracle gate output, per arm (a cell counts for every arm among its receipt copies)
    od = os.path.join(out, 'oracle')
    gate = json.load(open(os.path.join(od, 'gate.json')))
    cache = {}
    arm_q = lambda q: cache[q] if q in cache else cache.setdefault(q, recheck.arm_of(q, json.load(open(q)), None))
    O = {}
    for line in open(os.path.join(od, 'evals.jsonl')):
        c = json.loads(line)
        if c.get('source') != 'record':
            continue
        for arm in {arm_q(q) for q in c['paths'] if '/gates/' not in q}:
            o = O.setdefault(arm, dict.fromkeys(VERDICTS + ('text_owned',), 0))
            o[c['verdict']] = o.get(c['verdict'], 0) + 1
    for line in open(os.path.join(od, 'mismatches.jsonl')):
        c = json.loads(line)
        if c['owner'] in ('engine', 'owned-form'):
            for arm in {arm_q(q) for q in c['paths']}:
                O.setdefault(arm, dict.fromkeys(VERDICTS + ('text_owned',), 0))['text_owned'] += 1
    R = recheck.main(os.path.join(out, 'recheck'), [root])
    rows = []
    for arm in sorted(A):
        a, o, r = A[arm], O.get(arm), R.get(arm)
        rows.append({'arm': arm, 'guide': guides.get(arm, '?'), 'replies': a['replies'],
                     'reasoning_total': sum(a['rt']), 'reasoning_mean': round(statistics.mean(a['rt']), 1),
                     'completion_total': sum(a['ct']), 'completion_mean': round(statistics.mean(a['ct']), 1),
                     'solve': a['solve'], 'graded': a['graded'], 'grade_source': sorted(a['grade_source']), 'cap_hits': a['cap'],
                     'denial': a['denial'], 'mental_only': a['mental_only'], 'oracle': o,
                     'recheck': None if not r else {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}})
    doc = {'root': root, 'gate': {k: v for k, v in gate.items() if k != 'wrong'}, 'wrong': gate.get('wrong', []), 'arms': rows}
    json.dump(doc, open(os.path.join(out, 'window-report.json'), 'w'), indent=1)
    pct = lambda n, d: f'{100 * n / d:.1f}%' if d else '—'
    L = [f'# Window report: {root}', '',
         f"**owned-oracle gate: {gate.get('verdict') or ('PASS' if gate['pass'] else 'FAIL')}.** Records {gate['records']}: " +
         ', '.join(f"{v} {gate.get(v, 0)}" for v in VERDICTS) + f"; text-mode owned mismatches {gate['text_owned_mismatches']}.", '']
    for w in doc['wrong']:
        L.append(f"- WRONG {w['verdict']} [{w['form']}] span {w.get('span')!r} wrote {str(w.get('written'))[:60]!r} at {w['paths'][0]}")
    L += ['', '## Per arm', '',
          '| arm | guide | replies | reasoning tok (mean / total) | completion tok (mean / total) | solve | cap hits | denial | mental only |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        L.append(f"| {x['arm']} | {x['guide']} | {x['replies']} | {x['reasoning_mean']} / {x['reasoning_total']} | "
                 f"{x['completion_mean']} / {x['completion_total']} | "
                 f"{x['solve']}/{x['graded']} ({pct(x['solve'], x['graded'])}) | {x['cap_hits']} | "
                 f"{x['denial']}/{x['replies']} ({pct(x['denial'], x['replies'])}) | {x['mental_only']} |")
    L += ['', '## Owned cells and re-verification cost (native records)', '',
          '| arm | guide | records | ok | wrong-value | wrong-span | not-in-text | cut | short | unknown | unparsed | text-mode owned mismatches | '
          're-checked cells | re-check tokens | share of reasoning | replies owning cells | share in those | tokens / owned cell |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        o, r = x['oracle'] or {}, x['recheck']
        recs = sum(o.get(v, 0) for v in VERDICTS)
        rc = (f"{r['rechecked']} | {r['tokens']:.0f} | {pct(r['tokens'], r['reasoning'])} | {r['owning']} | "
              f"{pct(r['tokens'], r['owning_reasoning'])} | {r['tokens'] / max(1, r['owned']):.1f}") if r else '— | — | — | — | — | —'
        L.append(f"| {x['arm']} | {x['guide']} | {recs} | " + ' | '.join(str(o.get(v, 0)) for v in VERDICTS) +
                 f" | {o.get('text_owned', 0)} | {rc} |")
    by = {}
    for x in rows:
        g = by.setdefault(x['guide'], {'arms': [], 'replies': 0, 'rt': 0, 'solve': 0, 'graded': 0, 'cap': 0, 'denial': 0,
                                       'tokens': 0.0, 'reasoning': 0, 'owning_reasoning': 0, 'owned': 0})
        g['arms'].append(x['arm']); g['replies'] += x['replies']; g['rt'] += x['reasoning_total']; g['solve'] += x['solve']
        g['graded'] += x['graded']; g['cap'] += x['cap_hits']; g['denial'] += x['denial']
        if x['recheck']:
            for k in ('tokens', 'reasoning', 'owning_reasoning', 'owned'):
                g[k] += x['recheck'][k]
    L += ['', '## Per guide', '', '| guide | arms | replies | reasoning tok | solve | cap hits | denial | re-check share of reasoning | share in owning replies |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for k in sorted(by):
        g = by[k]
        L.append(f"| {k} | {', '.join(g['arms'])} | {g['replies']} | {g['rt']} | {g['solve']}/{g['graded']} | {g['cap']} | "
                 f"{g['denial']}/{g['replies']} ({pct(g['denial'], g['replies'])}) | {pct(g['tokens'], g['reasoning'])} | "
                 f"{pct(g['tokens'], g['owning_reasoning'])} |")
    L += ['', 'Solve source per arm: ' + '; '.join(f"{x['arm']} {', '.join(x['grade_source']) or 'not graded'}" for x in rows) + '.',
          'records: every engine record the gate judged (reasoning and content); re-check columns use the reasoning cells only.',
          'Re-check tokens: recheck.py (lower bound; precision 35/36 by hand). Denial: T5 detector.json, raw detector output '
          '(no hand corrections). Cap hits: finish `length`.']
    open(os.path.join(out, 'window-report.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         sys.argv[4] if len(sys.argv) > 4 else os.path.join(here, 'detector.json'))
