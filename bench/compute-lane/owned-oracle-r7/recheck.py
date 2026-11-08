"""recheck.py: what the model spends re-checking values the engine wrote (engine records, TF_DSV41_RUNNER_CELLS).

  recheck.py OUT ROOT...      (rank0-host/rank1-host only; ARM_GROUP regex like ORACLE_GROUP, else the receipt's arm, else its directory)

For each owned cell in the reasoning, an episode starts at
  - question: the write is questioned at once (`?` after it, or a verify/check/wait/compute cue in the next sentence),
  - restate:  a later sentence restates the cell's span (digits compared without grouping), not by another engine write,
  - cue:      a later sentence holds the value with a check word (verify, check, yes, correct, matches, ...),
and runs forward over the next sentences while they do arithmetic, until the value comes back as the result of the
model's own arithmetic (`... = value`). Only a `?` episode may stay open (no conclusion): it ends after its arithmetic
run (at most 6 sentences). restate needs the span on its own followed by `=`; cue needs a value of 3+ digits.
Tokens: the records' (byte, token) pairs and (end, reasoning_tokens) map byte offsets to tokens by linear interpolation;
engine-forced tokens inside an episode are not counted. A reply's re-check tokens are the union of its episodes;
restate-only episodes (a bare `S = V` copied forward, no arithmetic) and doubts (a lone `?` after the value) are
reported apart and not counted as re-checks. Cells: inline and trace (V of `[exact…: S = V]`); fence and REPL
outputs are not followed. Known gaps (lower bound): checks of a composite result, a recheck in a
parenthesis right after the write, and a second pass done through new engine cells are not counted.
Writes OUT/episodes.jsonl (one per episode, with its text), OUT/cells.jsonl, OUT/replies.jsonl and OUT/report.md
(per-arm totals; shares over all replies and over the replies that own cells)."""
import json, os, re, sys, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle import replies  # noqa: E402

CUE = re.compile(r'\b(verify|verif\w*|check\w*|double[- ]check|confirm\w*|re-?comput\w*|recalc\w*|sanity|indeed|correct|'
                 r'matches|consistent|yes|good|right|let\'?s (?:compute|calculate|do)|wait|hmm|actually)\b', re.I)
ARITH = re.compile(r'\d\s*[-+*/^×·]\s*[\d(]|[\d)]\s*=|=\s*-?\d|\bmod\b|\d\s*\^')
UNIT = re.compile(r'[^\n]*?(?:[.?!](?=\s+[A-Z(])|\n|$)')
MAX_RUN, MAX_UNITS = 6, 40


def norm(s):
    s = re.sub(r'(?<=\d),(?=\d{3})', '', s).replace('\\!', '')
    return re.sub(r'\s+', '', s).replace('**', '^').replace('×', '*')


def units(text):
    out = []
    for m in UNIT.finditer(text):
        if m.end() > m.start():
            out.append((m.start(), m.end()))
    return out


def value_re(v):
    plain = v.lstrip('-')
    forms = {plain}
    if re.fullmatch(r'\d+', plain) and len(plain) > 3:
        forms.add(f'{int(plain):,}')
    alt = '|'.join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    return re.compile(r'(?<![\d.,])(?:' + alt + r')(?!\d|[.,]\d)')


def tokmap(text, recs, total):
    b = text.encode()
    pts = [(0, 0)] + sorted((r['at'], r['token']) for r in recs if r.get('token') is not None) + [(len(b), total)]
    xs, ys = [], []
    for x, y in pts:
        if not xs or (x > xs[-1] and y >= ys[-1]):
            xs.append(x); ys.append(y)
    def tok(ci):
        x = len(text[:ci].encode())
        k = bisect.bisect_right(xs, x) - 1
        if k >= len(xs) - 1:
            return ys[-1]
        return ys[k] + (ys[k + 1] - ys[k]) * (x - xs[k]) / max(1, xs[k + 1] - xs[k])
    return tok


def cells_of(text, recs):
    tb, out = text.encode(), []
    for r in recs:
        v = r.get('value', '')
        if r.get('field') != 'reasoning' or not v or tb[r['at']:r['at'] + len(v.encode())] != v.encode():
            continue
        s = len(tb[:r['at']].decode('utf-8', 'ignore'))
        out.append(dict(r, s=s, e=s + len(v)))
    return out


def episodes(text, cells, us):
    eng = [(c['s'], c['e']) for c in cells]
    in_eng = lambda a, b: any(s <= a and b <= e for s, e in eng)
    masked = text
    for s, e in eng:  # the engine's own bytes (a trace check's `2*3 ≡ 6`) never read as the model restating a span
        masked = masked[:s] + ' ' * (e - s) + masked[e:]
    starts = [u[0] for u in us]
    for c in cells:
        v = c['value'].strip()
        if c.get('form') == 'trace':  # `[exact…: S = V]`: the value is V (`remainder` lines keep the quotient)
            m = re.search(r'=\s*(-?\d+(?:\.\d+)?)(?:\s+remainder\s+\d+)?\s*\]', v)
            v = m.group(1) if m else v
        span = norm(c.get('span') or '')
        num = re.fullmatch(r'-?\d+(?:\.\d+)?', v) is not None
        vre = value_re(v) if num else None
        long_ = num and len(re.sub(r'\D', '', v)) >= 3
        ops = [o for o in re.findall(r'\d+', span) if len(o) >= 2] or [norm(v)]
        k0 = bisect.bisect_right(starts, c['e']) - 1  # the unit holding the write
        later = c['e']
        eps = []

        def result_in(k, after):
            a, b = us[k]
            for m in vre.finditer(text, max(a, after), b):
                if in_eng(m.start(), m.end()):
                    continue
                left = text[a:m.start()]
                # the whole right side of the model's own `... = V`: not an operand or a summand (`= x*(V`, `+V`, `V*y`)
                if re.search(r'[=≡]\s*$', left) and ARITH.search(left[max(0, len(left) - 200):]) and not re.match(r'\s*[-+*/^×(]\s*[\d(]', text[m.end():m.end() + 4]):
                    return m.end()
            return None

        def run(k, start, kind):
            # forward from unit k: arithmetic or cue sentences, to the value's own result (closed) or the run's end (open)
            if not num:
                return None
            j, end = k, None
            while j < len(us) and j - k < MAX_UNITS:
                a, b = us[j]
                seg = text[max(a, start):b]
                r = result_in(j, start)
                if r:
                    return {'kind': kind, 'start': start, 'end': r, 'closed': True}
                if j > k and not (ARITH.search(seg) or CUE.search(seg)):
                    break
                end = b
                j += 1
            # open: only the sentences that still name one of the span's operands (else it is other work)
            end = start
            for i in range(k, min(j + 1, k + MAX_RUN + 1, len(us))):
                a, b = us[i]
                if i > k and not any(o in norm(text[a:b]) for o in ops):
                    break
                end = b
            return {'kind': kind, 'start': start, 'end': end, 'closed': False}

        # question: `?` right after the write (kept open or closed), or a cue opening the next sentence (kept if closed)
        tail = text[c['e']:c['e'] + 3]
        nxt = text[c['e']:us[k0 + 1][1]] if k0 + 1 < len(us) else text[c['e']:]
        q = tail.lstrip(' ').startswith('?')
        if q or re.match(r'[^A-Za-z]{0,4}(?:' + CUE.pattern + ')', nxt, re.I):
            ep = run(k0, c['e'], 'question')
            if ep and (q or ep['closed']):
                if not ep['closed'] and len(norm(text[ep['start']:ep['end']]).strip('?.!')) == 0:
                    ep['kind'] = 'doubt'  # a lone `?` after the value: questioned, nothing re-derived
                eps.append(ep); later = ep['end']
        span_eq = lambda body: re.fullmatch(r'.{0,40}?' + re.escape(span) + r'[A-Za-z]{0,10}[=≡][^\d]{0,12}' + re.escape(norm(v)), body)
        own = re.compile(r'(?:^|(?<=[A-Za-z(,:;=]))' + re.escape(span) + r'[A-Za-z]{0,10}[=≡]') if len(span) >= 3 else None
        for k in range(k0 + 1, len(us)):
            a, b = us[k]
            if b <= later:
                continue
            seg = text[a:b]
            hit = None
            if own and own.search(norm(masked[a:b])):  # the span on its own, then `=`: the model restating `S = ...` by hand
                restated = [x for x in cells if x is not c and x['s'] >= a and x['s'] < b and norm(x.get('span') or '') == span]
                if not restated:  # the same span through another engine write is the engine's, not a hand re-check
                    hit = 'restate'
            if not hit and long_ and vre.search(seg) and CUE.search(seg) and not all(in_eng(m.start() + a, m.end() + a) for m in vre.finditer(seg)):
                hit = 'cue'
            if hit:
                ep = run(k, a, hit)
                if ep and ep['closed']:
                    body = text[ep['start']:ep['end']]
                    bare = re.sub(r'\b[A-Za-z_]\w*\s*[=≡]', '', vre.sub('', body))  # `Good c0 = V`: an acknowledgement
                    if span_eq(norm(body)) or not ARITH.search(bare):
                        ep['kind'] = 'restate-only'  # `S = V` copied forward, no arithmetic of its own
                    eps.append(ep); later = ep['end']
        c['episodes'] = eps
    return cells


def arm_of(path, d, group):
    if group:
        m = group.search(os.path.basename(path))
        return m.group(1) if m else '(other)'
    for k in ('arm', 'variant'):  # measure and trace receipts name `arm`, S1's raw receipts `variant`
        if isinstance(d, dict) and isinstance(d.get(k), str):
            return d[k]
    return os.path.basename(os.path.dirname(path))


def main(out, roots):
    os.makedirs(out, exist_ok=True)
    group = re.compile(os.environ['ARM_GROUP']) if os.environ.get('ARM_GROUP') else None
    arms, seen = {}, set()
    with open(os.path.join(out, 'episodes.jsonl'), 'w') as eo, open(os.path.join(out, 'cells.jsonl'), 'w') as co, \
            open(os.path.join(out, 'replies.jsonl'), 'w') as ro:
        for root in roots:
            for dp, _, fs in os.walk(root):
                for f in sorted(fs):
                    if not f.endswith('.json'):
                        continue
                    p = os.path.join(dp, f)
                    try:
                        d = json.load(open(p))
                    except Exception:
                        continue
                    for field, text, counts, content, recs in replies(p):
                        if field != 'reasoning' or recs is None or (text, p) in seen:
                            continue
                        seen.add((text, p))
                        u = d.get('usage') or ((d.get('response') or {}).get('usage') if isinstance(d.get('response'), dict) else None) or {}
                        total = ((u.get('completion_tokens_details') or {}).get('reasoning_tokens')) or 0
                        if not total:
                            continue
                        tok = tokmap(text, recs, total)
                        cells = episodes(text, cells_of(text, recs), units(text))
                        spans, forced = [], [(c['s'], c['e'], c.get('landed') or 0) for c in cells]
                        for c in cells:
                            for ep in c['episodes']:
                                t = tok(ep['end']) - tok(ep['start']) - sum(n for s, e, n in forced if ep['start'] <= s and e <= ep['end'])
                                ep['tokens'] = max(0.0, t)
                                if ep['kind'] not in ('restate-only', 'doubt'):  # a bare `S = V` or a lone `?` is not a re-check
                                    spans.append((ep['start'], ep['end']))
                                eo.write(json.dumps({'path': p, 'arm': arm_of(p, d, group), 'cell': c.get('cell'), 'span': c.get('span'),
                                                     'value': c['value'], 'kind': ep['kind'], 'closed': ep['closed'],
                                                     'tokens': round(ep['tokens'], 1), 'start': ep['start'], 'end': ep['end'],
                                                     'before': text[max(0, c['s'] - 80):c['e']], 'text': text[ep['start']:ep['end']][:1500]}) + '\n')
                            co.write(json.dumps({'path': p, 'cell': c.get('cell'), 'span': c.get('span'), 'value': c['value'],
                                                 'episodes': len(c['episodes'])}) + '\n')
                        spans.sort()
                        union, cur = 0.0, None
                        for s, e in spans:
                            if cur and s <= cur[1]:
                                cur = (cur[0], max(cur[1], e))
                            else:
                                if cur:
                                    union += tok(cur[1]) - tok(cur[0])
                                cur = (s, e)
                        if cur:
                            union += tok(cur[1]) - tok(cur[0])
                        inside = sum(n for s, e, n in forced if any(a <= s and e <= b for a, b in spans))
                        a = arms.setdefault(arm_of(p, d, group), {'replies': 0, 'reasoning': 0, 'owned': 0, 'rechecked': 0, 'episodes': 0, 'tokens': 0.0, 'restate_only': 0.0, 'doubt': 0, 'owning': 0, 'owning_reasoning': 0})
                        a['replies'] += 1; a['reasoning'] += total; a['owned'] += len(cells)
                        real = [[ep for ep in c['episodes'] if ep['kind'] not in ('restate-only', 'doubt')] for c in cells]
                        a['doubt'] += sum(1 for c in cells for ep in c['episodes'] if ep['kind'] == 'doubt')
                        a['rechecked'] += sum(1 for r in real if r); a['episodes'] += sum(len(r) for r in real)
                        a['tokens'] += max(0.0, union - inside)
                        if cells:
                            a['owning'] += 1; a['owning_reasoning'] += total
                        ro.write(json.dumps({'path': p, 'arm': arm_of(p, d, group), 'reasoning': total, 'owned': len(cells),
                                             'recheck_tokens': round(max(0.0, union - inside), 1)}) + '\n')
                        a['restate_only'] += sum(ep['tokens'] for c in cells for ep in c['episodes'] if ep['kind'] == 'restate-only')
    L = ['| arm | replies | reasoning tokens | owned cells | re-checked cells | episodes | re-check tokens | restate-only tokens (not counted) | lone-`?` doubts (not counted) | share of reasoning | replies owning cells | share in those | tokens per owned cell |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    tot = {'replies': 0, 'reasoning': 0, 'owned': 0, 'rechecked': 0, 'episodes': 0, 'tokens': 0.0, 'restate_only': 0.0, 'doubt': 0, 'owning': 0, 'owning_reasoning': 0}
    for k in sorted(arms):
        a = arms[k]
        for x in tot:
            tot[x] += a[x]
        L.append(f"| {k} | {a['replies']} | {a['reasoning']} | {a['owned']} | {a['rechecked']} | {a['episodes']} | {a['tokens']:.0f} | {a['restate_only']:.0f} | {a['doubt']} | "
                 f"{100 * a['tokens'] / max(1, a['reasoning']):.2f}% | {a['owning']} | {100 * a['tokens'] / max(1, a['owning_reasoning']):.2f}% | "
                 f"{a['tokens'] / max(1, a['owned']):.1f} |")
    L.append(f"| **all** | {tot['replies']} | {tot['reasoning']} | {tot['owned']} | {tot['rechecked']} | {tot['episodes']} | {tot['tokens']:.0f} | {tot['restate_only']:.0f} | {tot['doubt']} | "
             f"{100 * tot['tokens'] / max(1, tot['reasoning']):.2f}% | {tot['owning']} | {100 * tot['tokens'] / max(1, tot['owning_reasoning']):.2f}% | "
             f"{tot['tokens'] / max(1, tot['owned']):.1f} |")
    open(os.path.join(out, 'report.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    return arms


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
