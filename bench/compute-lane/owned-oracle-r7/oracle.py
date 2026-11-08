#!/usr/bin/env python3
"""owned-oracle: check every value the in-model runner wrote into a reply against an independent evaluation.

  oracle.py extract  OUT ROOT...   receipts -> OUT/cells.jsonl   (text parsing only)
  oracle.py evaluate OUT           cells    -> OUT/evals.jsonl   (compute: run it in the capped, no-network container)
  oracle.py report   OUT           evals    -> OUT/report.md, OUT/mismatches.jsonl

A receipt whose runtime.runner has "cells" (TF_DSV41_RUNNER_CELLS, engine dsv41/imr-receipt) names every owned cell: its
parsed span, route and the bytes that landed at a byte offset. Those records are judged exactly, and every other value in
that reply is the model's. Receipts without them fall back to reconstructing cells from the text.

No engine code is imported. Inline and trace spans go through an exact integer/rational evaluator written here;
fence and REPL code runs in CPython (python3 -I) under rlimits inside the sandbox container (see run.sh).
"""
import ast, hashlib, json, os, re, subprocess, sys, tempfile
from fractions import Fraction
from math import comb, factorial, gcd, isqrt, lcm, perm

# ---------------------------------------------------------------- receipts

def replies(path):
    """(field, text, runner counts, final content, engine records or None) for each reply text in one receipt file."""
    try:
        d = json.load(open(path))
    except Exception:
        return []
    if not isinstance(d, dict):
        return []
    ex = d.get('extra') if isinstance(d.get('extra'), dict) else {}
    tf = d.get('tensorfold') or ex.get('tensorfold') or {}  # measure receipts: extra.tensorfold
    if isinstance(d.get('response'), dict) and d['response'].get('choices'):
        r = d['response']
        m = r['choices'][0].get('message') or {}
        tf = r.get('tensorfold') or tf
        d = {'reasoning': m.get('reasoning_content') or m.get('reasoning'), 'content': m.get('content')}
    run = tf.get('runner') if isinstance(tf.get('runner'), dict) else {}
    counts = {k: (run.get(k) or {}).get('owned', 0) for k in ('inline', 'trace', 'fence', 'repl')}
    recs = run.get('cells') if isinstance(run.get('cells'), list) else None
    out = []
    for field in ('reasoning', 'content'):
        t = d.get(field)
        if isinstance(t, str) and t:
            mine = None if recs is None else [r for r in recs if r.get('field') == field]
            out.append((field, t, counts, d.get('content') if field == 'reasoning' else None, mine))
    return out


CAPPED = ('length', 'cap', 'deadline')  # final receipts only: the token budget, or the server's own deadline (deadline_s)


def cap_info(path):
    """(capped, last, live): whether the reply may end inside its last record (a final receipt's length cap or server
    deadline, or live records only), (field, at) of its last engine record over both fields or None, and live: its
    records are the stream's live set (`live: true`, no final receipt). A live set's coverage of the client's text is
    unauthenticated (text and records are separate events), so a client close never verifies a cut: inconclusive."""
    try:
        d = json.load(open(path))
    except Exception:
        return False, None, False
    if not isinstance(d, dict):
        return False, None, False
    ex = d.get('extra') if isinstance(d.get('extra'), dict) else {}
    tf = d.get('tensorfold') or ex.get('tensorfold') or {}
    fins = [d.get('finish'), tf.get('finish')]
    if isinstance(d.get('response'), dict) and d['response'].get('choices'):
        r = d['response']
        fins.append(r['choices'][0].get('finish_reason'))
        tf = r.get('tensorfold') or tf
    run = tf.get('runner') if isinstance(tf.get('runner'), dict) else {}
    recs = [r for r in run.get('cells') or [] if isinstance(r, dict)]
    order = {'reasoning': 0, 'content': 1}
    last = max(recs, key=lambda r: (order.get(r.get('field'), 2), r.get('at') or 0), default=None)
    live = run.get('live') is True
    return any(f in CAPPED for f in fins) or live, (None if last is None else (last.get('field'), last.get('at'))), live


def locate(text, r):
    """Char offset of record r's landed bytes in text: (offset, 'exact' | 'shifted' | 'missing' | 'empty')."""
    tb = text.encode()
    v = bytes.fromhex(r['value_hex']) if 'value_hex' in r else r.get('value', '').encode()
    if not v:
        return None, 'empty'
    at = r.get('at', 0)
    if tb[at:at + len(v)] == v:
        pos, how = at, 'exact'
    else:
        hits = [m.start() for m in re.finditer(re.escape(v), tb)]
        if not hits:
            return None, 'missing'
        pos, how = min(hits, key=lambda h: abs(h - at)), 'shifted'
    return len(tb[:pos].decode('utf-8', 'ignore')), how


def record_cells(text, recs, cap=(False, None, False)):
    """One cell per engine record, with the reader's view around it (the expression left of `=`, earlier code).
    cap: the reply's (capped, last record's (field, at)); `capped_last` marks the record a length cap may have cut."""
    out = []
    blocks = block_cells(text)
    for r in recs:
        off, how = locate(text, r)
        value = bytes.fromhex(r['value_hex']).decode('utf-8', 'replace') if 'value_hex' in r else r.get('value', '')
        c = {'form': r.get('form'), 'source': 'record', 'offset': off, 'located': how, 'written': value,
             **{k: r.get(k) for k in ('cell', 'route', 'status', 'lang', 'span', 'span_sha256', 'span_len', 'display', 'at',
                                      'token', 'ids', 'landed', 'drafted', 'forced', 'path')}}
        c['capped_last'] = bool(cap[0]) and cap[1] == (r.get('field'), r.get('at'))
        c['live'] = bool(cap[2])
        if off is not None:
            start = text.rfind('\n', 0, off) + 1
            pre = text[start:off]
            eq = pre.rstrip().rfind('=')
            c['left'] = pre[seg_start(pre, eq):eq].strip() if eq >= 0 else ''
            c['line'] = text[start:text.find('\n', off) if '\n' in text[off:] else len(text)][:240]
            c['session'] = [b['code'] for b in blocks if b['form'] == 'repl' and b['offset'] <= off]
            if c['span'] is None and c.get('span_sha256'):
                for b in blocks:  # a long span: the reply's own copy, checked by its hash
                    if hashlib.sha256(b['code'].encode()).hexdigest() == c['span_sha256']:
                        c['span'] = b['code']
        out.append(c)
    return out

# ---------------------------------------------------------------- cells in a reply text

# no fractions: the engine writes integers or decimals, so `= 262144/2` is the write 262144 continued by `/2`
MARK = ' @engine'
MARK_W = re.compile(r'[ \t]?(-?\d+(?:\.\d+)?(?:/\d+)?) @engine(?![A-Za-z_])')
NUM_W = re.compile(r'[ \t]?(-?\d+(?:,\d{3})*(?:\.\d+)?)(?![\d.]\d)')
# the engine's trace line: `[exact: S = V]`, or the default since ce89c50a7, `[exact, computed by the engine: S = V]`
TRACE_OPEN = r'\[exact(?:,[^\]:=\n]*)?:'
TRACE = re.compile(TRACE_OPEN + r'\s*([^\]=]+?)\s*=\s*([^\]]+?)\]')
TRACE_ANY = re.compile(r'\[exact([^\]\n]*)\]')


def trace_parts(s):
    """(S, V) of the engine's trace line in s, any wording: TRACE, else `[exact…]` split at its last ` = ` (S after the
    last `:` before it), so a new attribution wording is judged, not left unparsed."""
    m = TRACE.search(s)
    if m:
        return m.group(1), m.group(2)
    m = TRACE_ANY.search(s)
    if not m or ' = ' not in m.group(1):
        return None
    left, v = m.group(1).rsplit(' = ', 1)
    return left.rsplit(':', 1)[-1].strip(), v.strip()


CHECK_MODULI = (9, 1000)


def check_exact(expr):
    """expr's exact value when it is one the opt-in trace check takes (decimal literals with + - * ^ and brackets, 1 to 4
    literals, each power a literal of at most 4, no unary minus), with its tokens; None otherwise."""
    if not re.fullmatch(r'[0-9 +\-*^()]+', expr):
        return None
    toks = re.findall(r'\d+|\S', expr)
    lits, prev = 0, None
    for i, t in enumerate(toks):
        if t.isdigit():
            if prev == '^':
                if int(t) > 4 or (i + 1 < len(toks) and toks[i + 1] == '^'):
                    return None  # a power past 4, or a chained one
            else:
                lits += 1
        elif t == '^' and (i + 1 >= len(toks) or not toks[i + 1].isdigit()):
            return None  # a power that is not a literal
        elif t == '-' and not (prev is not None and (prev.isdigit() or prev == ')')):
            return None  # unary minus
        prev = t
    if not 1 <= lits <= 4:
        return None
    try:
        v = eval(compile(ast.parse(re.sub(r'(?<!\d)0+(?=\d)', '', expr).replace('^', '**'), mode='eval'), '<check>', 'eval'), {'__builtins__': {}}, {})
    except Exception:
        return None
    return v if isinstance(v, int) else None


def check_shown(digits, r, m):
    """A residue as the check writes it: mod 1000 of a number of 3 or more digits keeps 3 digits (`012`)."""
    return f'{r:03d}' if m == 1000 and len(digits) >= 3 else str(r)


def check_text(display, value):
    """The opt-in trace check (runner_trace_check, S2 R7) the engine appends to `[exact…: display = value]`, computed here
    from the display and the value alone: ' check: L ≡ v (mod 9), L ≡ v (mod 1000)', '' when the line takes none, None
    when the value contradicts its own residues. Plain: L is the display with each literal (not a power) as its residue;
    divmod `A ÷ B = q remainder r`: L is `q*B + r` and v is A, an operand that is one literal keeping its digits' shape."""
    display, value = display.strip(), value.strip()
    dm = re.fullmatch(r'(\d+) remainder (\d+)', value)
    parts = []
    if dm:
        if ' ÷ ' not in display:
            return ''
        a, b = display.split(' ÷ ', 1)
        av, bv = check_exact(a), check_exact(b)
        if av is None or bv is None:
            return ''
        q, r = dm.groups()
        lone = lambda e, v, m: (lambda x: check_shown(x.group(1), v % m, m) if x else str(v % m))(re.fullmatch(r'\(?(\d+)\)?', e.strip()))
        for m in CHECK_MODULI:
            if (int(q) * bv + int(r) - av) % m:
                return None
            parts.append(f'{check_shown(q, int(q) % m, m)}*{lone(b, bv, m)} + {check_shown(r, int(r) % m, m)} ≡ {lone(a, av, m)} (mod {m})')
    else:
        dv = check_exact(display)
        if dv is None or not re.fullmatch(r'\d+', value):
            return ''
        for m in CHECK_MODULI:
            if (dv - int(value)) % m:
                return None
            lits, prev = [], None
            for t in re.findall(r'\d+|\D', display):
                lits.append(check_shown(t, int(t) % m, m) if t.isdigit() and prev != '^' else t)
                if t != ' ':
                    prev = t
            parts.append(f"{''.join(lits)} ≡ {check_shown(value, int(value) % m, m)} (mod {m})")
    return ' check: ' + ', '.join(parts)


def check_verdict(c):
    """A trace record's text after its `]`: None when there is none (the check is opt-in) or it is check_text of the
    line's own display and value, else 'wrong-check' (other text, a wrong residue, or a check where the line takes none)."""
    w = c['written'].rstrip('\n')
    got = w[w.index(']') + 1:] if ']' in w else ''
    if not got:
        return None
    m = trace_parts(c['written'])
    want = check_text(c.get('display') or (m[0] if m else ''), m[1] if m else '')
    c['check'], c['check_want'] = got, want
    return None if want and got == want else 'wrong-check'


def seg_start(line, end):
    """Where the visible left side of an `=` at `end` begins: after an earlier `=`, `:`, `;`, `?` or sentence end."""
    s = 0
    for m in re.finditer(r'(?<![=<>!])=(?!=)|[:;?]|\.\s+(?=[A-Z])', line[:end]):
        s = m.end()
    return s


def marked_inline_cells(text):
    """Explicit inline attribution, including text output fences; executable code fences remain excluded."""
    out, pos, fence = [], 0, None
    for raw in text.split('\n'):
        line = raw.strip()
        if line.startswith('```'):
            ticks = len(line) - len(line.lstrip('`'))
            if fence is None:
                lang = line[ticks:].strip().lower()
                fence = ('output' if lang in ('text', 'output', 'stdout') else 'code', ticks)
            elif ticks >= fence[1] and not line[ticks:].strip():
                fence = None
        elif fence is None or fence[0] == 'output':
            for eq in re.finditer(r'(?<![=<>!])=(?!=)', raw):
                w = MARK_W.match(raw, eq.end())
                if not w:
                    continue
                left = raw[seg_start(raw, eq.start()):eq.start()].strip().replace('{,}', ',')
                left = re.sub(r'(?<=[\d)])\s*mod\s*(?=[+-]?\s*[\d(])', ' % ', left, flags=re.I)
                if not re.search(r'\d', left):
                    continue
                out.append({'form': 'inline', 'offset': pos + w.start(1), 'left': left,
                            'written': w.group(1) + MARK, 'marked': True, 'line': raw[:240],
                            'chained': re.search(r'(?<![=<>!])=(?!=)', raw[:eq.start()]) is not None,
                            'continued': False, 'step': False})
        pos += len(raw) + 1
    return out


def marker_authority(text, cell, records):
    """Authenticate only the exact recorded suffix span; numeric equality or a nearby copy is not evidence."""
    if records is None:
        return 'model-imitation'  # no record names it: model text (run the arm with TF_DSV41_RUNNER_CELLS=1)
    raw = text.encode()
    end = len(text[:cell['offset'] + len(cell['written'])].encode())
    start = end - len(MARK.encode())
    covering = []
    for record in records:
        value = bytes.fromhex(record['value_hex']) if 'value_hex' in record else record.get('value', '').encode()
        at = record.get('at', 0)
        if raw[at:at + len(value)] != value:
            continue
        if at <= start and end <= at + len(value):
            covering.append(record)
            if record.get('form') == 'inline' and end == at + len(value) and value.endswith(MARK.encode()):
                return 'engine-inline' if record.get('landed') == record.get('ids') else 'engine-inline-partial'
    return 'engine-other-output' if covering else 'model-imitation'


def literal_marker_cells(text):
    # Catalog every literal cue, including prose, final answers and executable-code strings.
    return [{'form': 'inline', 'offset': hit.start(), 'written': MARK, 'marked': True}
            for hit in re.finditer(re.escape(MARK), text)]


def inline_cells(text):
    out = marked_inline_cells(text)
    pos = 0
    fence = False
    for raw in text.split('\n'):
        line = raw.replace('{,}', ',')  # LaTeX thousands separators
        if line.lstrip().startswith('```'):
            fence = not fence
        elif not fence and not re.search(TRACE_OPEN, line):
            for m in re.finditer(r'(?<![=<>!])=(?!=)', line):
                if MARK_W.match(line, m.end()):
                    continue  # explicit marked writes were extracted with their raw offsets above
                w = NUM_W.match(line, m.end())
                if not w or re.match(r'[A-Za-z_\d]', line[w.end():]):
                    continue
                # `= 4321*4000`, `= 1+1`: the right side goes on. A value the engine wrote can be continued by the
                # model (`φ(2^18)= 262144/2=131072`), so these stay, judged only for the engine's boundary signature
                continued = re.match(r'\s*(?:[-+*/^×÷·%(!\\]|mod\b)', line[w.end():]) is not None
                step = re.match(r'\s*=>', line[w.end():]) is not None
                left = line[seg_start(line, m.start()):m.start()]
                chained = re.search(r'(?<![=<>!])=(?!=)', line[:m.start()]) is not None
                if not re.search(r'\d', left) or not re.search(r'[-+*/^×÷·%!()]|\bmod\b|−', left):
                    continue  # no arithmetic on the left: a name or a restatement, never an inline cell
                out.append({'form': 'inline', 'offset': pos + w.start(1), 'left': left.strip(), 'written': w.group(1),
                            'line': line[:240], 'chained': chained, 'continued': continued, 'step': step})
        pos += len(raw) + 1
    for m in TRACE.finditer(text):
        if any(c.get('marked') and m.start(2) <= c['offset'] < m.end(2) for c in out):
            continue  # an INLINE mark inside a saved trace wrapper is not a second TRACE write
        out.append({'form': 'trace', 'offset': m.start(2), 'left': m.group(1), 'written': m.group(2).strip(), 'line': m.group(0)[:240]})
    return out


RESULT_LINE = re.compile(r'[ \t]*\**[A-Za-z][^\n]{0,40}:\**[ \t]*\n')


def block_cells(text):
    """REPL lines (`>>> code` then its output) and python fences followed by their output."""
    out = []
    lines = text.split('\n')
    starts = [0]
    for l in lines:
        starts.append(starts[-1] + len(l) + 1)
    session = []
    i = 0
    while i < len(lines):
        l = lines[i].strip()
        if l.startswith('>>> ') and not l.rstrip().endswith(':'):
            code = l[4:]
            outl = []
            j = i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith(('>>> ', '... ', '```')):
                outl.append(lines[j])
                j += 1
            session.append(code)
            if outl:
                out.append({'form': 'repl', 'offset': starts[i + 1], 'code': code, 'session': list(session),
                            'written': '\n'.join(outl), 'line': lines[i][:240]})
            i = j
            continue
        i += 1
    for m in re.finditer(r'```(?:python|py)[ \t]*\n(.*?)\n[ \t]*```[ \t]*\n', text, re.S):
        if m.group(1).lstrip().startswith('>>> '):
            continue  # a pasted REPL transcript: its `>>>` lines are the cells
        rest = text[m.end():]
        k = len(rest) - len(rest.lstrip('\n'))
        r = RESULT_LINE.match(rest, k)
        if r:
            k = r.end()
            k += len(rest[k:]) - len(rest[k:].lstrip('\n'))
        fenced = re.match(r'```[A-Za-z]*[ \t]*\n', rest[k:])
        if fenced:
            body = rest[k + fenced.end():]
            close = body.find('\n```')
            written = body[:close] if close >= 0 else None
            at = m.end() + k + fenced.end()
        else:
            if not r:
                continue  # prose after the fence, no result line: nothing was written as its output
            body = rest[k:]
            stop = body.find('\n\n')
            written = body[:stop] if stop >= 0 else body
            at = m.end() + k
        if written is None or not written.strip() or written.lstrip().startswith(('```', '>>>')):
            continue
        out.append({'form': 'fence', 'offset': at, 'code': m.group(1), 'written': written, 'line': text[m.start():m.start() + 240]})
    return out


def extract(out_dir, roots):
    seen = {}
    for root in roots:
        for dp, _, fs in os.walk(root):
            for f in sorted(fs):
                if f.endswith('.json'):
                    p = os.path.join(dp, f)
                    cap = cap_info(p)
                    for field, text, counts, content, recs in replies(p):
                        # one text with other counts or records (an off arm, an engine without records) is another
                        # reply, and so is one a length cap ended where a copy finished on its own (a cut is per receipt)
                        key = [counts, recs] if not cap[0] else [counts, recs, list(cap)]
                        h = hashlib.sha1((text + json.dumps(key, sort_keys=True)).encode()).hexdigest()
                        if h in seen:
                            seen[h]['paths'].append(p)
                        else:
                            seen[h] = {'paths': [p], 'field': field, 'text': text, 'counts': counts, 'content': content,
                                       'records': recs, 'cap': cap}
    n = 0
    with open(os.path.join(out_dir, 'cells.jsonl'), 'w') as o, open(os.path.join(out_dir, 'replies.jsonl'), 'w') as rj, open(os.path.join(out_dir, 'marks.jsonl'), 'w') as mj:
        for h, r in seen.items():
            text = r['text']
            cap = r.pop('cap')
            rj.write(json.dumps({'reply': h, **r}) + '\n')
            recs = r['records']
            for mark in literal_marker_cells(text):
                mj.write(json.dumps(dict(mark, authority=marker_authority(text, mark, recs), reply=h, field=r['field'], paths=r['paths'])) + '\n')
            cells = inline_cells(text) + block_cells(text)
            if recs is not None:
                # the engine names its writes: the text's other values are the model's (one at a record's place is the record)
                rc = record_cells(text, recs, cap)
                taken = [(c['offset'], c['offset'] + len(c['written'])) for c in rc if c['offset'] is not None]
                cells = rc + [dict(c, source='text') for c in cells
                              if not any(a - 2 <= c['offset'] <= b for a, b in taken)]
            for c in cells:
                at = (c['offset'] or 0) + len(c['written'])
                c.update(reply=h, field=r['field'], paths=r['paths'], counts=r['counts'], after=text[at:at + 160],
                         records=recs is not None)
                o.write(json.dumps(c) + '\n')
                n += 1
    live = sorted({p for r in seen.values() for p in r['paths'] if cap_info(p)[2]})
    json.dump({'replies': len(seen), 'cells': n, 'roots': roots, 'live': live}, open(os.path.join(out_dir, 'extract.json'), 'w'))
    print(f'replies {len(seen)}, cells {n}')

# ---------------------------------------------------------------- exact evaluator for written arithmetic

TOK = re.compile(r'\s*(?:(0[xX][0-9a-fA-F]+|\d{1,3}(?:,\d{3})+(?!\d)|\d+(?:\.\d+)?(?:e\d+)?)|(\*\*|[-+*/^×÷·⋅%!(),−–])|(mod\b)|([^\W\d][\w]*))', re.U)
NORM = [('\\times', '×'), ('\\cdot', '·'), ('\\bmod', ' mod '), ('\\pmod', ' mod '), ('\\mod', ' mod '), ('\\,', ''),
        ('$', ''), ('{', '('), ('}', ')'), ('\\left', ''), ('\\right', '')]
FUNCS = {'gcd', 'lcm', 'isqrt', 'sqrt', 'factorial', 'comb', 'C', 'perm', 'abs', 'min', 'max', 'pow', 'phi', 'φ', 'ϕ', 'totient'}


class Unknown(Exception):
    pass


def tokenize(s):
    """Tokens of the longest tokenizable tail of s (prose before an unknown character is dropped)."""
    for a, b in NORM:
        s = s.replace(a, b)
    s = re.sub(r'(?<=[\d)])\s*\*\*\s*(?=[\d(])', '^', s).replace('**', ' ')  # Python power, else markdown bold
    toks = []
    i = 0
    s = s.rstrip()
    while i < len(s):
        m = TOK.match(s, i)
        if not m or m.end() == i:
            toks = []
            i += 1
            continue
        num, op, mod, name = m.groups()
        if num is not None:
            toks.append(('n', num))
        elif op is not None:
            toks.append(('o', {'×': '*', '·': '*', '⋅': '*', '÷': '/', '−': '-', '–': '-', '^': '**'}.get(op, op)))
        elif mod:
            toks.append(('o', '%'))
        elif len(name) > 2 and name not in FUNCS:
            toks = []  # a prose word (`Then`, `verify`): the expression starts after it
        elif name not in FUNCS and re.match(r'\s+\(', s[m.end():]):
            toks = []  # `do (a+b)^2`: a short word before a spaced bracket is prose, not a call
        else:
            toks.append(('id', name))
        i = m.end()
    return toks


def number(t):
    t = t.replace(',', '')
    if t[:2] in ('0x', '0X'):
        return Fraction(int(t, 16))
    if 'e' in t:
        a, b = t.split('e')
        return Fraction(a) * 10 ** int(b)
    return Fraction(t)


def power(a, b):
    if b.denominator != 1:
        raise Unknown('fractional power')
    if abs(b) > 100000 or (abs(a) > 1 and abs(b) * max(1, len(str(abs(a.numerator)))) > 200000):
        raise Unknown('too large')
    return a ** int(b)


def totient(n):
    if n < 1:
        raise Unknown('phi domain')
    r, m, p = n, n, 2
    while p * p <= m:
        if p > 10 ** 7:
            raise Unknown('phi factor')
        if m % p == 0:
            while m % p == 0:
                m //= p
            r -= r // p
        p += 1
    if m > 1:
        r -= r // m
    return r


def call(name, args):
    ints = [int(a) for a in args if a.denominator == 1]
    if len(ints) != len(args):
        raise Unknown('non-int arg')
    if name in ('phi', 'φ', 'ϕ', 'totient') and len(ints) == 1:
        return totient(ints[0])
    if name == 'gcd':
        return gcd(*ints)
    if name == 'lcm':
        return lcm(*ints)
    if name in ('isqrt', 'sqrt') and len(ints) == 1 and ints[0] >= 0:
        r = isqrt(ints[0])
        if name == 'sqrt' and r * r != ints[0]:
            raise Unknown('irrational')
        return r
    if name == 'factorial' and len(ints) == 1 and 0 <= ints[0] <= 5000:
        return factorial(ints[0])
    if name in ('comb', 'C') and len(ints) == 2:
        return comb(*ints)
    if name == 'perm' and len(ints) == 2:
        return perm(*ints)
    if name == 'abs' and len(ints) == 1:
        return abs(ints[0])
    if name in ('min', 'max') and ints:
        return min(ints) if name == 'min' else max(ints)
    if name == 'pow' and len(ints) == 3 and ints[2] != 0 and ints[1] >= 0:
        return pow(*ints)
    if name == 'pow' and len(ints) == 2:
        return power(Fraction(ints[0]), Fraction(ints[1]))
    raise Unknown(f'call {name}')


class Parser:
    """Written arithmetic: + - * / % mod ^ ** ! ( ) and calls; ^ is power, `/` exact. Names parse but do not evaluate."""

    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self, kind, val=None):
        k, v = self.peek()
        if k == kind and (val is None or v == val):
            self.i += 1
            return v
        return None

    def full(self):
        v = self.sum()
        if self.i != len(self.t):
            raise SyntaxError
        return v

    def sum(self):
        v = self.term()
        while True:
            op = self.take('o', '+') or self.take('o', '-')
            if not op:
                return v
            r = self.term()
            v = lambda a=v, b=r, op=op: a() + b() if op == '+' else a() - b()

    def term(self):
        v = self.unary()
        while True:
            op = self.take('o', '*') or self.take('o', '/') or self.take('o', '%')
            if not op:
                return v
            r = self.unary()
            v = self.binop(op, v, r)

    def binop(self, op, a, b):
        if op == '*':
            return lambda: a() * b()
        if op == '/':
            return lambda: self.div(a(), b())
        def modulo():
            m = b()
            if m == 0 or m.denominator != 1:
                raise Unknown('mod')
            base = getattr(a, 'powpair', None)
            if base:  # a^e mod m without the full power
                x, e = base()
                if x.denominator == 1 and e.denominator == 1 and e >= 0:
                    return Fraction(pow(int(x), int(e), int(m)))
            return a() % m
        return modulo

    @staticmethod
    def div(x, y):
        if y == 0:
            raise Unknown('div0')
        return x / y

    def unary(self):
        if self.take('o', '-'):
            v = self.unary()
            return lambda: -v()
        if self.take('o', '+'):
            return self.unary()
        return self.power()

    def power(self):
        base = self.postfix()
        if self.take('o', '**'):
            e = self.unary()
            f = lambda: power(base(), e())
            f.powpair = lambda: (base(), e())
            return f
        return base

    def postfix(self):
        v = self.atom()
        while self.take('o', '!'):
            v = lambda a=v: Fraction(factorial(int(a()))) if a().denominator == 1 and 0 <= a() <= 5000 else (_ for _ in ()).throw(Unknown('!'))
        return v

    def atom(self):
        k, v = self.peek()
        if k == 'n':
            self.i += 1
            x = number(v)
            return lambda: x
        if self.take('o', '('):
            e = self.sum()
            if not self.take('o', ')'):
                raise SyntaxError
            return e
        if k == 'id':
            self.i += 1
            if self.take('o', '('):
                args = [self.sum()]
                while self.take('o', ','):
                    args.append(self.sum())
                if not self.take('o', ')'):
                    raise SyntaxError
                return lambda: Fraction(call(v, [a() for a in args]))
            return lambda: (_ for _ in ()).throw(Unknown(f'name {v}'))
        raise SyntaxError


def suffixes(text):
    """[(span text, token index, value or None)] for every complete expression ending at the `=`, longest first."""
    toks = tokenize(text)
    if not toks:
        return []
    out = []
    for i in range(len(toks)):
        if toks[i][0] == 'o' and toks[i][1] not in '(-':
            continue
        if toks[i] == ('o', '-') and i > 0 and (toks[i - 1][0] in ('n', 'id') or toks[i - 1] == ('o', ')')):
            continue  # a binary minus: its right side alone is no span
        try:
            f = Parser(toks[i:]).full()
        except (SyntaxError, IndexError):
            continue
        try:
            val = f()
        except (Unknown, ZeroDivisionError, ValueError, OverflowError):
            val = None
        out.append((render(toks[i:]), i, val))
    return out


def render(toks):
    return ''.join(v if k != 'o' else {'**': '^'}.get(v, v) for k, v in toks)


def written_value(w, inline_mark=False):
    if inline_mark and w.endswith(MARK):
        w = w[:-len(MARK)]
    try:
        return number(w) if '/' not in w else Fraction(w.replace(',', ''))
    except (ValueError, ZeroDivisionError):
        return None

# ---------------------------------------------------------------- evaluate

def run_python(code, repl_session=None, timeout=10):
    """CPython stdout for code (or for the last line of a REPL session), under rlimits, in an empty directory."""
    if repl_session is not None:
        prog = ('import sys\nlines=' + repr(repl_session) + '\nfor i,l in enumerate(lines):\n'
                '    last=i==len(lines)-1\n    if not last: sys.stdout=open("/dev/null","w")\n'
                '    else: sys.stdout=sys.__stdout__\n    try: exec(compile(l,"<stdin>","single"),globals())\n'
                '    except Exception as e:\n        if last: print(type(e).__name__+": "+str(e))\n')
    else:
        prog = code
    with tempfile.TemporaryDirectory() as d:
        try:
            p = subprocess.run(['prlimit', '--as=1073741824', '--cpu=5', '--nofile=64', '--nproc=64',
                                sys.executable, '-I', '-c', prog], cwd=d, capture_output=True, text=True,
                               timeout=timeout, env={'PATH': '/usr/bin:/bin', 'HOME': d})
            return p.returncode, p.stdout, p.stderr[-400:]
        except subprocess.TimeoutExpired:
            return None, '', 'timeout'


def block_output(v):
    """A fence/REPL write's program output: the header, closing fence and cut notes the engine adds taken off."""
    v = v[1:] if v.startswith('\n') else v
    v = v[len('Result:\n'):] if v.startswith('Result:\n') else v
    for tail in ('\n```\n', '\n```', '\n'):
        if v.endswith(tail):
            v = v[:-len(tail)]
            break
    return re.sub(r'\n\[(output cut|machine fault|stopped after \d+ steps)\]$', '', v)


DIVMOD_SPAN = re.compile(r'\s*divmod\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*')
DIVMOD_SHOWN = re.compile(r'\s*\(?(-?\d+)\)?\s*÷\s*\(?(-?\d+)\)?\s*')


def trace_own(c, echo=None):
    """(problem, value, display) of a trace record from its own span alone (T5's r6 OBJECT): value is the span's exact
    value (divmod: 'q remainder r'), None when the span is not one evaluated here; problem is 'wrong-span' when the
    record's display or the line's visible echo computes to anything else (divmod: other operands), else ''."""
    span, disp = c.get('span') or '', c.get('display')
    ab = DIVMOD_SPAN.fullmatch(span)
    if ab:
        a, b = int(ab.group(1)), int(ab.group(2))
        if not b:
            return '', None, disp
        q, r = divmod(a, b)
        for x in (disp, echo):
            m = DIVMOD_SHOWN.fullmatch(x) if x is not None else None
            if x is not None and (not m or (int(m.group(1)), int(m.group(2))) != (a, b)):
                return 'wrong-span', f'{q} remainder {r}', disp
        return '', f'{q} remainder {r}', disp or f'{a} ÷ {b}'
    sp = suffixes(span)
    own = sp[0][2] if sp and sp[0][1] == 0 else None
    if own is None:
        return '', None, disp
    for x in (disp, echo):
        sx = suffixes(x) if x is not None else None
        v = sx[0][2] if sx and sx[0][1] == 0 else None
        if v is not None and v != own:
            return 'wrong-span', own, disp
    return '', own, disp or span


def cut_write(c):
    """A length cap's cut of the reply's last record (fewer result tokens landed than the engine gave, at their offset):
    'cut' when the landed bytes are a byte prefix of the whole write its own span gives (computed here: the exact
    evaluator for inline and trace values, CPython for fence and REPL output), 'wrong-value' on any landed byte that
    differs, None when the whole write cannot be computed here (the record is then judged as any other, so it fails)."""
    w, span = c['written'], c.get('span') or ''
    if c['form'] == 'inline':
        sp = suffixes(span)
        own = sp[0][2] if sp and sp[0][1] == 0 else None
        if own is None or own.denominator != 1:
            return None  # the engine's rendering of a non-integer is not reproduced here
        c['span_value'] = str(own)
        vis = suffixes(c.get('left') or '')
        if vis and vis[0][2] is not None and vis[0][2] != own and len(vis[0][0]) > len(render(tokenize(span))):
            c['visible'], c['visible_value'], c['parsed_span'] = vis[0][0], str(vis[0][2]), span
            return 'wrong-span'  # the reader's expression is larger than the span: no prefix of the span's value is right
        whole = ' ' + str(own.numerator)
        return 'cut' if (whole + MARK).startswith(w) else 'wrong-value'
    if c['form'] == 'trace':
        # the whole line from the span alone (divmod: `a ÷ b = q remainder r`); a display that is not the span fails
        bad, own, disp = trace_own(c)
        if own is None or (not isinstance(own, str) and own.denominator != 1):
            return None
        val = own if isinstance(own, str) else str(own.numerator)
        c['span_value'] = val
        if bad:
            return bad
        shown = disp if isinstance(own, str) else render(tokenize(disp))  # divmod's echo is `a ÷ b`, never `a/b`
        if len(w) <= len('[exact'):
            return 'cut' if '[exact'.startswith(w) else 'wrong-value'
        t = w + ' ' if w.endswith(' =') else w  # cut between `=` and the value's space
        left, eq, value = t.rpartition(' = ') if ' = ' in t else (t, '', '')
        colon = left.rfind(':')  # the header ends at its last colon (`[exact (engine: graph) : S = V]` too)
        if colon < 0:
            return 'cut' if re.fullmatch(r'\[exact[^\]\n=]*', w) else 'wrong-value'  # cut inside the header
        if not re.fullmatch(r'\[exact[^\]\n=]*:', left[:colon + 1]):
            return 'wrong-value'
        echo = re.sub(r'\s+', '', left[colon + 1:])
        want = re.sub(r'\s+', '', shown)
        if not (want.startswith(echo) if not eq else echo == want):
            return None  # the engine's echo of its span is not one this tool renders: judged as any other record
        chk = check_text(disp, val)  # the line may go on with its opt-in check
        if chk is None:
            return 'wrong-value'  # the value contradicts its own display's residues: never approve the bare line
        tails = [']\n'] + ([']' + chk + '\n'] if chk else [])
        return 'cut' if any((val + t).startswith(value) for t in tails) else 'wrong-value'
    if c['form'] in ('fence', 'repl') and span:
        sess = None
        if c['form'] == 'repl':
            sess = list(c.get('session') or [])
            if not sess or sess[-1] != span:
                sess.append(span)
        rc, out, err = run_python(span, sess)
        c['code'], c['cpython_rc'], c['cpython_stdout'], c['cpython_stderr'] = span, rc, out, err
        if rc != 0 or not out:
            return None
        wholes = [pre + head + out + tail for pre in ('\n', '') for head in ('Result:\n', '') for tail in ('```\n', '')]
        return 'cut' if any(x.startswith(w) for x in wholes) else 'wrong-value'
    return None


def judge_record(c):
    """An engine record: its own span evaluated here, against the bytes that landed and the reader's expression."""
    span = c.get('span')
    if c.get('capped_last') and (c.get('landed') or 0) < (c.get('ids') or 0) and c['located'] == 'exact':
        v = cut_write(c)
        if v is not None:
            return 'inconclusive' if v == 'cut' and c.get('live') else v  # a live set may lag the text: never a cut
    if c['located'] != 'exact' and c['form'] == 'inline' and c['written'].rstrip().endswith(MARK):
        return 'not-in-text'  # never relocate a recorded mark onto an unrecorded imitation
    if c['located'] == 'missing':
        return 'not-in-text'  # the receipt says these bytes landed, the reply text does not have them
    if c['form'] in ('inline', 'trace'):
        if c['form'] == 'trace':
            # the value from the span alone; the display and the line's visible echo must compute to it (T5's r6 OBJECT)
            m = trace_parts(c['written'])
            bad, own, _ = trace_own(c, m[0] if m else None)
            if isinstance(own, str):  # `a ÷ b = q remainder r`: the engine's divmod line
                dm = re.fullmatch(r'(-?\d+)\s+remainder\s+(-?\d+)', m[1].strip()) if m else None
                c['span_value'] = own
                c['written_value'] = f'{dm.group(1)} remainder {dm.group(2)}' if dm else (m[1].strip() if m else None)
                if not m:
                    return 'unparsed'
                return 'wrong-value' if c['written_value'] != own else bad or check_verdict(c) or 'ok'
            w = written_value(m[1].strip()) if m else None
            sp = [('', 0, own)] if own is not None else []
        else:
            raw = c['written'].strip()
            if (c.get('landed') or 0) < (c.get('ids') or 0):
                for n in range(1, len(MARK)):
                    if raw.endswith(MARK[:n]) and written_value(raw[:-n]) is not None:
                        raw = raw[:-n]
                        break
            w = written_value(raw, inline_mark=True)
            sp = suffixes(span or '')
        own = sp[0][2] if sp and sp[0][1] == 0 else None
        vis = suffixes(c.get('left') or '') if c['form'] == 'inline' else []
        c['written_value'] = str(w) if w is not None else None
        c['span_value'] = str(own) if own is not None else None
        c['visible'] = vis[0][0] if vis else None
        c['visible_value'] = str(vis[0][2]) if vis and vis[0][2] is not None else None
        if w is None:
            return 'unparsed'
        if own is None:
            return 'unknown'
        if own != w:
            return 'wrong-value'  # the engine's own span has another value
        if c['form'] == 'trace' and bad:
            return bad  # right for its span, but the display or the visible echo is another computation
        if vis and vis[0][2] is not None and vis[0][2] != w and len(vis[0][0]) > len(render(tokenize(span or ''))):
            c['parsed_span'] = span
            return 'wrong-span'  # right for the span it parsed, but the reader's expression is larger (`mod 10^18`)
        return (check_verdict(c) if c['form'] == 'trace' else None) or 'ok'
    if not span:
        return 'unknown'  # a long span whose hash matched no fence in the reply
    sess = None
    if c['form'] == 'repl':
        sess = list(c.get('session') or [])
        if not sess or sess[-1] != span:
            sess.append(span)
    rc, out, err = run_python(span, sess)
    c['code'], c['cpython_rc'], c['cpython_stdout'], c['cpython_stderr'] = span, rc, out, err
    if rc is None or rc < 0 or (rc and 'NameError' in err):
        return 'unknown'  # a limit, or a name from an earlier cell (the notebook is not replayed)
    want = [l.rstrip() for l in out.rstrip('\n').split('\n')]
    got = [l.rstrip() for l in block_output(c['written']).split('\n')]
    return 'ok' if want == got else 'wrong-value'


def evaluate(out_dir):
    with open(os.path.join(out_dir, 'cells.jsonl')) as i, open(os.path.join(out_dir, 'evals.jsonl'), 'w') as o:
        for line in i:
            c = json.loads(line)
            if c.get('source') == 'record':
                c['verdict'] = judge_record(c)
                if c['verdict'] == 'ok' and (c.get('landed') or 0) < (c.get('ids') or 0):
                    c['verdict'] = 'short'  # fewer of the result's tokens landed than the engine gave, no cap cut it
            elif c['form'] in ('inline', 'trace'):
                w = written_value(c['written'], inline_mark=c['form'] == 'inline')
                sx = suffixes(c['left'])
                c['written_value'] = str(w) if w is not None else None
                c['spans'] = [(s, k, str(v) if v is not None else None) for s, k, v in sx]
                full = sx[0] if sx else None
                c['visible'] = full[0] if full else None
                c['visible_value'] = str(full[2]) if full and full[2] is not None else None
                if c['form'] == 'trace':
                    c['verdict'] = ('unparsed' if not full else 'unknown' if full[2] is None or w is None
                                    else 'ok' if full[2] == w else 'value')
                elif w is None or not sx:
                    c['verdict'] = 'unparsed'
                elif not c.get('marked') and not re.search(r'[-+*/^%!]|[A-Za-z]\(', full[0].lstrip('-')):
                    c['verdict'] = 'no-cell'  # a bare number left of `=`: no arithmetic for any trigger to own
                elif full[2] is not None and full[2] == w:
                    c['verdict'] = 'ok'
                else:
                    # a computed span only: the engine never writes a value after a bare literal, `(0)` included
                    sub = [(s, k, v) for s, k, v in sx[1:] if v is not None and v == w and re.search(r'[-+*/^%!]', s.lstrip('-('))]
                    if c.get('continued'):
                        # a continued right side counts only glued to `mod`/`%` or to a call's name (`φ(2^18)`)
                        toks = tokenize(c['left'])
                        sub = [x for x in sub if x[1] > 0 and (toks[x[1] - 1] == ('o', '%') or toks[x[1] - 1][0] == 'id')]
                        if not sub:
                            c['verdict'] = 'no-cell'
                            o.write(json.dumps(c) + '\n')
                            continue
                    if sub:
                        c['verdict'] = 'boundary'  # the written value is a strictly shorter span's: the engine's cell signature
                        c['parsed_span'] = sub[0][0]
                    elif full[2] is None:
                        c['verdict'] = 'unknown'
                    else:
                        c['verdict'] = 'value'
            else:
                rc, out, err = run_python(c['code'], c.get('session') if c['form'] == 'repl' else None)
                c['cpython_rc'], c['cpython_stdout'], c['cpython_stderr'] = rc, out, err
                want = [l.rstrip() for l in out.strip('\n').split('\n')] if out.strip() else []
                got = [l.rstrip() for l in c['written'].strip('\n').split('\n')]
                if rc is None or rc < 0:
                    c['verdict'] = 'unknown'  # timeout or a CPU/memory limit
                elif want and [l.strip() for l in got[:len(want)]] == [l.strip() for l in want]:
                    c['verdict'] = 'ok'  # the output's lines, then the model's own text
                else:
                    c['verdict'] = 'value'
            o.write(json.dumps(c) + '\n')

# ---------------------------------------------------------------- report

def used(c, reply):
    if c['offset'] is None:
        return {'reappears_later': 0, 'in_final_content': False, 'questioned': False}
    w = c['written'].strip() if c.get('source') == 'record' and c['form'] == 'inline' else c['written']
    plain = w.replace(',', '')
    grouped = f'{int(plain):,}' if re.fullmatch(r'-?\d+', plain) else w
    later = reply['text'][c['offset'] + len(w):]
    again = sum(len(re.findall(r'(?<![\d,])' + re.escape(x) + r'(?![\d,]\d)', later)) for x in {plain, grouped} if len(x) >= 2)
    content = reply.get('content') or ''
    final = any(re.search(r'(?<![\d,])' + re.escape(x) + r'(?![\d,]\d)', content) for x in {plain, grouped} if len(x) >= 3)
    q = re.match(r'\s*(\?|\.?\s*(Wait|Hmm|Actually|No[,.]|But))', c['after'])
    return {'reappears_later': again, 'in_final_content': final, 'questioned': bool(q)}


def owned(c):
    """trace: the engine's own line. inline `value`: matches no span, so not an exact engine value for any cell it can
    parse (the model's arithmetic, or an engine compute error the receipts cannot tell apart). Else by the form's count."""
    if c.get('source') == 'record':
        return 'engine (record)'
    if c.get('records'):
        return 'model (records: not owned)'  # the receipt names every owned cell, and this is none of them
    n = c['counts'].get(c['form'], 0)
    if c['form'] == 'trace':
        return 'engine'
    if c['form'] == 'inline' and c.get('marked'):
        return 'model (@engine imitation: no record)'  # a text-only mark proves nothing: the model can type it
    if not n:
        return 'model (form owned 0)'
    if c['form'] == 'inline' and ',' in c['written']:
        return 'model (grouped digits)'  # the engine writes plain digits
    if c['form'] == 'inline' and c.get('step'):
        return 'model? (=> step)'  # `a-b*c = d =>e`: the model's partial term before its own result
    if c['form'] == 'inline' and c.get('chained'):
        return 'model? (chained line)'  # the inline trigger refuses a cell with an earlier `=` on its line
    if c['form'] == 'inline' and c['verdict'] == 'value':
        return 'unexplained (model or engine)'
    return 'owned-form'


BAD = ('boundary', 'value', 'wrong-value', 'wrong-span', 'not-in-text', 'short', 'wrong-check')


def arms(rows, bad, texts, group):
    """Per-arm counts (ORACLE_GROUP: a regex whose first group, searched in a receipt's file name, names its arm)."""
    arm = lambda path: (lambda m: m.group(1) if m else '(other)')(group.search(os.path.basename(path)))
    owned, receipts, recorded = {}, {}, {}
    for r in texts.values():
        for p in r['paths']:
            if p not in receipts:  # a receipt's counts once, though its two fields are two texts
                a = arm(p)
                receipts[p] = a
                owned[a] = owned.get(a, 0) + sum(r['counts'].values())
                recorded[a] = recorded.get(a, False) or r.get('records') is not None
    keys = {'ok': 0, 'cut': 0, 'inconclusive': 0, 'unknown': 0}
    judged = {a: dict(keys) for a in owned}  # records mode: explicit verdicts, a receipt's record once per receipt
    for c in rows:
        if c.get('source') == 'record' and c['verdict'] in keys:
            for p in dict.fromkeys(c['paths']):
                judged.setdefault(arm(p), dict(keys))[c['verdict']] += 1
    keys = {'wrong-span': 0, 'wrong-value': 0, 'not-in-text': 0, 'short': 0, 'unparsed': 0, 'wrong-check': 0, 'unexplained': 0}
    tally = {a: dict(keys) for a in owned}
    for c in bad:
        t = tally.setdefault(arm(c['paths'][0]), dict(keys))
        o, v = c['owner'], c['verdict']
        if o == 'engine (record)':
            t[v] += 1
        elif o in ('owned-form', 'engine'):
            t['wrong-span' if v == 'boundary' else 'wrong-value'] += 1
        elif o == 'unexplained (model or engine)':
            t['unexplained'] += 1
    L = ['| arm | receipts | owned | correct | wrong-value | wrong-span | not-in-text | short | unparsed | wrong-check | cut | '
         'inconclusive | unknown | unexplained | mode |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for a in sorted(tally):
        t, n, j = tally[a], owned.get(a, 0), judged.get(a, keys)
        wrong = t['wrong-value'] + t['wrong-span'] + t['not-in-text'] + t['short'] + t['unparsed'] + t['wrong-check']
        mode = 'records' if recorded.get(a) else 'text'
        rec = lambda v: v if mode == 'records' else 'n/a'
        L.append(f"| {a} | {sum(1 for x in receipts.values() if x == a)} | {n} | {j['ok'] if mode == 'records' else n - wrong} | "
                 f"{t['wrong-value']} | {t['wrong-span']} | {rec(t['not-in-text'])} | {rec(t['short'])} | "
                 f"{rec(t['unparsed'])} | {rec(t['wrong-check'])} | {rec(j['cut'])} | {rec(j['inconclusive'])} | {rec(j['unknown'])} | {t['unexplained']} | {mode} |")
    L += ['', 'owned: the receipts\' runner owned counts. correct: records mode, the records judged `ok` (a `cut`, '
          '`unknown` or missing record is never counted correct); text mode, owned less wrong (not contradicted, as text '
          'reconstruction cannot see every owned cell). wrong-span: text mode\'s owned-form boundary. '
          'unexplained: an inline value in an owned reply that matches no span (read it).']
    return L


def report(out_dir):
    rows = [json.loads(l) for l in open(os.path.join(out_dir, 'evals.jsonl'))]
    texts = {r['reply']: r for r in map(json.loads, open(os.path.join(out_dir, 'replies.jsonl')))}
    ex = json.load(open(os.path.join(out_dir, 'extract.json')))
    bad = []
    for c in rows:
        if c['verdict'] in BAD or (c['verdict'] == 'unparsed' and c.get('source') == 'record'):
            c['owner'] = owned(c)
            c.update(used(c, texts[c['reply']]))
            bad.append(c)
    tally = {}
    for c in rows:
        key = (c['form'], c['verdict'], c['owner'] if 'owner' in c else '')
        tally[key] = tally.get(key, 0) + 1
    with open(os.path.join(out_dir, 'mismatches.jsonl'), 'w') as o:
        for c in bad:
            o.write(json.dumps({k: c.get(k) for k in ('form', 'verdict', 'owner', 'paths', 'field', 'offset', 'visible', 'visible_value',
                                                       'parsed_span', 'written', 'code', 'cpython_stdout', 'line', 'after',
                                                       'reappears_later', 'in_final_content', 'questioned', 'counts',
                                                       'source', 'span', 'route', 'path', 'located', 'at', 'token', 'ids',
                                                       'landed', 'drafted', 'forced', 'span_value')}) + '\n')
    rank = {'engine (record)': 0, 'model (records: not owned)': 6, 'engine': 0, 'owned-form': 1, 'unexplained (model or engine)': 2, 'model? (=> step)': 3, 'model? (chained line)': 3,
            'model (grouped digits)': 4, 'model (form owned 0)': 5, 'model (@engine imitation: no record)': 6}
    sev = lambda c: (rank[c['owner']], c['verdict'] not in ('boundary', 'wrong-span'), c['form'])
    bad.sort(key=sev)
    L = [f"# owned-oracle report", '', f"Replies (unique texts): {ex['replies']}; cells checked: {len(rows)}.", '',
         '| form | verdict | owner | cells |', '|---|---|---|---:|']
    for (f, v, ow), n in sorted(tally.items()):
        L.append(f'| {f} | {v} | {ow} | {n} |')
    L += ['', 'Verdicts: ok = written value equals the independent value of the visible expression (inline), the span (trace) '
          'or CPython stdout (fence/repl). boundary = the value of a strictly shorter span glued inside a larger expression '
          '(the engine-cell signature: `3^400 mod 10^18 = 10^18`, `φ(2^18) = 2^18`). value = a different value. '
          'unknown/unparsed = not decidable here. owner: engine (trace form), owned-form (the reply owned cells of this form), '
          'model (this form owned nothing in the reply, so the text is the model\'s).', '']
    recs = sum(1 for c in rows if c.get('source') == 'record')
    L.insert(3, f"Engine records (runtime.runner.cells): {recs} owned cells judged exactly; replies without records "
                f"fall back to text reconstruction.")
    L += ['Record verdicts: wrong-value = the engine\'s own parsed span has another value; wrong-span = right for the span '
          'it parsed, but the reader\'s expression around it is larger; not-in-text = the landed bytes are not in the '
          'reply; short = every byte of the value landed but fewer result tokens than the engine gave, with no length cap; '
          'cut (counted, never a verified value, not listed here) = the length cap ended the reply inside its last write '
          'and the landed bytes are a prefix of the whole write; unparsed = the landed bytes hold no value this '
          'oracle can read, so the write is unchecked (fails the gate).', '']
    for c in bad:
        if c.get('source') == 'record':
            L.append(f"- **{c['verdict']}** [{c['form']} record, {c['route']}, {c['path']}] span {c['span']!r} "
                     f"= {c.get('span_value') or c.get('cpython_stdout', '')!r}; wrote {c['written'][:80]!r}; "
                     f"reader `{c.get('visible')}` = {c.get('visible_value')}")
            L.append(f"  - at {c['paths'][0]} ({c['field']} byte {c['at']}, token {c['token']}, located {c['located']}; "
                     f"landed {c['landed']}/{c['ids']}, drafted {c['drafted']})")
            L.append(f"  - model: reappears later {c['reappears_later']}x, in final content {c['in_final_content']}, "
                     f"questioned {c['questioned']}")
            continue
        what = (f"`{c['visible']}` = {c['visible_value']}; wrote {c['written']}; span {c.get('parsed_span')}"
                if c['form'] in ('inline', 'trace') else f"code {c['code'][:120]!r}; wrote {c['written'][:80]!r}; CPython {c['cpython_stdout'][:80]!r}")
        L.append(f"- **{c['verdict']}** [{c['form']}, {c['owner']}] {what}")
        L.append(f"  - at {c['paths'][0]} ({c['field']} offset {c['offset']}; {len(c['paths'])} receipt copies)")
        L.append(f"  - model: reappears later {c['reappears_later']}x, in final content {c['in_final_content']}, questioned "
                 f"{c['questioned']}; next {c['after'][:100]!r}")
    group = os.environ.get('ORACLE_GROUP')
    if group:
        L[4:4] = arms(rows, bad, texts, re.compile(group)) + ['']
    open(os.path.join(out_dir, 'report.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L[:len(tally) + 6]))


FAIL = ('wrong-value', 'wrong-span', 'not-in-text', 'unparsed', 'wrong-check')  # an unreadable owned write is unchecked: fail


def gate(out_dir):
    """Exit 1 on any wrong engine write: a record judged wrong-value/wrong-span/not-in-text, or (receipts without records)
    an owned mismatch, owner engine or owned-form. A record judged unparsed fails too (its value went unchecked).
    cut (a length cap's cut of the reply's last write, its landed bytes a verified prefix: never a verified value),
    short and unknown are counted, not failed. A receipt with only live records (a client close: its text may run past
    its newest record) cannot pass: exit 3 INCONCLUSIVE when nothing failed (its correct partial write: inconclusive)."""
    rows = [json.loads(l) for l in open(os.path.join(out_dir, 'evals.jsonl'))]
    bad = [json.loads(l) for l in open(os.path.join(out_dir, 'mismatches.jsonl'))]
    rec = [c for c in rows if c.get('source') == 'record']
    n = {v: sum(1 for c in rec if c['verdict'] == v) for v in ('ok',) + FAIL + ('cut', 'inconclusive', 'short', 'unknown')}
    ep = os.path.join(out_dir, 'extract.json')
    live = (json.load(open(ep)).get('live') or []) if os.path.exists(ep) else []
    text = [c for c in bad if c['owner'] in ('engine', 'owned-form')]
    wrong = [c for c in bad if c['owner'] == 'engine (record)' and c['verdict'] in FAIL] + text
    mp = os.path.join(out_dir, 'marks.jsonl')
    marks = [json.loads(l)['authority'] for l in open(mp)] if os.path.exists(mp) else []
    m = {a: marks.count(a) for a in ('engine-inline', 'engine-inline-partial', 'engine-other-output', 'model-imitation')}
    word = 'FAIL' if wrong else 'INCONCLUSIVE' if live or n['inconclusive'] else 'PASS'
    g = {'pass': word == 'PASS', 'verdict': word, 'live_receipts': live, 'records': len(rec), **n,
         'text_owned_mismatches': len(text), 'marks': m,
         'wrong': [{k: c.get(k) for k in ('verdict', 'form', 'span', 'visible', 'written', 'paths', 'field', 'at', 'offset')} for c in wrong]}
    json.dump(g, open(os.path.join(out_dir, 'gate.json'), 'w'), indent=1)
    for c in wrong:
        print(f"WRONG {c['verdict']} [{c['form']}] span {c.get('span') or c.get('parsed_span')!r} wrote {c['written'][:60]!r} "
              f"reader {c.get('visible')!r} at {c['paths'][0]} ({c['field']} {c.get('at') if c.get('at') is not None else c.get('offset')})")
    print(f"owned-oracle gate {word}: records {len(rec)} (" + ', '.join(f'{k} {v}' for k, v in n.items())
          + f"), text-mode owned mismatches {len(text)}" + (f"; @engine marks: engine {m['engine-inline']}, partial "
          f"{m['engine-inline-partial']}, program output {m['engine-other-output']}, model imitation {m['model-imitation']}" if marks else '')
          + (f"; live-only receipts {len(live)} (coverage unauthenticated)" if live else ''))
    sys.exit({'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 3}[word])


if __name__ == '__main__':
    cmd, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    {'extract': lambda: extract(out, sys.argv[3:]), 'evaluate': lambda: evaluate(out), 'report': lambda: report(out),
     'gate': lambda: gate(out)}[cmd]()
