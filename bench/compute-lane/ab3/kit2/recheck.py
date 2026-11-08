#!/usr/bin/env python3
"""Re-check spans: reasoning a reply spends re-evaluating, doubting or re-running a result the engine already landed.
One deterministic policy for every arm (runner on or off), over the native cell records (VERSION below; W8's figures
came from v1, commit 5b90aee74):
  landed result  an `ok` cell in the reasoning field: its display (or span), the value's last number and the
                 display's operands
  trigger        a line (only the part after the write, for the write's own line) about a result landed before it:
                 cue    it refers to the result (its display of 4+ characters, or its value's 6+ digits; spaces,
                        commas and bars ignored) and carries a doubt/verify cue, or a `?` straight after those digits
                 rerun  it restates the landed operation: the display's operands in order, at most 6 non-digits
                        apart, leading zeros and 3-digit grouping commas ignored (only for 2+ operands with 5+ digits
                        in all, so small operations recurring in new work don't count)
  extension      the lines after a trigger while they compute (digits with an operator, `=` or `|` groups); a blank
                 line continues only into another trigger line; a new write with a display not landed before ends it
                 (new work, not a re-check)
v2 against v1, on W8's blind review (recheck_eval.py; development data, so optimistic): the rerun rule, and `ensure`
dropped as a cue (4 of its 10 flagged spans judged not a re-check); a stricter extension was tried and lost recall.
Spans are character ranges of the reasoning; a token counts once (by its end offset) however many spans hold it.
Usage: import recheck; recheck.spans(reasoning, cells) -> [{'start', 'end', 'trigger', 'cell', 'display', 'rule'}]"""
import re

VERSION = 'recheck-v2'
CUE = re.compile(r"\b(?:verif\w*|double[- ]?check\w*|re-?check\w*|cross-?check\w*|check(?:s|ed|ing)?|confirm\w*|"
                 r"re-?comput\w*|re-?calc\w*|re-?deriv\w*|re-?do|independent\w*|again|sanity|"
                 r"wrong|mistake|recall|wait|hmm|by hand|manual\w*|is (?:that|this|it) (?:right|correct)|"
                 r"compar\w*|let(?:'?s| me) (?:re-?)?(?:do|compute|calculate|check|count|recount))\b", re.I)
MIN_DIGITS, MIN_DISPLAY = 6, 4  # shorter results/displays recur inside unrelated numbers and new work
MIN_OPS, MIN_OP_DIGITS = 2, 5  # a restated operation must be distinctive enough not to recur by chance
NUMBER = re.compile(r'-?\d[\d,]*(?:\.\d+)?')
COMPUTE = re.compile(r'\d\s*[-+*/×x^=|<>]\s*\d|=\s*-?\d|\|\s*\d|\d\s*\||^\s*[-+*/=]')
GROUPED = re.compile(r'(?<=\d),(?=\d{3}(?!\d))')


def squash(s):
    return re.sub(r'[\s,_|]', '', s)


def plain(s):
    """Grouping commas out, so 785,370,248 reads as one number."""
    return GROUPED.sub('', s)


def landed(reasoning, cells):
    """The reasoning-field writes in order, with what a later line can refer to or restate."""
    raw, out = reasoning.encode(), []
    for c in cells or []:
        if c.get('field') != 'reasoning' or c.get('status') != 'ok' or c.get('at') is None or not c.get('value'):
            continue
        v = c['value'].encode()
        if raw[c['at']:c['at'] + len(v)] != v:
            continue
        s = len(raw[:c['at']].decode(errors='ignore'))
        nums = NUMBER.findall(c['value'].rsplit('=', 1)[-1]) or NUMBER.findall(c['value'])
        digits = squash(nums[-1]).lstrip('-') if nums else ''
        shown = c.get('display') or c.get('span') or ''
        ops = [o.lstrip('0') or '0' for o in re.findall(r'\d+', plain(shown))]
        rerun = None
        if len(ops) >= MIN_OPS and sum(map(len, ops)) >= MIN_OP_DIGITS:
            rerun = re.compile(r'\D{0,6}?'.join(rf'(?<!\d)0*{o}(?!\d)' for o in ops))
        out.append({'start': s, 'end': s + len(c['value']), 'display': squash(shown), 'digits': digits,
                    'ops': ops, 'rerun': rerun, 'cell': c.get('cell')})
    return sorted(out, key=lambda w: w['start'])


def refers(text, w):
    t = squash(text)
    return ((len(w['digits']) >= MIN_DIGITS and w['digits'] in t)
            or (len(w['display']) >= MIN_DISPLAY and w['display'] in t))


def reruns(text, w):
    return w['rerun'] is not None and w['rerun'].search(plain(text)) is not None


def doubted(text, w):
    if CUE.search(text):
        return True
    if len(w['digits']) >= MIN_DIGITS:
        i = squash(text).find(w['digits'])
        return i >= 0 and squash(text)[i + len(w['digits']):i + len(w['digits']) + 2].startswith('?')
    return False


def trigger(reasoning, writes, s, e):
    """(write, rule, start) for the first landed result this line re-checks, else None."""
    for w in writes:
        if w['end'] > e:
            break
        a = max(s, w['end'])  # on the write's own line, only what follows it
        done = {x['display'] for x in writes if x['end'] <= a}
        b = min([x['start'] for x in writes if a <= x['start'] < e and x['display'] not in done] + [e])
        free = reasoning[a:b]
        if not free.strip():
            continue
        if refers(free, w) and doubted(free, w):
            return w, 'cue', a
        if reruns(free, w):
            return w, 'rerun', a
    return None


def spans(reasoning, cells):
    writes = landed(reasoning, cells)
    if not writes:
        return []
    lines, pos = [], 0
    for part in reasoning.split('\n'):
        lines.append((pos, pos + len(part)))
        pos += len(part) + 1
    out, i = [], 0
    while i < len(lines):
        s, e = lines[i]
        hit = trigger(reasoning, writes, s, e)
        if hit is None:
            i += 1
            continue
        w, rule, start = hit
        seen = {x['display'] for x in writes if x['end'] <= start}
        end, j = e, i + 1
        while j < len(lines):
            a, b = lines[j]
            text = reasoning[a:b]
            if any(a <= x['start'] < b and x['display'] not in seen for x in writes):
                break
            if not text.strip():
                k = j + 1
                while k < len(lines) and not reasoning[lines[k][0]:lines[k][1]].strip():
                    k += 1
                if k < len(lines) and trigger(reasoning, writes, *lines[k]) is not None:
                    j = k
                    continue
                break
            if not COMPUTE.search(text):
                break
            end, j = b, j + 1
        out.append({'start': start, 'end': end, 'trigger': reasoning[start:min(e, start + 160)], 'cell': w['cell'],
                    'display': w['display'], 'rule': rule})
        i = j
    return out


def tokens_in(spans_, ends):
    """How many tokens (by end offset) fall inside any span: each token once."""
    return sum(any(s['start'] < x <= s['end'] for s in spans_) for x in ends)
