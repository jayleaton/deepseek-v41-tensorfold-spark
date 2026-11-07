"""Where a reply's runner results sit, and what they should be: a client-side port of TE's scanner (runner_scan.zig at
f7cf2f67e) for reports, never for serving. Cells are numbered in close order, as runner.zig numbers them.

fence:  a ```lang block (empty tag allowed, result tags not) closed by a bare ``` line; a header line ("Result:") within
        the lookahead after it triggers it; the result starts after the header line, or after "Result: " when the
        model goes on writing on that line (T5's partial header).
repl:   a `>>> code` line (not ending in ':'); it triggers at once; the result is the next line.
inline: a line ending at `=` or `= ` whose tail is one whole arithmetic expression (inlineExpr's rules, including the
        range refusal after a word); only the value is owned, after one space the runner adds if the model wrote none.
Text only: the runner sees token boundaries; here an inline cell also needs a number right after its `=` (where an
owned or written value sits), so `-1/4=(10-3)/12` is no cell. The report's exact-signature count shows mismatches."""
import os
import re
from fractions import Fraction

PARTIAL_HEADER = True  # T5's scanner fix: "Result: <value>" on one line triggers too (window 4 on)
LOOKAHEAD = 64 * 4  # runner LOOKAHEAD is 64 tokens; about 4 chars a token in these replies
RESULT_TAGS = {'output', 'text', 'console', 'stdout', 'plaintext', 'txt', 'result'}
HEADER_WORDS = {'result', 'results', 'output', 'outputs', 'it prints', 'this prints', 'prints', 'running it gives',
                'running this gives', 'expected output', 'the output', 'this outputs', 'it outputs', 'the result',
                'output is', 'running it', 'running this', 'executing it', 'executing this'}
VALUE = re.compile(r'-?\d[\d,]*(?:\.\d+)?(?:/\d+)?')
EXPR_CHARS = set('0123456789 \t+-*/^().×÷−')


def indent_ok(line):
    t = line.lstrip(' ')
    return t if len(line) - len(t) <= 3 else None


def fence_open(line):
    t = indent_ok(line)
    if t is None or not t.startswith('```'):
        return None
    tag = t[3:].strip(' \t')
    if not all(c.isascii() and (c.isalnum() or c in '+#-_.') for c in tag) or tag.lower() in RESULT_TAGS:
        return None
    return tag.lower()


def fence_close(line):
    t = indent_ok(line)
    return t is not None and t.rstrip(' \t') == '```'


def header(line):
    t = line.strip(' \t*#->_')
    if len(t) < 2 or len(t) > 60 or t[-1] != ':':
        return False
    return t[:-1].strip(' \t*_').lower() in HEADER_WORDS


def partial_header(line):
    """Where a header's result starts when the model goes on writing on the header's own line ("Result: 2059…"), or
    None. T5's blank-delimited partial header: a header word, ':', then a blank; the runner owns from after the blank."""
    k = line.find(':')
    if k < 0 or line[k + 1:k + 2] not in (' ', '\t') or not line[k + 2:].strip():
        return None
    return k + 2 if header(line[:k + 1]) else None


def repl_line(line):
    t = indent_ok(line)
    if t is None or not t.startswith('>>> '):
        return None
    src = t[4:].strip(' \t')
    return None if not src or src.endswith(':') else src


def inline_expr(line):
    """(expr, space) when the line ends at `=` or `= ` after one whole expression, else None (inlineExpr)."""
    t, space = line, True
    if t.endswith(' '):
        t, space = t[:-1], False
    if len(t.encode()) < 4 or not t.endswith('='):
        return None
    before = t[:-1]
    if any(c in before for c in '=<>!'):
        return None
    start = len(before)
    while start > 0 and before[start - 1] in EXPR_CHARS:
        start -= 1
    lead, expr = before[:start], before[start:].strip(' \t')
    if len(expr.encode()) < 3 or len(expr.encode()) > 120 or '..' in expr:
        return None
    if lead:
        stop = lead[-1]
        if stop in ',$%_' or stop.isascii() and stop.isalnum():
            if before[start] not in ' \t' or stop == ',':
                return None
        word = lead.rstrip(' \t')
        if word and word[-1].isascii() and word[-1].isalpha():
            if any(expr[k] == '-' and expr[k - 1].isdigit() and expr[k + 1].isdigit() for k in range(1, len(expr) - 1)):
                return None  # a range ("positions 12-20 ="), not a subtraction
    if not (expr[0].isdigit() or expr[0] in '(-'):
        return None
    if expr[0] == '-' and lead.rstrip(' \t') and not lead.rstrip(' \t').endswith(':'):
        return None
    if not (expr[-1].isdigit() or expr[-1] == ')'):
        return None
    depth, ops, prev, gap, i = 0, 0, None, False, 0
    while i < len(expr):
        c = expr[i]
        if c in ' \t':
            gap = True
            i += 1
            continue
        if c.isdigit() or c == '.':
            if prev == 'num' and gap or prev == 'close':
                return None
            prev = 'num'
        elif c == '(':
            if prev in ('num', 'close'):
                return None
            depth, prev = depth + 1, 'open'
        elif c == ')':
            depth -= 1
            if depth < 0 or prev in ('op', 'open'):
                return None
            prev = 'close'
        else:
            unary = c == '-' and prev in (None, 'open', 'op')
            if not unary:
                if prev not in ('num', 'close'):
                    return None
                ops += 1
            if c == '*' and i + 1 < len(expr) and expr[i + 1] == '*':
                i += 1
            prev = 'op'
        gap = False
        i += 1
    if depth or not ops:
        return None
    return expr, space


def exact(expr):
    """The exact value of an inline expression (Fraction), or None when it is not plain arithmetic. `^` is power, as
    exact inline reads it since TE bf2ab7ec5."""
    src = expr.replace('×', '*').replace('÷', '/').replace('−', '-').replace('^', '**')
    if not re.fullmatch(r'[\d \t.()+\-*/]+', src) or '//' in src:
        return None
    for m in re.finditer(r'\*\*\s*\(?\s*(-?[\d.]+)', src):
        if '.' in m.group(1) or abs(int(m.group(1))) > 4096:
            return None  # a fractional or huge exponent: not exact here
    src = re.sub(r'\d+(?:\.\d+)?', lambda m: f'Fraction("{m.group(0)}")', src)
    try:
        v = eval(src, {'__builtins__': {}, 'Fraction': Fraction})  # digits, operators and Fraction only
    except (ZeroDivisionError, SyntaxError, TypeError, OverflowError, ValueError):
        return None
    return v if isinstance(v, Fraction) else None


def number(text):
    """A reply's value text as a Fraction (commas dropped), or None."""
    m = VALUE.match(text.strip())
    if not m:
        return None
    try:
        return Fraction(m.group(0).replace(',', ''))
    except (ValueError, ZeroDivisionError):
        return None


# the think block's end ends a line (runner images with T5 94c65d209); T4_THINK_ENDS_LINE=0 for older images' windows
LINE_END = re.compile(r'\n|</think>' if os.environ.get('T4_THINK_ENDS_LINE', '1') != '0' else r'\n')


def lines(text):
    """(line, start, next line's start). The think block's end ends a line too, as the runner's scanner takes it
    (T5 94c65d209): an answer fence may open right after `</think>`."""
    pos = 0
    while True:
        m = LINE_END.search(text, pos)
        if not m:
            yield text[pos:], pos, len(text) + 1
            return
        yield text[pos:m.start()], pos, m.end()
        pos = m.end()


def cells(text, forms=('fence', 'repl')):
    """Every closed cell in close order: dict(index, form, lang, code, start, end, at, prior). `at` is where its result
    starts (None when no trigger follows); `prior` holds the earlier cells' code."""
    found, state, pending, after_end = [], 'prose', None, 0
    lang, code, start = '', [], 0
    for raw, line_start, nxt in lines(text):
        line_end = line_start + len(raw)
        line = raw.rstrip('\r')
        if state == 'fence':
            if fence_close(line):
                pending = {'form': 'fence', 'lang': lang, 'code': ''.join(code), 'start': start, 'end': line_end,
                           'at': None}
                found.append(pending)
                state, after_end = 'after', line_end + LOOKAHEAD
            else:
                code.append(line + '\n')
            continue
        if state == 'after' and line_start > after_end:
            state = 'prose'
        if state == 'after' and header(line):
            pending['at'] = min(nxt, len(text))
            state = 'prose'
            continue
        if state == 'after' and PARTIAL_HEADER and partial_header(line) is not None:
            pending['at'] = line_start + partial_header(line)
            state = 'prose'
            continue
        tag = fence_open(line) if 'fence' in forms else None
        if tag is not None:
            lang, code, start, state = tag, [], line_start, 'fence'
            continue
        src = repl_line(line) if 'repl' in forms else None
        if src is not None:
            found.append({'form': 'repl', 'lang': 'python', 'code': src, 'start': line_start, 'end': line_end,
                          'at': nxt if line_end < len(text) else None})
            state = 'prose'
            continue
        if 'inline' in forms and '=' in line:
            k = line.index('=')
            cut = k + 2 if line[k + 1:k + 2] == ' ' else k + 1
            x = inline_expr(line[:cut])
            # a value follows a real trigger (owned, or the model's own); `=(` or `=` then prose came in one token
            if x is not None and VALUE.match(line[cut:].lstrip(" ")):
                found.append({'form': 'inline', 'lang': 'python', 'code': x[0], 'start': line_start,
                              'end': line_start + cut, 'at': line_start + cut})
    for i, c in enumerate(found):
        c['index'] = i
        c['prior'] = [p['code'] for p in found[:i]]
    return found


def value_at(text, at, form):
    """The reply's text at a result position: the value (inline), the line (repl) or up to a fence/blank line."""
    rest = text[at:]
    if form == 'inline':
        rest = rest[1:] if rest.startswith(' ') else rest  # the space the runner adds before a value
        m = VALUE.match(rest)
        return m.group(0) if m else rest[:24]
    if form == 'repl':
        return rest.split('\n', 1)[0]
    end = min([i for i in (rest.find('```'), rest.find('\n\n')) if i >= 0] or [len(rest)])
    return rest[:end].rstrip('\n')


NOTEBOOK = '''import contextlib, io
for _src in {prior!r}:
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            exec(compile(_src, "<prior>", "exec"))
        except BaseException:
            pass
'''


def truth(cell):
    """What the cell's result must be: ('text', s) for fence/repl (CPython's stdout), ('value', Fraction) for inline,
    or None when it cannot be known here (another language, a run error)."""
    if cell['form'] == 'inline':
        v = exact(cell['code'])
        return None if v is None else ('value', v)
    if cell['lang'] not in ('python', 'py', 'python3', ''):
        return None
    import sandbox
    code = cell['code'] if cell['form'] == 'fence' else f'print(repr({cell["code"]}))'
    st, out, _ = sandbox.run(NOTEBOOK.format(prior=cell['prior']) + code, timeout_s=10)
    if st != 'ok':
        return None
    return 'text', out[:-1] if out.endswith('\n') else out


def right(t, text):
    """Does a result text match the truth? Inline compares the value exactly; fence/repl compare the text."""
    if t is None or text is None:
        return None
    if t[0] == 'value':
        return number(text) == t[1]
    want = t[1]
    return text.rstrip() == want.rstrip() or text.startswith(want) and (len(text) == len(want) or text[len(want)] == '\n')
