"""Writes selftest/receipts/{cut-record-*,live-record-*,bad-record-capcut-*}.json: a length cap ending a reply inside its last engine
record (owned-oracle `cut`), and the negative controls that must fail. Run once when the shapes change:
  python3 -I selftest/make_cut_receipts.py"""
import sys
import json, os
if __name__ == '__main__' and sys.argv[1:] == ['--help']:
    print(__doc__)
    raise SystemExit(0)

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'receipts')
MARK = ' @engine'


def build(parts):
    """parts: text strings and (record, landed text) pairs; returns (text, records with byte offsets)."""
    text, recs = '', []
    for p in parts:
        if isinstance(p, str):
            text += p
            continue
        r, landed = p
        recs.append(dict(r, field='reasoning', at=len(text.encode()), value=landed, route='graph', status='ok',
                         lang='python', drafted=0, path='held', token=len(recs) + 1, forced=r['landed']))
        text += landed
    return text, recs


def inline(cell, span, ids, landed):
    return {'cell': cell, 'form': 'inline', 'span': span, 'ids': ids, 'landed': landed}


def measure(name, parts, finish):
    """A measure-style receipt (reasoning + tensorfold at the top, `finish` beside them)."""
    text, recs = build(parts)
    doc = {'reasoning': text, 'finish': finish, 'tensorfold': {'finish': 'done' if finish == 'stop' else finish,
                                                               'runner': {'cells': recs}}}
    json.dump(doc, open(os.path.join(HERE, name + '.json'), 'w'), indent=1)


def openai(name, parts, finish):
    """An S1-style receipt: the OpenAI response with finish_reason, tensorfold in the response."""
    text, recs = build(parts)
    doc = {'key': name, 'request': {'max_tokens': 256}, 'response': {
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': None, 'reasoning_content': text},
                     'finish_reason': finish}],
        'tensorfold': {'runner': {'cells': recs}}}}
    json.dump(doc, open(os.path.join(HERE, name + '.json'), 'w'), indent=1)


FULL0 = (inline(0, '314159*271000', 5, 5), ' 85137089000')
FULL1 = (inline(1, '271828*300000', 5, 5), ' 81548400000')
LEAD = ['We compute pieces. 314159*271000 =', FULL0, '. Then 271828*300000 =', FULL1, '. And 271828*14000 =']
TRACE = {'cell': 0, 'form': 'trace', 'span': '2**37-3**19', 'ids': 12}
FENCE = {'cell': 0, 'form': 'fence', 'span': 'print(271828 * 14000)\n', 'ids': 4}
FENCE_TEXT = 'Run it:\n```python\nprint(271828 * 14000)\n```'

# a cap's cut, landed bytes a prefix of the whole write: `cut` (S1 032339Z's shape first: one landed space, then the
# model's own last token `3` outside the record)
openai('cut-record-s1', LEAD + [(inline(2, '271828*14000', 5, 1), ' '), '3'], 'length')
measure('cut-record-digits', LEAD + [(inline(2, '271828*14000', 5, 2), ' 380')], 'length')
measure('cut-record-mark', LEAD + [(inline(2, '271828*14000', 7, 6), ' 3805592000 @en')], 'cap')
measure('cut-record-trace', ['So ', (dict(TRACE, landed=9), '[exact, computed by the engine: 2^37 - 3^19 = 1362')], 'length')
measure('cut-record-trace-head', ['So ', (dict(TRACE, landed=3), '[exact, computed by the en')], 'length')
measure('cut-record-fence', [FENCE_TEXT, (dict(FENCE, landed=2), '\n38055')], 'length')

# a client deadline (measure.py's finish `timeout`) inside the last write: only the stream's live records (extra.tensorfold),
# whose coverage of the text is unauthenticated (T5's r3 OBJECT): its correct partial is `inconclusive`, gate rc 3
def live(name, parts, finish):
    text, recs = build(parts)
    doc = {'reasoning': text, 'finish': finish, 'extra': {'tensorfold': {'runner': {'live': True, 'cells': recs}}}}
    json.dump(doc, open(os.path.join(HERE, name + '.json'), 'w'), indent=1)


live('live-record-timeout', LEAD + [(inline(2, '271828*14000', 5, 2), ' 380')], 'timeout')
# the server's own deadline (deadline_s): finish_reason `deadline`, the final response with its receipt
openai('cut-record-deadline', LEAD + [(inline(2, '271828*14000', 5, 2), ' 380')], 'deadline')

# negative controls: each must fail the gate
measure('bad-record-capcut-wrong', LEAD + [(inline(2, '271828*14000', 5, 2), ' 381')], 'length')
measure('bad-record-capcut-sign', LEAD + [(inline(2, '271828*14000', 5, 1), ' -')], 'length')
measure('bad-record-capcut-notlast', ['First 271828*14000 =', (inline(0, '271828*14000', 5, 1), ' '), '3. Then 314159*271000 =',
                                      (inline(1, '314159*271000', 5, 5), ' 85137089000'), '.'], 'length')
measure('bad-record-capcut-nocap', LEAD + [(inline(2, '271828*14000', 5, 1), ' '), '3'], 'stop')
measure('bad-record-capcut-trace', ['So ', (dict(TRACE, landed=9), '[exact, computed by the engine: 2^37 - 3^19 = 1363')], 'length')
# T5's case: the record's span is shorter than the reader's expression (2*3 of 2*3*4): wrong-span, never a cut
measure('bad-record-capcut-span', ['Compute 2*3*4 =', (inline(0, '2*3', 4, 1), ' 6')], 'length')
live('bad-record-capcut-timeout', LEAD + [(inline(2, '271828*14000', 5, 2), ' 381')], 'timeout')
openai('bad-record-capcut-deadline', LEAD + [(inline(2, '271828*14000', 5, 2), ' 381')], 'deadline')
measure('bad-record-capcut-fence', [FENCE_TEXT, (dict(FENCE, landed=2), '\n38056')], 'length')

# a divmod trace line (`a ÷ b = q remainder r`, its opt-in check) cut anywhere, T5's r5 OBJECT: every prefix of the whole
# line is a cut, the same cut with one landed byte changed fails, a live set's correct prefix is inconclusive
DIVMOD = {'cell': 0, 'form': 'trace', 'span': 'divmod(1000,7)', 'display': '1000 ÷ 7', 'ids': 40}
DM_LINE = '[exact, computed by the engine: 1000 ÷ 7 = 142 remainder 6] check: 7*7 + 6 ≡ 1 (mod 9), 142*7 + 6 ≡ 000 (mod 1000)\n'
for tag, upto in (('quotient', '= 14'), ('remainder-word', '142 remai'), ('remainder', 'remainder 6'), ('check9', 'check: 7*7'),
                  ('mod9', '≡ 1 (mod'), ('check1000', '142*7 + 6 ≡ 00'), ('mod1000', '(mod 10')):
    cut = DM_LINE[:DM_LINE.index(upto) + len(upto)]
    measure(f'cut-record-divmod-{tag}', ['So ', (dict(DIVMOD, landed=3), cut)], 'deadline')
    bad = cut[:-1] + ('5' if cut[-1] != '5' else '4') if cut[-1].isdigit() else cut[:-2] + 'X' + cut[-1]
    measure(f'bad-record-capcut-divmod-{tag}', ['So ', (dict(DIVMOD, landed=3), bad)], 'deadline')
live('live-record-divmod', ['So ', (dict(DIVMOD, landed=3), DM_LINE[:DM_LINE.index('(mod 10') + 7])], 'deadline')

# T5's r6 OBJECT: a trace line's value comes from its span alone; a display or visible echo that is another computation
# fails, cut or complete (divmod operands, and a plain expression), and a check that contradicts its line never passes
WRONG_SHOWN = {'cell': 0, 'form': 'trace', 'span': 'divmod(1000,7)', 'display': '999 ÷ 7', 'ids': 40}
measure('bad-record-span-divmod-display-cut', ['So ', (dict(WRONG_SHOWN, landed=3),
                                                       '[exact, computed by the engine: 999 ÷ 7 = 142 remainder 6]')], 'deadline')
measure('bad-record-span-divmod-display-whole', ['So ', (dict(WRONG_SHOWN, landed=40),
                                                         '[exact, computed by the engine: 999 ÷ 7 = 142 remainder 6]\n')], 'stop')
measure('bad-record-span-divmod-echo-whole', ['So ', (dict(DIVMOD, landed=40),
                                                      '[exact, computed by the engine: 1001 ÷ 7 = 142 remainder 6]\n')], 'stop')
measure('bad-record-span-plain-echo-whole', ['So ', (dict(TRACE, landed=12),
                                                     '[exact, computed by the engine: 2^37 - 3^18 = 136276692005]\n')], 'stop')
measure('bad-record-span-plain-display-whole', ['So ', (dict(TRACE, display='2^37 - 3^18', landed=12),
                                                        '[exact, computed by the engine: 2^37 - 3^18 = 136276692005]\n')], 'stop')
measure('good-record-span-divmod-whole', ['So ', (dict(DIVMOD, landed=40), DM_LINE)], 'stop')
print('written')
