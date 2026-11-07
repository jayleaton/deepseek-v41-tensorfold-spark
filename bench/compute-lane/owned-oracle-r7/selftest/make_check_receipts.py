"""Writes selftest/receipts/*-check-*.json: trace records with S2 R7's opt-in check (runner_trace_check) after `]`, the
negative controls that must fail, and a model-written checked line with no record. Expected lines are worked by hand:
  python3 -I selftest/make_check_receipts.py"""
import json, os

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'receipts')
HEAD = '[exact, computed by the engine: '
# 123456789012 has digit sum 48 (3 mod 9), 987654321098 62 (8); 3*8 = 24 = 6 mod 9; 012*098 = 1176 = 176 mod 1000
PLAIN = ('123456789012*987654321098', '121932631136585886175176',
         ' check: 3*8 ≡ 6 (mod 9), 012*098 ≡ 176 (mod 1000)')
# 142*7 + 6 = 1000: mod 9, 142 = 7 and 1000 = 1 (7*7 + 6 = 55 = 1); mod 1000, 1000 keeps 3 digits as 000
DIVMOD = ('1000 ÷ 7', '142 remainder 6', ' check: 7*7 + 6 ≡ 1 (mod 9), 142*7 + 6 ≡ 000 (mod 1000)')


def receipt(name, lead, line, display, landed=None, finish='stop', ids=30, tail='Done.\n'):
    text = lead + line + tail
    rec = {'cell': 0, 'form': 'trace', 'route': 'graph', 'status': 'ok', 'lang': 'python', 'field': 'reasoning',
           'at': len(lead.encode()), 'token': 3, 'drafted': 0, 'path': 'held', 'span': display.replace('^', '**'),
           'display': display, 'value': line, 'ids': ids, 'landed': ids if landed is None else landed}
    rec['forced'] = rec['landed']
    doc = {'reasoning': text, 'content': '', 'finish': finish,
           'tensorfold': {'finish': 'done' if finish == 'stop' else finish, 'runner': {'trace': {'owned': 1}, 'cells': [rec]}}}
    json.dump(doc, open(os.path.join(HERE, name + '.json'), 'w'), indent=1, ensure_ascii=False)


def line(t, check=None):
    return HEAD + f'{t[0]} = {t[1]}]' + (t[2] if check is None else check) + '\n'


receipt('good-record-check-plain', 'Compute:\n', line(PLAIN), PLAIN[0])
receipt('good-record-check-divmod', 'Divide:\n', line(DIVMOD), DIVMOD[0])
receipt('good-record-check-off', 'Compute:\n', line(PLAIN, ''), PLAIN[0])  # the check is opt-in: none written is fine
# a cap inside the check: the landed bytes are a prefix of the whole line, so `cut`
cut_line = line(PLAIN)
receipt('cut-record-check', 'Compute:\n', cut_line[:cut_line.index('(mod 9)') + 3], PLAIN[0], landed=20, finish='length', tail='')
# negative controls
receipt('bad-record-check-residue', 'Compute:\n', line(PLAIN, ' check: 3*8 ≡ 6 (mod 9), 012*098 ≡ 177 (mod 1000)'), PLAIN[0])
receipt('bad-record-check-literal', 'Compute:\n', line(PLAIN, ' check: 3*8 ≡ 6 (mod 9), 12*98 ≡ 176 (mod 1000)'), PLAIN[0])
receipt('bad-record-check-divmod', 'Divide:\n', line(DIVMOD, ' check: 7*7 + 6 ≡ 1 (mod 9), 142*7 + 6 ≡ 0 (mod 1000)'), DIVMOD[0])
# 2^37 - 3^19: powers past 4 take no check, so any text after `]` is wrong
receipt('bad-record-check-none', 'Compute:\n', HEAD + '2^37 - 3^19 = 136276692005] check: 2^37 - 3^19 ≡ 2 (mod 9)\n', '2^37 - 3^19')
receipt('bad-record-capcut-check', 'Compute:\n', cut_line[:cut_line.index('(mod 9)') - 2] + '7 ', PLAIN[0], landed=18,
        finish='length', tail='')
# text mode: a reply with records whose model writes a checked trace line itself (wrong residues even) is not the engine's
lead = 'Compute:\n'
own = line(PLAIN)
model = HEAD + '7*8 = 56] check: 7*8 ≡ 3 (mod 9), 7*8 ≡ 56 (mod 1000)\n'
text = lead + own + 'Then by hand:\n' + model
rec = {'cell': 0, 'form': 'trace', 'route': 'graph', 'status': 'ok', 'lang': 'python', 'field': 'reasoning', 'at': len(lead.encode()),
       'token': 3, 'drafted': 0, 'path': 'held', 'span': PLAIN[0], 'display': PLAIN[0], 'value': own, 'ids': 30, 'landed': 30, 'forced': 30}
json.dump({'reasoning': text, 'content': '', 'tensorfold': {'runner': {'trace': {'owned': 1}, 'cells': [rec]}}},
          open(os.path.join(HERE, 'good-check-imitation.json'), 'w'), indent=1, ensure_ascii=False)
print('written')
