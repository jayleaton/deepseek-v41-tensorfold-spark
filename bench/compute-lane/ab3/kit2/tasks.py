#!/usr/bin/env python3
"""The §14 task sets, one shared prompt list for every arm, and their exact checks.

    python3 -I tasks.py build MBPP_JSON OUT.jsonl     writes the prompt set (deterministic: same seed, same file)
    python3 -I tasks.py selfcheck TASKS.jsonl         every checker accepts its reference answer and refuses a wrong one
    python3 -I tasks.py sample TASKS.jsonl first      the ids of a named sample (SAMPLES), the c=1 subset marked *

Sets (ids are stable; `set` names the group in reports):
  add30 / mul30      NUPA-style 30-digit integer add and multiply, 40 each
  mul_big / div_big  DeepMind mathematics arithmetic__mul_big / __div_big style, 40 each (decimals in mul_big)
  bignum             20 big-number reasoning tasks (multi-step, exact integers)
  mbpp               MBPP-40: sanitized task ids 11-92, the lane A/B's prompt and hidden asserts
  algo               40 stdin/stdout problems (algo.py), 12 hidden tests each, graded by the reference's output
  build              one falling-block game in one HTML file, graded by game_check.py's scripted checklist
Arithmetic answers are the last number in the reply's final content, compared exactly (commas and spaces in digit
groups allowed). Code answers are the last ```python block of the final content."""
import hashlib
import json
import math
import random
import re
import sys
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20261101
MBPP_IDS = (11, 92)
MBPP_INSTRUCTION = ("Write the Python function described below. You may run code to test it. "
                    "End your answer with the complete solution in a single ```python code block.")
ALGO_INSTRUCTION = ("Solve the programming problem below. Write a complete Python 3 program that reads standard input "
                    "and writes standard output; it must handle the largest inputs within a few seconds. End your answer "
                    "with the complete program in a single ```python code block.")
ALGO_TESTS = 12
# Expected completion tokens per reply by set (thinking uncapped), for scheduling (longest first) and window estimates:
# the runner-off arm's means in window 2 (2026-11-01, imr-measure-20261101T215640Z); build-01 runs to the 32K stop.
PRIOR_TOKENS = {'add30': 1550, 'mul30': 10120, 'mul_big': 430, 'div_big': 545, 'bignum': 4830, 'mbpp': 345,
                'algo': 780, 'build': 32768}
# Named samples: per-set counts drawn with a fixed seed (stratified; ids sorted, then random.Random(seed-set).sample),
# and a c=1 subset (drawn from the sample) for latency and decode tok/s. "all" is every task, no c=1 subset.
SAMPLES = {
    'first': {'seed': 20261102,
              'counts': {'add30': 8, 'mul30': 2, 'mul_big': 8, 'div_big': 6, 'bignum': 2, 'mbpp': 15, 'algo': 4,
                         'build': 1},
              'c1': {'add30': 2, 'mul_big': 2, 'div_big': 1, 'mbpp': 3}},
}


def sample(tasks, name):
    """(tasks of the named sample in file order, ids of its c=1 subset). `all` keeps everything."""
    if name in (None, '', 'all'):
        return list(tasks), set()
    spec = SAMPLES[name]
    by_set = {}
    for t in tasks:
        by_set.setdefault(t['set'], []).append(t['id'])
    keep, c1 = set(), set()
    for group, n in spec['counts'].items():
        ids = sorted(by_set.get(group, []))
        chosen = random.Random(f'{spec["seed"]}-{group}').sample(ids, min(n, len(ids)))
        keep.update(chosen)
        k = spec['c1'].get(group, 0)
        c1.update(random.Random(f'{spec["seed"]}-{group}-c1').sample(sorted(chosen), min(k, len(chosen))))
    return [t for t in tasks if t['id'] in keep], c1


def digits(rng, n):
    return rng.randint(10 ** (n - 1), 10 ** n - 1)


def dec_text(x):
    """A Decimal or int as plain text: no exponent, no trailing fractional zeros."""
    d = Decimal(x)
    t = format(d, 'f')
    if '.' in t:
        t = t.rstrip('0').rstrip('.')
    return '0' if t in ('-0', '') else t


def arithmetic(rng):
    out = []
    for i in range(40):
        a, b = digits(rng, 30), digits(rng, 30)
        out.append(('add30', f'What is {a} + {b}? Give the exact integer.', str(a + b)))
    for i in range(40):
        a, b = digits(rng, 30), digits(rng, 30)
        out.append(('mul30', f'What is {a} * {b}? Give the exact integer.', str(a * b)))
    for i in range(40):
        kind = rng.randrange(3)
        if kind == 0:
            a, b = digits(rng, rng.randint(6, 12)) * rng.choice((1, -1)), digits(rng, rng.randint(4, 9))
        elif kind == 1:
            a = Decimal(digits(rng, rng.randint(3, 7))).scaleb(-rng.randint(1, 4)) * rng.choice((1, -1))
            b = digits(rng, rng.randint(4, 8))
        else:
            a = Decimal(digits(rng, rng.randint(3, 6))).scaleb(-rng.randint(1, 3))
            b = Decimal(digits(rng, rng.randint(3, 6))).scaleb(-rng.randint(1, 3)) * rng.choice((1, -1))
        form = rng.choice(('Calculate {a}*{b}.', 'Multiply {a} and {b}.', 'What is {a} times {b}?',
                           'Product of {a} and {b}.'))
        out.append(('mul_big', form.format(a=dec_text(a), b=dec_text(b)) + ' Give the exact value.',
                    dec_text(Decimal(a) * Decimal(b))))
    for i in range(40):
        q = digits(rng, rng.randint(5, 12)) * rng.choice((1, -1))
        d = digits(rng, rng.randint(2, 8)) * rng.choice((1, -1))
        form = rng.choice(('Divide {x} by {d}.', 'What is {x} divided by {d}?', 'Calculate {x} divided by {d}.',
                           '{x} divided by {d}'))
        out.append(('div_big', form.format(x=q * d, d=d) + ' Give the exact value.', str(q)))
    return out


def bignum(rng):
    """Multi-step big-number questions; each truth is computed here, exactly."""
    out = []
    for i in range(20):
        k = i % 5
        if k == 0:
            a, e, m = digits(rng, 25), rng.randint(20, 60), digits(rng, 12)
            out.append((f'What is {a}^{e} mod {m}? Give the exact integer.', str(pow(a, e, m))))
        elif k == 1:
            b, e = rng.randint(2, 9), rng.randint(60, 140)
            out.append((f'What is the sum of the decimal digits of {b}^{e}?', str(sum(map(int, str(b ** e))))))
        elif k == 2:
            p, q = rng.randint(80, 160), rng.randint(30, 70)
            out.append((f'Compute 2^{p} - 3^{q} exactly.', str(2 ** p - 3 ** q)))
        elif k == 3:
            n = rng.randint(25, 40)
            m = n - rng.randint(4, 9)
            out.append((f'What is {n}! / {m}! exactly?', str(math.factorial(n) // math.factorial(m))))
        else:
            a, b, c = digits(rng, 20), digits(rng, 18), digits(rng, 15)
            out.append((f'A warehouse holds {a} units. It ships {b} units every day for {c % 900 + 100} days, then '
                        f'receives {c} units. How many units does it hold now (negative means a shortfall)?',
                        str(a - b * (c % 900 + 100) + c)))
    return out


def mbpp(path):
    data = {t['task_id']: t for t in json.loads(Path(path).read_text())}
    tasks = [data[i] for i in sorted(data) if MBPP_IDS[0] <= i <= MBPP_IDS[1]]
    assert len(tasks) == 40, len(tasks)
    return tasks


def algo_tests(p, rng_seed):
    """The 12 tests of one problem: sizes from the problem, then small random ones; expected = the reference."""
    rng = random.Random(rng_seed)
    sizes = list(p['sizes']) + [rng.randint(1, 12) for _ in range(ALGO_TESTS - len(p['sizes']))]
    tests = []
    for n in sizes:
        inp = p['gen'](rng, n)
        tests.append((inp, p['solve'](inp)))
    return tests


BUILD_PROMPT = (Path(HERE / 'build_task.txt').read_text() if (HERE / 'build_task.txt').exists() else '')


def build(mbpp_path, out):
    rng = random.Random(SEED)
    rows = []
    for i, (group, prompt, truth) in enumerate(arithmetic(rng)):
        rows.append({'id': f'{group}-{i % 40 + 1:02d}', 'set': group, 'prompt': prompt,
                     'check': {'kind': 'number', 'truth': truth}})
    for i, (prompt, truth) in enumerate(bignum(rng)):
        rows.append({'id': f'bignum-{i + 1:02d}', 'set': 'bignum', 'prompt': prompt,
                     'check': {'kind': 'number', 'truth': truth}})
    for t in mbpp(mbpp_path):
        rows.append({'id': f'mbpp-{t["task_id"]}', 'set': 'mbpp',
                     'prompt': f'{MBPP_INSTRUCTION}\n\n{t["prompt"]}\nYour code should pass this test:\n{t["test_list"][0]}',
                     'check': {'kind': 'asserts', 'imports': t.get('test_imports', []), 'tests': t['test_list'],
                               'reference': t['code']}})
    import algo
    for p in algo.PROBLEMS:
        seed = int(hashlib.sha256(f'{SEED}-{p["id"]}'.encode()).hexdigest()[:8], 16)
        tests = algo_tests(p, seed)
        ex_in, ex_out = tests[0]
        prompt = (f'{ALGO_INSTRUCTION}\n\n{p["title"]}\n{p["statement"]}\n\nExample input:\n{ex_in.rstrip()}\n\n'
                  f'Example output:\n{ex_out}')
        rows.append({'id': p['id'], 'set': 'algo', 'prompt': prompt,
                     'check': {'kind': 'stdio', 'seed': seed, 'tests_sha256': tests_sha(tests)}})
    assert BUILD_PROMPT, 'build_task.txt is missing'
    rows.append({'id': 'build-01', 'set': 'build', 'prompt': BUILD_PROMPT, 'check': {'kind': 'game'}})
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'{len(rows)} tasks -> {out} sha256 {hashlib.sha256(Path(out).read_bytes()).hexdigest()}')


def tests_sha(tests):
    return hashlib.sha256(json.dumps(tests).encode()).hexdigest()


# ---- checks -------------------------------------------------------------------------------------------------------
NUM = re.compile(r'-?\d{1,3}(?:[,   ]\d{3})+(?:\.\d+)?(?!\d)|-?\d+(?:\.\d+)?|-?\.\d+')


BOXED = re.compile(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}')


def numbers(text):
    """Every number in the text, digit-group separators removed."""
    text = (text or '').replace('−', '-').replace('{,}', ',').replace('\\,', ',').replace('\\!', '')
    return [re.sub(r'[,   ]', '', n) for n in NUM.findall(text)]


def last_number(text):
    """The last number in the text with digit-group separators removed; None when there is none."""
    found = numbers(text)
    return found[-1] if found else None


def final_number(text):
    """Sign-aware final: the last \\boxed{} number, else the last number, negative if its last earlier mention is."""
    boxed = [n for b in BOXED.findall(text or '') for n in numbers(b)[-1:]]
    if boxed:
        return boxed[-1]
    found = numbers(text)
    if not found:
        return None
    last = found[-1]
    if not last.startswith('-'):
        same = [n for n in found[:-1] if n.lstrip('-') == last]
        if same and same[-1].startswith('-'):
            return same[-1]
    return last


def same_number(got, truth):
    try:
        return got is not None and Decimal(got) == Decimal(truth)
    except ArithmeticError:
        return False


def final_code(text):
    blocks = re.findall(r'```(?:python|py|python3)?[ \t]*\n(.*?)```', text or '', re.S)
    return blocks[-1] if blocks else None


def check(task, content):
    """{ok: bool, ...detail} for one reply's final content (the answer channel, not the reasoning)."""
    c = task['check']
    if c['kind'] == 'number':
        got, signed = last_number(content), final_number(content)
        return {'ok': same_number(got, c['truth']), 'got': got,
                'ok_signed': same_number(signed, c['truth']), 'got_signed': signed}
    if c['kind'] in ('asserts', 'stdio'):
        import sandbox
        code = final_code(content)
        if code is None:
            return {'ok': False, 'why': 'no python block'}
        if c['kind'] == 'asserts':
            st, out, err = sandbox.run(code + '\n\n' + '\n'.join(c['imports'] + c['tests']) + '\n', timeout_s=30)
            return {'ok': st == 'ok', 'why': None if st == 'ok' else (st + ' ' + err[-300:])}
        if isinstance(c.get('tests'), list):  # W9 rows (T5's frozen manifest) carry their tests
            tests = [(t['input'], t['output']) for t in c['tests']]
        else:
            import algo
            p = next(x for x in algo.PROBLEMS if x['id'] == task['id'])
            tests = algo_tests(p, c['seed'])
            if tests_sha(tests) != c['tests_sha256']:
                raise RuntimeError(f'{task["id"]}: regenerated tests differ from the task file '
                                   '(Python random changed?)')
        results = sandbox.run_batch(code, [inp for inp, _ in tests], timeout_s=10)
        passed = 0
        for i, ((inp, want), (st, out, err)) in enumerate(zip(tests, results)):
            if st != 'ok' or out.split() != want.split():
                return {'ok': False, 'passed': passed, 'why': f'test {i}: {st} {err[-200:]}'.rstrip()}
            passed += 1
        if passed < len(tests):
            return {'ok': False, 'passed': passed, 'why': 'stopped early'}
        return {'ok': True, 'passed': passed}
    if c['kind'] == 'game':
        import game_check
        return game_check.check(content)
    raise ValueError(c['kind'])


def reference_answer(task):
    """A reply that must pass (selfcheck and the fake server's oracle replies)."""
    c = task['check']
    if c['kind'] == 'number':
        t = c['truth']
        return f'The answer is **{t}**.'
    if c['kind'] == 'asserts':
        return f'```python\n{c["reference"]}\n```'
    if c['kind'] == 'stdio' and c.get('reference'):  # a standalone reference program (W9 rows)
        return f'```python\n{c["reference"]}\n```'
    if c['kind'] == 'stdio':
        main = f"\nimport sys\nprint(next(p for p in PROBLEMS if p['id'] == {task['id']!r})['solve'](sys.stdin.read()))\n"
        return f'```python\n{(HERE / "algo.py").read_text()}{main}```'
    if c['kind'] == 'game':
        return '```html\n' + (HERE / 'game_reference.html').read_text() + '\n```'
    raise ValueError(c['kind'])


def wrong_answer(task):
    c = task['check']
    if c['kind'] == 'number':
        t = c['truth']
        return f'The answer is {t[:-1] + str((int(t[-1]) + 1) % 10)}.'
    if c['kind'] == 'game':
        return '```html\n' + (HERE / 'game_reference.html').read_text().replace('const ROTATE_OK = true', 'const ROTATE_OK = false') + '\n```'
    return '```python\ndef f(*a):\n    return None\nprint(0)\n```'


def selfcheck(path):
    import sandbox
    ok, detail = sandbox.probe()
    print(f'jail probe: {"PASS" if ok else "FAIL"} ({detail})')
    if not ok:
        return False
    tasks = [json.loads(l) for l in open(path)]
    bad = 0
    for t in tasks:
        good, wrong = check(t, reference_answer(t)), check(t, wrong_answer(t))
        if not good['ok'] or wrong['ok']:
            bad += 1
            print('SELFCHECK FAIL', t['id'], good, wrong)
    print(f'selfcheck: {len(tasks) - bad}/{len(tasks)} tasks accept their reference and refuse a wrong answer')
    return bad == 0


if __name__ == '__main__':
    sys.path.insert(0, str(HERE))
    if sys.argv[1] == 'build':
        build(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'selfcheck':
        sys.exit(0 if selfcheck(sys.argv[2]) else 1)
    elif sys.argv[1] == 'sample':
        chosen, c1 = sample([json.loads(l) for l in open(sys.argv[2])], sys.argv[3])
        print(len(chosen), 'tasks,', len(c1), 'at c=1:', ' '.join(t['id'] + ('*' if t['id'] in c1 else '') for t in chosen))
