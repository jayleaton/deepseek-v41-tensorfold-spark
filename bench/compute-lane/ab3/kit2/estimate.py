#!/usr/bin/env python3
"""Window length for a sample and the window configs, from tasks.PRIOR_TOKENS and today's measured costs.

    python3 -I estimate.py TASKS.jsonl SAMPLE [window-configs.json]

Per arm: requests go out longest expected first over the arm's lanes (4, or 1 for :c1); a reply takes its set's prior
tokens at the per-reply decode rate plus 1 s of prefill. Rates: T6 census per-reply tok/s at 4 workers (big-number
37.8, algorithmic 29.6, build 30.8); at c=1 taken as 1.5x (unmeasured: the c=1 subset measures it). Fixed costs:
STOP + locks 3 min; boot to gates done 8 min (6.5-8.0 measured in today's efast2 runs); stop + cool between boots
4 min; decode check 2.5 min where configured; G restore 8 min. Replay: owned cells per reply (inline boot: 4 for
arithmetic sets, 1 otherwise; fence,repl boot: 0.3), 2 validity positions per reply on 60% of off replies; each is a
prefill of half the reply at 1500 tok/s plus 24 (inline) or 160 tokens, 4 at once. `worst` lets every mul30 and
bignum reply run to the 32768-token stop. Configured arm caps (cap_s, cap_c1_s) bound each arm."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tasks as T  # noqa: E402

ARITH = {'add30', 'mul30', 'mul_big', 'div_big', 'bignum'}
RATE4 = {s: (37.8 if s in ARITH else 30.0) for s in T.PRIOR_TOKENS}
C1_SPEEDUP = 1.5


def lanes(costs, n):
    ends = [0.0] * n
    for c in sorted(costs, reverse=True):
        i = ends.index(min(ends))
        ends[i] += c
    return max(ends) if costs else 0.0


def arm_seconds(tasks, c1, arm, worst):
    sub = [t for t in tasks if t['id'] in c1] if arm.endswith(':c1') else tasks
    n = 1 if arm.endswith(':c1') else 4
    speed = C1_SPEEDUP if n == 1 else 1.0
    costs = []
    for t in sub:
        tok = 32768 if worst and t['set'] in ('mul30', 'bignum') else T.PRIOR_TOKENS[t['set']]
        costs.append(tok / (RATE4[t['set']] * speed) + 1.0)
    return lanes(costs, n), sub


def replay_seconds(sub, arm, forms):
    total = 0.0
    for t in sub:
        tok = T.PRIOR_TOKENS[t['set']]
        if arm[0] == 'r':
            k = (4 if t['set'] in ARITH else 1) if 'inline' in forms else 0.3
            out = 24 if 'inline' in forms else 160
        elif arm.endswith(':c1'):
            continue
        else:
            k, out = 2 * 0.6, 24
        total += k * (tok / 2 / 1500 + out / RATE4[t['set']])
    return total / 4


def estimate(tasks, c1, configs, worst=False):
    rows, total = [], 3 * 60.0
    for i, cfg in enumerate(configs):
        m = cfg['measure']
        boot = 8 * 60.0 + (4 * 60.0 if i else 0) + (150.0 if m.get('decode_check') else 0)
        rows.append((cfg['name'], 'boot+gates+checks', boot))
        total += boot
        keep = set(m['sets'].split(',')) if m.get('sets') else None
        mine = [t for t in tasks if keep is None or t['set'] in keep]
        for arm in m['arms'].split(','):
            s, sub = arm_seconds(mine, c1, arm, worst)
            cap = m.get('cap_c1_s' if arm.endswith(':c1') else 'cap_s', 0)
            s = min(s, cap) if cap else s  # the arm cap bounds it: nothing starts after, in-flight is cut
            r = replay_seconds(sub, arm, m['forms'])
            rows.append((cfg['name'], f'{arm} ({len(sub)} tasks)', s + r))
            total += s + r
    rows.append(('-', 'G restore', 8 * 60.0))
    return rows, total + 8 * 60.0


def main():
    tasks = [json.loads(l) for l in open(sys.argv[1])]
    chosen, c1 = T.sample(tasks, sys.argv[2])
    configs = json.loads(Path(sys.argv[3] if len(sys.argv) > 3 else HERE / 'window-configs.json').read_text())
    tok = sum(T.PRIOR_TOKENS[t['set']] for t in chosen)
    print(f'sample {sys.argv[2]}: {len(chosen)} tasks ({len(c1)} at c=1), {tok / 1000:.0f}K expected tokens per c=4 arm')
    for label, worst in (('expected', False), ('worst (mul30/bignum to 32K)', True)):
        rows, total = estimate(chosen, c1, configs, worst)
        print(f'\n{label}:')
        for boot, part, sec in rows:
            print(f'  {boot:4} {part:28} {sec / 60:6.1f} min')
        print(f'  TOTAL {total / 60:.0f} min ({total / 3600:.2f} h)')


if __name__ == '__main__':
    main()
