"""Compare complete token IDs and preserve all cross-arm and within-arm consistency failures."""

import argparse
import json
from pathlib import Path


def difference(a, b):
    if a == b:
        return None
    i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return {'index': i, 'left_token': a[i] if i < len(a) else None,
            'right_token': b[i] if i < len(b) else None}


def load(folder):
    out = {}
    for p in sorted(folder.glob('*.json')):
        value = json.loads(p.read_text())
        if 'response' in value:
            out[value['case']] = value['response']
    assert len(out) == 24, 'incomplete suite'
    return out


def within(records):
    checks = []
    for name, r in records.items():
        root = name.removesuffix('-plain').removesuffix('-concurrent')
        if root == name:
            continue
        a = records[root]['tensorfold']['token_ids']
        b = r['tensorfold']['token_ids']
        checks.append({'case': name, 'kind': 'draft_plain' if name.endswith('-plain') else 'concurrent_solo',
                       'token_ids_equal': a == b, 'first_difference': difference(a, b)})
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('baseline', type=Path)
    ap.add_argument('candidate', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    aa, bb = load(args.baseline), load(args.candidate)
    assert aa.keys() == bb.keys()
    cases = []
    for name, a in aa.items():
        b = bb[name]
        x, y = a['tensorfold']['token_ids'], b['tensorfold']['token_ids']
        cases.append({'case': name, 'token_ids_equal': x == y, 'first_difference': difference(x, y),
                      'prompt_tokens': a['usage']['prompt_tokens'],
                      'baseline_prefill_s': a['tensorfold']['prefill_s'],
                      'candidate_prefill_s': b['tensorfold']['prefill_s'],
                      'baseline_ttft_s': a['tensorfold']['ttft_s'],
                      'candidate_ttft_s': b['tensorfold']['ttft_s']})
    internal = {'baseline': within(aa), 'candidate': within(bb)}
    failures = sum(not c['token_ids_equal'] for c in cases)
    internal_failures = sum(not c['token_ids_equal'] for checks in internal.values() for c in checks)
    responses = list(aa.values()) + list(bb.values())
    all_cold = all(r['tensorfold']['cached'] == 0 for r in responses)
    stops = sum(r['choices'][0]['finish_reason'] == 'stop' for r in responses)
    report = {'cross_arm_passed': len(cases) - failures, 'cross_arm_failed': failures,
              'within_arm_passed': 22 - internal_failures, 'within_arm_failed': internal_failures,
              'responses': len(responses), 'all_cold': all_cold, 'stop_completions': stops,
              'full_gate_pass': not failures and not internal_failures and all_cold and stops == 48,
              'comparisons': cases, 'within_arm_comparisons': internal}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('comparisons', 'within_arm_comparisons')}))
    return int(not report['full_gate_pass'])


if __name__ == '__main__':
    raise SystemExit(main())
