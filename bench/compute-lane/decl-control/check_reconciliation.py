#!/usr/bin/env python3
"""Fake responses only. No network, model or code-execution calls."""
import copy
import hashlib
import json
from pathlib import Path
from reconciliation import compare_declared_templates, reconcile


def main():
    root = Path(__file__).resolve().parent
    draft = json.loads((root/'DRAFT-requests-NOT-QUALIFIED.json').read_text())
    prompts = {tuple(r['key']): r for r in map(json.loads, (root/'prompt-arrays.jsonl').read_text().splitlines())}
    pairs = {}
    for template in draft['requests']:
        key = tuple(template['key'])
        pairs.setdefault(key, {})[template['arm']] = template
        field = {'plain':'plain_ids', 'declared_off':'declared_off_ids', 'lane':'declared_lane_ids'}[template['arm']]
        n = len(prompts[key][field])
        fake = {'actual_request_wire': template['wire'], 'usage': {'prompt_tokens': n},
                'sse_frames': [{'usage': {'prompt_tokens': n}}]}
        result = reconcile(fake, template, prompts[key])
        assert result['qualified_under_proposed_count_contract']
        assert result['served_array_equality_proven'] is False
        assert result['all_round_prompt_tokens'] is None
        bads = [dict(fake, usage=None), dict(fake, usage={'prompt_tokens': True}),
                dict(fake, usage={'prompt_tokens': -1}), dict(fake, usage={'prompt_tokens': n+1}),
                dict(fake, actual_request_wire=None), dict(fake, actual_request_wire=template['wire']+' '),
                dict(fake, sse_frames=[{'usage': {'prompt_tokens': n+1}}])]
        for bad in bads:
            assert not reconcile(bad, template, prompts[key])['qualified_under_proposed_count_contract']
    for key, arms in pairs.items():
        assert compare_declared_templates(arms['declared_off'], arms['lane'], prompts[key])
    d, l = pairs[next(iter(pairs))]['declared_off'], pairs[next(iter(pairs))]['lane']
    altered = copy.deepcopy(l)
    altered['wire'] += ' '
    try:
        compare_declared_templates(d, altered, prompts[tuple(d['key'])])
    except ValueError:
        pass
    else:
        raise AssertionError('wire mismatch accepted')
    result = {'verdict':'PASS', 'fake_requests':552, 'paired_wire_checks':184,
              'adverse_response_controls': 7*552, 'network':False, 'amendment_approved':False,
              'collection_qualified':False,
              'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [root/'reconciliation.py', root/'check_reconciliation.py']}}
    with (root/'reconciliation-check.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
