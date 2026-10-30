"""Replay the public STREAM_RB=4 cold-input suite against an already configured two-rank server."""

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', default='http://127.0.0.1:18100')
    ap.add_argument('--fixtures', type=Path, default=Path('fixtures.zip'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--short-only', action='store_true', help='quick concurrency/draft probe without long inputs')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.fixtures) as archive:
        cases = json.loads(archive.read('manifest.json'))
        bodies = {c['id']: archive.read(c['id'] + '.json') for c in cases}
    for c in cases:
        assert hashlib.sha256(bodies[c['id']]).hexdigest() == c['sha256']

    def request(case, draft=True, suffix=''):
        body = json.loads(bodies[case['id']])
        body['draft'] = draft
        req = urllib.request.Request(args.base.rstrip('/') + '/v1/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=1800) as response:
            value = json.load(response)
        name = case['id'] + suffix
        result = {'case': name, 'draft': draft, 'wall_s': time.monotonic() - start, 'response': value}
        (args.out / (name + '.json')).write_text(json.dumps(result, indent=2) + '\n')
        stats = value['tensorfold']
        text = value['choices'][0]['text']
        assert stats['token_ids']
        assert value['usage']['prompt_tokens'] == case['prompt_tokens']
        assert stats['cached'] == 0
        assert value['choices'][0]['finish_reason'] == 'stop'
        assert case['expected_contains'] is None or case['expected_contains'] in text
        if case['id'] == 'short-json':
            assert json.loads(text) == {'ok': True, 'count': 3}
        print(json.dumps({'case': name, 'prefill_s': stats.get('prefill_s'), 'ttft_s': stats.get('ttft_s')}), flush=True)
        return result

    short = [c for c in cases if c['id'].startswith('short-')]
    results = [request(c) for c in short]
    if not args.short_only:
        results += [request(c) for c in cases if not c['id'].startswith('short-')]
    results += [request(c, False, '-plain') for c in short]
    if not args.short_only:
        mid = next(c for c in cases if c['prompt_tokens'] == 131072)
        results.append(request(mid, False, '-plain'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(request, c, True, '-concurrent') for c in short[:4]]
        results += [f.result() for f in futures]
    lookup = {r['case']: r['response']['tensorfold']['token_ids'] for r in results}
    checks = []
    for name, ids in lookup.items():
        root = name.removesuffix('-plain').removesuffix('-concurrent')
        if root != name:
            checks.append({'case': name, 'token_ids_equal': ids == lookup[root]})
    summary = {'requests_passed': len(results), 'consistency_checks': checks,
               'consistency_failures': sum(not c['token_ids_equal'] for c in checks)}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return int(summary['consistency_failures'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
