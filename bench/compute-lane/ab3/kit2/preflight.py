#!/usr/bin/env python3
"""W9 turn-1 preflight (Lead's go): one fixture request through the real HTTP path with a short server deadline,
before any W9 request. PASS needs all four: finish_reason `deadline`, usage present, native ids whose sha256 is the
engine's token_sha256 (return_token_ids), and the final runner receipt (not live records). The fixture is not a W9
task and its reply is in no W9 number.
Usage: python3 -B preflight.py BASE OUT.json [DEADLINE_S]   (exit 0 PASS, 1 FAIL)"""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure import ids_check, stream_chat  # noqa: E402

PROMPT = ('What is 735214987 × 418903251? Work it out by hand, then list every prime number below 20000, '
          'one per line.')


def main():
    base, out = sys.argv[1], Path(sys.argv[2])
    deadline_s = float(sys.argv[3]) if len(sys.argv) > 3 else 20
    model = json.load(urllib.request.urlopen(base + '/v1/models', timeout=30))['data'][0]['id']
    guide = json.loads((Path(__file__).resolve().parent / 'guides.json').read_text())['trust']
    body = {'model': model, 'messages': [{'role': 'user', 'content': PROMPT}], 'max_tokens': 32768, 'stream': True,
            'stream_options': {'include_usage': True}, 'chat_template_kwargs': {'enable_thinking': True},
            'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'seed': 1, 'runner_guide': guide,
            'return_token_ids': True, 'deadline_s': deadline_s}
    task = {'id': 'preflight-deadline', 'set': 'preflight', 'prompt': PROMPT}
    rec = stream_chat(base, model, task, 'r', 32768, deadline=None, body=body)  # no client deadline: the server's
    tf = rec['extra'].get('tensorfold')
    runner = tf.get('runner') if isinstance(tf, dict) else None
    checks = {'finish_deadline': rec.get('finish') == 'deadline',
              'usage': bool((rec.get('usage') or {}).get('completion_tokens')),
              'token_ids_match': ids_check(rec) == 'match',
              'runner_receipt': isinstance(runner, dict) and not runner.get('live') and isinstance(runner.get('cells'), list)}
    verdict = 'PASS' if all(checks.values()) else 'FAIL'
    summary = {'verdict': verdict, 'checks': checks, 'deadline_s': deadline_s, 'finish': rec.get('finish'),
               'elapsed_s': rec['elapsed_s'], 'usage': rec.get('usage'), 'ids_check': ids_check(rec),
               'cells': len(runner.get('cells') or []) if isinstance(runner, dict) else None,
               'error': rec.get('error'), 'model': model}
    out.write_text(json.dumps({'summary': summary, 'record': rec}) + '\n')
    print('PREFLIGHT', verdict, json.dumps(summary), flush=True)
    sys.exit(0 if verdict == 'PASS' else 1)


if __name__ == '__main__':
    main()
