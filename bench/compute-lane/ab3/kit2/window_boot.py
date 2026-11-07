#!/usr/bin/env python3
"""One booted config of the window (called by imr-measure.py once /v1/models answers and the gates ran):
1. with "decode_check": the placement and decode check (t4-live runner_decode_check.py, about 2 minutes: decode alone
   and beside cells, tf-runner threads and per-CPU busy), labelled off or on;
2. measure.py over the config's sample (T4_SAMPLE overrides; `all` = the 261) with its arms, forms and caps, this
   boot's rank0.log, the replay validity check, and (with "replay_from", on the runner-off boot) the replay of the
   runner boots' owned cells.
Usage: python3 -B window_boot.py OUT/<config> '<config json>'
Env: T4_SETS (comma list, default all); T4_BASE and T4_LIMIT only for the dry run (a fake server, N tasks a set)."""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
out = Path(sys.argv[1])
cfg = json.loads(sys.argv[2])
m = cfg['measure']
base = os.environ.get('T4_BASE', 'http://127.0.0.1:8000')
if m.get('decode_check'):
    check = next(p for p in (HERE / 'runner_decode_check.py', HERE.parent / 't4-live' / 'runner_decode_check.py') if p.exists())
    with (out / 'decode-check.txt').open('w') as f:
        rc = subprocess.run(['python3', '-I', str(check), m['place']],
                            stdout=f, stderr=subprocess.STDOUT, timeout=900, env=dict(os.environ, BASE=base)).returncode
    print('decode-check rc', rc, flush=True)
if os.environ.get('T4_PREFLIGHT_DEADLINE') and not (out.parent / 'PREFLIGHT').exists():
    # W9 turn 1, its first boot only: the server deadline checked end to end before any W9 request; a FAIL stops the
    # turn (the driver skips every later config on ABORT)
    rc = subprocess.run(['python3', '-B', str(HERE / 'preflight.py'), base, str(out / 'preflight-deadline.json')],
                        timeout=900).returncode
    (out.parent / 'PREFLIGHT').write_text(f'rc={rc} {out.name}\n')
    if rc:
        (out.parent / 'ABORT').write_text(f'preflight rc={rc} on {out.name}\n')
        sys.exit(7)
cmd = ['python3', '-B', str(HERE / 'measure.py'), '--base', base, '--boot', cfg['name'],
       '--arms', m['arms'], '--tasks', str(HERE / m.get('tasks', 'tasks.jsonl')), '--out', str(out),
       '--forms', m['forms'],
       '--concurrency', str(m.get('concurrency', 4)), '--max-tokens', str(m.get('max_tokens', 32768)), '--replay',
       '--sample', os.environ.get('T4_SAMPLE', m.get('sample', 'first')), '--cap-s', str(m.get('cap_s', 0)),
       '--cap-c1-s', str(m.get('cap_c1_s', 0))]
if any(x.startswith('r') for x in m['arms'].split(',')):
    cmd += ['--rank-log', str(out / 'rank0.log')]
if m.get('arm_opts'):  # "@name" = that guide's text from guides.json beside this file (S1's frozen guide texts)
    guides = json.loads((HERE / 'guides.json').read_text())
    opts = {arm: {k: guides[v[1:]] if isinstance(v, str) and v.startswith('@') else v for k, v in o.items()}
            for arm, o in m['arm_opts'].items()}
    cmd += ['--arm-opts', json.dumps(opts)]
if m.get('replay_from'):  # the runner boots of this run, measured before this runner-off boot
    cmd += ['--replay-from', ','.join(str(out.parent / b) for b in m['replay_from'])]
if m.get('request_deadline_s'):
    cmd += ['--request-deadline-s', str(m['request_deadline_s'])]
if m.get('arm_ids'):  # frozen per-arm task subsets (W8's seed-3 split)
    cmd += ['--arm-ids', json.dumps(m['arm_ids'])]
if m.get('server_deadline_field'):  # W9: the server ends a late reply itself and still sends usage, records, ids
    cmd += ['--server-deadline-field', m['server_deadline_field'], '--deadline-grace-s',
            str(m.get('deadline_grace_s', 30))]
if m.get('templates'):  # W9: T5's frozen request bodies for this boot, in their frozen order
    cmd += ['--templates', str(HERE / m['templates'])]
if m.get('token_ids'):
    cmd += ['--token-ids']
if m.get('no_count_fallback'):
    cmd += ['--no-count-fallback']
if os.environ.get('T4_LIMIT'):
    cmd += ['--limit', os.environ['T4_LIMIT']]
if os.environ.get('T4_SETS') or m.get('sets'):
    cmd += ['--sets', os.environ.get('T4_SETS') or m['sets']]
sys.exit(subprocess.run(cmd).returncode)
