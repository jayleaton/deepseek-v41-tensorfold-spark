#!/usr/bin/env python3
"""lane's stale-name-binding check (s3-trace-default-risk/scripts/stale.py, unchanged) over a window's live trace
claims: each native trace record becomes one of its rows (display, value, the claim's place and, as the statement line,
the line before the claim); then a count per arm and the replies with one.
Usage: python3 -B stale_live.py STALE_SCRIPTS_DIR RUN OUTDIR > stale.json"""

import sys as _archive_sys
import contextlib
import io
import json
import re
import sys
from pathlib import Path

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

sys.path.insert(0, sys.argv[1])
import stale  # noqa: E402


def char_at(text, at):
    """Records place a claim by byte; stale.py slices characters. The claim's own line starts there either way."""
    for i in (len(text.encode()[:at].decode('utf-8', 'ignore')), at):
        if text.startswith('[exact', i) or text.startswith('\n[exact', i):
            return i, True
    return len(text.encode()[:at].decode('utf-8', 'ignore')), False


def main():
    run, out = Path(sys.argv[2]), Path(sys.argv[3])
    (out / 'text').mkdir(parents=True, exist_ok=True)
    rows, arms, placed = [], {}, [0, 0]
    for armdir in sorted(p.parent for p in run.glob('*/*/arm.json')):
        label = f'{armdir.parent.name}/{armdir.name}'
        arms[label] = {'claims': 0, 'stale': 0, 'replies': []}
        for f in sorted(armdir.glob('*.json')):
            if f.name == 'arm.json':
                continue
            r = json.loads(f.read_text())
            for c in (((r.get('extra') or {}).get('tensorfold') or {}).get('runner') or {}).get('cells') or []:
                if c['form'] != 'trace' or c.get('status') != 'ok':
                    continue
                text = r.get(c.get('field')) or ''
                name = f'{label.replace("/", "__")}__{r["id"]}__{c["field"]}'
                (out / 'text' / name).write_text(text)
                at, ok = char_at(text, c['at'])
                placed[ok] += 1
                rows.append({'name': name, 'display': c.get('display') or c.get('span') or '', 'value': c.get('value') or '',
                             'at': at, 'line': text[:at].rstrip('\n').split('\n')[-1], 'stated_only': False})
                arms[label]['claims'] += 1
    (out / 'shapes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        stale.main(str(out / 'shapes.jsonl'), str(out / 'none'), str(out / 'text'))
    report = buf.getvalue()
    (out / 'stale.txt').write_text(report)
    for m in re.finditer(r'^== (\S+)', report, re.M):
        label, tid = m.group(1).split('__')[:2], m.group(1).split('__')[2]
        a = arms['/'.join(label)]
        a['stale'] += 1
        a['replies'] = sorted(set(a['replies']) | {tid})
    json.dump({'arms': arms, 'claims_placed_on_their_line': placed[1], 'claims_not_placed': placed[0],
               'summary': report.strip().splitlines()[-1] if report.strip() else ''}, sys.stdout, indent=1)


if __name__ == '__main__':
    main()
