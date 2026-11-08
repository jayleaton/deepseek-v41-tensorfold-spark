"""Runs window cells' code in the grading jail (sandbox.py) and writes {key: stdout}: python3 -I run_cells.py IN OUT.
IN is w3_paired.py's output; only its handed-off cells ("b") run. Model code never runs outside the jail."""

import sys as _archive_sys
import json
import sys

import sandbox

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

ok, detail = sandbox.probe()
if not ok:
    sys.exit(f'jail probe failed: {detail}')
out = {}
for c in json.load(open(sys.argv[1]))['b']:
    st, stdout, err = sandbox.run(c['code'])
    out[c['key']] = stdout if st == 'ok' else ''
    print(c['key'], st, stdout.strip()[:80], err.strip()[-120:], flush=True)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
