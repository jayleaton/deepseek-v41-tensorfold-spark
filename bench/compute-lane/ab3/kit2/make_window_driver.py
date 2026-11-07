#!/usr/bin/env python3
"""Builds imr-measure.py from TE's imr-turn.py (holds, cooling, thermal guard, rank start/stop, G gates, configs run in
order): the only change is that a config with "measure" runs window_boot.py (placement check, then measure.py with
replay) instead of TE's smoke, with a 6 h cap per boot; and a config may carry "start_by_s" (not started later than
that many seconds into the config loop) and "needs" (names that must have run first, e.g. its ABBA pair's other boot);
out/<run>/ABORT (window_boot.py's failed preflight) skips every config after it.
Usage: python3 make_window_driver.py ~/work/imr-te/imr-turn.py imr-measure.py"""
import re
import sys
from pathlib import Path

src = open(sys.argv[1]).read()
# TE's smoke block, whatever its arguments: `if cfg.get('smoke'):` through the subprocess.run(...).returncode it ends in
old = re.compile(r"^    if cfg\.get\('smoke'\):\n(?:        .*\n)*?.*\.returncode\n", re.M)
new = """    if cfg.get('measure'):
        with (out / name / 'window-boot.log').open('w') as log:
            attempt['measure_rc'] = subprocess.run(['python3', '-B', '@WINDOW_BOOT@',
                                                    str(out / name), json.dumps(cfg)], stdout=log,
                                                   stderr=subprocess.STDOUT, timeout=21600).returncode
"""
new = new.replace('@WINDOW_BOOT@', str(Path(__file__).resolve().parent / 'window_boot.py'))  # this kit copy's
assert len(old.findall(src)) == 1, 'TE imr-turn.py changed: the smoke block is not where make_window_driver.py expects it'
out = old.sub(lambda _: new, src)
loop = """    for i, cfg in enumerate(configs):
        if i and ranks:
"""
guard = """    t_loop, ran = time.monotonic(), set()
    for i, cfg in enumerate(configs):
        late = cfg.get('start_by_s') and time.monotonic() - t_loop > cfg['start_by_s']
        abort = (out / 'ABORT').exists()  # a boot's preflight failed: nothing more in this turn
        if late or abort or not set(cfg.get('needs', [])) <= ran:
            print('CONFIG-SKIPPED', cfg['name'], 'abort' if abort else 'late' if late else 'needs', flush=True)
            status.setdefault('skipped', []).append(cfg['name'])
            continue
        ran.add(cfg['name'])
        if ran != {cfg['name']} and ranks:
"""
assert out.count(loop) == 1, 'TE imr-turn.py changed: the config loop is not where make_window_driver.py expects it'
out = out.replace(loop, guard)
compile(out, sys.argv[2], 'exec')
assert "import json" in out and "import time" in out and "attempt['measure_rc']" in out
open(sys.argv[2], 'w').write(out)
print('ok')
