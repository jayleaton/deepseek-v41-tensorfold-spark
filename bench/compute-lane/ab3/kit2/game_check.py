"""The build task's scripted checklist: the reply's last ```html block runs in headless Chrome (no network: a dead
proxy; a throwaway profile) with a checker script appended. Items: interface, rotation, line clear, game over (the
three DESIGN §14 items) plus move, gravity and a stopped board after game over. Solved = all items pass."""
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

CHECKS = r'''<script>
(function () {
  const out = {};
  const norm = cs => { const r = Math.min(...cs.map(c => c[0])), k = Math.min(...cs.map(c => c[1]));
    return cs.map(c => (c[0] - r) + ',' + (c[1] - k)).sort().join(' '); };
  const cols = s => s.piece.cells.map(c => c[1]);
  function run() {
    const g = window.game;
    out.interface = !!g && ['reset', 'state', 'press', 'tick', 'setBoard'].every(f => typeof g[f] === 'function');
    if (!out.interface) return;
    g.reset(1);
    let s = g.state();
    out.board = s.width === 10 && s.height === 20 && s.board.length === 20 && s.board.every(r => r.length === 10 && r.every(v => !v)) && !!s.piece && s.piece.cells.length === 4;
    const c0 = Math.min(...cols(s)); g.press('left'); s = g.state();
    out.move = Math.min(...cols(s)) === c0 - 1 || c0 === 0;
    const r0 = Math.max(...s.piece.cells.map(c => c[0])); g.tick(); s = g.state();
    out.gravity = Math.max(...s.piece.cells.map(c => c[0])) === r0 + 1;
    out.rotate = false;
    for (let seed = 1; seed <= 40 && !out.rotate; seed++) {
      g.reset(seed); g.tick(); g.tick(); s = g.state();
      const before = norm(s.piece.cells);
      if (before === '0,0 0,1 1,0 1,1') continue;
      g.press('rotate');
      const after = norm(g.state().piece.cells);
      out.rotate = after !== before && g.state().piece.cells.length === 4;
    }
    // line clear: learn where seed 7's first piece lands, then leave exactly those bottom-row holes
    g.reset(7); g.press('drop'); s = g.state();
    const bottom = s.board[19].map(v => !!v);
    out.line_clear = false;
    if (bottom.some(v => v)) {
      g.reset(7);
      const rows = Array.from({length: 20}, () => Array(10).fill(0));
      rows[19] = bottom.map(v => v ? 0 : 1);
      g.setBoard(rows); const l0 = g.state().lines; g.press('drop'); s = g.state();
      out.line_clear = s.lines === l0 + 1 && !s.board[19].every(v => v);
    }
    g.reset(3);
    for (let i = 0; i < 400 && !g.state().over; i++) g.press('drop');
    s = g.state();
    out.game_over = !!s.over;
    const frozen = JSON.stringify(s.board); g.press('drop'); g.tick();
    out.stopped = out.game_over && JSON.stringify(g.state().board) === frozen;
  }
  try { run(); } catch (e) { out.error = String(e).slice(0, 300); }
  const pre = document.createElement('pre'); pre.id = 't4-result'; pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
</script>'''
ITEMS = ('interface', 'board', 'move', 'gravity', 'rotate', 'line_clear', 'game_over', 'stopped')


def chrome():
    for name in ('google-chrome', 'chromium', 'chromium-browser'):
        if shutil.which(name):
            return name
    return None


def html_of(content):
    blocks = re.findall(r'```html[ \t]*\n(.*?)```', content or '', re.S)
    return blocks[-1] if blocks else None


def check(content, timeout_s=60):
    html = html_of(content)
    if html is None:
        return {'ok': False, 'why': 'no html block'}
    exe = chrome()
    if exe is None:
        raise RuntimeError('no headless Chrome on this host')
    page = html + CHECKS if '</body>' not in html else html.replace('</body>', CHECKS + '</body>', 1)
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / 'game.html').write_text(page)
        cmd = [exe, '--headless=new', '--disable-gpu', '--no-first-run', '--proxy-server=127.0.0.1:9',
               f'--user-data-dir={d}/profile', '--virtual-time-budget=5000', '--dump-dom', f'file://{d}/game.html']
        try:
            p = subprocess.run(cmd, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {'ok': False, 'why': 'chrome timeout'}
    m = re.search(r'<pre id="t4-result">(.*?)</pre>', p.stdout.decode(errors='replace'), re.S)
    if not m:
        return {'ok': False, 'why': 'checker did not run (script error before it?)'}
    items = json.loads(m.group(1).replace('&quot;', '"').replace('&amp;', '&'))
    return {'ok': all(items.get(k) for k in ITEMS), 'items': items}
