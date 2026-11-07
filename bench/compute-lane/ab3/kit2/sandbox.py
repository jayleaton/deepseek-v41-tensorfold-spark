"""Runs model-written Python for grading, in one throwaway container per reply: no network, read-only root, user
nobody, no capabilities, no new privileges, 512 MiB memory, 128 pids, one CPU at the lowest CPU weight (nice 19 does
not reach containers), a 64 MiB scratch /tmp and the job files read-only. (The Sparks refuse unprivileged user
namespaces, so bwrap cannot run there.) The image is the serving image, so its CPython is the compute lane's; the
prelude is the lane's too (lane-stage ab.py), so MBPP grades as the A/B did. Inside, a driver runs each test as its
own process with its own timeout. T4_GRADE_IMAGE overrides the image."""
import json
import os
import shutil
import subprocess
import tempfile
import uuid

IMAGE = os.environ.get('T4_GRADE_IMAGE', 'tensorfold-dsv41-imr:r-94ce5be47f85')
PRELUDE = r'''import sys, ast, io, linecache, random, traceback
random.seed(0)
src = open(sys.argv[1]).read()
linecache.cache["<code>"] = (len(src), None, src.splitlines(True), "<code>")
try:
    tree = ast.parse(src, "<code>")
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    ns = {"__name__": "__main__"}
    exec(compile(tree, "<code>", "exec"), ns)
    if last is not None:
        value = eval(compile(ast.Expression(last.value), "<code>", "eval"), ns)
        if value is not None:
            print(repr(value))
except SystemExit:
    raise
except BaseException as e:
    tb = None if isinstance(e, SyntaxError) else e.__traceback__.tb_next
    traceback.print_exception(type(e), e, tb)
    sys.exit(1)
'''
DRIVER = r'''import json, subprocess, sys
job = json.load(open("/job/job.json"))
out = []
for stdin in job["inputs"]:
    try:
        p = subprocess.run([sys.executable, "-I", "-B", "/job/prelude.py", "/job/code.py"], input=stdin.encode(),
                           capture_output=True, timeout=job["timeout"], env={"PYTHONHASHSEED": "0", "PATH": "/usr/bin",
                           "PYTHONINTMAXSTRDIGITS": "0", "HOME": "/tmp"}, cwd="/tmp")
        out.append(["ok" if p.returncode == 0 else "exit_error", p.stdout[:4 << 20].decode(errors="replace"),
                    p.stderr[-4096:].decode(errors="replace")])
    except subprocess.TimeoutExpired:
        out.append(["timeout", "", ""])
    if job["stop_on_fail"] and out[-1][0] != "ok":
        break
print(json.dumps(out))
'''


def have_jail():
    return shutil.which('docker') is not None


def run_batch(code, inputs, timeout_s=10.0, stop_on_fail=True):
    """[(status, stdout, stderr)] per input (stops after the first failure unless stop_on_fail is False); status is
    ok / exit_error / timeout, or the container's error for every input when the jail itself failed."""
    if not have_jail():
        raise RuntimeError('docker is missing: model code never runs unjailed')
    with tempfile.TemporaryDirectory() as d:
        os.chmod(d, 0o755)
        for name, text in (('code.py', code), ('prelude.py', PRELUDE), ('driver.py', DRIVER),
                           ('job.json', json.dumps({'inputs': inputs, 'timeout': timeout_s, 'stop_on_fail': stop_on_fail}))):
            with open(f'{d}/{name}', 'w') as f:
                f.write(text)
            os.chmod(f'{d}/{name}', 0o644)
        name = f't4-grade-{uuid.uuid4().hex[:12]}'
        cmd = ['docker', 'run', '--rm', '--name', name, '--network', 'none', '--read-only', '--user', '65534:65534',
               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--memory', '512m', '--memory-swap', '512m',
               '--pids-limit', '128', '--cpus', '1', '--cpu-shares', '2', '--cpuset-cpus', '0-4',  # host slow cores
               '--tmpfs', '/tmp:size=64m', '-v', f'{d}:/job:ro',
               '--entrypoint', 'nice', IMAGE, '-n', '19', 'python3', '-I', '-B', '/job/driver.py']
        try:
            p = subprocess.run(cmd, capture_output=True, timeout=timeout_s * len(inputs) + 60)
        except subprocess.TimeoutExpired:
            subprocess.run(['docker', 'kill', name], capture_output=True, timeout=60)
            return [('timeout', '', 'container timeout')] * len(inputs)
        try:
            return [tuple(x) for x in json.loads(p.stdout)]
        except ValueError:
            err = ('jail ' + p.stderr.decode(errors='replace')[-300:]).strip()
            return [('exit_error', '', err)] * len(inputs)


def run(code, stdin='', timeout_s=10.0):
    return run_batch(code, [stdin], timeout_s)[0]


PROBE = '''import os, socket
try:
    socket.create_connection(("1.1.1.1", 53), timeout=2)
    net = "open"
except OSError:
    net = "blocked"
try:
    open("/usr/t4-probe", "w")
    root = "writable"
except OSError:
    root = "read-only"
print(net, root, os.getuid(), sum(map(int, input().split())))
'''


def probe():
    """The jail's own check: (passed, detail). Network blocked, root read-only, nobody, stdin works, timeouts fire."""
    st, out, err = run(PROBE, '2 3 4')
    slow = run('while True: pass', timeout_s=2)[0]
    ok = st == 'ok' and out.split() == ['blocked', 'read-only', '65534', '9'] and slow == 'timeout'
    return ok, f'{st} {out.strip()} {err[-200:].strip()} | busy loop: {slow}'
