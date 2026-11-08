#!/usr/bin/env python3
"""Re-runs ab3_toolcheck.py's jobs the way the engine's compute lane ran them, in a jail. Per job, one throwaway
container of the qualified image: `/usr/bin/python3 -P -s -B -u -X utf8 -c RUNNER`, the source on stdin, the engine's
five environment variables, cwd /, RLIMIT_AS 1 GiB and a 10 s wall (core/compute_sandbox.zig:17,95;
core/compute_lane.zig:12,16,17 at 3bf584af1c54). RUNNER is core/compute_runner.zig's `runner`, verbatim in
engine_runner.txt (sha256 pinned below). Jail: no network, read-only root, nobody, no capabilities, no new
privileges, 128 pids, --memory 2g --memory-swap 2g --cpus 1 --cpuset-cpus 0-4, nice 19 inside. The first row is the
jail's probe; ab3_toolcheck.py refuses a PASS without a passing one. Run on a node in a 2 GiB scope, never during a
GPU window, never on dev-host.

    python3 -I run_jobs2.py JOBS.jsonl RESULTS.jsonl

Rows: {"key", "code_sha256", "status": ok|exit_error|timeout, "exit_code", "stdout", "stderr"}, or {"key",
"jail_error"}."""
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE = 'tensorfold-dsv41-imr:r-3bf584af1c54'
RUNNER_SHA256 = '80177f5c8a086de532b1372e15cf9b49eeacaa490da2ed4312fc68d2b885a7d0'
ENV = ['PYTHONHASHSEED=0', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONIOENCODING=utf-8', 'LANG=C.UTF-8', 'PATH=/usr/bin']
WALL_S, AS_BYTES = 10.0, 1024 << 20
# Inside the container: RLIMIT_AS and the wall alarm (a pending alarm survives exec), then the engine's exact argv
# with only its environment. SIGALRM ends the job: docker exits 128 + 14.
LAUNCH = ('import os, resource, signal, sys; resource.setrlimit(resource.RLIMIT_AS, (%d, %d)); os.chdir("/"); '
          'env = dict(x.split("=", 1) for x in sys.argv[3:8]); signal.alarm(int(sys.argv[2])); '
          'os.execve("/usr/bin/python3", ["/usr/bin/python3", "-P", "-s", "-B", "-u", "-X", "utf8", "-c", '
          'sys.argv[1]], env)') % (AS_BYTES, AS_BYTES)
ALARM_EXIT = 128 + 14


def run(runner, code, wall_s=WALL_S):
    """(status, exit_code, stdout, stderr) of one job, as the engine's sandbox would have produced them."""
    name = f'ab3-rerun-{uuid.uuid4().hex[:12]}'
    cmd = ['docker', 'run', '--rm', '-i', '--name', name, '--network', 'none', '--read-only', '--user', '65534:65534',
           '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--memory', '2g', '--memory-swap', '2g',
           '--pids-limit', '128', '--cpus', '1', '--cpu-shares', '2', '--cpuset-cpus', '0-4',
           '--tmpfs', '/tmp:size=64m',
           '--entrypoint', 'nice', IMAGE, '-n', '19', '/usr/bin/python3', '-I', '-c', LAUNCH, runner,
           str(max(1, round(wall_s))), *ENV]
    try:
        p = subprocess.run(cmd, input=code.encode(), capture_output=True, timeout=wall_s + 60)
    except subprocess.TimeoutExpired:
        subprocess.run(['docker', 'kill', name], capture_output=True, timeout=60)
        return 'timeout', None, '', ''
    if p.returncode in (125, 126, 127):  # docker itself failed: never a result
        raise RuntimeError('docker ' + p.stderr.decode(errors='replace')[-300:])
    if p.returncode == ALARM_EXIT:
        return 'timeout', None, p.stdout.decode(errors='replace'), p.stderr.decode(errors='replace')
    return ('ok' if p.returncode == 0 else 'exit_error', p.returncode, p.stdout.decode(errors='replace'),
            p.stderr.decode(errors='replace'))


PROBE = '''import os, socket, sys
try:
    socket.create_connection(("1.1.1.1", 53), timeout=2)
    net = "open"
except OSError:
    net = "blocked"
try:
    open("/usr/ab3-probe", "w")
    root = "writable"
except OSError:
    root = "read-only"
print(net, root, os.getuid(), os.getcwd(), sys.flags.utf8_mode, sys.flags.safe_path, sys.get_int_max_str_digits())
2 ** 10
'''


def probe(runner):
    st, code, out, err = run(runner, PROBE)
    want = ['blocked', 'read-only', '65534', '/', '1', 'True', '4300', '1024']
    slow = run(runner, 'while True: pass', wall_s=3)[0]
    return st == 'ok' and out.split() == want and slow == 'timeout', f'{st} {out.split()} {err[-200:]} | loop {slow}'


def main():
    runner = (HERE / 'engine_runner.txt').read_text()
    assert hashlib.sha256(runner.encode()).hexdigest() == RUNNER_SHA256, 'engine_runner.txt is not the pinned runner'
    if shutil.which('docker') is None:
        sys.exit('no docker here: model code never runs unjailed')
    jobs = [json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
    passed, detail = probe(runner)
    with open(sys.argv[2], 'w') as w:
        w.write(json.dumps({'probe': {'passed': passed, 'detail': detail, 'image': IMAGE,
                                      'runner_sha256': RUNNER_SHA256,
                                      'run_jobs_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}})
                + '\n')
        w.flush()
        if not passed:
            sys.exit(f'jail probe failed: {detail}')
        for j in jobs:
            row = {'key': j['key'], 'code_sha256': hashlib.sha256(j['code'].encode()).hexdigest()}
            try:
                st, rc, out, err = run(runner, j['code'], min(float(j.get('timeout_s', WALL_S)), WALL_S))
                row.update(status=st, exit_code=rc, stdout=out, stderr=err)
            except Exception as e:  # the jail itself failed: the check reads this call as INCONCLUSIVE
                row['jail_error'] = f'{type(e).__name__}: {e}'[:300]
            w.write(json.dumps(row, ensure_ascii=False) + '\n')
            w.flush()
    print(f'{len(jobs)} jobs -> {sys.argv[2]}')


if __name__ == '__main__':
    main()
