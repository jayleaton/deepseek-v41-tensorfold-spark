#!/usr/bin/env python3
"""rank1-host guard; rank0-host controller supplies authoritative queue state on stdin."""
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path


def children(pid):
    result = [pid]
    try:
        for value in Path(f'/proc/{pid}/task/{pid}/children').read_text().split():
            result.extend(children(int(value)))
    except FileNotFoundError:
        pass
    return result


def send(pid, sig):
    for exact_pid in reversed(children(pid)):
        try:
            os.kill(exact_pid, sig)
        except ProcessLookupError:
            pass


def memory_ok():
    available = next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    return available >= 6 * 1024 * 1024


def jail_pids():
    job = os.environ.get('STDBENCH_CPU_JOB')
    if not job or not job.startswith('stdbench-cpu-'):
        raise RuntimeError('owned CPU job label required')
    found = subprocess.run(['docker', 'ps', '-q', '--filter', 'label=stdbench.cpu_job=' + job],
                           text=True, capture_output=True, timeout=10, check=True).stdout.split()
    pids = []
    for container in found:
        import json
        inspected = subprocess.run(['docker', 'inspect', container], text=True, capture_output=True, timeout=10)
        if inspected.returncode:
            continue  # --rm containers can finish between ps and inspect.
        data = json.loads(inspected.stdout)[0]
        if data['Config'].get('Labels', {}).get('stdbench.cpu_job') != job:
            raise RuntimeError('container ownership label drift')
        pid = data['State']['Pid']
        if pid > 1:
            pids.append(pid)
    return pids


def main():
    assert sys.argv[1] == '--'
    state, seen = 'PAUSE', 0.0
    # Read the first controller state before starting any test.
    first = sys.stdin.readline().strip()
    if first != 'RUN' or not memory_ok():
        print('CPU admission refused: queue, QUIET or memory', flush=True)
        return 75
    scopes = subprocess.check_output(['systemctl', '--user', 'list-units', '--type=scope', '--state=running', '--no-legend'], text=True)
    if len([x for x in scopes.splitlines() if '.scope' in x and 'init.scope' not in x]) >= 4:
        print('CPU admission refused: scope count', flush=True)
        return 75
    state, seen = first, time.monotonic()
    proc = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL)
    started, paused, jails, next_scan = time.monotonic(), False, [], 0.0
    try:
        while proc.poll() is None:
            if time.monotonic() >= next_scan:
                jails = jail_pids()
                next_scan = time.monotonic() + 1
            if select.select([sys.stdin], [], [], 0.1)[0]:
                line = sys.stdin.readline()
                if not line:
                    state = 'PAUSE'
                else:
                    state, seen = line.strip(), time.monotonic()
            if state in ('BOOT', 'LOWMEM') or not memory_ok() or time.monotonic() - started > 1200:
                for pid in jails:
                    send(pid, signal.SIGCONT)
                    send(pid, signal.SIGKILL)
                send(proc.pid, signal.SIGCONT)
                send(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    send(proc.pid, signal.SIGKILL)
                    proc.wait()
                print('CPU guard aborted own job: boot/memory/time limit', flush=True)
                return 75
            should_pause = state != 'RUN' or time.monotonic() - seen > 5
            if should_pause != paused:
                for pid in jails:
                    send(pid, signal.SIGSTOP if should_pause else signal.SIGCONT)
                send(proc.pid, signal.SIGSTOP if should_pause else signal.SIGCONT)
                paused = should_pause
        return proc.returncode
    finally:
        if proc.poll() is None:
            for pid in jail_pids():
                send(pid, signal.SIGCONT)
                send(pid, signal.SIGKILL)
            send(proc.pid, signal.SIGCONT)
            send(proc.pid, signal.SIGTERM)
            proc.wait()


if __name__ == '__main__':
    raise SystemExit(main())
