#!/usr/bin/env python3
"""T4 -> TE: does the in-model runner slow decode? About 2 minutes, run on rank 0's Spark beside the serving engine.

Run it once in the imr-off phase and once in imr-on (same engine flags otherwise):
    python3 -I runner_decode_check.py off   >  decode-off.txt
    python3 -I runner_decode_check.py on    >  decode-on.txt
Compare the two `DECODE ... beside-cells` lines: on/off >= 0.95 passes. In `on`, `RUNNER threads` must name the
runner's CPUs (TF_DSV41_RUNNER_CPUS; default: the small cores minus the engine's) and none on the engine's CPU.

Parts: (1) a fixed greedy coding probe alone, 3 times; (2) the same probe while 3 other lanes stream replies that
write Python cells the runner owns (multiply, loop), 3 times; (3) throughout (2), every 0.5 s, the engine's
tf-runner threads (allowed CPUs, last CPU, CPU time) and per-CPU busy %. Stdlib only; reads /proc, writes nothing.
"""
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

BASE = os.environ.get("BASE", "http://localhost:8000")
PROBE = "Write a Python function that merges overlapping integer intervals, with a docstring and three asserts. Code only."
CELLS = [
    "Use Python to compute 123456789012345678901234567890 * 987654321098765432109876543210. "
    "Write the code in a ```python block that prints the product, then give the result.",
    "Use Python to compute the sum of i*i for i in range(10) with a for loop. "
    "Write the code in a ```python block that prints the sum, then give the result.",
]


def model():
    return json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=10))["data"][0]["id"]


def stream(name, prompt, max_tokens, stop=None):
    """Streams one greedy reply; returns (prefill s, chunks, decode tok/s)."""
    body = {"model": name, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    stamps = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            if stop is not None and stop.is_set():
                break
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            for ch in json.loads(line[5:]).get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    stamps.append(time.monotonic())
    n = len(stamps)
    rate = (n - 1) / (stamps[-1] - stamps[0]) if n > 1 else 0.0
    return (stamps[0] - t0 if stamps else 0.0), n, rate


def engine_pid():
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                with open(f"/proc/{pid}/comm") as f:
                    if f.read().strip() == "tensorfold-dsv4":
                        return int(pid)
            except OSError:
                pass
    return None


def cpu_times():
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                p = line.split()
                v = list(map(int, p[1:]))
                out[p[0]] = (sum(v), v[3] + v[4])
    return out


def sample_runner(pid, stop, seen):
    """Every 0.5 s: each tf-runner thread's allowed CPUs, last CPU and CPU ticks."""
    while not stop.is_set():
        try:
            for tid in os.listdir(f"/proc/{pid}/task"):
                try:
                    with open(f"/proc/{pid}/task/{tid}/comm") as f:
                        if f.read().strip() != "tf-runner":
                            continue
                    with open(f"/proc/{pid}/task/{tid}/status") as f:
                        allowed = next(l.split()[1] for l in f if l.startswith("Cpus_allowed_list"))
                    with open(f"/proc/{pid}/task/{tid}/stat") as f:
                        rest = f.read().rsplit(")", 1)[1].split()
                    seen[tid] = (allowed, rest[36], int(rest[11]) + int(rest[12]))
                except (OSError, StopIteration):
                    pass
        except OSError:
            pass
        stop.wait(0.5)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    name = model()
    pid = engine_pid()
    if pid:
        with open(f"/proc/{pid}/status") as f:
            main_cpus = next(l.split()[1] for l in f if l.startswith("Cpus_allowed_list"))
        print(f"engine pid {pid}, main thread CPUs {main_cpus}", flush=True)
    else:
        print("engine process not visible from here: no thread sampling", flush=True)
    stream(name, PROBE, 16)  # warm-up
    alone = [stream(name, PROBE, 256)[2] for _ in range(3)]
    print(f"DECODE {label} alone: " + " ".join(f"{r:.2f}" for r in alone) + f" tok/s (median {statistics.median(alone):.2f})", flush=True)
    stop = threading.Event()
    seen = {}
    sampler = threading.Thread(target=sample_runner, args=(pid, stop, seen), daemon=True) if pid else None
    counts = [0, 0, 0]

    def lane(i):
        while not stop.is_set():
            try:
                stream(name, CELLS[i % len(CELLS)], 400, stop)
                counts[i] += 1
            except Exception as e:  # a lane error is reported, not fatal
                print(f"cell lane {i}: {e}", flush=True)
                stop.wait(1)

    lanes = [threading.Thread(target=lane, args=(i,), daemon=True) for i in range(3)]
    t0 = cpu_times()
    if sampler:
        sampler.start()
    for t in lanes:
        t.start()
    time.sleep(3)  # the cell lanes are decoding before the probe starts
    beside = [stream(name, PROBE, 256)[2] for _ in range(3)]
    t1 = cpu_times()
    stop.set()
    for t in lanes:
        t.join(timeout=30)
    print(f"DECODE {label} beside-cells: " + " ".join(f"{r:.2f}" for r in beside) + f" tok/s (median {statistics.median(beside):.2f}); cell replies finished {sum(counts)}", flush=True)
    busy = {c: 100 * ((t1[c][0] - t0[c][0]) - (t1[c][1] - t0[c][1])) / max(t1[c][0] - t0[c][0], 1) for c in t1}
    print("CPU busy % during cells: " + " ".join(f"{c[3:]}={busy[c]:.0f}" for c in sorted(busy, key=lambda c: int(c[3:]))), flush=True)
    if pid:
        by = {}
        for allowed, last, ticks in seen.values():
            k = by.setdefault(allowed, [0, set(), 0])
            k[0] += 1
            k[1].add(last)
            k[2] += ticks
        if by:
            for allowed, (n, lasts, ticks) in by.items():
                print(f"RUNNER threads {label}: {n} on CPUs {allowed} (last seen on {','.join(sorted(lasts, key=int))}), {ticks * 10} ms CPU", flush=True)
        else:
            print(f"RUNNER threads {label}: none seen (runner off, or cells settled between samples)", flush=True)


if __name__ == "__main__":
    main()
