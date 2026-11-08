#!/usr/bin/env python3
"""One boot's share of the §14 measurement: every task of the shared prompt set, streamed, greedy (unless an arm
samples), thinking on and uncapped (no budget; max_tokens is only a runaway stop), each chunk timestamped; then, on
the runner-off boot, the model's own guess at every result position the runner boots owned (replay).

    python3 -B measure.py --base URL --boot NAME --arms a,d --tasks tasks.jsonl --out DIR [--sets mul30,mbpp]
                          [--concurrency 4] [--max-tokens 32768] [--forms fence,repl] [--rank-log R0]
                          [--replay] [--replay-from BOOTDIR,...] [--sample first] [--cap-s S --cap-c1-s S] [--limit N]
                          [--request-deadline-s S] [--arm-ids '{"X:s3": [task ids]}']

Arms (the request is the only difference inside a boot):
  a  runner off (G): plain chat            d  G + the compute lane: tools [{"type": "code_interpreter"}]
  r  runner on (the boot's TF_DSV41_RUNNER_FORMS decide fence,repl or +inline): plain chat
  r<x>  (rg, rk, rt, ...) runner arms whose requests add --arm-opts fields (runner_guide text, runner_trace false/true)
  X:c1  arm X one request at a time over the sample's c=1 subset (latency and decode tok/s); directory X-c1
  X:sN  arm X with request seed N (with a temperature in its --arm-opts: one sampled run per seed); directory X-sN
--sample NAME takes a named stratified sample (tasks.SAMPLES, fixed seed; `all` = every task). Requests go out longest
expected first (tasks.PRIOR_TOKENS) so long replies overlap the rest.
Writes DIR/<arm>/<task id>.json, DIR/<arm>/arm.json (timing, placement samples, this arm's byte range of --rank-log)
and, with --replay, DIR/<arm>/replay.jsonl. With a cap, requests still streaming at the cap are cut (DIR/<arm>/cut.jsonl,
no reply file) and later ones never start; the report counts both, and a rerun without the cap resumes them.
A request past --request-deadline-s is cut but kept as a reply with finish `timeout` (not resumed).

Replay runs on the runner-off boot only: on a runner boot the runner also runs cells inside /v1/completions (seen on
bf2ab7ec5: ~300 runner replies for 58 chats), so the "guess" there would be the runner's value. Same weights, same
template: for each result position a runner reply owned (its boot's rank0.log, matched to the reply), the prompt is the
chat prompt's ids (/v1/tokenize, messages) plus the reply text up to that position (/v1/tokenize, prompt); the greedy
continuation is the model's own guess. --replay-from names the runner boots' directories of this run (run that boot
last); --replay alone is the validity check on this boot's runner-off replies (the continuation must equal what the
reply really says next). Rows go to the runner arm's own replay.jsonl, so analyze.py reads them where it did."""
import argparse
import copy
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spans  # noqa: E402
import tasks as T  # noqa: E402

THINK_CLOSE = os.environ.get('T4_THINK_CLOSE', '</think>')


def post(base, path, body, timeout=3600):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream_chat(base, model, task, arm, max_tokens, deadline=None, opts=None, body=None):
    if body is None:  # else a frozen request body, sent as it is (W9's templates)
        body = {'model': model, 'messages': [{'role': 'user', 'content': task['prompt']}], 'temperature': 0,
                'max_tokens': max_tokens, 'stream': True, 'stream_options': {'include_usage': True},
                'chat_template_kwargs': {'enable_thinking': True}}
        if arm == 'd':
            body['tools'] = [{'type': 'code_interpreter'}]
    body.update(opts or {})  # the arm's own request fields (runner_guide, runner_trace), or W9's bindings
    rec = {'id': task['id'], 'set': task['set'], 'arm': arm, 'request': body, 'chunks': [], 'reasoning': '',
           'content': '', 'usage': None, 'extra': {}, 'sent_unix': time.time()}
    req = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith('data:') or line == 'data: [DONE]':
                    continue
                c = json.loads(line[5:])
                t = round(time.monotonic() - t0, 4)
                if c.get('usage'):
                    rec['usage'] = c['usage']
                for k in c:
                    if k not in ('id', 'object', 'created', 'model', 'choices', 'usage', 'system_fingerprint'):
                        rec['extra'][k] = c[k]  # the engine's own fields (tensorfold timing, compute calls)
                tfc = c.get('tensorfold')
                live = (tfc.get('runner') or {}) if isinstance(tfc, dict) else {}
                if live.get('live'):  # T2's records on the stream: the newest set the client has the bytes of
                    rec['live_cells'] = live.get('cells')
                if deadline is not None and time.monotonic() > deadline:
                    rec['finish'] = 'cut'  # the arm's cap: closing the stream cancels the request
                    break
                for ch in c.get('choices', []):
                    d = ch.get('delta', {})
                    rc, ct = d.get('reasoning_content') or '', d.get('content') or ''
                    if rc or ct:
                        rec['chunks'].append([t, len(rc), len(ct)])
                    rec['reasoning'] += rc
                    rec['content'] += ct
                    if d.get('tool_calls'):
                        rec.setdefault('tool_calls', []).append(d['tool_calls'])
                    if ch.get('finish_reason'):
                        rec['finish'] = ch['finish_reason']
    except Exception as exc:  # recorded; the arm goes on
        rec['error'] = repr(exc)[:500]
    tf = rec['extra'].get('tensorfold')
    if rec.get('live_cells') is not None and not (isinstance(tf, dict) and (tf.get('runner') or {}).get('cells')):
        # no finished receipt after the live records (a cut stream, or a later chunk without them): keep what was seen
        if not isinstance(tf, dict):
            tf = rec['extra']['tensorfold'] = {}
        tf['runner'] = dict(tf.get('runner') or {}, live=True, cells=rec['live_cells'])
    rec['elapsed_s'] = round(time.monotonic() - t0, 3)
    return rec


def bindings(a, opts):
    """(request fields with native ids and the server deadline added, the client's own deadline or None)."""
    opts = dict(opts)
    if a.token_ids:
        opts['return_token_ids'] = True  # native ids in the final tensorfold block (what token_sha256 hashes)
    own = None
    if a.request_deadline_s:
        own = time.monotonic() + a.request_deadline_s
        if a.server_deadline_field:  # the server ends the reply itself and still sends usage/records/ids
            opts[a.server_deadline_field] = a.request_deadline_s
            own += a.deadline_grace_s
    return opts, own


def settle(a, rec, own, limit, arm):
    """A finished request's record: its own deadline as finish `timeout`, the token checks and counts."""
    if rec.get('finish') == 'cut' and own is not None and own == limit:
        rec['finish'] = 'timeout'  # its own deadline: kept as a reply (in the denominator), not left for a rerun
        rec['closed_by'] = 'client'  # anything the server sends at its end never arrived
        if a.no_count_fallback:
            rec['usage_missing'] = not rec.get('usage')  # missing stays missing (T5-W9-SPEC)
        elif not rec.get('usage'):  # usage comes at the stream's end: count what streamed
            try:
                n = count_tokens(a.base, rec['reasoning'] + rec['content'])
                rec['usage'] = {'completion_tokens': n, 'counted': True}
            except Exception as exc:
                rec['usage_counted_error'] = repr(exc)[:200]
    rec['arm'] = arm
    rec['ids_check'] = ids_check(rec)
    u = rec.get('usage') or {}
    if not a.no_count_fallback and not (u.get('completion_tokens_details') or {}).get('reasoning_tokens') \
            and rec['reasoning']:
        try:
            rec['reasoning_tokens_counted'] = count_tokens(a.base, rec['reasoning'])
        except Exception as exc:
            rec['reasoning_tokens_counted_error'] = repr(exc)[:200]
    return rec


def save(out, tid, rec):
    if rec.get('finish') != 'cut':
        (out / f'{tid}.json').write_text(json.dumps(rec) + '\n')
    else:
        with (out / 'cut.jsonl').open('a') as f:
            f.write(json.dumps({'id': tid, 'elapsed_s': rec['elapsed_s'],
                                'chars': len(rec['reasoning']) + len(rec['content'])}) + '\n')


def ids_check(rec):
    """'match' when the returned native ids hash to the engine's token_sha256, else 'mismatch', 'no-sha' or 'no-ids'."""
    tf = (rec.get('extra') or {}).get('tensorfold') or {}
    ids = tf.get('token_ids')
    if not isinstance(ids, list):
        return 'no-ids'
    if not tf.get('token_sha256'):
        return 'no-sha'
    return 'match' if hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest() == tf['token_sha256'] else 'mismatch'


def count_tokens(base, text):
    if not text:
        return 0
    return post(base, '/v1/tokenize', {'prompt': text, 'add_special_tokens': False}, timeout=120)['count']


# ---- placement: the engine's runner threads, sampled from the host's /proc while an arm runs ---------------------
def engine_pids():
    out = []
    for pid in os.listdir('/proc'):
        if pid.isdigit():
            try:
                with open(f'/proc/{pid}/comm') as f:
                    if f.read().strip() in ('tensorfold-dsv4', 'tf-dsv41 follow', 'tensorfold-dsv41'):
                        out.append(int(pid))
            except OSError:
                pass
    return out


def sample_placement(stop, samples):
    """Every second: each tf-runner thread's allowed CPUs and last CPU, and the engine threads' last CPUs."""
    while not stop.is_set():
        snap = {'t': round(time.time(), 1), 'runner': [], 'engine_cpus': []}
        for pid in engine_pids():
            try:
                tids = os.listdir(f'/proc/{pid}/task')
            except OSError:
                continue
            for tid in tids:
                try:
                    with open(f'/proc/{pid}/task/{tid}/comm') as f:
                        comm = f.read().strip()
                    with open(f'/proc/{pid}/task/{tid}/stat') as f:
                        rest = f.read().rsplit(')', 1)[1].split()
                    if comm == 'tf-runner':
                        with open(f'/proc/{pid}/task/{tid}/status') as f:
                            allowed = next(l.split()[1] for l in f if l.startswith('Cpus_allowed_list'))
                        snap['runner'].append([int(tid), allowed, int(rest[36]), int(rest[11]) + int(rest[12])])
                    elif int(rest[11]) + int(rest[12]) > 0 and rest[0] == 'R':
                        snap['engine_cpus'].append(int(rest[36]))
                except (OSError, StopIteration, IndexError, ValueError):
                    pass
        samples.append(snap)
        stop.wait(1.0)


def log_size(a):
    return os.path.getsize(a.rank_log) if a.rank_log and os.path.exists(a.rank_log) else 0


def request_opts(a, arm):
    """The arm's request fields: its --arm-opts (the whole arm's key first, else its name before any `:`), plus `seed`
    from an `:sN` suffix."""
    opts = dict(a.arm_opts.get(arm, a.arm_opts.get(arm.split(':')[0])) or {})
    seed = next((p[1:] for p in arm.split(':')[1:] if p[:1] == 's' and p[1:].isdigit()), None)
    if seed is not None:
        opts['seed'] = int(seed)
    return opts


def run_arm(a, arm, tasks, model):
    letter, c1 = arm[0], arm.endswith(':c1')
    out = Path(a.out) / arm.replace(':', '-')
    out.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if not (out / f'{t["id"]}.json').exists()]  # a rerun resumes
    todo.sort(key=lambda t: -T.PRIOR_TOKENS.get(t['set'], 0))
    concurrency = 1 if c1 else a.concurrency
    cap = a.cap_c1_s if c1 else a.cap_s
    deadline = time.monotonic() + cap if cap else None
    log_start = log_size(a)
    lock = threading.Lock()
    samples, stop = [], threading.Event()
    sampler = threading.Thread(target=sample_placement, args=(stop, samples), daemon=True)
    sampler.start()
    t0 = time.time()
    order = []

    def worker():
        while True:
            with lock:
                if not todo or deadline is not None and time.monotonic() > deadline:
                    return  # past the cap nothing new starts; the rest stays unrun (a rerun resumes it)
                task = todo.pop(0)
            opts, own = bindings(a, request_opts(a, arm))
            limit = min(x for x in (deadline, own) if x is not None) if deadline or own else None
            rec = settle(a, stream_chat(a.base, model, task, letter, a.max_tokens, limit, opts), own, limit, arm)
            u = rec.get('usage') or {}
            save(out, task['id'], rec)
            with lock:
                order.append(task['id'])
                print(f'{arm} {task["id"]} {rec.get("finish")} {rec["elapsed_s"]:.1f}s '
                      f'{u.get("completion_tokens")} tok {rec.get("error", "")}', flush=True)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    sampler.join(timeout=5)
    opts = request_opts(a, arm)
    arm_meta = {'boot': a.boot, 'arm': arm, 'base': a.base, 'model': model, 'concurrency': concurrency,
                'request_opts': {k: (hashlib.sha256(v.encode()).hexdigest() if k == 'runner_guide' else v)
                                 for k, v in opts.items()},
                'max_tokens': a.max_tokens, 'forms': a.forms, 'sample': a.sample, 'started_unix': t0,
                'ended_unix': time.time(), 'finished_order': order, 'placement': samples,
                'rank_log_bytes': [log_start, log_size(a)], 'cap_s': cap,
                'planned': [t['id'] for t in tasks]}
    prev = out / 'arm.json'
    if prev.exists():  # resumed: keep the earlier part's samples too
        old = json.loads(prev.read_text())
        arm_meta['placement'] = old.get('placement', []) + samples
        arm_meta['finished_order'] = old.get('finished_order', []) + order
        arm_meta['started_unix'] = old.get('started_unix', t0)
        arm_meta['rank_log_bytes'] = [old.get('rank_log_bytes', [log_start])[0], log_size(a)]
    prev.write_text(json.dumps(arm_meta) + '\n')
    return out


def template_sha(body):
    """T5's template hash: sorted keys, no spaces, UTF-8 as written (`×` unescaped)."""
    text = json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def run_templates(a, tasks, model):
    """W9 (T5's frozen request-templates.json): this boot's requests in their frozen order (seeds interleaved), each
    body sent as frozen with only the model and the observability bindings (native ids, server deadline) added.
    No sorting, sampling or caps; replies go to OUT/<arm>-s<seed>/<task id>.json, one arm.json per seed."""
    spec = json.loads(Path(a.templates).read_text())
    assert spec['boot'] == a.boot, (spec['boot'], a.boot)
    arm, rows = spec['arm'], {t['id']: t for t in tasks}
    todo = sorted(spec['requests'], key=lambda r: r['within_boot'])
    assert [r['within_boot'] for r in todo] == list(range(1, len(todo) + 1)), 'within_boot not 1..n'
    for r in todo:
        assert template_sha(r['body_template']) == r['body_template_sha256'], f'{r["pair_id"]}: template sha'
        assert r['body_template']['seed'] == r['seed'] and r['pair_id'] == f'{r["task_id"]}:s{r["seed"]}', r['pair_id']
        assert r['body_template']['messages'] == [{'role': 'user', 'content': rows[r['task_id']]['prompt']}], \
            f'{r["pair_id"]}: prompt differs from the task row'
    outs = {}
    for r in todo:
        o = outs.setdefault(r['seed'], Path(a.out) / f'{arm}-s{r["seed"]}')
        o.mkdir(parents=True, exist_ok=True)
    todo = [r for r in todo if not (outs[r['seed']] / f'{r["task_id"]}.json').exists()]  # a rerun resumes
    log_start = log_size(a)
    lock = threading.Lock()
    samples, stop = [], threading.Event()
    sampler = threading.Thread(target=sample_placement, args=(stop, samples), daemon=True)
    sampler.start()
    t0 = time.time()
    order = {s: [] for s in outs}

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                r = todo.pop(0)
            body = copy.deepcopy(r['body_template'])
            body['model'] = model
            opts, own = bindings(a, {})
            rec = stream_chat(a.base, model, rows[r['task_id']], arm[0], body['max_tokens'], own, opts, body=body)
            rec = settle(a, rec, own, own, f'{arm}:s{r["seed"]}')
            rec.update(pair_id=r['pair_id'], within_boot=r['within_boot'], template_sha256=r['body_template_sha256'])
            save(outs[r['seed']], r['task_id'], rec)
            u = rec.get('usage') or {}
            with lock:
                order[r['seed']].append(r['task_id'])
                print(f'{arm}:s{r["seed"]} #{r["within_boot"]} {r["task_id"]} {rec.get("finish")} '
                      f'{rec["elapsed_s"]:.1f}s {u.get("completion_tokens")} tok {rec.get("error", "")}', flush=True)

    threads = [threading.Thread(target=worker) for _ in range(a.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    sampler.join(timeout=5)
    for seed, o in outs.items():
        mine = [r for r in spec['requests'] if r['seed'] == seed]
        meta = {'boot': a.boot, 'arm': f'{arm}:s{seed}', 'base': a.base, 'model': model, 'concurrency': a.concurrency,
                'templates': a.templates,
                'templates_sha256': hashlib.sha256(Path(a.templates).read_bytes()).hexdigest(),
                'request_opts': {k: v for k, v in bindings(a, {})[0].items()}, 'forms': a.forms, 'sample': 'templates',
                'started_unix': t0, 'ended_unix': time.time(), 'finished_order': order[seed], 'placement': samples,
                'rank_log_bytes': [log_start, log_size(a)], 'cap_s': 0,
                'planned': [r['task_id'] for r in sorted(mine, key=lambda r: r['within_boot'])]}
        prev = o / 'arm.json'
        if prev.exists():
            old = json.loads(prev.read_text())
            meta['finished_order'] = old.get('finished_order', []) + order[seed]
            meta['started_unix'] = old.get('started_unix', t0)
        prev.write_text(json.dumps(meta) + '\n')


# ---- replay ---------------------------------------------------------------------------------------------------------
def prompt_ids(base, task):
    return post(base, '/v1/tokenize', {'messages': [{'role': 'user', 'content': task['prompt']}],
                                       'add_generation_prompt': True,
                                       'chat_template_kwargs': {'enable_thinking': True}}, timeout=120)['tokens']


def stream_text(rec):
    """The reply as the model wrote it: reasoning, the think close, then content."""
    return rec['reasoning'] + (THINK_CLOSE + rec['content'] if rec['content'] or rec.get('finish') else '')


def guess(base, model, ids, prefix, n):
    more = post(base, '/v1/tokenize', {'prompt': prefix, 'add_special_tokens': False}, timeout=120)['tokens'] if prefix else []
    r = post(base, '/v1/completions', {'model': model, 'prompt': ids + more, 'max_tokens': n, 'temperature': 0},
             timeout=600)
    return r['choices'][0]['text']


def replay(a, src, tasks, model, forms, rank_log=None, validity=False):
    """One line per result position in arm directory src: the reply's text there and the model's own continuation from
    the same prefix, decoded on this (runner-off) boot. Runner arms with their boot's rank_log replay exactly the cells
    the runner owned (runner.zig's lines matched to replies as analyze.py matches them); without it, the first
    --replay-max positions; the validity check takes 2 per reply."""
    import analyze  # the scanner view and log pairing the report uses
    path = src / ('replay-validity.jsonl' if validity else 'replay.jsonl')
    done = {(r['id'], r['pos']) for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    forms = tuple(forms.split(','))
    owned = None
    if rank_log and not validity:
        recs = {p.stem: json.loads(p.read_text()) for p in src.glob('*.json') if p.name != 'arm.json'}
        meta = json.loads((src / 'arm.json').read_text())
        logs = analyze.log_replies(Path(rank_log), *meta['rank_log_bytes'])
        mapping, _ = analyze.match_replies(logs, recs, forms, meta['concurrency'])
        owned = {}
        for rid, tid in mapping.items():  # the owned cells' result positions, paired by form order (trace cells have none)
            p = src / f'{tid}.json'
            if not p.exists():
                continue
            paired = analyze.pair_cells(logs[rid], analyze.visible_cells(stream_text(json.loads(p.read_text())), forms))
            owned[tid] = {paired[i]['at'] for i, c in logs[rid]['cells'].items() if c['outcome'] == 'owned' and i in paired}
    jobs = []
    for task in tasks:
        p = src / f'{task["id"]}.json'
        if not p.exists():
            continue
        text = stream_text(json.loads(p.read_text()))
        cells = [c for c in analyze.visible_cells(text, forms) if c['at'] is not None]
        if validity:
            cells = cells[:2]
        elif owned is not None:
            cells = [c for c in cells if c['at'] in owned.get(task['id'], ())]
        else:
            cells = cells[:a.replay_max]
        jobs += [(task, text, c) for c in cells if (task['id'], c['at']) not in done]
    lock = threading.Lock()
    ids = {}

    def worker():
        while True:
            with lock:
                if not jobs:
                    return
                task, text, c = jobs.pop(0)
            n = 24 if c['form'] == 'inline' else 160
            try:
                if task['id'] not in ids:
                    ids[task['id']] = prompt_ids(a.base, task)
                g, err = guess(a.base, model, ids[task['id']], text[:c['at']], n), None
            except Exception as exc:
                g, err = None, repr(exc)[:200]
            row = {'id': task['id'], 'pos': c['at'], 'index': c['index'], 'form': c['form'], 'replayed_on': a.boot,
                   'reply': text[c['at']:c['at'] + 600], 'guess': g, 'error': err}
            with lock:
                with path.open('a') as f:
                    f.write(json.dumps(row) + '\n')

    threads = [threading.Thread(target=worker) for _ in range(a.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--boot', required=True)
    ap.add_argument('--arms', required=True)
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sets', default='')
    ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--max-tokens', type=int, default=32768)
    ap.add_argument('--forms', default='fence,repl')
    ap.add_argument('--arm-opts', default='{}', type=json.loads,
                    help='JSON: arm name -> extra request fields, e.g. {"rg": {"runner_guide": "...", "runner_trace": false}}')
    ap.add_argument('--replay', action='store_true', help="the validity check on this boot's runner-off arms")
    ap.add_argument('--replay-from', default='', help='runner boot dirs of this run whose owned cells replay here')
    ap.add_argument('--replay-max', type=int, default=4, help='positions replayed per reply when there is no rank log')
    ap.add_argument('--rank-log', help="this boot's rank0.log: replay the owned cells only")
    ap.add_argument('--limit', type=int, default=0, help='first N tasks of each set (dry runs)')
    ap.add_argument('--sample', default='all', help='a named sample from tasks.SAMPLES, or all')
    ap.add_argument('--cap-s', type=float, default=0, help='per c=4 arm: no new request after this, in-flight cut')
    ap.add_argument('--cap-c1-s', type=float, default=0, help='the same for :c1 arms')
    ap.add_argument('--request-deadline-s', type=float, default=0, help='per request: cut and kept as finish timeout')
    ap.add_argument('--server-deadline-field', default='',
                    help='request field carrying --request-deadline-s to the server; the client waits the grace more')
    ap.add_argument('--deadline-grace-s', type=float, default=30, help='with --server-deadline-field')
    ap.add_argument('--token-ids', action='store_true', help='ask for native ids (return_token_ids), check the hash')
    ap.add_argument('--no-count-fallback', action='store_true',
                    help='never fill missing usage with /v1/tokenize counts (usage_missing instead)')
    ap.add_argument('--arm-ids', default='{}', type=json.loads, help='JSON: arm -> its task ids (a frozen subset)')
    ap.add_argument('--templates', default='', help="W9: this boot's frozen request bodies (w9_kit.py), in order")
    a = ap.parse_args()
    if a.templates:  # the frozen requests are the whole boot: no sets, sample, limit, caps or replay
        tasks = [json.loads(l) for l in open(a.tasks)]
        model = json.load(urllib.request.urlopen(a.base + '/v1/models', timeout=30))['data'][0]['id']
        Path(a.out).mkdir(parents=True, exist_ok=True)
        print(f'BOOT {a.boot} templates {a.templates} model {model}', flush=True)
        t0 = time.monotonic()
        run_templates(a, tasks, model)
        print(f'ARM-DONE {a.boot} templates {time.monotonic() - t0:.0f}s', flush=True)
        return
    tasks = everything = [json.loads(l) for l in open(a.tasks)]
    if a.sets:
        keep = set(a.sets.split(','))
        tasks = [t for t in tasks if t['set'] in keep]
    tasks, c1 = T.sample(tasks, a.sample)
    if a.limit:
        seen = {}
        tasks = [t for t in tasks if seen.setdefault(t['set'], []).append(t) or len(seen[t['set']]) <= a.limit]
        c1 = c1 or {t['id'] for t in tasks[::max(1, len(tasks) // 4)]}
    model = json.load(urllib.request.urlopen(a.base + '/v1/models', timeout=30))['data'][0]['id']
    Path(a.out).mkdir(parents=True, exist_ok=True)
    print(f'BOOT {a.boot} arms {a.arms} tasks {len(tasks)} (c=1 subset {len(c1)}) sample {a.sample} model {model}',
          flush=True)
    for arm in a.arms.split(','):
        subset = [t for t in tasks if t['id'] in c1] if arm.endswith(':c1') else tasks
        if arm in a.arm_ids:
            subset = [t for t in subset if t['id'] in set(a.arm_ids[arm])]
        if request_opts(a, arm).get('temperature') and 'seed' not in request_opts(a, arm):
            print(f'WARN {arm}: sampled with no seed (not repeatable)', flush=True)
        t0 = time.monotonic()
        run_arm(a, arm, subset, model)
        print(f'ARM-DONE {a.boot} {arm} {time.monotonic() - t0:.0f}s', flush=True)
        greedy = not request_opts(a, arm).get('temperature')  # a greedy continuation checks only a greedy reply
        if a.replay and greedy and arm[0] in 'ad' and not arm.endswith(':c1'):  # validity at c=4; runners elsewhere
            t0 = time.monotonic()
            replay(a, Path(a.out) / arm, subset, model, a.forms, validity=True)
            print(f'REPLAY-DONE {a.boot} {arm} {time.monotonic() - t0:.0f}s', flush=True)
    for boot in filter(None, a.replay_from.split(',')):
        boot = Path(boot)
        for meta_path in sorted(boot.glob('r*/arm.json')):
            meta = json.loads(meta_path.read_text())
            t0 = time.monotonic()
            replay(a, meta_path.parent, everything, model, meta['forms'], rank_log=boot / 'rank0.log')
            print(f'REPLAY-DONE {a.boot} from {boot.name}/{meta_path.parent.name} {time.monotonic() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
