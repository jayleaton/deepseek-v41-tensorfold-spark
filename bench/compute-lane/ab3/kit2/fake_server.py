#!/usr/bin/env python3
"""A fake TensorFold server for the CPU dry run: OpenAI-style routes that replay T6's census replies, so measure.py,
the replay and analyze.py run end to end without a GPU.

    python3 -B fake_server.py --port 18181 --census ../t6-census/raw --tasks tasks.jsonl --mode off|fence|inline
                              --rank-log DIR/rank0.log [--speed 20000]

Each task gets a census reply of a matching category (by a hash of the id). Its answer is replaced by the task's
reference (1/3), a wrong answer (1/3) or kept as recorded (1/3), so every checker sees passes and failures.
Runner modes act like the runner on the reasoning: `inline` owns every inline expression's value (the exact value
replaces what the census reply wrote); `fence` adds one python cell with an `Output:` header to arithmetic replies and
owns its result. Owned results pause the stream (held time) and log TE's runner.zig lines to --rank-log, with replies
numbered in admission order. /v1/tokenize gives bytes as tokens; /v1/completions on a known prefix (from any fake's
reply) returns what the census reply (the model) wrote at that point, as a runner-off boot would, so replay on the
off fake recovers the model's own guesses."""
import argparse
import hashlib
import itertools
import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spans  # noqa: E402
import tasks as T  # noqa: E402

MODEL = 'DeepSeek-V4.1-Flash-TF'
CATEGORY = {'add30': 'arithmetic', 'mul30': 'big-number', 'mul_big': 'arithmetic', 'div_big': 'arithmetic',
            'bignum': 'big-number', 'mbpp': 'algorithmic', 'algo': 'algorithmic', 'build': 'algorithmic'}
OPS = {'add30': '+', 'mul30': '*'}


def h(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def fmt(v):
    return str(v.numerator) if v.denominator == 1 else f'{v.numerator}/{v.denominator}'


def mine(text, forms):
    """The cells the runner sees: the scanner's, less those on an [exact: ...] line (the runner's own trace text)."""
    def line(c):
        end = text.find('\n', c['start'])
        return text[c['start']:len(text) if end < 0 else end]
    return [c for c in spans.cells(text, forms) if '[exact:' not in line(c)]


class State:
    def __init__(self, a):
        self.a = a
        self.lock = threading.Lock()
        self.next_reply = 1
        self.census = {}
        for p in sorted(Path(a.census).glob('*.json')):
            r = json.loads(p.read_text())
            if 'response' in r:
                self.census.setdefault(r['task']['category'], []).append(r['response']['choices'][0]['message'])
        self.tasks = {t['prompt']: t for t in map(json.loads, open(a.tasks))}
        self.log = open(a.rank_log, 'a') if a.rank_log else None

    def log_line(self, s):
        if self.log:
            with self.lock:
                self.log.write(f'info(runner): {s}\n')
                self.log.flush()

    def compose(self, prompt, mode, trace=False, guided=False):
        """What a boot in `mode` serves for prompt: (task, tid, rng, reasoning, content, raw events, originals). Pure:
        the same prompt and mode give the same reply on every fake, as the same weights do on every boot. `guided` (a
        runner_guide sent) hands an arithmetic task off at the start, as S1's guides make the model do: one python cell,
        then `Result:`, where the runner fires at the colon (T5 94c65d209) and puts the value on the next line."""
        task = self.tasks.get(prompt)
        tid = task['id'] if task else prompt[:40]
        pool = self.census[CATEGORY.get(task['set'], 'algorithmic') if task else 'arithmetic']
        msg = pool[h(tid) % len(pool)]
        reasoning, content = msg.get('reasoning_content') or '', msg.get('content') or ''
        if task:
            pick = h(tid + 'answer') % 3
            content = T.reference_answer(task) if pick == 0 else T.wrong_answer(task) if pick == 1 else content
        events, originals = [], {}
        rng = random.Random(h(tid))
        if trace and task and task['set'] in OPS:  # trace: one owned [exact: ...] line after the first reasoning line
            x, y = [int(w) for w in task['prompt'].replace('?', ' ').split() if w.isdigit()][:2]
            v = x + y if OPS[task['set']] == '+' else x * y
            cut = reasoning.find('\n') + 1 if '\n' in reasoning else len(reasoning)
            lead = '' if cut == 0 or reasoning[cut - 1] == '\n' else '\n'  # the runner owns a whole line
            reasoning = reasoning[:cut] + f'{lead}[exact: {x} {OPS[task["set"]]} {y} = {v}]\n' + reasoning[cut:]
            events.append((None, 9, 'trace'))
        if mode == 'fence' and task and task['set'] in OPS:
            x, y = [int(w) for w in task['prompt'].replace('?', ' ').split() if w.isdigit()][:2]
            truth = str(x + y if OPS[task['set']] == '+' else x * y)
            guess = truth if rng.random() < 0.5 else truth[:-1] + str((int(truth[-1]) + 3) % 10)
            cell = f'\n\nLet me compute it.\n```python\nprint({x} {OPS[task["set"]]} {y})\n```\nOutput:\n'
            at = len(reasoning) + len(cell)
            reasoning = reasoning + cell + truth + '\n'
            originals[at] = guess + '\n' + reasoning[at + len(truth) + 1:]
            events.append((at, len(truth), 'python fence'))
        elif mode == 'inline':
            if guided and task and task['set'] in OPS:
                x, y = [int(w) for w in task['prompt'].replace('?', ' ').split() if w.isdigit()][:2]
                truth = str(x + y if OPS[task['set']] == '+' else x * y)
                guess = truth if rng.random() < 0.5 else truth[:-1] + str((int(truth[-1]) + 3) % 10)
                cell = f'Need compute. Use python.\n```python\nprint({x} {OPS[task["set"]]} {y})\n```\nResult:\n'
                rest = '\n' + reasoning
                reasoning = cell + truth + rest
                originals[len(cell)] = guess + rest[:200]
                events.append((len(cell), len(truth), 'python fence'))
            out, last = [], 0
            for c in mine(reasoning, ('inline',)):
                v = spans.exact(c['code'])
                if v is None:  # the runner claims it and abstains (unsupported), as runner.zig logs it
                    events.append((sum(map(len, out)) + c['at'] - last, 0, 'abstain'))
                    continue
                written = spans.value_at(reasoning, c['at'], 'inline')
                out.append(reasoning[last:c['at']])
                at = sum(map(len, out))
                out.append(fmt(v))
                last = c['at'] + len(written)
                originals[at] = written + reasoning[last:last + 200]
                events.append((at, len(fmt(v)), 'python inline_expr'))
            out.append(reasoning[last:])
            reasoning = ''.join(out)
        return task, tid, rng, reasoning, content, events, originals

    def model_text(self, prompt, prefix):
        """The model's own text after prefix: what a runner-off boot decodes there (the runner never runs on
        /v1/completions here). A prefix from a runner reply (any mode's) continues with the model's original guess."""
        for mode in dict.fromkeys((self.a.mode, 'off', 'fence', 'inline')):  # its own replies first, as a real boot
            for trace, guided in itertools.product((False, True), (False, True)):
                _, _, _, reasoning, content, _, originals = self.compose(prompt, mode, trace, guided)
                served = reasoning + '</think>' + content
                if served.startswith(prefix):
                    return originals.get(len(prefix), served[len(prefix):])
        return ''

    def reply(self, prompt, lane, trace=False, guided=False):
        """(reasoning, content, owned events [(pos in stream text, held s)], log lines, original guesses)."""
        mode = self.a.mode
        task, tid, rng, reasoning, content, events, originals = self.compose(prompt, mode, trace and mode == 'inline',
                                                                             guided and mode == 'inline')
        lines = []
        if mode != 'off':
            with self.lock:
                rid = self.next_reply
                self.next_reply += 1
            forms = ('fence', 'repl', 'inline') if mode == 'inline' else ('fence', 'repl')
            closed = mine(reasoning + '</think>' + content, forms)
            shift = 2 if any(k == 'trace' for _, _, k in events) else 0  # the trace cell, and one unclaimed trace candidate
            index = {c['at']: c['index'] + shift for c in closed}
            timed = []
            for at, n, kind in events:
                if kind == 'abstain':
                    lines.append(f'reply {rid} cell {index[at]} (python inline_expr): abstain (unsupported), held '
                                 f'{rng.randint(1, 9)} ms')
                    continue
                if kind == 'trace':
                    held = rng.randint(2, 40)
                    lines.append(f'reply {rid} cell 0 (trace): owned {n} tokens, {rng.randint(10, 400)} steps, run 1 ms, '
                                 f'held {held} ms')
                    continue
                if at not in index:
                    raise RuntimeError(f'{tid}: the scanner does not see the owned cell at {at}')
                held = rng.randint(2, 40)
                lines.append(f'reply {rid} cell {index[at]} ({kind}): owned {max(1, n // 3)} tokens, ok after '
                             f'{rng.randint(30, 400)} steps, run {rng.randint(0, 3)} ms, held {held} ms')
                timed.append((at, held / 1000))
            events = timed
            if closed or shift:
                lines.append(f'reply {rid}: {len(closed) + shift} cells closed, {len(events)} owned, 0 abstained at a header, '
                             f'{len(closed) - len(events)} never claimed')
        extra = {'tensorfold': {'tokens_per_second': 30.0 + rng.random() * 8}}
        if lane and task and task['set'] in ('mbpp', 'algo'):
            extra['tensorfold']['compute'] = {'calls': [{'exec_ms': rng.randint(5, 80)}], 'inserted_tokens': 40}
        return reasoning, content, events, lines, extra


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == '/v1/models':
            return self.send({'data': [{'id': MODEL}]})
        self.send({'error': 'no route'}, 404)

    def do_POST(self):
        st = self.server.state
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/v1/tokenize':
            text = (f'<U>{body["messages"][0]["content"]}<A><think>' if 'messages' in body else body['prompt'])
            ids = list(text.encode())
            return self.send({'count': len(ids), 'tokens': ids})
        if self.path == '/v1/completions':
            text = bytes(body['prompt']).decode(errors='replace')
            prompt, _, prefix = text[3:].partition('<A><think>')
            n = body.get('max_tokens', 16) * 4
            cont = st.model_text(prompt, prefix)[:n]
            return self.send({'choices': [{'index': 0, 'text': cont, 'finish_reason': 'length'}]})
        if self.path != '/v1/chat/completions':
            return self.send({'error': 'no route'}, 404)
        lane = any(t.get('type') == 'code_interpreter' for t in body.get('tools', []))
        reasoning, content, events, lines, extra = st.reply(body['messages'][0]['content'], lane,
                                                             st.a.trace and body.get('runner_trace') is not False,
                                                             bool(body.get('runner_guide')))
        if body.get('runner_guide') and st.a.mode != 'off':  # as TE's contract: the guide's hash comes back
            extra['tensorfold']['runner_guide'] = {'sha256': hashlib.sha256(body['runner_guide'].encode()).hexdigest()}
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        speed = st.a.speed
        pos, held = 0, dict(events)
        t_start = time.monotonic()
        stop_at = body.get(st.a.deadline_field) if st.a.deadline_field else None  # a server-side request deadline
        late = lambda: stop_at is not None and time.monotonic() - t_start > stop_at  # noqa: E731

        def chunk(delta, finish=None):
            c = {'id': 'fake', 'object': 'chat.completion.chunk', 'model': MODEL,
                 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
            self.wfile.write(f'data: {json.dumps(c)}\n\n'.encode())
            self.wfile.flush()
        time.sleep(0.02)
        cuts = sorted(set(list(range(0, len(reasoning), 16)) + list(held) + [len(reasoning)]))
        logged = 0
        for i, start in enumerate(cuts[:-1]):
            if late():
                break
            if start in held:
                time.sleep(held[start])
                st.log_line(lines[logged])
                logged += 1
            piece = reasoning[start:cuts[i + 1]]
            chunk({'reasoning_content': piece})
            time.sleep(len(piece) / speed)
            pos += len(piece)
        for line in lines[logged:]:
            st.log_line(line)
        sent = ''
        for i in range(0, len(content), 16):
            if late():
                break
            chunk({'content': content[i:i + 16]})
            sent += content[i:i + 16]
            time.sleep(16 / speed)
        ended = late() or pos < len(reasoning)
        reasoning, content = reasoning[:pos], sent
        chunk({}, 'deadline' if ended else 'stop')
        u = {'prompt_tokens': 40, 'completion_tokens': (len(reasoning) + len(content)) // 4,
             'completion_tokens_details': {'reasoning_tokens': len(reasoning) // 4}}
        ids = list(reasoning.encode()) + [0] + list(content.encode())  # bytes as tokens, as /v1/tokenize here
        extra['tensorfold']['token_sha256'] = hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()
        if body.get('return_token_ids'):
            extra['tensorfold']['token_ids'] = ids
        self.wfile.write(f'data: {json.dumps(dict({"id": "fake", "choices": [], "usage": u}, **extra))}\n\n'.encode())
        self.wfile.write(b'data: [DONE]\n\n')
        self.wfile.flush()
        self.close_connection = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--census', required=True)
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--mode', choices=('off', 'fence', 'inline'), default='off')
    ap.add_argument('--trace', action='store_true', help='the boot has the trace form: requests without runner_trace false get one')
    ap.add_argument('--rank-log')
    ap.add_argument('--speed', type=float, default=20000.0, help='chars a second per reply')
    ap.add_argument('--deadline-field', default='', help='a request field the fake honours as a server-side deadline')
    a = ap.parse_args()
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), Handler)
    srv.state = State(a)
    print(f'fake server :{a.port} mode {a.mode}', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
