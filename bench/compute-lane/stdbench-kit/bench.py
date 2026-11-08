#!/usr/bin/env python3
"""Serial, stdlib-only paired benchmark. Offline unless --live AND a GO file."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL = "DeepSeek-V4.1-Flash-TF"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    with open(path, "x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")


def validate_tasks(tasks):
    if not tasks:
        raise ValueError("empty task set")
    seen = set()
    for task in tasks:
        key = (task["suite"], task["id"])
        if not all(isinstance(x, str) and x for x in key) or key in seen:
            raise ValueError("missing/duplicate/non-string task identity: " + repr(key))
        seen.add(key)
        if not isinstance(task.get("prompt"), str) or not task["prompt"]:
            raise ValueError("task prompt required")
        if task.get("kind") not in ("integer", "numeric", "math_manual", "humaneval", "mbpp"):
            raise ValueError("unsupported task kind")
        if task["kind"] in ("humaneval", "mbpp"):
            if not isinstance(task.get("test"), str) or not task["test"].strip():
                raise ValueError("coding task requires nonempty base tests")
            if not re.fullmatch(r"[A-Za-z_]\w*", task.get("entry_point", "")):
                raise ValueError("coding task requires entry_point")
        elif "answer" not in task:
            raise ValueError("math task requires answer")
    return tasks


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("redirect refused; explicitly lock/use the final endpoint")


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise ValueError("dataset redirect must remain HTTPS without credentials")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def locked_sources(args):
    lock_path = Path(args.lock).resolve()
    lock = json.loads(lock_path.read_text())
    if lock.get("version") != 1:
        raise ValueError("unsupported lock version")
    entries = lock["datasets"]
    suites = args.suites.split(",") if args.suites else [x["suite"] for x in entries]
    if len(set(suites)) != len(suites):
        raise ValueError("duplicate suites")
    if len({x["suite"] for x in entries}) != len(entries):
        raise ValueError("duplicate locked suite")
    selected = [x for x in entries if x["suite"] in suites]
    if set(suites) != {x["suite"] for x in selected}:
        raise ValueError("suite absent from lock")
    for entry in selected:
        revision, sha = entry["revision"], entry["sha256"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("lock must pin immutable 40-hex revision and SHA256")
        url = urllib.parse.urlsplit(entry["url"])
        if url.scheme != "https" or revision not in url.path.split("/"):
            raise ValueError("URL must contain immutable revision path component")
        path = (lock_path.parent / entry["path"]).resolve()
        if lock_path.parent not in path.parents:
            raise ValueError("locked path escapes kit directory")
        if args.command == "fetch":
            if not args.fetch_network:
                raise ValueError("fetch requires explicit --fetch-network; normalize is offline")
            if not path.exists():
                with urllib.request.build_opener(HTTPSRedirect).open(entry["url"], timeout=60) as response:
                    blob = response.read()
                if hashlib.sha256(blob).hexdigest() != sha:
                    raise ValueError("download SHA256 mismatch: " + entry["suite"])
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "xb") as stream:
                    stream.write(blob)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("source SHA256 mismatch: " + str(path))
        yield entry, path


def normalize(args):
    tasks = []
    for source, path in locked_sources(args):
        if args.command == "fetch":
            continue
        for index, row in enumerate(read_jsonl(path)):
            fmt = source["format"]
            task = {"suite": source["suite"], "source_sha256": source["sha256"],
                    "source_revision": source["revision"]}
            if fmt == "gsm8k":
                if "####" not in row["answer"]:
                    raise ValueError("GSM8K reference lacks #### delimiter")
                task.update(id=str(row.get("id", index)), prompt=row["question"] + "\n\nReturn final answer after ####.",
                            answer=row["answer"].rsplit("####", 1)[1].strip(), kind="numeric")
            elif fmt == "math500":
                task.update(id=str(row.get("unique_id", row.get("id", index))),
                            prompt=row["problem"], answer=row["answer"], kind="math_manual")
            elif fmt == "humaneval":
                task.update(id=str(row["task_id"]),
                            prompt=row["prompt"] + "\n\nReturn the complete function in a single Python code fence.",
                            code_prefix=row["prompt"], test=row["test"],
                            entry_point=row["entry_point"], kind="humaneval")
            elif fmt == "mbpp":
                # MBPP mapping must explicitly supply an entry point; never infer by executing tests.
                tests = row.get("test", row.get("test_list"))
                task.update(id=str(row.get("task_id", row.get("id", index))),
                            prompt=row.get("prompt", row.get("text")),
                            test="\n".join(tests) if isinstance(tests, list) else tests,
                            entry_point=row["entry_point"], kind="mbpp")
            elif fmt == "normalized":
                task.update(row)
                task.update(suite=source["suite"], source_sha256=source["sha256"],
                            source_revision=source["revision"])
                task["id"] = str(task["id"])
            else:
                raise ValueError("unsupported source format: " + fmt)
            tasks.append(task)
    if args.command == "fetch":
        return
    validate_tasks(tasks)
    selected = []
    for suite in sorted({t["suite"] for t in tasks}):
        rows = [t for t in tasks if t["suite"] == suite]
        rows.sort(key=lambda t: (hashlib.sha256(("stdbench-v1:" + t["id"]).encode()).hexdigest(), t["id"]))
        selected.extend(rows[:args.limit] if args.limit else rows)
    write_jsonl(args.out, selected)


def request_payload(task, seed, arm, model, max_tokens, deadline_s=270):
    payload = {"model": model, "messages": [{"role": "user", "content": task["prompt"]}],
               "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": seed,
               "max_tokens": max_tokens, "stream": False,
               "reasoning_effort": "high", "deadline_s": deadline_s,
               "chat_template_kwargs": {"enable_thinking": True}}
    if arm == "lane":
        payload["tools"] = [{"type": "code_interpreter"}]
    return payload


def pair_key(row):
    return (row["suite"], row["id"], row["seed"])


class Replay:
    def __init__(self, rows):
        self.rows = {}
        for row in rows:
            key = pair_key(row)
            if key in self.rows:
                raise ValueError("duplicate replay key: " + repr(key))
            self.rows[key] = row
        self.used = set()

    def __call__(self, wire, task, seed):
        key = (task["suite"], task["id"], seed)
        if key in self.used or key not in self.rows:
            raise ValueError("duplicate/missing replay pair: " + repr(key))
        self.used.add(key)
        row = self.rows[key]
        # Decode the actual serialized request even offline. Optional exact request assertion.
        request = json.loads(wire)
        if "request" in row and row["request"] != request:
            raise ValueError("replay request does not match serialized request")
        return {k: row[k] for k in ("raw_response", "status", "error", "http_status", "elapsed_seconds") if k in row}

    def finish(self):
        if self.used != set(self.rows):
            raise ValueError("unused replay pairs")


class Live:
    def __init__(self, endpoint, timeout, go):
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("explicit http(s) chat-completions endpoint required")
        if not parsed.path.endswith("/chat/completions"):
            raise ValueError("endpoint must be the full /chat/completions URL")
        self.endpoint, self.timeout, self.go = endpoint, timeout, Path(go)
        self.opener = urllib.request.build_opener(NoRedirect)

    def __call__(self, wire, task, seed):
        if not self.go.is_file() or self.go.read_text().strip() != "GO":
            raise ValueError("live requires an existing GO file containing exactly GO")
        headers = {"Content-Type": "application/json"}
        token = os.environ.get("OPENAI_API_KEY")
        if token:
            headers["Authorization"] = "Bearer " + token
        req = urllib.request.Request(self.endpoint, data=wire, headers=headers, method="POST")
        if signal.getitimer(signal.ITIMER_REAL)[0]:
            raise ValueError("live serial client requires no pre-existing SIGALRM timer")
        def deadline_expired(signum, frame):
            raise TimeoutError("whole-request wall timeout exceeded")
        old_handler = signal.signal(signal.SIGALRM, deadline_expired)
        signal.setitimer(signal.ITIMER_REAL, self.timeout)
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                return {"status": "ok", "http_status": response.status,
                        "raw_response": response.read().decode("utf-8", errors="replace")}
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read().decode("utf-8", errors="replace")
                return {"status": "error", "http_status": exc.code, "error": str(exc), "raw_response": raw}
            except (TimeoutError, socket.timeout) as body_error:
                return {"status": "timeout", "http_status": exc.code, "error": str(body_error), "raw_response": None}
            except OSError as body_error:
                return {"status": "error", "http_status": exc.code, "error": str(body_error), "raw_response": None}
            finally:
                exc.close()
        except (TimeoutError, socket.timeout) as exc:
            return {"status": "timeout", "error": str(exc), "raw_response": None}
        except urllib.error.URLError as exc:
            return {"status": "timeout" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "error",
                    "error": str(exc), "raw_response": None}
        except (OSError, ValueError) as exc:
            return {"status": "error", "error": str(exc), "raw_response": None}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


def usage_counts(usage):
    if not isinstance(usage, dict):
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None, "reasoning_tokens": None}
    def number(value):
        return value if type(value) is int and value >= 0 else None
    prompt, completion = number(usage.get("prompt_tokens")), number(usage.get("completion_tokens"))
    details = usage.get("completion_tokens_details") or {}
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion if prompt is not None and completion is not None else None,
            "reasoning_tokens": number(details.get("reasoning_tokens")) if isinstance(details, dict) else None}


def parse_response(record):
    record.update(content=None, finish_reason=None, usage=None)
    if record["status"] == "ok":
        try:
            body = json.loads(record["raw_response"])
            choices = body["choices"]
            if len(choices) != 1:
                raise ValueError("expected exactly one choice")
            choice = choices[0]
            record["finish_reason"] = choice.get("finish_reason")
            record["usage"] = body.get("usage")
            record["content"] = choice["message"].get("content")
            if "tool_calls" in choice["message"]:
                raise ValueError("tool_calls are not a final answer; no client tool runner")
            if not isinstance(record["content"], str):
                raise ValueError("response content is not text")
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            record.update(status="error", error="invalid response: " + str(exc))
    record["tokens"] = usage_counts(record["usage"])


def run_records(tasks, seeds, arm, model, max_tokens, transport, emit, deadline_s=270, order_seed=20261104):
    ordered = list(tasks)
    random.Random(order_seed).shuffle(ordered)
    for task in ordered:
        for seed in seeds:
            payload = request_payload(task, seed, arm, model, max_tokens, deadline_s)
            wire = canonical(payload).encode("utf-8")
            start = time.monotonic()
            result = transport(wire, task, seed)
            observed = time.monotonic() - start
            if result.get("status") not in ("ok", "error", "timeout"):
                raise ValueError("transport/replay status must be ok/error/timeout")
            elapsed = result.get("elapsed_seconds", observed)
            if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("invalid recorded elapsed time")
            record = {"suite": task["suite"], "id": task["id"], "seed": seed,
                      "task_sha256": digest(task), "arm": arm, "request": payload,
                      "status": result["status"], "raw_response": result.get("raw_response"),
                      "http_status": result.get("http_status"), "error": result.get("error"),
                      "elapsed_seconds": elapsed, "transport_observed_seconds": observed,
                      "timing_source": "recorded" if "elapsed_seconds" in result else "monotonic"}
            parse_response(record)
            emit(record)
    if isinstance(transport, Replay):
        transport.finish()


def final_answer(text):
    # Never pick the last arbitrary number from a chain of reasoning.
    if not isinstance(text, str):
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = text.strip()
    boxes = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start, depth = match.end(), 1
        for position in range(start, len(text)):
            if text[position] == "{" and (position == 0 or text[position - 1] != "\\"):
                depth += 1
            elif text[position] == "}" and (position == 0 or text[position - 1] != "\\"):
                depth -= 1
            if depth == 0:
                boxes.append(text[start:position].strip())
                break
        else:
            return None
    if boxes:
        if len(set(boxes)) != 1:
            return None
        return boxes[-1].strip()
    if "####" in text:
        return text.rsplit("####", 1)[1].strip()
    match = re.search(r"(?:Final answer|Answer)\s*:\s*([^\n]+)\s*$", text, re.I)
    return match.group(1).strip() if match else text


def numeric(text, integer=False):
    text = final_answer(str(text))
    if text is None:
        return None
    text = text.strip()
    if text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()
    # Thousands separators only when syntactically valid. No units, rounding, eval, or tolerance.
    pattern = r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?"
    if not re.fullmatch(pattern, text):
        return None
    try:
        result = Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None
    if integer and result != result.to_integral_value():
        return None
    return result


def code_program(task, content):
    content = content.strip("\r\n")
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1].strip("\r\n")
    fence = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", content.strip(), re.S)
    if fence:
        content = fence.group(1)
    if task["kind"] == "humaneval":
        if re.search(r"(?m)^def\s+" + re.escape(task["entry_point"]) + r"\s*\(", content):
            candidate = content
        else:
            # HumanEval's native completion protocol appends a function body to its prompt.
            candidate = task.get("code_prefix", task["prompt"]) + "\n" + content
        return candidate + "\n" + task["test"] + "\ncheck(" + task["entry_point"] + ")\n"
    return content + "\n" + task["test"] + "\n"


def code_grade(task, content, timeout, memory_mb, isolated, trusted=False):
    if not trusted:
        marker = Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()
        if not isolated or not marker:
            return {"grade": "manual_review", "reason": "untrusted coding requires explicitly acknowledged isolated container"}
    program = code_program(task, content)
    with tempfile.TemporaryDirectory(prefix="stdbench-grade-") as cwd:
        receipt_read, receipt_write = os.pipe()
        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", str(ROOT / "grader_worker.py"), str(max(1, math.ceil(timeout))),
                 str(memory_mb * 1024 * 1024), str(receipt_write)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                pass_fds=(receipt_write,), cwd=cwd,
                env={"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            os.close(receipt_write)
            receipt_write = None
            timed_out = False
            try:
                proc.communicate(canonical({"program": program}).encode(), timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Only this exact child PID; the required container forbids descendants.
                if proc.poll() is None:
                    proc.kill()
                proc.communicate()
            os.set_blocking(receipt_read, False)
            try:
                receipt = os.read(receipt_read, 64)
            except BlockingIOError:
                receipt = b""
        finally:
            os.close(receipt_read)
            if receipt_write is not None:
                os.close(receipt_write)
    if timed_out or proc.returncode == -signal.SIGXCPU:
        return {"grade": "incorrect", "reason": "grading_timeout", "grading_timeout": True}
    passed = proc.returncode == 0 and receipt == b"stdbench-tests-complete"
    return {"grade": "correct" if passed else "incorrect",
            "reason": "base_tests_pass" if passed else "base_tests_fail_or_missing_receipt",
            "harness_receipt": receipt == b"stdbench-tests-complete",
            "worker_returncode": proc.returncode}


def rational(text):
    value = final_answer(text)
    if value is None:
        return None
    value = value.strip().strip("$").strip()
    number = numeric(value)
    if number is not None:
        return Fraction(number)
    fraction = re.fullmatch(r"\\(?:d?frac)\{([+-]?\d+)\}\{([+-]?\d+)\}", value)
    if fraction:
        numerator, denominator = map(int, fraction.groups())
    else:
        fraction = re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", value)
        if not fraction:
            return None
        numerator, denominator = map(int, fraction.groups())
    return Fraction(numerator, denominator) if denominator else None


def math_grade(answer, content):
    expected, actual = final_answer(answer), final_answer(content)
    def clean(value):
        return re.sub(r"\s+", " ", value.strip().strip("$").strip()) if value is not None else None
    if clean(expected) and clean(expected) == clean(actual):
        return {"grade": "correct", "reason": "exact_canonical_string"}
    gold, prediction = rational(answer), rational(content)
    if gold is not None and prediction is not None:
        return {"grade": "correct" if gold == prediction else "incorrect", "reason": "exact_rational"}
    return {"grade": "manual_review", "reason": "MATH_equivalence_unsupported"}


def grade_record(task, record, timeout=3, memory_mb=256, isolated=False, trusted=False):
    if record["status"] != "ok":
        return {"grade": "incorrect", "reason": "generation_" + record["status"]}
    if record["finish_reason"] == "length":
        return {"grade": "incorrect", "reason": "truncated"}
    if record["finish_reason"] != "stop":
        return {"grade": "incorrect", "reason": "unsupported_finish_reason"}
    kind = task["kind"]
    if kind in ("humaneval", "mbpp"):
        return code_grade(task, record["content"], timeout, memory_mb, isolated, trusted)
    if kind == "math_manual":
        return math_grade(task["answer"], record["content"])
    expected = numeric(task["answer"], kind == "integer")
    actual = numeric(record["content"], kind == "integer")
    if expected is None:
        return {"grade": "manual_review", "reason": "unsupported_reference_number"}
    return {"grade": "correct" if actual is not None and actual == expected else "incorrect",
            "reason": "exact_numeric", "normalized_answer": str(actual) if actual is not None else None}


def index_records(rows):
    result = {}
    for row in rows:
        key = pair_key(row)
        if key in result:
            raise ValueError("duplicate task+seed pair: " + repr(key))
        if row.get("grade") not in ("correct", "incorrect", "manual_review"):
            raise ValueError("ungraded record")
        result[key] = row
    if not result:
        raise ValueError("empty results")
    return result


def safe_ratio(a, b):
    return a / b if b else None


def binomial_exact(wins, losses):
    n = wins + losses
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2 ** n))


def paired_metrics(pairs):
    scored = [(a, b) for a, b in pairs if "manual_review" not in (a["grade"], b["grade"])]
    ac = sum(a["grade"] == "correct" for a, b in scored)
    bc = sum(b["grade"] == "correct" for a, b in scored)
    result = {"pairs": len(pairs), "scored_pairs": len(scored), "manual_excluded_pairs": len(pairs) - len(scored),
              "plain_accuracy": ac / len(scored) if scored and len(scored) == len(pairs) else None,
              "lane_accuracy": bc / len(scored) if scored and len(scored) == len(pairs) else None,
              "accuracy_delta_lane_minus_plain": (bc - ac) / len(scored) if scored and len(scored) == len(pairs) else None,
              "resolved_subset": {"pairs": len(scored), "plain_accuracy": ac / len(scored) if scored else None,
                                  "lane_accuracy": bc / len(scored) if scored else None,
                                  "accuracy_delta": (bc - ac) / len(scored) if scored else None}}
    for metric in ("prompt_tokens", "completion_tokens", "total_tokens", "elapsed_seconds"):
        def value(row):
            return row[metric] if metric == "elapsed_seconds" else row["tokens"][metric]
        common = [(value(a), value(b)) for a, b in pairs if value(a) is not None and value(b) is not None]
        av, bv = sum(a for a, b in common), sum(b for a, b in common)
        result[metric] = {"common_pairs": len(common), "missing_pairs": len(pairs) - len(common),
                          "plain_sum": av if common else None, "lane_sum": bv if common else None,
                          "plain_mean": av / len(common) if common else None,
                          "lane_mean": bv / len(common) if common else None,
                          "delta_lane_minus_plain": bv - av if common else None,
                          "ratio_lane_over_plain": safe_ratio(bv, av) if common else None}
    return result


def flatten_metrics(metrics):
    result = {"accuracy_delta_lane_minus_plain": metrics["accuracy_delta_lane_minus_plain"]}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens", "elapsed_seconds"):
        for field in ("delta_lane_minus_plain", "ratio_lane_over_plain"):
            result[name + "." + field] = metrics[name][field]
    return result


def cluster_bootstrap(pairs, repetitions, seed):
    groups = defaultdict(list)
    for a, b in pairs:
        groups[(a["suite"], a["id"])].append((a, b))
    # Stratify by suite, resample tasks, retain every seed of each sampled task.
    strata = defaultdict(list)
    for (suite, task_id), rows in sorted(groups.items()):
        strata[suite].append(rows)
    rng = random.Random(seed)
    samples = defaultdict(list)
    for _ in range(repetitions):
        drawn = []
        for suite in sorted(strata):
            pool = strata[suite]
            for _ in pool:
                drawn.extend(rng.choice(pool))
        for name, value in flatten_metrics(paired_metrics(drawn)).items():
            if value is not None:
                samples[name].append(value)
    result = {}
    for name in flatten_metrics(paired_metrics(pairs)):
        values = sorted(samples[name])
        result[name] = {"low": values[int((len(values) - 1) * .025)] if values else None,
                        "high": values[int((len(values) - 1) * .975)] if values else None,
                        "valid_replicates": len(values)}
    return {"method": "95% percentile, task clusters retaining all seeds, stratified by suite",
            "replicates": repetitions, "seed": seed, "intervals": result}


def summarize(pairs, repetitions, seed):
    result = paired_metrics(pairs)
    wins = sum(a["grade"] == "incorrect" and b["grade"] == "correct" for a, b in pairs)
    losses = sum(a["grade"] == "correct" and b["grade"] == "incorrect" for a, b in pairs)
    result["mcnemar_task_seed_descriptive"] = {"lane_only_correct": wins, "plain_only_correct": losses,
                               "discordant_pairs": wins + losses, "two_sided_p": binomial_exact(wins, losses),
                               "caveat": "task+seed binomial; repeated seeds are correlated; use task bootstrap CI"}
    primary = [(a, b) for a, b in pairs if a["seed"] == 101]
    resolved = [(a, b) for a, b in primary if "manual_review" not in (a["grade"], b["grade"])]
    primary_wins = sum(a["grade"] == "incorrect" and b["grade"] == "correct" for a, b in resolved)
    primary_losses = sum(a["grade"] == "correct" and b["grade"] == "incorrect" for a, b in resolved)
    result["mcnemar_primary_seed101"] = {"independent_tasks": len(primary), "resolved_tasks": len(resolved),
        "lane_only_correct": primary_wins, "plain_only_correct": primary_losses,
        "two_sided_p": binomial_exact(primary_wins, primary_losses) if primary and len(primary) == len(resolved) else None,
        "withheld_for_manual_or_missing_seed101": not primary or len(primary) != len(resolved)}
    for label, position in (("plain", 0), ("lane", 1)):
        rows = [pair[position] for pair in pairs]
        wall = sorted(r["elapsed_seconds"] for r in rows)
        middle = len(wall) // 2
        result[label] = {"records": len(rows), "grade_counts": dict(Counter(r["grade"] for r in rows)),
                         "wall_mean_seconds": sum(wall) / len(wall),
                         "wall_median_seconds": wall[middle] if len(wall) % 2 else (wall[middle - 1] + wall[middle]) / 2,
                         "generation_errors": sum(r["status"] == "error" for r in rows),
                         "generation_timeouts": sum(r["status"] == "timeout" for r in rows),
                         "grading_timeouts": sum(bool(r.get("grading_timeout")) for r in rows),
                         "truncated": sum(r["finish_reason"] == "length" for r in rows),
                         "missing_usage": sum(r["usage"] is None for r in rows)}
    result["bootstrap"] = cluster_bootstrap(pairs, repetitions, seed)
    return result


def make_report(plain, lane, tasks, seeds, repetitions=20000, seed=20261104):
    left, right = index_records(plain), index_records(lane)
    validate_tasks(tasks)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("empty/duplicate manifest seeds")
    expected = {(t["suite"], t["id"], s) for t in tasks for s in seeds}
    task_map = {(t["suite"], t["id"]): t for t in tasks}
    if left.keys() != expected or right.keys() != expected:
        raise ValueError("missing/unexpected pairs against task+seed manifest")
    if left.keys() != right.keys():
        raise ValueError("missing pairs: plain-only=" + repr(sorted(left.keys() - right.keys())) +
                         "; lane-only=" + repr(sorted(right.keys() - left.keys())))
    pairs = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if a["arm"] != "plain" or b["arm"] != "lane":
            raise ValueError("arm labels must be plain and lane")
        if a["task_sha256"] != b["task_sha256"]:
            raise ValueError("paired task content differs")
        if a["task_sha256"] != digest(task_map[key[:2]]):
            raise ValueError("paired task differs from manifest")
        ra, rb = dict(a["request"]), dict(b["request"])
        if "tools" in ra or rb.pop("tools", None) != [{"type": "code_interpreter"}] or ra != rb:
            raise ValueError("paired request settings differ beyond native lane tool")
        pairs.append((a, b))
    # Every task within a suite must have exactly the same seeds, even if both arms omit a pair.
    suite_seeds = defaultdict(set)
    task_seeds = defaultdict(set)
    for suite, task_id, task_seed in left:
        suite_seeds[suite].add(task_seed)
        task_seeds[(suite, task_id)].add(task_seed)
    for (suite, task_id), seeds in task_seeds.items():
        if seeds != suite_seeds[suite]:
            raise ValueError("incomplete seed grid for " + repr((suite, task_id)))
    return {"schema_version": 1, "direction": "lane minus plain; ratios lane / plain",
            "usage_policy": "total = prompt + completion; reasoning is a completion subset, never added; null stays missing",
            "accuracy_policy": "primary accuracy withheld if any manual pair remains; resolved subset shown separately; errors/timeouts/truncation incorrect",
            "overall": summarize(pairs, repetitions, seed),
            "per_suite": {suite: summarize([(a, b) for a, b in pairs if a["suite"] == suite], repetitions, seed)
                          for suite in sorted(suite_seeds)},
            "paired_tasks": [{"suite": a["suite"], "id": a["id"], "seed": a["seed"],
                              "plain_grade": a["grade"], "lane_grade": b["grade"],
                              "accuracy_delta": (int(b["grade"] == "correct") - int(a["grade"] == "correct"))
                              if "manual_review" not in (a["grade"], b["grade"]) else None,
                              "metrics": paired_metrics([(a, b)])} for a, b in pairs]}


def manifest(tasks, seeds, arm, model, max_tokens, mode, endpoint, request_timeout, deadline_s=270, order_seed=20261104):
    return {"version": 1, "tasks_sha256": digest(tasks), "tasks": tasks, "seeds": seeds,
            "arm": arm, "model": model, "underlying_build": "1017 (operator must verify server identity)",
            "max_tokens": max_tokens, "mode": mode, "endpoint": endpoint, "request_timeout": request_timeout,
            "deadline_s": deadline_s, "order_seed": order_seed,
            "workers": 1, "retries": 0}


def run(args):
    tasks = validate_tasks(read_jsonl(args.tasks))
    seeds = [int(x) for x in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if args.live:
        if args.replay or not args.endpoint or not args.go_file:
            raise ValueError("live requires --endpoint and --go-file and forbids --replay")
        transport = Live(args.endpoint, args.request_timeout, args.go_file)
    else:
        if not args.replay:
            raise ValueError("offline default requires --replay; use selftest for built-in fake fixtures")
        transport = Replay(read_jsonl(args.replay))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "manifest.json", manifest(tasks, seeds, args.arm, args.model, args.max_tokens,
               "live" if args.live else "replay", args.endpoint if args.live else None, args.request_timeout,
               args.deadline_s, args.order_seed))
    with open(out / "responses.jsonl", "x", encoding="utf-8") as stream:
        def emit(row):
            stream.write(canonical(row) + "\n")
            stream.flush()
        run_records(tasks, seeds, args.arm, args.model, args.max_tokens, transport, emit, args.deadline_s, args.order_seed)


def grade(args):
    directory = Path(args.run)
    meta = json.loads((directory / "manifest.json").read_text())
    tasks = validate_tasks(meta["tasks"])
    if digest(tasks) != meta["tasks_sha256"]:
        raise ValueError("manifest task hash mismatch")
    expected = {(t["suite"], t["id"], s) for t in tasks for s in meta["seeds"]}
    task_map = {(t["suite"], t["id"]): t for t in tasks}
    records, seen = read_jsonl(directory / "responses.jsonl"), set()
    for row in records:
        key = pair_key(row)
        if key in seen or key not in expected:
            raise ValueError("duplicate/unexpected response pair")
        seen.add(key)
        task = task_map[(row["suite"], row["id"])]
        if row["task_sha256"] != digest(task) or row["arm"] != meta["arm"]:
            raise ValueError("response does not match task manifest")
        if row["request"] != request_payload(task, row["seed"], meta["arm"], meta["model"], meta["max_tokens"], meta["deadline_s"]):
            raise ValueError("response request does not match manifest")
    if seen != expected:
        raise ValueError("missing response pairs")
    # Validate full grid before executing any coding completion.
    for row in records:
        row.update(grade_record(task_map[(row["suite"], row["id"])], row,
                               args.code_timeout, args.memory_mb, args.isolated_container))
    write_jsonl(args.out, records)


def report(args):
    plain_meta = json.loads((Path(args.plain_run) / "manifest.json").read_text())
    lane_meta = json.loads((Path(args.lane_run) / "manifest.json").read_text())
    for meta in (plain_meta, lane_meta):
        if digest(meta["tasks"]) != meta["tasks_sha256"]:
            raise ValueError("manifest task hash mismatch")
    if plain_meta["tasks"] != lane_meta["tasks"] or plain_meta["seeds"] != lane_meta["seeds"]:
        raise ValueError("run manifests differ in tasks/seeds")
    for key in ("model", "max_tokens", "deadline_s", "request_timeout", "order_seed"):
        if plain_meta[key] != lane_meta[key]:
            raise ValueError("run manifests differ in " + key)
    write_json(args.out, make_report(read_jsonl(args.plain), read_jsonl(args.lane),
               plain_meta["tasks"], plain_meta["seeds"], args.bootstrap, args.bootstrap_seed))


def fake_response(text, finish="stop", usage=True):
    value = {"choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}]}
    if usage:
        value["usage"] = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
                          "completion_tokens_details": {"reasoning_tokens": 12}}
    return canonical(value)


def selftest(args):
    # This is the only trusted-code bypass. Fixtures below are constants, never user input.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    tasks = [{"suite": "fixture_math", "id": str(i), "prompt": "Return the integer 7.",
              "answer": "7", "kind": "integer"} for i in range(6)]
    tasks += [{"suite": "fixture_code", "id": "base", "prompt": "def add(a, b):\n    \"\"\"Add.\"\"\"\n",
               "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0",
               "entry_point": "add", "kind": "humaneval"},
              {"suite": "fixture_math500", "id": "manual", "prompt": "Return sqrt(4).", "answer": r"\sqrt{4}", "kind": "math_manual"}]
    validate_tasks(tasks)
    write_jsonl(out / "tasks.jsonl", tasks)
    seeds = [101, 102]
    arms = {}
    for arm in ("plain", "lane"):
        replay = []
        for task in tasks:
            for seed in seeds:
                fixture = {"suite": task["suite"], "id": task["id"], "seed": seed,
                           "request": request_payload(task, seed, arm, MODEL, 128),
                           "status": "ok", "elapsed_seconds": 1.0 if arm == "plain" else 1.5}
                if task["suite"] == "fixture_code":
                    text = "def add(a, b):\n    return " + ("a + b" if arm == "lane" else "a - b")
                elif task["suite"] == "fixture_math500":
                    text = "2"
                else:
                    text = "7" if (int(task["id"]) + (arm == "lane")) % 2 == 0 else "8"
                fixture["raw_response"] = fake_response(text, "length" if task["id"] == "3" else "stop",
                                                         usage=task["id"] != "4")
                if task["id"] == "2":
                    fixture.update(status="timeout", raw_response=None, error="trusted recorded timeout")
                if task["id"] == "5":
                    fixture.update(status="error", raw_response="recorded server error", error="HTTP 500", http_status=500)
                replay.append(fixture)
        records = []
        run_records(tasks, seeds, arm, MODEL, 128, Replay(replay), records.append)
        run_dir = out / (arm + "-run")
        run_dir.mkdir()
        write_json(run_dir / "manifest.json", manifest(tasks, seeds, arm, MODEL, 128, "fake", None, 300))
        write_jsonl(run_dir / "responses.jsonl", records)
        task_map = {(t["suite"], t["id"]): t for t in tasks}
        for row in records:
            row.update(grade_record(task_map[(row["suite"], row["id"])], row, trusted=True))
        write_jsonl(out / (arm + "-replay.jsonl"), replay)
        write_jsonl(out / (arm + "-graded.jsonl"), records)
        arms[arm] = records
    result = make_report(arms["plain"], arms["lane"], tasks, seeds, args.bootstrap, 20261104)
    grades = [row for rows in arms.values() for row in rows]
    assert {r["grade"] for r in grades} == {"correct", "incorrect", "manual_review"}
    assert any(r["status"] == "timeout" for r in grades)
    assert any(r["status"] == "error" for r in grades)
    assert any(r["finish_reason"] == "length" for r in grades)
    assert any(r["usage"] is None for r in grades)
    assert all(r["tokens"]["total_tokens"] == 30 for r in grades if r["usage"] is not None)
    assert result["overall"]["plain_accuracy"] is None
    assert result["overall"]["lane_accuracy"] is None
    assert result["overall"]["resolved_subset"]["plain_accuracy"] == 4 / 14
    assert result["overall"]["resolved_subset"]["lane_accuracy"] == 4 / 14
    assert numeric(r"\boxed{007}", True) == Decimal(7)
    assert numeric("Reasoning 7 then 8", True) is None
    assert numeric("1,23") is None
    assert numeric("1,234") == Decimal(1234)
    assert binomial_exact(3, 0) == .25
    assert final_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert math_grade(r"\frac{1}{2}", r"\boxed{0.5}")["grade"] == "correct"
    assert math_grade("2", "3")["grade"] == "incorrect"
    assert math_grade(r"\sqrt{4}", "2")["grade"] == "manual_review"
    early_exit = dict(tasks[-2])
    early_exit["id"] = "trusted-early-exit"
    early_result = code_grade(early_exit, "import os\nos._exit(0)\ndef add(a, b):\n    return a + b", 3, 256, False, trusted=True)
    assert early_result["grade"] == "incorrect" and not early_result["harness_receipt"]
    assert early_result["worker_returncode"] == 0
    tool_body = json.loads(fake_response("7"))
    tool_body["choices"][0]["message"]["tool_calls"] = [{"type": "function", "id": "fixture", "function": {"name": "fixture", "arguments": "{}"}}]
    tool_record = {"status": "ok", "raw_response": canonical(tool_body)}
    parse_response(tool_record)
    assert tool_record["status"] == "error"
    assert grade_record(tasks[0], tool_record)["grade"] == "incorrect"
    write_json(out / "guard-fixtures.json", {"early_exit": early_result, "tool_calls": tool_record,
               "math_pending": math_grade(r"\sqrt{4}", "2")})
    assert request_payload(tasks[0], 1, "lane", MODEL, 128)["tools"] == [{"type": "code_interpreter"}]
    for bad in (arms["lane"][:-1], arms["lane"] + [arms["lane"][0]]):
        try:
            make_report(arms["plain"], bad, tasks, seeds, 1, 42)
        except ValueError:
            pass
        else:
            raise AssertionError("missing/duplicate pairs must fail")
    write_json(out / "report.json", result)
    write_json(out / "selftest.json", {"passed": True, "network": "no network transport constructed",
               "trusted_fixture_only": True, "all_passing": False, "workers": 1,
               "primary_accuracy": None, "expected_plain_resolved_accuracy": 4 / 14,
               "expected_lane_resolved_accuracy": 4 / 14})
    print("Selftest artifacts:", out)


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("fetch", "normalize"):
        p = sub.add_parser(command)
        p.add_argument("--lock", default=str(ROOT / "datasets.lock.json"))
        p.add_argument("--suites", help="comma-separated locked suites; defaults to all")
        p.add_argument("--fetch-network", action="store_true")
        p.add_argument("--out", default="tasks.jsonl")
        p.add_argument("--limit", type=positive_int, help="maximum tasks per suite")
        p.set_defaults(func=normalize)
    p = sub.add_parser("run")
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", required=True, help="new output directory")
    p.add_argument("--arm", choices=("plain", "lane"), required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--seeds", default="101,102")
    p.add_argument("--max-tokens", type=positive_int, default=8192)
    p.add_argument("--request-timeout", type=positive_float, default=300)
    p.add_argument("--deadline-s", type=positive_float, default=270)
    p.add_argument("--order-seed", type=int, default=20261104)
    p.add_argument("--live", action="store_true")
    p.add_argument("--endpoint", help="explicit full chat-completions URL; no default")
    p.add_argument("--go-file", help="file containing exactly GO; checked before every request")
    p.add_argument("--replay", help="offline recorded JSONL responses")
    p.set_defaults(func=run)
    p = sub.add_parser("grade")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--isolated-container", action="store_true", help="acknowledge a properly isolated container")
    p.add_argument("--code-timeout", type=positive_float, default=3)
    p.add_argument("--memory-mb", type=positive_int, default=256)
    p.set_defaults(func=grade)
    p = sub.add_parser("report")
    p.add_argument("--plain", required=True)
    p.add_argument("--lane", required=True)
    p.add_argument("--plain-run", required=True, help="run directory containing complete task+seed manifest")
    p.add_argument("--lane-run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bootstrap", type=positive_int, default=20000)
    p.add_argument("--bootstrap-seed", type=int, default=20261104)
    p.set_defaults(func=report)
    p = sub.add_parser("selftest", help="offline fake/recorded trusted-fixture check, one worker")
    p.add_argument("--out", required=True, help="new output directory")
    p.add_argument("--bootstrap", type=positive_int, default=20000)
    p.set_defaults(func=selftest)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
