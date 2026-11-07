#!/usr/bin/env python3
"""Container supervisor and its sole resource-limited child. No host execution."""
import ast
import base64
import hashlib
import io
import json
import linecache
import os
import random
import re
import resource
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import traceback

WALL_SECONDS = 3
OUTPUT_LIMIT = 65536
RECEIPT = b"stdbench-base-tests-complete-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate(payload):
    if not isinstance(payload, dict) or payload.get("mode") not in ("reference", "humaneval"):
        raise ValueError("reference or humaneval mode required")
    if payload["mode"] == "reference":
        if not isinstance(payload.get("code"), str):
            raise ValueError("reference code must be text")
    else:
        task = payload.get("task")
        if not isinstance(task, dict) or task.get("kind") != "humaneval":
            raise ValueError("normalized HumanEval task required")
        for key in ("id", "prompt", "test", "entry_point"):
            if not isinstance(task.get(key), str) or not task[key].strip():
                raise ValueError("missing task field: " + key)
        if not re.fullmatch(r"[A-Za-z_]\w*", task["entry_point"]):
            raise ValueError("invalid entry_point")
        if not isinstance(payload.get("content"), str):
            raise ValueError("delivered final content required")
    if len(canonical(payload).encode()) > 2 * 1024 * 1024:
        raise ValueError("payload exceeds 2MiB")


def extract_program(task, content):
    """Parse without repair. All parsing occurs in the constrained child."""
    fences = list(re.finditer(r"(?m)^```(?:python|py)[ \t]*\r?\n(.*?)^```[ \t]*(?:\r?\n|$)",
                              content, re.DOTALL))
    candidate = fences[-1].group(1) if fences else content
    prefix = task.get("code_prefix", task["prompt"])
    if not isinstance(prefix, str):
        raise ValueError("code_prefix must be text")
    try:
        tree = ast.parse(candidate, "<delivered-final>")
    except (SyntaxError, IndentationError):
        tree = None
    full = tree is not None and any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                                    and node.name == task["entry_point"] for node in tree.body)
    if full:
        # Only exact original import statements, never inferred dependencies.
        prefix_tree = ast.parse(prefix, "<original-prompt>")
        imports = [ast.get_source_segment(prefix, node) for node in prefix_tree.body
                   if isinstance(node, (ast.Import, ast.ImportFrom))]
        program = "\n".join(imports) + ("\n" if imports else "") + candidate
    else:
        # Native body completion: original prefix plus exact delivered bytes.
        program = prefix + candidate
    parsed = ast.parse(program, "<candidate>")
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
               node.name == task["entry_point"] for node in parsed.body):
        raise ValueError("no complete entry-point function")
    return program, "last_complete_python_fence" if fences else "unfenced_function_or_completion"


def reference(code):
    # Independent CPython implementation of the frozen native REPL convention.
    random.seed(0)
    sys.stdin = io.StringIO()
    linecache.cache["<code>"] = (len(code), None, code.splitlines(True), "<code>")
    try:
        tree = ast.parse(code, filename="<code>")
        expression = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            expression = tree.body.pop().value
        namespace = {"__name__": "__main__"}
        exec(compile(tree, "<code>", "exec"), namespace)
        if expression is not None:
            value = eval(compile(ast.Expression(expression), "<code>", "eval"), namespace)
            if value is not None:
                print(repr(value))
    except SystemExit:
        raise
    except BaseException as exc:
        tb = None if isinstance(exc, SyntaxError) else exc.__traceback__.tb_next
        traceback.print_exception(type(exc), exc, tb)
        sys.stdout.flush()
        sys.stderr.flush()
        raise SystemExit(1)


def child(receipt_fd):
    resource.setrlimit(resource.RLIMIT_CPU, (2, 3))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    payload = json.load(sys.stdin)
    validate(payload)
    if payload["mode"] == "reference":
        reference(payload["code"])
    else:
        task = payload["task"]
        program, extraction = extract_program(task, payload["content"])
        namespace = {"__name__": "__main__"}
        # Capture harness operations before candidate code; a receipt protects
        # premature exits, not a claim of resistance to deliberate test forgery.
        execute, compile_source, write_receipt = exec, compile, os.write
        report = canonical(dict(receipt=RECEIPT.decode(), program_sha256=digest(program),
                                test_sha256=digest(task["test"]), extraction=extraction)).encode()
        execute(compile_source(program, "<candidate>", "exec"), namespace)
        execute(compile_source(task["test"], "<official-base-test>", "exec"), namespace)
        namespace["check"](namespace[task["entry_point"]])
        write_receipt(receipt_fd, report)
    sys.stdout.flush()
    sys.stderr.flush()


def normalized(stdout, stderr, status, returncode):
    body = stdout
    if stderr:
        body += (b"\n[stderr]\n" if body else b"[stderr]\n") + stderr
    if status != "ok":
        if body and not body.endswith(b"\n"):
            body += b"\n"
        if status == "timeout":
            body += b"[timed out after 3000 ms]"
        elif status == "output_limit":
            body += b"[output truncated at 65536 bytes]"
        elif returncode is not None and returncode < 0:
            body += ("[killed by signal %d]" % -returncode).encode()
        else:
            body += ("[exit status %d]" % (returncode or 0)).encode()
    if body.endswith(b"\n"):
        body = body[:-1]
    return body or b"[no output]"


def supervise(payload):
    validate(payload)
    receipt_read, receipt_write = os.pipe()
    process = None
    selector = selectors.DefaultSelector()
    collected = {"stdout": bytearray(), "stderr": bytearray(), "receipt": bytearray()}
    status = None
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="candidate-", dir="/tmp") as work:
            process = subprocess.Popen(
                ["/usr/bin/python3", "-I", "-S", "/kit/jail_worker.py", "--child", str(receipt_write)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                pass_fds=(receipt_write,), cwd=work, close_fds=True,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8",
                     "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            os.close(receipt_write)
            receipt_write = None
            for channel, fd in (("stdout", process.stdout.fileno()),
                                ("stderr", process.stderr.fileno()), ("receipt", receipt_read)):
                os.set_blocking(fd, False)
                selector.register(fd, selectors.EVENT_READ, channel)
            # Payload <=2MiB; child reads stdin before candidate code. Bound this
            # write too, so a failed child cannot block a supervisor forever.
            pending = memoryview(canonical(payload).encode())
            input_fd = process.stdin.fileno()
            os.set_blocking(input_fd, False)
            selector.register(input_fd, selectors.EVENT_WRITE, "stdin")
            while selector.get_map() or process.poll() is None:
                if time.monotonic() - started >= WALL_SECONDS and status is None:
                    status = "timeout"
                if status is not None and process.poll() is None:
                    process.kill()  # Exact sole child PID only.
                for key, _ in selector.select(0.02):
                    channel = key.data
                    if channel == "stdin":
                        try:
                            count = os.write(key.fd, pending[:65536])
                            pending = pending[count:]
                        except BrokenPipeError:
                            pending = pending[:0]
                        if not pending:
                            selector.unregister(key.fd)
                            process.stdin.close()
                        continue
                    data = os.read(key.fd, 8192)
                    if not data:
                        selector.unregister(key.fd)
                        continue
                    if channel == "receipt":
                        remaining = max(0, 4096 - len(collected[channel]))
                    else:
                        remaining = max(0, OUTPUT_LIMIT - len(collected["stdout"]) - len(collected["stderr"]))
                    collected[channel].extend(data[:remaining])
                    if len(data) > remaining and status is None:
                        status = "output_limit"
            process.wait(timeout=1)
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=1)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        selector.close()
        os.close(receipt_read)
        if receipt_write is not None:
            os.close(receipt_write)
    returncode = process.returncode
    if status is None:
        status = "timeout" if returncode == -signal.SIGXCPU else ("ok" if returncode == 0 else "exit_error")
    stdout, stderr = bytes(collected["stdout"]), bytes(collected["stderr"])
    body = normalized(stdout, stderr, status, returncode)
    receipt = None
    try:
        receipt = json.loads(collected["receipt"].decode())
    except (ValueError, UnicodeError):
        pass
    harness_receipt = isinstance(receipt, dict) and receipt.get("receipt") == RECEIPT.decode()
    result = dict(status=status, exit_code=returncode, stdout=stdout.decode("utf-8", "replace"),
                  stderr=stderr.decode("utf-8", "replace"), stdout_base64=base64.b64encode(stdout).decode(),
                  stderr_base64=base64.b64encode(stderr).decode(), normalized_body=body.decode("utf-8", "replace"),
                  normalized_body_base64=base64.b64encode(body).decode(),
                  normalized_body_sha256=hashlib.sha256(body).hexdigest(),
                  elapsed_seconds=time.monotonic() - started, output_limited=status == "output_limit",
                  harness_receipt=harness_receipt, receipt=receipt,
                  native_result_token_cap=512, needs_native_token_cap_check=True,
                  native_error_validation_separate=status != "ok")
    if payload["mode"] == "humaneval":
        passed = status == "ok" and harness_receipt
        result.update(grade="correct" if passed else "incorrect",
                      reason="base_tests_pass" if passed else
                      ("grading_timeout" if status == "timeout" else "base_tests_fail_or_missing_receipt"))
    return result


def main():
    if not container_marker():
        raise SystemExit("container marker required; never execute this worker on host")
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(int(sys.argv[2]))
    elif sys.argv[1:] == ["--supervise"]:
        try:
            result = supervise(json.load(sys.stdin))
        except Exception as exc:
            result = dict(status="harness_error", grade="incorrect", harness_receipt=False,
                          reason=type(exc).__name__ + ": " + str(exc))
        print(canonical(result), flush=True)
    else:
        raise SystemExit("use --supervise or --child FD")


def container_marker():
    return os.path.isfile("/.dockerenv") or os.path.isfile("/run/.containerenv")


if __name__ == "__main__":
    main()
