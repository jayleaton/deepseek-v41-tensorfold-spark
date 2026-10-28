"""bench/tooleval/run.sh end to end with a dummy key: stand-in tool-eval-bench and spark-bench that echo their argv,
environment and request headers into every report, log and dump, and a localhost fake server that checks the key.
The key goes in through API_KEY (the separate --api-key form) and TEB_ARGS / SB_ARGS (--api-key=KEY); it must not
appear in any saved file or on the terminal."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# built at runtime: no credential literal in the tree
KEY = "DUMMY" + "-tooleval-" + "5f3c9a0e1b7d"
MODEL = "DeepSeek-V4.1-Flash-TF"

TEB = """#!/bin/sh
: > "$RAW_TOOL_CALLED"
json=""; prev=""
for a in "$@"; do [ "$prev" = --json-file ] && json=$a; prev=$a; done
echo "tool-eval-bench argv: $*"
echo "tool-eval-bench using key $API_KEY" >&2
printf '{"argv": "%s", "headers": {"Authorization": "Bearer %s"}}\\n' "$*" "$API_KEY" > "$json"
"""
SB = """#!/bin/sh
: > "$RAW_TOOL_CALLED"
echo "spark-bench argv: $*"
echo "spark-bench key=$SPARK_BENCH_API_KEY"
printf '{"request_headers": {"x-api-key": "%s"}, "argv": "%s"}\\n' "$SPARK_BENCH_API_KEY" "$*" \\
    > "$SPARK_BENCH_DUMP_DIR/raw_dump.jsonl"
"""


@pytest.fixture
def server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            ok = self.headers.get("Authorization") == "Bearer " + KEY
            body = (
                {"status": "ok"}
                if self.path == "/health"
                else {"data": [{"id": MODEL}]}
            )
            self.send_response(200 if ok else 401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(body if ok else {"error": "invalid_api_key"}).encode()
            )

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", seen
    httpd.shutdown()


def install_fakes(cache: Path) -> None:
    for path, text in (
        (cache / "tool-eval-bench/.venv/bin/tool-eval-bench", TEB),
        (cache / "spark-bench/.venv/bin/python", SB),
    ):
        path.parent.mkdir(parents=True)
        path.write_text(text)
        path.chmod(0o755)


@pytest.mark.skipif(shutil.which("curl") is None, reason="run.sh needs curl")
@pytest.mark.parametrize("mode", ["teb", "sb", "both", "chains"])
def test_unadapted_third_party_writers_are_refused_before_any_write(tmp_path, server, mode):
    url, seen = server
    install_fakes(tmp_path / "cache")
    env = dict(
        os.environ,
        BASE_URL=url + "/v1",
        MODEL=MODEL,
        API_KEY=KEY,
        OUT=str(tmp_path / "out"),
        TOOLEVAL_CACHE=str(tmp_path / "cache"),
        TEB_ARGS="--api-key=" + KEY,
        SB_ARGS="--api-key=" + KEY,
        RAW_TOOL_CALLED=str(tmp_path / "raw-tool-called"),
    )
    r = subprocess.run(
        ["bash", str(ROOT / "bench/tooleval/run.sh"), mode, "off"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 78
    assert not seen, "unadapted mode contacted the server before its safety check"
    assert not (tmp_path / "raw-tool-called").exists(), "unadapted writer was invoked"
    assert not (tmp_path / "out").exists(), "unadapted mode created report files"
    leaked = KEY in r.stdout + r.stderr
    assert not leaked, "credential reached refusal output"
    assert "pre-write sanitization is unavailable" in r.stderr
