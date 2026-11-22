"""The guard reads the pinned engine's /health wire as well as the recorded one, and fails closed on bad counts."""

import http.server
import json
import threading
import unittest

from backend import Backend, native_health
from guard_fixtures import backend_config

# The pinned engine's idle /health with eight slots and the adaptive window on (captured on the pair).
PINNED_IDLE = {
    "status": "ok", "model": "DeepSeek-V4.1-Flash-TF", "model_ids": ["DeepSeek-V4.1-Flash-TF"],
    "max_batch_size": 8, "warming": False,
    "memory": {"active": 109314999948, "cache": 0, "peak": 109315726384},
    "live": {"connections": 0, "waiting": 0, "decode_tokens_per_second": 2.0, "prefill_tokens_per_second": 84.7},
    "context": {"window": 1048576, "active_chats": 1, "budget": 4194304, "floor": 262144},
    "version": "1.2.5", "backend": "cuda",
}


def pinned(**live):
    return dict(PINNED_IDLE, live=dict(PINNED_IDLE["live"], **live))


class NativeHealthTests(unittest.TestCase):
    def test_pinned_idle_wire_maps_to_the_guard_fields(self):
        view = native_health(PINNED_IDLE)
        self.assertIs(view["ok"], True)
        self.assertEqual((view["queued"], view["requests_running"], view["inflight"]), (0, 0, 0))
        self.assertEqual(view["streams"], {"max": 8})
        self.assertEqual(view["context"], PINNED_IDLE["context"])

    def test_running_is_connections_less_waiting(self):
        view = native_health(pinned(connections=7, waiting=2))
        self.assertEqual((view["queued"], view["requests_running"], view["inflight"]), (2, 5, 7))

    def test_recorded_wire_is_unchanged(self):
        recorded = {"ok": True, "queued": 0, "inflight": 1, "requests_running": 1, "streams": {"max": 4}}
        self.assertEqual(native_health(recorded), recorded)
        self.assertEqual(native_health({}), {})

    def test_not_ready_engine_is_not_ok(self):
        for body in (dict(PINNED_IDLE, warming=True), dict(PINNED_IDLE, status="stalled"),
                     {k: v for k, v in PINNED_IDLE.items() if k != "warming"}):
            with self.subTest(body={k: body.get(k) for k in ("status", "warming")}):
                self.assertIs(native_health(body)["ok"], False)

    def test_bad_counts_stay_absent(self):
        for live in ({"connections": 1, "waiting": 2}, {"connections": -1, "waiting": 0},
                     {"connections": True, "waiting": 0}, {"connections": 1.0, "waiting": 0},
                     {"connections": 1}, {"waiting": 0}):
            with self.subTest(live=live):
                view = native_health(dict(PINNED_IDLE, live=live))
                for key in ("queued", "requests_running", "inflight"):
                    self.assertNotIn(key, view)
        self.assertNotIn("queued", native_health(dict(PINNED_IDLE, live=None)))
        for lanes in (0, -8, True, "8", None):
            with self.subTest(lanes=lanes):
                self.assertNotIn("streams", native_health(dict(PINNED_IDLE, max_batch_size=lanes)))


class Engine(http.server.BaseHTTPRequestHandler):
    body = PINNED_IDLE

    def log_message(self, *_):
        pass

    def do_GET(self):
        payload = json.dumps(type(self).body).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class PinnedEngineBackendTests(unittest.TestCase):
    def setUp(self):
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Engine)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.backend = Backend(backend_config(upstream=f"http://127.0.0.1:{self.server.server_port}",
                                              limits={"max_inflight": 8}))
        Engine.body = PINNED_IDLE

    def test_idle_proof_and_eight_slot_boot_proof_over_http(self):
        self.assertEqual(self.backend.idle()["inflight"], 0)
        self.backend.identity = lambda *_: {"id": "exact"}
        self.backend.run = lambda *_: "boot growth worst serving point 7.35 GiB"
        self.assertEqual(len(self.backend.boot_proof("sha")["ranks"]), 2)

    def test_busy_or_malformed_engine_is_never_idle(self):
        for body in (pinned(connections=1, waiting=0), pinned(connections=1, waiting=1),
                     dict(PINNED_IDLE, live={"connections": 2, "waiting": 3}), dict(PINNED_IDLE, warming=True)):
            Engine.body = body
            with self.subTest(live=body["live"], warming=body["warming"]), self.assertRaises(ValueError):
                self.backend.idle()

    def test_slot_count_must_still_equal_the_guard_inflight(self):
        self.backend.config["limits"]["max_inflight"] = 4
        self.backend.identity = lambda *_: {"id": "exact"}
        self.backend.run = lambda *_: "boot growth worst serving point 7.35 GiB"
        with self.assertRaisesRegex(ValueError, "max_inflight"):
            self.backend.boot_proof("sha")


if __name__ == "__main__":
    unittest.main()
