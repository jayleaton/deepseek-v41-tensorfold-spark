"""Approved thermal window and container configuration for the native serving integration."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import Backend
from common import POLICY, container_names, inference_environment
from policy import Policy


class NativeIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.policy = Policy(self.root / "state", self.root / "events", now=1000)

    def sample(self, now, temperature=93, memory=8, owned=True):
        self.policy.update({"0": dict(epoch=now, max_zone_c=temperature, available_gib=memory)},
                           owned, "owned" if owned else "operator_lease", now)

    def test_sustained_temperature_stops_after_30_seconds(self):
        for now in range(1001, 1031):
            self.sample(now)
            self.assertFalse(self.policy.data["hard_stop"])
        self.sample(1031)
        self.assertTrue(self.policy.data["hard_stop"])
        self.assertTrue(json.loads((self.root / "state").read_text())["hard_stop"])

    def test_cooling_gap_and_lease_restart_temperature_window(self):
        for reset in ("cool", "gap", "lease"):
            with self.subTest(reset=reset), tempfile.TemporaryDirectory() as directory:
                self.policy = Policy(Path(directory) / "state", Path(directory) / "events", now=1000)
                for now in range(1001, 1030):
                    self.sample(now)
                if reset == "cool":
                    self.sample(1030, temperature=92)
                elif reset == "lease":
                    self.sample(1030, owned=False)
                else:
                    self.sample(1036)
                self.sample(1037)
                self.assertFalse(self.policy.data["hard_stop"])

    def test_immediate_temperature_stops_without_waiting(self):
        self.sample(1001, temperature=93.9)
        self.assertFalse(self.policy.data["hard_stop"])
        self.sample(1002, temperature=100)
        self.assertTrue(self.policy.data["hard_stop"])
        fired = self.policy.data["stop_evidence"]["fired"]
        self.assertEqual(fired, [{"condition": "temperature>=94", "rank": "0", "value": 100}])
        self.assertTrue(json.loads((self.root / "state").read_text())["hard_stop"])
        self.assertEqual(POLICY["immediate_hard_c"], 94)

    def test_one_fresh_sample_at_94_latches_stop_and_aborts_active_work(self):
        # The production policy, not the instantaneous ScenarioPolicy: the 93 C hold does not delay the 94 C stop.
        aborted = []
        self.policy.active["request"] = type("Connection", (), {"shutdown": lambda _, how: aborted.append(how)})()
        self.sample(1001, temperature=94)
        state = json.loads((self.root / "state").read_text())
        self.assertTrue(state["hard_stop"] and state["terminal_stop"])
        self.assertEqual(state["pending_action"], "stop_owned")
        self.assertEqual(aborted, [2])
        self.sample(1002, temperature=70)
        self.assertTrue(self.policy.data["hard_stop"])  # Durable: cooling does not clear it.

    def test_memory_floor_still_stops_immediately(self):
        self.sample(1001, temperature=80, memory=3.99)
        self.assertTrue(self.policy.data["hard_stop"])

    def test_resume_requires_both_nodes_at_86(self):
        self.sample(1001, temperature=92)
        for temperature, expected in ((87, True), (86, False)):
            now = 1002 + (87 - temperature)
            rows = {rank: dict(epoch=now, max_zone_c=temperature, available_gib=8) for rank in ("0", "1")}
            self.policy.update(rows, True, "owned", now)
            self.assertEqual(self.policy.data["paused"], expected)
        self.assertEqual(POLICY["hard_seconds"], 30)

    def test_prefix_default_override_and_rejection(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(container_names({}), {"0": "tensorfold-zig-r0", "1": "tensorfold-zig-r1"})
            self.assertEqual(container_names({"container_prefix": "native"})["1"], "native-r1")
            with patch.dict("os.environ", {"CONTAINER_PREFIX": "alternate"}):
                self.assertEqual(container_names({"container_prefix": "native"})["0"], "alternate-r0")
            for prefix in ("-flag", "bad;command", "", 1):
                with self.assertRaises(ValueError):
                    container_names({"container_prefix": prefix})

    def test_clock_changes_refused_without_permission(self):
        backend = Backend({"upstream": "http://127.0.0.1:8001", "clock_ladder_mhz": [2100, 2000]})
        with self.assertRaisesRegex(ValueError, "explicit operator approval"):
            backend.lower("profile", lambda: None)

    def test_profile_includes_network_and_allocation_settings(self):
        env = ["TF_TP_ROCE_HCA=mlx5_0", "NCCL_CUMEM_ENABLE=0", "CUDA_DEVICE_MAX_CONNECTIONS=1",
               "TF_DSV41_LAUNCH_T0=123", "PATH=/usr/bin"]
        actual = inference_environment({"Config": {"Env": env}})
        self.assertEqual(actual, dict(value.split("=", 1) for value in env[:3]))


if __name__ == "__main__":
    unittest.main()
