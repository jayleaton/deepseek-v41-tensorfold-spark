#!/usr/bin/env python3
"""CPU fake checks, for parent execution in a Spark 2G scope only.

No subprocess, network, engine, Docker or SSH is invoked. No campaign or GO
is generated. This is staged evidence, not qualification of a future spec.
"""
import ast
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def read_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LifecycleCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.controller = read_module('compute_dl_controller_check', 'controller.py')
        cls.ranks = read_module('compute_dl_rank_check', 'rank_ops.py')

    def test_no_process_or_network_in_fake_checks(self):
        # This test suite has a process/network tripwire for every test below.
        self.assertEqual(self.controller.ARM_NAMES, {'D': 'declared_off', 'L': 'lane', 'P': 'plain'})

    def setUp(self):
        for target in ('subprocess.run', 'subprocess.Popen', 'subprocess.check_output',
                       'urllib.request.urlopen', 'socket.socket'):
            guard = patch(target, side_effect=AssertionError('external operation forbidden in fake check'))
            guard.start()
            self.addCleanup(guard.stop)

    def test_gate_before_campaign_read_or_directory_creation(self):
        with patch.object(self.controller, 'require_collection_ready', side_effect=ValueError('SPEC_PENDING')) as gate, \
             patch.object(self.controller, 'load', side_effect=AssertionError('campaign read preceded gate')), \
             patch.object(Path, 'mkdir', side_effect=AssertionError('directory created before gate')):
            with self.assertRaisesRegex(ValueError, 'SPEC_PENDING'):
                self.controller.Turn('nonexistent-campaign.json', 't01')
            gate.assert_called_once_with('nonexistent-campaign.json')

    def test_actual_parent_gate_is_staged(self):
        import protocol
        with self.assertRaisesRegex(ValueError, 'SPEC_PENDING|CPU_GATES_PENDING'):
            protocol.require_collection_ready({})

    def test_execute_rechecks_gate_before_output(self):
        turn = object.__new__(self.controller.Turn)
        turn.path = Path('/nonexistent-campaign.json')
        with patch.object(self.controller, 'require_collection_ready', side_effect=ValueError('SPEC_PENDING')), \
             patch.object(Path, 'mkdir', side_effect=AssertionError('directory created')):
            with self.assertRaisesRegex(ValueError, 'SPEC_PENDING'):
                turn.execute()

    def test_remote_control_scope_and_timeout_floor(self):
        args = self.controller.remote_rank(['python3', '-B', 'rank_ops.py', 'verify'], 7200)
        command = args[-1]
        for text in ('stdbench-compute-dl-rctl-', 'MemoryMax=2G', 'MemorySwapMax=0',
                     'MAKEFLAGS=-j1', 'nice -n 19', 'ionice -c3', 'timeout 7200'):
            self.assertIn(text, command)
        self.assertEqual(args[-2], 'root@<rank1-link-ip>')
        with self.assertRaises(ValueError):
            self.controller.remote_rank(['false'], 7199)

    def test_pins_and_publication_refusal(self):
        import generate_envs3
        self.assertEqual(self.ranks.IMAGE_IDS, generate_envs3.IMAGE_IDS)
        self.assertEqual(self.ranks.REVISION, generate_envs3.REVISION)
        for flag in ('IMR_PUBLISH', 'KV_SPLIT_DEDUP', 'KV_SPLIT_EARLY', 'TF_TP_ROCE_CHUNKED_KB'):
            with self.assertRaises(ValueError):
                self.ranks.reject_banned({flag: '0'})

    def test_owned_path_escape_refused(self):
        root = Path('<rank0-work>/stdbench/compute-dl/stdbench-compute-dl-fake')
        for path in (root, root / '../outside', Path('/tmp/not-owned')):
            with self.assertRaises(ValueError):
                self.ranks.owned(path, root)

    def test_schema_and_staged_source_contract(self):
        source = (HERE / 'controller.py').read_text()
        ast.parse(source)
        for text in ("index['schema'] != 1", "len(index['turns']) < 2", "self.cfg['layout']['boots']",
                     "self.cfg['rank_manifest_sha256']", "exec 9</tmp/rank1-host-gpu.lock", 'actual_steps[-1]'):
            self.assertIn(text, source)
        for filename in ('launch.sh', 'turn.sh'):
            shell = (HERE / filename).read_text()
            self.assertIn('--protocol-check', shell)
            self.assertIn('set -euo pipefail', shell)
            self.assertIn('gate=$(bash <rank0-work>/stdbench/phase2/cpu-rank0-host.sh', shell)
        shell = (HERE / 'launch.sh').read_text()
        self.assertLess(shell.index('--protocol-check'), shell.index('exec systemd-run'))


def run_checks():
    output = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LifecycleCheck)
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    return {'passed': result.wasSuccessful(), 'tests': result.testsRun,
            'failures': len(result.failures), 'errors': len(result.errors),
            'qualification': 'CPU_FAKE_ONLY_GPU_BLOCKED_SPEC_PENDING',
            'details': output.getvalue()}


def main():
    result = run_checks()
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == '__main__':
    raise SystemExit(0 if main()['passed'] else 1)
