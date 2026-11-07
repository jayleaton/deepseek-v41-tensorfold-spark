#!/usr/bin/env python3
"""Small CPU fake checks for runtime_stage; parent runs inside Spark CPU guard.

No task generator, full dataset, native renderer, process or network is used.
Temporary materialization fixtures are deleted when each check exits.
"""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runtime_stage as runtime


class RuntimeCheck(unittest.TestCase):
    def setUp(self):
        for target in ('subprocess.run', 'subprocess.Popen', 'subprocess.check_output', 'socket.socket'):
            guard = patch(target, side_effect=AssertionError('external operation forbidden'))
            guard.start()
            self.addCleanup(guard.stop)

    def test_pair_wire_and_key_order_survive_json_roundtrip(self):
        task = {'suite': 'compute_dl', 'id': 'fake-only', 'prompt': 'FAKE CPU CHECK'}
        body = dict(runtime.SETTINGS)
        body['messages'] = [{'content': 'old', 'role': 'user'}]
        body['seed'] = 101
        tools = [{'type': 'function', 'function': {'name': 'python', 'parameters': {'type': 'object'}}}]
        body['tools'] = copy.deepcopy(tools)
        d = runtime.request_template(task, 102, 'D', body, tools)
        l = runtime.request_template(task, 102, 'L', body, tools)
        self.assertEqual(d['wire'], l['wire'])
        self.assertEqual(d['key'], ['compute_dl', 'fake-only', 102])
        self.assertEqual(d['execution_policy'], {'TF_DSV41_COMPUTE_LANE': '0'})
        self.assertEqual(l['execution_policy'], {'TF_DSV41_COMPUTE_LANE': '1'})
        reloaded = runtime.load_text(runtime.json_text(d))
        self.assertEqual(runtime.wire(reloaded['body_template']), d['wire'])
        self.assertEqual(list(reloaded['body_template']), list(body))
        self.assertEqual(d['task_sha256'], runtime.digest(runtime.canonical(task).encode()))
        self.assertEqual(d['wire_sha256'], runtime.digest(d['wire'].encode()))

    def test_rank_paths_and_rank0_host_manifest_shadow(self):
        files = {'rank0/configs/t01-D-r0.env': '', 'rank1/configs/t01-D-r1.env': '',
                 'rank0/boots/t01-D/manifest.json': '', 'rank1/boots/t01-D/manifest.json': '',
                 'frozen-v1/manifest.json': '', 'stage-campaign.json': ''}
        first = runtime.materialization_plan(files, 0)
        second = runtime.materialization_plan(files, 1)
        self.assertEqual(first['rank0/configs/t01-D-r0.env'], 'configs/t01-D-r0.env')
        self.assertIn('rank1/boots/t01-D/manifest.json', first)
        self.assertNotIn('rank1/configs/t01-D-r1.env', first)
        self.assertNotIn('rank0/boots/t01-D/manifest.json', second)
        self.assertEqual(str(runtime.prefix_root(1)), '<rank1-work>/stdbench/compute-dl/' + runtime.PREFIX)

    def test_turn_halves_family_order_and_fresh_shuffle_streams(self):
        # Metadata-only fixture: no inputs, prompts or generated study instances.
        tasks = [{'id': f'FAKE-{family}-{slot:02}', 'family': family, 'slot': slot}
                 for family in reversed(runtime.FAMILIES) for slot in reversed(range(20))]
        streams = []

        class FakeRandom:
            def __init__(self, seed):
                self.seed = seed
                self.before = None
                streams.append(self)

            def shuffle(self, units):
                self.before = [(task['id'], seed) for task, seed in units]
                units.reverse()

        with patch.object(runtime.random, 'Random', FakeRandom):
            first = runtime.ordered_pairs(tasks, 't01')
            second = runtime.ordered_pairs(tasks, 't02')
        self.assertEqual(len(streams), 12)
        self.assertTrue(all(stream.seed == 20261105 for stream in streams))
        self.assertTrue(all(stream.before == sorted(stream.before) for stream in streams))
        for pairs, low in ((first, 0), (second, 10)):
            self.assertEqual(len(pairs), 120)
            self.assertTrue(all(low <= task['slot'] < low + 10 for task, _ in pairs))
            self.assertEqual([pairs[index * 20][0]['family'] for index in range(6)], list(runtime.FAMILIES))

    def test_materialization_exclusive_and_tamper_refusal(self):
        with tempfile.TemporaryDirectory(prefix='compute-dl-runtime-fake-') as temporary:
            root = Path(temporary)
            stage = runtime.new_directory(root / 'stage')
            data = b'FAKE CONFIG\n'
            relative = 'rank0/configs/t01-D-r0.env'
            runtime.write_new(stage, relative, data)
            manifest = {'prefix': runtime.PREFIX, 'spec_sha256': runtime.SPEC_SHA256,
                        'files': {relative: runtime.digest(data)}}
            runtime.write_new(stage, 'stage-manifest.json', runtime.json_text(manifest).encode())
            destination = root / 'materialized'
            result = runtime.materialize(stage, 0, destination)
            self.assertFalse(result['collection_qualified'])
            self.assertEqual((destination / 'configs/t01-D-r0.env').read_bytes(), data)
            self.assertFalse((destination / 'GO').exists())
            self.assertFalse((destination / 'campaign-v1.json').exists())
            with self.assertRaises(FileExistsError):
                runtime.materialize(stage, 0, destination)
            (stage / relative).write_bytes(b'TAMPERED FAKE')
            refused = root / 'refused'
            with self.assertRaisesRegex(ValueError, 'drift'):
                runtime.materialize(stage, 0, refused)
            self.assertFalse(refused.exists())

    def test_json_duplicates_nonfinite_and_unsafe_rank_refused(self):
        for data in ('{"id":1,"id":2}', '{"id":NaN}'):
            with self.assertRaises(ValueError):
                runtime.load_text(data)
        for rank in (-1, 2, True):
            with self.assertRaises(ValueError):
                runtime.spark_root(rank)

    def test_stage_bad_spec_before_any_output(self):
        with patch.object(runtime, 'sha', return_value='wrong'), \
             patch.object(runtime, 'read_tasks', side_effect=AssertionError('read before spec')), \
             patch.object(runtime, 'new_directory', side_effect=AssertionError('output before spec')):
            with self.assertRaisesRegex(ValueError, 'spec SHA'):
                runtime.stage('/fake-output', '/fake-input')


def run_checks():
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeCheck))
    return {'passed': result.wasSuccessful(), 'tests': result.testsRun,
            'failures': len(result.failures), 'errors': len(result.errors),
            'qualification': 'CPU_FAKE_ONLY_NO_RENDER_OR_COLLECTION', 'details': output.getvalue()}


def main():
    result = run_checks()
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == '__main__':
    raise SystemExit(0 if main()['passed'] else 1)
