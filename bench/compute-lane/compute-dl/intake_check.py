#!/usr/bin/env python3
"""TEST-ONLY toy negative controls; parent runs under guarded CPU scopes.

No production generation, study tasks, endpoints, engine imports or GPU authority.
run_checks() writes only tiny hand-written fixtures in a temporary directory and
injects fake verifier responses: no subprocess is started. CLI prints a
FAKE_NOT_COLLECTION receipt; passing cannot authorize collection or GPU work.
Import itself runs nothing. Parent separately supplies AST/frozen import and
turn-preflight checks; this module does not claim those checks passed.
"""

import copy
import json
from pathlib import Path
import tempfile
import types


def _load_intake():
    # Resolve sibling explicitly rather than importing a same-named ambient module.
    path = Path(__file__).resolve().with_name('intake.py')
    # Execute the actual sibling bytes without leaving a __pycache__ artifact.
    module = types.ModuleType('_compute_dl_intake_check_target')
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def run_checks():
    """Run fake-response controls only when explicitly called by the parent."""
    api = _load_intake()

    def forbid_process(*args):
        raise AssertionError('test controls must never start a verifier process')

    # Even a regression opening a production gate cannot launch a real process.
    api.run_verifier = forbid_process
    passed = []
    with tempfile.TemporaryDirectory(prefix='FAKE_NOT_COLLECTION-') as directory:
        root = Path(directory)
        spec_file, task_file = root / 'TEST-ONLY-spec.txt', root / 'TEST-ONLY-tasks.json'
        spec_file.write_bytes(b'TEST-ONLY toy fixture. Not a study specification.\n')
        spec_pin = api.digest(spec_file.read_bytes())
        # Deliberately hand-written; these are not generated study units.
        document = {'schema_version': 1, 'spec_sha256': spec_pin, 'tasks': [
            {'id': 'TEST-ONLY-add', 'suite': 'TEST-ONLY', 'family': 'toy-add',
             'prompt': 'Add the two integers.', 'payload': {'values': [2, 3]},
             'answer': {'kind': 'integer', 'value': 5}},
            {'id': 'TEST-ONLY-text', 'suite': 'TEST-ONLY', 'family': 'toy-text',
             'prompt': 'Return the given word.', 'payload': {'word': 'toy'},
             'answer': {'kind': 'string', 'value': 'toy'}}]}
        task_file.write_bytes(api.canonical(document))
        # Distinct source bytes are ONLY pin fixtures, never independence evidence.
        source_paths = [root / 'TEST-ONLY-first.py', root / 'TEST-ONLY-second.py']
        source_paths[0].write_bytes(b'# TEST-ONLY first pin fixture; never executed\n')
        source_paths[1].write_bytes(b'# TEST-ONLY second pin fixture; never executed\n')
        pins = [api.digest(path.read_bytes()) for path in source_paths]
        config = {'schema_version': 1, 'spec_sha256': spec_pin,
                  'taskset_sha256': api.digest(task_file.read_bytes()),
                  'expected_task_ids': ['TEST-ONLY-add', 'TEST-ONLY-text'],
                  'verifier_timeout_seconds': 7200,
                  'verifiers': [{'name': name, 'path': str(path), 'sha256': pin}
                                for name, path, pin in zip(('first', 'second'), source_paths, pins)],
                  'review': {'operator': 'TEST-ONLY fake operator', 'spec_sha256': spec_pin,
                             'taskset_sha256': api.digest(task_file.read_bytes()),
                             'source_sha256': pins, 'independence_basis': 'TEST-ONLY no independence claim',
                             'public_input_review': 'TEST-ONLY hand-written toys',
                             'spec_source_review_complete': False}}
        response = {'schema_version': 1, 'answers': [
            {'id': 'TEST-ONLY-add', 'answer': {'kind': 'rational', 'numerator': 10, 'denominator': 2}},
            {'id': 'TEST-ONLY-text', 'answer': {'kind': 'string', 'value': 'toy'}}]}
        calls = []

        def fake(source, public, timeout):
            if source not in [path.read_bytes() for path in source_paths] or timeout != 7200:
                raise AssertionError('unexpected runner source or timeout')
            visible = api.decode(public)
            for row in visible['tasks']:
                if set(row) != {'id', 'suite', 'family', 'prompt', 'payload'}:
                    raise AssertionError('private task fields leaked')
            calls.append(public)
            return api.canonical(response)

        def invoke(cfg=None, runner=fake):
            return api.intake(task_file, spec_file=spec_file,
                              config=config if cfg is None else cfg,
                              runner=runner, test_only=True)

        def reject(name, action):
            try:
                action()
            except api.IntakeError:
                passed.append(name)
            else:
                raise AssertionError(name + ': accepted invalid intake')

        receipt = invoke()
        if (receipt['status'] != 'FAKE_NOT_COLLECTION' or receipt['gpu_authorized']
                or receipt['collection_authorized'] or receipt['qualification'] != 'PLUMBING_ONLY'
                or len(calls) != 2 or calls[0] != calls[1]):
            raise AssertionError('fake receipt or identical public input contract failed')
        passed.append('exact-rational-and-string-success-same-public-bytes')

        def changed_tasks(name, mutate):
            changed = copy.deepcopy(document)
            mutate(changed)
            task_file.write_bytes(api.canonical(changed))
            cfg = copy.deepcopy(config)
            cfg['taskset_sha256'] = api.digest(task_file.read_bytes())
            cfg['review']['taskset_sha256'] = cfg['taskset_sha256']
            try:
                reject(name, lambda: invoke(cfg))
            finally:
                task_file.write_bytes(api.canonical(document))

        changed_tasks('reference-answer-mutation', lambda doc: doc['tasks'][0]['answer'].update(value=6))
        changed_tasks('missing-task', lambda doc: doc['tasks'].pop())
        changed_tasks('duplicate-task', lambda doc: doc['tasks'].append(copy.deepcopy(doc['tasks'][0])))
        changed_tasks('extra-task', lambda doc: doc['tasks'].append(dict(doc['tasks'][0], id='TEST-ONLY-extra')))
        changed_tasks('nested-answer-leak', lambda doc: doc['tasks'][0]['payload'].update(nested={'expected_answer': 5}))
        changed_tasks('seed-leak', lambda doc: doc['tasks'][0]['payload'].update(seed=1))
        changed_tasks('arm-leak', lambda doc: doc['tasks'][0]['payload'].update(arm='toy'))
        changed_tasks('task-controlled-verifier', lambda doc: doc['tasks'][0].update(verifier_path='/tmp/no.py'))
        changed_tasks('bool-answer-rejected', lambda doc: doc['tasks'][0]['answer'].update(value=True))
        changed_tasks('zero-denominator', lambda doc: doc['tasks'][0].update(
            answer={'kind': 'rational', 'numerator': 1, 'denominator': 0}))
        changed_tasks('task-spec-mismatch', lambda doc: doc.update(spec_sha256='0' * 64))

        original_spec = spec_file.read_bytes()
        spec_file.write_bytes(original_spec + b'drift')
        reject('spec-drift', invoke)
        spec_file.write_bytes(original_spec)
        reject('missing-spec', lambda: api.intake(task_file, spec_file=root / 'absent',
                                                  config=config, runner=fake, test_only=True))
        source = source_paths[0].read_bytes()
        source_paths[0].write_bytes(source + b'# drift\n')
        reject('source-pin-drift', invoke)
        source_paths[0].write_bytes(source)
        task_file.write_bytes(api.canonical(document) + b' ')
        reject('taskset-pin-drift', invoke)
        task_file.write_bytes(api.canonical(document))

        for name, mutate in (
            ('missing-verifier-answer', lambda obj: obj['answers'].pop()),
            ('duplicate-verifier-answer', lambda obj: obj['answers'].append(copy.deepcopy(obj['answers'][0]))),
            ('extra-verifier-answer', lambda obj: obj['answers'].append(dict(obj['answers'][0], id='extra'))),
            ('mismatched-verifier-answer', lambda obj: obj['answers'][1]['answer'].update(value='wrong')),
        ):
            bad = copy.deepcopy(response)
            mutate(bad)
            reject(name, lambda: invoke(runner=lambda source, public, timeout: api.canonical(bad)))
        reject('duplicate-json-key', lambda: api.decode(b'{"id":1,"id":2}'))
        reject('nonfinite-json', lambda: api.decode(b'{"value":NaN}'))
        reject('malformed-verifier-output', lambda: invoke(runner=lambda *args: b'not JSON'))
        cfg = copy.deepcopy(config)
        cfg['verifier_timeout_seconds'] = 7199
        reject('timeout-floor', lambda: invoke(cfg))
        cfg = copy.deepcopy(config)
        cfg['review']['source_sha256'] = list(reversed(pins))
        reject('review-source-binding', lambda: invoke(cfg))
        cfg = copy.deepcopy(config)
        cfg['spec_sha256'] = ''
        reject('missing-spec-pin', lambda: invoke(cfg))
        cfg = copy.deepcopy(config)
        cfg['expected_task_ids'].append(cfg['expected_task_ids'][0])
        reject('duplicate-expected-id', lambda: invoke(cfg))
        cfg = copy.deepcopy(config)
        cfg['verifiers'][1] = copy.deepcopy(cfg['verifiers'][0])
        reject('same-verifier-twice', lambda: invoke(cfg))
        # These fail before default runner selection; no subprocess can start.
        reject('production-review-gate', lambda: api.intake(task_file, spec_file=spec_file, config=config))
        reject('production-fake-runner-gate', lambda: api.intake(
            task_file, spec_file=spec_file, config=config, runner=fake))
        # First verifier succeeds; second must independently pass too.
        serial_calls = []

        def second_bad(source, public, timeout):
            serial_calls.append(source)
            if len(serial_calls) == 1:
                return api.canonical(response)
            return b'{"schema_version":1,"answers":[]}'

        reject('second-verifier-must-pass', lambda: invoke(runner=second_bad))
        if len(serial_calls) != 2:
            raise AssertionError('both serial verifier calls expected')
        return {'status': 'FAKE_NOT_COLLECTION', 'gpu_authorized': False,
                'collection_authorized': False, 'checks': passed,
                'intake_source_sha256': api.digest(Path(api.__file__).read_bytes()),
                'check_source_sha256': api.digest(Path(__file__).read_bytes()),
                'functional_scope': 'TEST-ONLY injected runner; no production verifier execution'}


if __name__ == '__main__':
    print(json.dumps(run_checks(), sort_keys=True))
