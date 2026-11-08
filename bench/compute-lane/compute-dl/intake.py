#!/usr/bin/env python3
"""CPU-only frozen generated-task intake; no generator, client, or GPU authority.

Task file (strict JSON): {schema_version: 1, spec_sha256: HEX, tasks: [...]}
Each task has exactly id, suite, family, prompt, payload, answer. Payload is a
JSON object. Answers are tagged objects: {kind: 'integer', value: INT},
{kind: 'string', value: STR}, or {kind: 'rational', numerator: INT,
denominator: NONZERO_INT}. Integers/rationals compare as exact fractions;
strings compare verbatim. No float or bool coercion, text extraction or tolerance.

Trusted operator config is a separate JSON file, never embedded in tasks:
  schema_version, spec_sha256, taskset_sha256, expected_task_ids,
  verifier_timeout_seconds, verifiers, review
verifiers is exactly two {name, path, sha256} objects; paths must be absolute
Python source paths supplied by the operator. review has exactly operator,
spec_sha256, taskset_sha256, source_sha256 (ordered like verifiers),
independence_basis, public_input_review, spec_source_review_complete (bool).
The operator must qualify algorithm independence against the actual spec and
sources, including transitive dependencies. Different source hashes do not prove
independence. Production rejects incomplete review. Test-only intake can inspect
plumbing without that review and always emits FAKE_NOT_COLLECTION.

Verifier protocol: stdin is canonical UTF-8 JSON {schema_version: 1, tasks:
[{id, suite, family, prompt, payload}, ...]}, sorted by id. Stdout is strict JSON
{schema_version: 1, answers: [{id, answer}, ...]}. Both get identical bytes, no
reference answer, spec, seed, arm, paths or config. Public field names are
screened recursively for forbidden metadata. Semantic/covert leakage in ordinary
text cannot be decided by this module: public_input_review attests source review.

intake(task_file, spec_file=..., config=..., runner=None, test_only=False) is the
importable entry point. An injected runner(source_bytes, public_bytes, timeout)
returns stdout bytes and is allowed only with test_only=True. The default runner
executes pinned source snapshots serially with isolated stdlib Python, no shell.
This is NOT a security sandbox for malicious Python or an enforcement mechanism
for reviewed dependencies. Parent must use guarded CPU scopes; the >=7200 second
production timeout is a protocol floor, not permission to exceed guard budgets.
No import-time execution or writes. CLI prints a receipt only after both pass;
it never writes task sets or authorizes collection. No study sizes/rules assumed.
"""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


class IntakeError(ValueError):
    """A closed intake gate, invalid artifact, or verifier protocol failure."""


def canonical(value):
    """Canonical JSON bytes used for all public input and normalized receipts."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _require(condition, message):
    if not condition:
        raise IntakeError(message)


def _keys(obj, keys, context):
    _require(type(obj) is dict and set(obj) == set(keys),
             context + ': unexpected or missing fields')


def _text(value, context):
    _require(type(value) is str and bool(value.strip()), context + ': nonempty string required')
    return value


def _sha(value, context):
    _require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None,
             context + ': lowercase SHA256 required')
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate JSON object key')
        result[key] = value
    return result


def _no_constant(value):
    raise IntakeError('nonfinite JSON constant forbidden')


def decode(raw):
    """Strict JSON loader, including duplicate-key and nonfinite rejection."""
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs,
                          parse_constant=_no_constant)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise IntakeError('invalid strict UTF-8 JSON') from error


def _read(path, context):
    try:
        return Path(path).read_bytes()
    except (OSError, TypeError, ValueError) as error:
        raise IntakeError(context + ': readable file required') from error


def _ids(values, context):
    _require(type(values) is list and bool(values), context + ': nonempty ID list required')
    for value in values:
        _text(value, context)
    _require(len(values) == len(set(values)), context + ': duplicate ID')
    return set(values)


def normalize_answer(answer):
    """Return an exact, JSON-safe answer; integer and rational share one form."""
    _require(type(answer) is dict, 'tagged answer required')
    kind = answer.get('kind')
    if kind == 'string':
        _keys(answer, ('kind', 'value'), 'string answer')
        _require(type(answer['value']) is str, 'string answer value required')
        return dict(answer)
    if kind == 'integer':
        _keys(answer, ('kind', 'value'), 'integer answer')
        _require(type(answer['value']) is int, 'integer answer value required')
        value = Fraction(answer['value'])
    elif kind == 'rational':
        _keys(answer, ('kind', 'numerator', 'denominator'), 'rational answer')
        _require(type(answer['numerator']) is int
                 and type(answer['denominator']) is int and answer['denominator'] != 0,
                 'exact rational numerator/nonzero denominator required')
        value = Fraction(answer['numerator'], answer['denominator'])
    else:
        raise IntakeError('unsupported answer kind')
    return {'kind': 'rational', 'numerator': value.numerator, 'denominator': value.denominator}


# Conservative field-name screen; ordinary prompt/payload semantics need review.
_FORBIDDEN = {'answer', 'answers', 'reference', 'references', 'expected', 'solution',
              'solutions', 'seed', 'seeds', 'arm', 'arms', 'verifier', 'verifiers',
              'oracle', 'private', 'secret', 'groundtruth'}


def _public_json(value):
    if type(value) is dict:
        for key, child in value.items():
            _require(type(key) is str, 'public JSON keys must be strings')
            tokens = re.findall('[a-z0-9]+', re.sub(r'([a-z])([A-Z])', r'\1_\2', key).lower())
            _require(not (_FORBIDDEN & set(tokens))
                     and ''.join(tokens) not in _FORBIDDEN,
                     'forbidden answer/seed/arm/verifier metadata in public input')
            _public_json(child)
    elif type(value) is list:
        for child in value:
            _public_json(child)
    else:
        _require(value is None or type(value) in (str, int, bool),
                 'public payload supports JSON strings/integers/bools/null, no floats')


def public_tasks(document, expected_ids, spec_sha256):
    """Validate full task coverage and return public document plus private answers."""
    _keys(document, ('schema_version', 'spec_sha256', 'tasks'), 'task document')
    _require(type(document['schema_version']) is int and document['schema_version'] == 1,
             'unsupported task schema')
    _require(document['spec_sha256'] == spec_sha256, 'task spec binding differs')
    tasks = document['tasks']
    _require(type(tasks) is list and bool(tasks), 'nonempty task list required')
    public, answers = [], {}
    for task in tasks:
        _keys(task, ('id', 'suite', 'family', 'prompt', 'payload', 'answer'), 'task')
        for field in ('id', 'suite', 'family', 'prompt'):
            _text(task[field], 'task ' + field)
        _require(task['id'] not in answers, 'duplicate task ID')
        _require(type(task['payload']) is dict, 'structured payload object required')
        answers[task['id']] = normalize_answer(task['answer'])
        row = {key: task[key] for key in ('id', 'suite', 'family', 'prompt', 'payload')}
        _public_json(row)
        public.append(row)
    _require(set(answers) == _ids(expected_ids, 'expected_task_ids'), 'task ID set differs')
    return {'schema_version': 1, 'tasks': sorted(public, key=lambda row: row['id'])}, answers


def _verifier_answers(raw, expected_ids):
    document = decode(raw)
    _keys(document, ('schema_version', 'answers'), 'verifier output')
    _require(type(document['schema_version']) is int and document['schema_version'] == 1,
             'unsupported verifier schema')
    _require(type(document['answers']) is list, 'verifier answers list required')
    result = {}
    for row in document['answers']:
        _keys(row, ('id', 'answer'), 'verifier answer')
        task_id = _text(row['id'], 'verifier ID')
        _require(task_id not in result, 'duplicate verifier answer ID')
        result[task_id] = normalize_answer(row['answer'])
    _require(set(result) == set(expected_ids), 'missing/extra verifier answer IDs')
    return result


def run_verifier(source, public_input, timeout_seconds):
    """Run a pinned snapshot, serially at the caller; external CPU guard required.

    Only public input is written to stdin. Temporary cwd contains only source;
    environment is empty, Python uses -I -S -B, no shell or task-controlled argv.
    Captured output is subject to the enclosing guard's memory/disk/wall limits.
    """
    try:
        with tempfile.TemporaryDirectory(prefix='compute-dl-verifier-') as directory:
            script = Path(directory) / 'verifier.py'
            script.write_bytes(source)
            result = subprocess.run(
                [sys.executable, '-I', '-S', '-B', str(script)], input=public_input,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=directory,
                env={}, timeout=timeout_seconds, check=False, shell=False)
            _require(result.returncode == 0, 'verifier process failed')
            return result.stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        raise IntakeError('verifier process unavailable or timed out') from error


def intake(task_file, *, spec_file, config, runner=None, test_only=False):
    """Validate supplied frozen files, run two pinned verifiers, return receipt.

    config must be trusted operator data, not task-controlled input. runner is an
    optional fake callable for parent negative controls, never production intake.
    No result (including reviewed production intake) grants GPU/collection GO.
    """
    _require(type(test_only) is bool, 'test_only must be bool')
    _require(runner is None or test_only, 'injected runner requires test_only')
    _keys(config, ('schema_version', 'spec_sha256', 'taskset_sha256',
                   'expected_task_ids', 'verifier_timeout_seconds', 'verifiers', 'review'),
          'operator config')
    # Snapshot caller data so injected code cannot mutate gates or expectations.
    config = decode(canonical(config))
    _require(type(config['schema_version']) is int and config['schema_version'] == 1,
             'unsupported operator config schema')
    spec_pin = _sha(config['spec_sha256'], 'spec pin')
    task_pin = _sha(config['taskset_sha256'], 'taskset pin')
    spec_raw = _read(spec_file, 'spec')
    _require(bool(spec_raw.strip()) and digest(spec_raw) == spec_pin, 'missing/empty/drifted spec')
    task_raw = _read(task_file, 'task file')
    _require(digest(task_raw) == task_pin, 'taskset source pin differs')
    public, expected = public_tasks(decode(task_raw), config['expected_task_ids'], spec_pin)
    public_raw = canonical(public)
    timeout = config['verifier_timeout_seconds']
    _require(type(timeout) is int and timeout >= 7200, 'verifier timeout floor is 7200 seconds')
    verifiers = config['verifiers']
    _require(type(verifiers) is list and len(verifiers) == 2, 'exactly two supplied verifiers required')
    sources, paths, pins, names = [], [], [], []
    for verifier in verifiers:
        _keys(verifier, ('name', 'path', 'sha256'), 'verifier config')
        names.append(_text(verifier['name'], 'verifier name'))
        path = Path(_text(verifier['path'], 'trusted verifier path'))
        _require(path.is_absolute() and path.suffix == '.py', 'absolute Python verifier path required')
        paths.append(path.resolve())
        pin = _sha(verifier['sha256'], 'verifier source pin')
        source = _read(path, 'verifier source')
        _require(bool(source.strip()) and digest(source) == pin, 'verifier source pin differs')
        sources.append(source)
        pins.append(pin)
    _require(len(set(paths)) == len(set(pins)) == len(set(names)) == 2,
             'two distinct supplied verifier paths, names and source pins required')
    review = config['review']
    _keys(review, ('operator', 'spec_sha256', 'taskset_sha256', 'source_sha256',
                   'independence_basis', 'public_input_review', 'spec_source_review_complete'),
          'review')
    for field in ('operator', 'independence_basis', 'public_input_review'):
        _text(review[field], 'review ' + field)
    _require(review['spec_sha256'] == spec_pin and review['taskset_sha256'] == task_pin
             and review['source_sha256'] == pins, 'review bindings differ')
    _require(type(review['spec_source_review_complete']) is bool, 'explicit review gate required')
    _require(test_only or review['spec_source_review_complete'],
             'production intake blocked pending spec/source independence and public-input review')
    receipts = []
    execute = run_verifier if runner is None else runner
    for name, source, pin in zip(names, sources, pins):
        raw = execute(source, public_raw, timeout)
        _require(type(raw) is bytes, 'verifier runner must return stdout bytes')
        actual = _verifier_answers(raw, expected)
        _require(actual == expected, 'verifier answers mismatch supplied exact references')
        receipts.append({'name': name, 'source_sha256': pin,
                         'public_input_sha256': digest(public_raw),
                         'stdout_sha256': digest(raw), 'stdout_bytes': len(raw),
                         'answers_by_id': actual, 'normalized_answers_sha256': digest(canonical(actual)),
                         'runner': 'injected' if runner is not None else 'isolated_python'})
    # Detect file drift during execution as well as checking the initial snapshot.
    _require(digest(_read(spec_file, 'spec')) == spec_pin, 'spec drift during intake')
    _require(digest(_read(task_file, 'tasks')) == task_pin, 'taskset drift during intake')
    for verifier, pin in zip(verifiers, pins):
        _require(digest(_read(verifier['path'], 'verifier')) == pin, 'verifier drift during intake')
    return {'schema_version': 1, 'status': 'FAKE_NOT_COLLECTION' if test_only else 'CPU_INTAKE_VERIFIED',
            'qualification': 'PLUMBING_ONLY' if test_only else 'OPERATOR_SPEC_SOURCE_REVIEWED',
            'gpu_authorized': False, 'collection_authorized': False,
            'independence_proven_by_hashes': False, 'review': review,
            'spec_sha256': spec_pin, 'taskset_sha256': task_pin,
            'operator_config_sha256': digest(canonical(config)),
            'intake_source_sha256': digest(Path(__file__).read_bytes()),
            'public_input_sha256': digest(public_raw), 'public_input_bytes': len(public_raw),
            'task_ids': sorted(expected), 'task_count': len(expected),
            'expected_answers_sha256': digest(canonical(expected)), 'verifiers': receipts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--config', required=True, help='explicit trusted operator JSON config')
    parser.add_argument('--test-only', action='store_true', help='always FAKE_NOT_COLLECTION')
    args = parser.parse_args(argv)
    try:
        receipt = intake(args.tasks, spec_file=args.spec, config=decode(_read(args.config, 'config')),
                         test_only=args.test_only)
        print(canonical(receipt).decode('utf-8'))
    except (IntakeError, OSError, TypeError, ValueError) as error:
        parser.exit(2, 'intake rejected: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
