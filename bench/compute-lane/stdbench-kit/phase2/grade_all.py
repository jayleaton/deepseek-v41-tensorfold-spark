#!/usr/bin/env python3
"""Grade the complete frozen cohort offline; untrusted Python runs only through Jail."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from analysis import ARMS, SEEDS, canonical, censored, digest, identity, manifest_tasks, number

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'kit'))
import bench


def validate_responses(rows, manifest, *, _trusted_selftest=False):
    tasks = manifest_tasks(manifest)
    if manifest['seeds'] != list(SEEDS):
        raise ValueError('frozen seeds must be [101,102]')
    task_map = {(t['suite'], t['id']): t for t in tasks}
    if not task_map or len(task_map) != len(tasks):
        raise ValueError('empty/duplicate frozen tasks')
    for suite, task_id in task_map:
        if not all(isinstance(v, str) and v for v in (suite, task_id)):
            raise ValueError('invalid frozen task identity')
    if 'tasks_sha256' in manifest and digest(tasks) != manifest['tasks_sha256']:
        raise ValueError('frozen task hash mismatch')
    if not _trusted_selftest:
        if len(tasks) != 92:
            raise ValueError('exactly92 frozen tasks/368 replies required')
        kinds = {kind: [t for t in tasks if t.get('kind') in supported] for kind, supported in
                 (('gsm', ('integer', 'numeric')), ('math', ('math_manual',)), ('he', ('humaneval',)))}
        if [len(kinds[k]) for k in ('gsm', 'math', 'he')] != [48, 20, 24]:
            raise ValueError('full 48 GSM /20 MATH /24 original HumanEval cohort required')
        if any(len({t['suite'] for t in kinds[k]}) != 1 for k in kinds):
            raise ValueError('each registered task kind must have exactly one suite')
        if len({t['suite'] for t in tasks}) != 3:
            raise ValueError('three distinct registered suites required')
    expected = {(s, i, seed, arm) for s, i in task_map for seed in SEEDS for arm in ARMS}
    indexed = {}
    for row in rows:
        if row.get('diagnostic') or row.get('included_in_cohort') is False:
            raise ValueError('diagnostics must not enter cohort grading inputs')
        key = identity(row)
        if type(row['seed']) is not int or key not in expected or key in indexed:
            raise ValueError('duplicate/unexpected dataset response: ' + repr(key))
        if not isinstance(row.get('status'), str) or not row['status']:
            raise ValueError('generation status required')
        if not number(row.get('elapsed_seconds')):
            raise ValueError('observed elapsed_seconds required, including errors/timeouts')
        if row.get('status') == 'ok' and not isinstance(row.get('content'), str):
            raise ValueError('delivered final content must be text')
        if row.get('task_sha256') is not None and row['task_sha256'] != digest(task_map[key[:2]]):
            raise ValueError('response/frozen task hash mismatch')
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError('missing scheduled dataset replies: ' + repr(sorted(expected - set(indexed))[:8]))
    return task_map, indexed


def gsm_final(text):
    """Only final-channel text; choose final balanced box before explicit final markers."""
    if not isinstance(text, str):
        return None
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    elif '<think>' in text:
        return None
    text = text.strip()
    boxes = []
    for match in re.finditer(r'\\boxed\s*\{', text):
        depth = 1
        for pos in range(match.end(), len(text)):
            if text[pos] == '{' and (pos == 0 or text[pos - 1] != '\\'):
                depth += 1
            elif text[pos] == '}' and (pos == 0 or text[pos - 1] != '\\'):
                depth -= 1
            if depth == 0:
                boxes.append(text[match.end():pos].strip())
                break
    if boxes:
        return boxes[-1]
    if '\\boxed' in text:
        return None
    if '####' in text:
        return text.rsplit('####', 1)[1].strip()
    labels = list(re.finditer(r'(?:^|\n)\s*Final\s+answer\s*:\s*', text, re.I))
    if labels:
        return text[labels[-1].end():].strip()
    return text  # Must pass the entire strict numeric grammar below.


def exact_number(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value.endswith('.'):
        value = value[:-1].rstrip()
    if value.startswith('$'):
        value = value[1:].lstrip()
    elif value.startswith(('-$', '+$')):
        value = value[0] + value[2:].lstrip()
    decimal = r'[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?'
    fraction = r'(?:[+-]?\d+\s*/\s*[+-]?\d+|\\(?:d?frac)\{[+-]?\d+\}\{[+-]?\d+\})'
    if not re.fullmatch('(?:' + decimal + '|' + fraction + ')', value):
        return None
    return bench.rational(value)


def gsm_grade(task, content):
    reference = task['answer']
    if not isinstance(reference, str):
        raise ValueError('official normalized reference string required')
    if '####' in reference:
        reference = reference.rsplit('####', 1)[1]
    gold = exact_number(reference)
    if gold is None:
        raise ValueError('invalid official GSM reference under frozen grammar')
    extracted = gsm_final(content)
    actual = exact_number(extracted)
    return {'grade': 'correct' if actual is not None and actual == gold else 'incorrect',
            'reason': 'exact_final_fraction' if actual is not None else 'ambiguous_or_missing_final',
            'extracted_final': extracted, 'extracted_fraction': str(actual) if actual is not None else None,
            'reference_final': reference.strip(), 'reference_fraction': str(gold)}


def grade_all(rows, manifest, image_id, *, _trusted_selftest=False, _trusted_jail=None):
    if not _trusted_selftest and not re.fullmatch(r'sha256:[0-9a-f]{64}', image_id):
        raise ValueError('full pinned Jail image ID required')
    task_map, indexed = validate_responses(rows, manifest, _trusted_selftest=_trusted_selftest)
    jail = _trusted_jail if _trusted_selftest else None
    grader_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = []
    for key in sorted(indexed):
        source = indexed[key]
        row, task = dict(source), task_map[key[:2]]
        row['censored'] = censored(row)
        row.setdefault('finish_reason', None)
        row.setdefault('tokens', bench.usage_counts(row.get('usage')))
        # Every MATH response goes to the all-answer blind review, including failures.
        if task.get('kind') == 'math_manual':
            result = {'grade': 'manual_review', 'reason': 'all_MATH_requires_independent_blinded_review'}
        elif row['status'] != 'ok':
            result = {'grade': 'incorrect', 'reason': 'generation_' + row['status']}
        elif task.get('kind') in ('integer', 'numeric'):
            result = gsm_grade(task, row['content'])
        elif task.get('kind') == 'humaneval':
            if jail is None:
                from jail import Jail
                jail = Jail(image_id)
            evidence = jail.grade_humaneval(task, row['content'])
            passed = (evidence.get('grade') == 'correct' and evidence.get('status') == 'ok'
                      and evidence.get('harness_receipt') is True)
            result = {'grade': 'correct' if passed else 'incorrect',
                      'reason': evidence.get('reason', 'isolated_original_HumanEval_tests'),
                      'jail_evidence': evidence}
        else:
            raise ValueError('unsupported registered task kind')
        # Prevent stale automatic grading/adjudication from masquerading as this run.
        row.pop('math_review', None)
        row['grading'] = result
        row['grade'] = result['grade']
        row['grader'] = {'name': 'phase2.grade_all', 'manifest_sha256': digest(manifest),
                         'source_sha256': grader_hash}
        output.append(row)
    return output


def selftest():
    for content, answer in [('$-1,234.', '-1234'), ('-$1,234', '-1234'),
                            ('Final answer: -3/2.', '-1.5'),
                            (r'Old \boxed{1}; final \boxed{-2}', '-2'),
                            ('work: 5\n#### $1,234.', '1234'), (r'\boxed{\frac{1}{2}}', '0.5')]:
        assert gsm_grade({'answer': answer}, content)['grade'] == 'correct'
    for content in ('There were 42 apples.', '2 or 3', '$1,23', 'Final answer: 2\nMaybe 3',
                    'Answer: 2', '#### 2 dollars', '2..', r'\boxed{2'):
        assert gsm_grade({'answer': '2'}, content)['grade'] == 'incorrect'
    task = {'suite': 'fixture', 'id': 'a', 'kind': 'numeric', 'answer': '-2'}
    manifest = {'tasks': [task], 'seeds': list(SEEDS)}
    rows = [dict(suite='fixture', id='a', seed=s, arm=a, content='Final answer: -2.',
                 status='ok', finish_reason='deadline' if s == 101 else 'length', elapsed_seconds=270)
            for s in SEEDS for a in ARMS]
    graded = grade_all(rows, manifest, 'unused-trusted-fixture', _trusted_selftest=True)
    assert all(r['grade'] == 'correct' and r['censored'] for r in graded)
    failed = [dict(r, status='timeout') for r in rows]
    assert all(r['grade'] == 'incorrect' for r in
               grade_all(failed, manifest, 'unused', _trusted_selftest=True))
    math_manifest = dict(manifest, tasks=[dict(task, kind='math_manual')])
    assert all(r['grade'] == 'manual_review' for r in
               grade_all(rows, math_manifest, 'unused', _trusted_selftest=True))
    for bad in (rows[:-1], rows + rows[:1]):
        try:
            validate_responses(bad, manifest, _trusted_selftest=True)
        except ValueError:
            pass
        else:
            raise AssertionError('missing/duplicate accepted')
    return {'selftest': 'passed', 'candidate_execution': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--image-id', help='full pinned Jail image ID; required for production')
    parser.add_argument('--inputlist', nargs='+', help='all pilot/main responses.jsonl paths, both arms; no diagnostics')
    parser.add_argument('--manifest', help='full frozen analysis-{arm}.json')
    parser.add_argument('--out', help='new canonical graded JSONL')
    args = parser.parse_args()
    if args.selftest:
        print(canonical(selftest()))
        return
    if not all((args.image_id, args.inputlist, args.manifest, args.out)):
        parser.error('--image-id --inputlist --manifest --out are required')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', args.image_id):
        parser.error('--image-id must be a full immutable sha256 image ID')
    if Path(args.out).exists():
        raise ValueError('refuse to overwrite grading output')
    rows = [r for p in args.inputlist for r in bench.read_jsonl(p)]
    output = grade_all(rows, json.loads(Path(args.manifest).read_text()), args.image_id)
    descriptor = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        for row in output:
            stream.write(canonical(row) + '\n')


if __name__ == '__main__':
    main()
