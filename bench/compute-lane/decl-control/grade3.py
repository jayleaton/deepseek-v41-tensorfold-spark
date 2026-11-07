#!/usr/bin/env python3
"""Three-arm offline grading adapter; never execute a client-visible tool attempt.

Public API: grade_all(rows, manifest, image_id), source_hashes(), gsm_sensitivity().
Seal source_hashes() before collection and pass it as expected_hashes when grading.
HumanEval execution remains exclusively inside the frozen phase2 Jail.
"""
import argparse
import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'phase2'))
import analysis as phase2

ARMS = ('plain', 'declared_off', 'lane')
SEEDS = (101, 102)
STRICT_SHA256 = 'dcc8ef766247a1b02c993c9a50186b70b68c4fb5b6493e1bb42b3f7727186d36'
WORKER_SHA256 = 'ea45e134f5e38b40b13870c1b1efa3d54e9dae752074d126bda2d8cb74eae2b5'
spec = importlib.util.spec_from_file_location('_decl_control_strict', ROOT.parent / 'phase2/grade_all.py')
strict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strict)
strict.ARMS = ARMS


def source_hashes():
    paths = {'strict': ROOT.parent / 'phase2/grade_all.py',
             'worker': ROOT.parent / 'kit/grader_worker.py', 'wrapper': Path(__file__)}
    return {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}


def verify_sources(expected_hashes=None):
    hashes = source_hashes()
    if hashes['strict'] != STRICT_SHA256 or hashes['worker'] != WORKER_SHA256:
        raise ValueError('frozen strict grader/worker source changed')
    if expected_hashes is not None and hashes != expected_hashes:
        raise ValueError('precollection grader/wrapper hashes changed')
    return hashes


NUM = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:\s*/\s*[+-]?\d+)?"
PATTERN = r"\s*[A-Za-z\s'’]*\$?\s*(" + NUM + r")[A-Za-z\s\-'’]*[.!]?\s*"
BLOCKED = r"\b(or|about|approximately|million|billion|thousand|hundred|percent)\b"


def gsm_sensitivity(task, content):
    primary = strict.gsm_grade(task, content)
    result = dict(primary, strict_grade=primary['grade'], changed=False,
                  rule='ONE-NUMERAL FINAL-SPAN secondary extraction sensitivity')
    if primary['grade'] == 'correct':
        return result
    span = primary['extracted_final']
    match = re.fullmatch(PATTERN, span) if isinstance(span, str) else None
    actual = None
    if match and not re.search(BLOCKED, span, re.I):
        try:
            actual = Fraction(match.group(1).replace(',', '').replace(' ', ''))
        except (ValueError, ZeroDivisionError):
            pass
    if actual is not None and actual == Fraction(primary['reference_fraction']):
        result.update(grade='correct', changed=True, reason='one_numeral_final_span')
    result['sensitivity_fraction'] = str(actual) if actual is not None else None
    return result


def grade_all(rows, manifest, image_id, *, expected_hashes=None,
              _trusted_selftest=False, _trusted_jail=None):
    hashes = verify_sources(expected_hashes)
    if expected_hashes is None and not _trusted_selftest:
        raise ValueError('production grading requires precollection expected_hashes')
    output = strict.grade_all(rows, manifest, image_id, _trusted_selftest=_trusted_selftest,
                              _trusted_jail=_trusted_jail)
    tasks = {(t['suite'], t['id']): t for t in phase2.manifest_tasks(manifest)}
    for row in output:
        row['grader']['adapter'] = 'decl-control.grade3'
        row['grader']['precollection_hashes'] = hashes
        task = tasks[row['suite'], row['id']]
        if task['kind'] in ('integer', 'numeric'):
            row['gsm_sensitivity'] = (gsm_sensitivity(task, row['content']) if row['status'] == 'ok'
                                      else {'grade': 'incorrect', 'changed': False,
                                            'reason': 'generation_' + row['status']})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hashes', action='store_true')
    p.add_argument('--manifest'); p.add_argument('--inputlist', nargs='+')
    p.add_argument('--image-id'); p.add_argument('--precollection-hashes'); p.add_argument('--out')
    a = p.parse_args()
    if a.hashes:
        print(phase2.canonical(verify_sources())); return
    if not all((a.manifest, a.inputlist, a.image_id, a.precollection_hashes, a.out)):
        p.error('manifest, inputlist, image-id, precollection-hashes and out required')
    rows = [r for path in a.inputlist for r in phase2.read_jsonl(path)]
    result = grade_all(rows, json.loads(Path(a.manifest).read_text()), a.image_id,
                       expected_hashes=json.loads(Path(a.precollection_hashes).read_text()))
    with open(a.out, 'x', encoding='utf-8') as stream:
        for row in result:
            stream.write(phase2.canonical(row) + '\n')


if __name__ == '__main__':
    main()
