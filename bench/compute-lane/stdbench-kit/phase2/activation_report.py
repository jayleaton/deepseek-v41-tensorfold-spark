#!/usr/bin/env python3
"""Descriptive in-dataset native activation; never drops error/censored outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path

from analysis import ARMS, SEEDS, canonical, digest, identity, read_jsonl, write_json
from grade_all import validate_responses


def report(rows, safety_rows, manifest, *, _trusted_selftest=False):
    _, indexed = validate_responses(rows, manifest, _trusted_selftest=_trusted_selftest)
    safety = {}
    diagnostic_safety = 0
    for item in safety_rows:
        if item.get('suite') == 'diagnostic':
            diagnostic_safety += 1
            continue
        key = identity(item)
        if key not in indexed or key in safety:
            raise ValueError('unexpected/duplicate safety response identity')
        safety[key] = item
    suites = {}
    for suite in sorted({r['suite'] for r in rows}):
        suite_report = {}
        lane_calls = 0
        for arm in ARMS:
            by_seed = {}
            for seed in SEEDS:
                selected = [r for r in rows if (r['suite'], r['arm'], r['seed']) == (suite, arm, seed)]
                calls_total = delivered = successful = independently_verified = replies_using = 0
                executors, failed_safety, missing_safety = Counter(), [], []
                for row in selected:
                    key = identity(row)
                    compute = ((row.get('tensorfold') or {}).get('compute') or {})
                    calls = compute.get('calls', [])
                    if not isinstance(calls, list) or any(not isinstance(c, dict) for c in calls):
                        raise ValueError('native compute.calls must be a list of call objects')
                    calls_total += len(calls)
                    replies_using += bool(calls)
                    evidence = safety.get(key)
                    if evidence is None:
                        missing_safety.append({'id': row['id'], 'seed': seed})
                    elif evidence.get('qualified') is not True:
                        failed_safety.append({'id': row['id'], 'seed': seed})
                    bound_calls = ((evidence or {}).get('binding') or {}).get('calls', [])
                    if not isinstance(bound_calls, list) or any(not isinstance(c, dict) for c in bound_calls):
                        raise ValueError('safety binding.calls must be a list')
                    native_ids = [c.get('id') for c in calls]
                    if len(set(native_ids)) != len(native_ids) or any(i is None for i in native_ids):
                        raise ValueError('native call IDs must be present and unique per reply')
                    bound_ids = [c.get('id') for c in bound_calls]
                    if len(set(bound_ids)) != len(bound_ids) or not set(bound_ids) <= set(native_ids):
                        raise ValueError('duplicate/foreign bound tool call IDs')
                    refs = (evidence or {}).get('references', [])
                    ref_ids = [r.get('call_id') for r in refs]
                    if len(set(ref_ids)) != len(ref_ids) or not set(ref_ids) <= set(native_ids):
                        raise ValueError('duplicate/foreign reference tool call IDs')
                    for call in calls:
                        executors[str(call.get('executor', 'UNRECORDED'))] += 1
                    for call in bound_calls:
                        delivered += call.get('delivered') is True
                        okay = (call.get('delivered') is True and call.get('status') == 'ok'
                                and call.get('native_bound') is True)
                        successful += okay
                        independently_verified += okay and any(
                            r.get('call_id') == call['id'] and r.get('matches') is True for r in refs)
                n = len(selected)
                by_seed[str(seed)] = {
                    'scheduled_replies': n, 'calls': calls_total, 'delivered_results': delivered,
                    'successful_delivered_results': successful,
                    'independently_verified_successful_results': independently_verified,
                    'replies_using_tool': replies_using, 'fraction_replies_using_tool': replies_using / n,
                    'executors_reported_by_native_calls': dict(executors),
                    'executors_native_bound': dict(Counter(
                        c.get('executor', 'UNRECORDED') for r in selected
                        for c in ((safety.get(identity(r), {}).get('binding') or {}).get('calls', []))
                        if c.get('delivered') is True)),
                    'safety_rows_present': n - len(missing_safety),
                    'missing_safety': missing_safety, 'failed_safety': failed_safety,
                    'generation_status_counts': dict(Counter(r['status'] for r in selected))}
                if arm == 'lane':
                    lane_calls += calls_total
            suite_report[arm] = by_seed
        suite_report['lane_dataset_activation'] = 'DECLARED BUT UNUSED' if lane_calls == 0 else 'NATIVE CALLS OBSERVED'
        suite_report['execution_benefit_claim_supported'] = False
        suite_report['interpretation'] = ('Declaration/configuration contrast only; zero lane dataset calls.'
                                          if lane_calls == 0 else
                                          'Descriptive activation only; calls do not establish an efficacy benefit.')
        suite_report['plain_zero_calls'] = all(v['calls'] == 0 for v in suite_report['plain'].values())
        suites[suite] = suite_report
    return {'suites': suites, 'response_rows': len(rows), 'safety_rows': len(safety),
            'responses_sha256': digest(sorted(rows, key=identity)),
            'safety_sha256': digest(sorted(safety.values(), key=identity)),
            'diagnostic_safety_rows_excluded': diagnostic_safety,
            'qualified_efficacy_claim': False, 'outcome_filtering': False}


def load_safety(response_paths, safety_paths):
    """Controller safety rows omit arm: bind only via their exact boot directory."""
    directory_arms = {}
    for path in response_paths:
        rows = read_jsonl(path)
        arms = {r['arm'] for r in rows}
        if len(arms) == 1:
            directory = Path(path).resolve().parent
            previous = directory_arms.setdefault(directory, next(iter(arms)))
            if previous != next(iter(arms)):
                raise ValueError('response directory has conflicting arm bindings')
    output = []
    for path in safety_paths:
        for row in read_jsonl(path):
            row = dict(row)
            if row.get('suite') == 'diagnostic':
                output.append(row)
                continue
            if 'arm' not in row:
                directory = Path(path).resolve().parent
                if directory not in directory_arms:
                    raise ValueError('armless safety must share its one-arm response boot directory')
                row['arm'] = directory_arms[directory]
            output.append(row)
    return output


def selftest():
    task = {'suite': 'fixture', 'id': 'a', 'kind': 'numeric', 'answer': '2'}
    manifest = {'tasks': [task], 'seeds': list(SEEDS)}
    rows = [dict(suite='fixture', id='a', seed=s, arm=a, content='2', status='ok',
                 elapsed_seconds=1, tensorfold={'compute': {'calls': []}}) for s in SEEDS for a in ARMS]
    safety = [dict(suite=r['suite'], id=r['id'], seed=r['seed'], arm=r['arm'], qualified=True,
                   binding={'calls': []}, references=[]) for r in rows]
    result = report(rows, safety, manifest, _trusted_selftest=True)
    assert result['suites']['fixture']['lane_dataset_activation'] == 'DECLARED BUT UNUSED'
    assert result['response_rows'] == 4 and not result['qualified_efficacy_claim']
    lane = next(r for r in rows if r['arm'] == 'lane')
    lane['tensorfold'] = {'compute': {'calls': [{'id': 'c1', 'executor': 'code_tool', 'status': 'ok'}]}}
    match = next(s for s in safety if identity(s) == identity(lane))
    match['binding'] = {'calls': [{'id': 'c1', 'executor': 'code_tool', 'status': 'ok',
                                  'delivered': True, 'native_bound': True}]}
    match['references'] = [{'call_id': 'c1', 'matches': True}]
    result = report(rows, safety, manifest, _trusted_selftest=True)['suites']['fixture']
    assert result['lane']['101']['successful_delivered_results'] == 1
    assert result['lane']['101']['fraction_replies_using_tool'] == 1
    assert result['lane']['101']['independently_verified_successful_results'] == 1
    assert not result['execution_benefit_claim_supported']
    return {'selftest': 'passed', 'candidate_execution': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--inputlist', nargs='+', help='all dataset response JSONL paths, no diagnostics')
    parser.add_argument('--safety', nargs='+', help='all corresponding safety JSONL paths')
    parser.add_argument('--manifest', help='full frozen analysis manifest')
    parser.add_argument('--out', help='new report JSON')
    args = parser.parse_args()
    if args.selftest:
        print(canonical(selftest()))
        return
    if not all((args.inputlist, args.safety, args.manifest, args.out)):
        parser.error('--inputlist --safety --manifest --out required')
    rows = [r for path in args.inputlist for r in read_jsonl(path)]
    safety = load_safety(args.inputlist, args.safety)
    write_json(args.out, report(rows, safety, json.loads(Path(args.manifest).read_text())))


if __name__ == '__main__':
    main()
