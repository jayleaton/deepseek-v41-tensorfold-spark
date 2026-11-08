#!/usr/bin/env python3
"""Offline record audit. Never sends requests, executes candidates, or reruns references."""
import argparse
import ast
import base64
import hashlib
import ipaddress
import json
import math
import re
import sys
from pathlib import Path

SPEC_SHA = '9c7327007cfd15825a608f2f6b06f2b39e33b88548c5bb7deb1b209b7395d240'
REVISION = '3a429d6f122d9eb2900c43e622933d311e4be44f'
IMAGE_IDS = {
    'sha256:5be7d7d40f267590b9e7b41410f48d40048af24e004203629a347d3d7c16a8cd',
    'sha256:80beaba732d1ed9439e012b007d8dbe19ec2f8962f50fbac29a18c8cd6fda306',
}
BOOT_NAMES = ['pilot-plain', 'pilot-lane'] + [
    f't{i:02d}-{arm}' for i in range(1, 7) for arm in ('plain', 'lane')]
ACCESS = re.compile(
    r'^\[tensorfold\] (?P<ip>\S+) "(?P<method>[A-Z]+) (?P<path>[^" ]+) '
    r'HTTP/1\.1" (?P<status>[0-9]{3}) -\s*$')
POST_HINT = re.compile(r'\bPOST\s')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def object_sha(value):
    return digest(canonical(value).encode())


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def strict_json(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result

    def invalid(_value):
        raise ValueError('nonfinite JSON constant')

    return json.loads(blob, object_pairs_hook=pairs, parse_constant=invalid)


class Inputs:
    """Hash only explicitly consumed files; never traverse reviews or private keys."""
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        path = Path(path).resolve()
        blob = path.read_bytes()
        sha = digest(blob)
        require(str(path) not in self.hashes or self.hashes[str(path)] == sha,
                'input changed while reading')
        self.hashes[str(path)] = sha
        return blob

    def json(self, path):
        return strict_json(self.read(path))

    def rows(self, path):
        return [strict_json(line) for line in self.read(path).splitlines() if line.strip()]

    def stable(self):
        for path, sha in self.hashes.items():
            require(digest(Path(path).read_bytes()) == sha, 'input changed during audit')


def inside(root, relative):
    require(isinstance(relative, str), 'invalid frozen file path')
    path = (root / relative).resolve()
    require(path.is_relative_to(root), 'frozen file escapes root')
    return path


def frozen_read(inputs, root, manifest, relative):
    blob = inputs.read(inside(root, relative))
    require(digest(blob) == manifest['files_sha256'].get(relative), 'frozen file hash drift')
    return blob


def definitions(blob, names, namespace):
    """Load only named, unchanged trusted definitions; no controller/transport imports."""
    tree = ast.parse(blob)
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    require({node.name for node in selected} == set(names), 'frozen API definitions missing')
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<frozen-audit-api>', 'exec'), namespace)
    return namespace


def load_apis(inputs, root, manifest, tokenizer, tokenizer_sha):
    sources = {}
    for name in ('client.py', 'native_frames.py', 'controller.py', 'jail_worker.py'):
        matches = [entry for entry in manifest['code_sources'] if Path(entry['path']).name == name]
        require(len(matches) == 1, 'missing/ambiguous frozen API source: ' + name)
        entry = matches[0]
        blob = frozen_read(inputs, root, manifest, entry['file'])
        require(digest(blob) == entry['sha256'], 'code source hash drift')
        sources[name] = blob
    common = dict(hashlib=hashlib, json=json, Path=Path)
    client = definitions(sources['client.py'],
                         ('canonical', 'sha_ids', 'accounting', 'NativeDecoder'), dict(common))
    # native_frames has only stdlib imports, constants and definitions. Drop its
    # imports and all top-level executable statements, preserving exact constants.
    tree = ast.parse(sources['native_frames.py'])
    body = [node for node in tree.body if isinstance(node, (ast.Assign, ast.FunctionDef))]
    native = dict(common)
    exec(compile(ast.Module(body=body, type_ignores=[]), '<frozen-native-frames>', 'exec'), native)
    compare = definitions(sources['controller.py'], ('reference_matches',), {'base64': base64})
    require(re.fullmatch(r'[0-9a-f]{64}', tokenizer_sha) is not None, 'invalid tokenizer SHA')
    require(digest(inputs.read(tokenizer)) == tokenizer_sha, 'tokenizer SHA drift')
    decoder = client['NativeDecoder'](tokenizer, tokenizer_sha)  # Exactly once.
    require(digest(inputs.read(tokenizer)) == tokenizer_sha, 'tokenizer changed during load')
    return client['accounting'], native['authenticate'], compare['reference_matches'], decoder, digest(sources['jail_worker.py'])


def identity(row):
    key = tuple(row.get(field) for field in ('suite', 'id', 'seed', 'arm', 'boot'))
    require(all(isinstance(value, str) for value in (key[0], key[1], key[3], key[4]))
            and type(key[2]) is int, 'invalid record identity')
    return key


def indexed(rows):
    result = {}
    for row in rows:
        require(isinstance(row, dict), 'record must be an object')
        key = identity(row)
        require(key not in result, 'duplicate record identity')
        result[key] = row
    return result


def body_matches(row, template):
    return (row.get('task_sha256') == template['task_sha256']
            and object_sha(row.get('request')) == template['body_sha256']
            and ('body_sha256' not in row or row['body_sha256'] == template['body_sha256']))


def http_audit(text, expected):
    result = dict(expected_posts=expected, successful_completion_posts=0, parsed_access_lines=0,
                  post_lines=0, foreign_access_lines=0, unexpected_post_lines=0,
                  unrecognized_post_lines=0, qualified=False, reasons=[])
    for line in text.splitlines():
        match = ACCESS.fullmatch(line)
        if match is None:
            if POST_HINT.search(line):
                result['unrecognized_post_lines'] += 1
            continue
        result['parsed_access_lines'] += 1
        fields = match.groupdict()
        try:
            loopback = ipaddress.ip_address(fields['ip']).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            result['foreign_access_lines'] += 1
        if fields['method'] == 'POST':
            result['post_lines'] += 1
            if fields['path'] == '/v1/chat/completions' and fields['status'] == '200':
                result['successful_completion_posts'] += 1
            else:
                result['unexpected_post_lines'] += 1
    if not result['parsed_access_lines']:
        result['reasons'].append('access format/coverage missing; parent must inspect actual log format')
    if result['successful_completion_posts'] != expected or result['post_lines'] != expected:
        result['reasons'].append('POST count/coverage differs from frozen dataset plus two diagnostics')
    for field in ('foreign_access_lines', 'unexpected_post_lines', 'unrecognized_post_lines'):
        if result[field]:
            result['reasons'].append(field)
    result['qualified'] = not result['reasons']
    return result


def reference_audit(calls, safety, compare, worker_sha):
    gaps = []
    refs = safety.get('references')
    if not isinstance(refs, list):
        return ['missing retained reference list']
    delivered = [call for call in calls if call.get('inserted') is True]
    ids = [call.get('id') for call in calls]
    if any(not isinstance(value, (str, int)) or isinstance(value, bool) for value in ids):
        return ['invalid call ID']
    if len(set(ids)) != len(ids):
        return ['duplicate native call ID']
    by_id = {}
    for ref in refs:
        if not isinstance(ref, dict) or type(ref.get('call_id')) not in (str, int):
            gaps.append('invalid retained reference identity')
            continue
        key = ref['call_id']
        if key in by_id:
            gaps.append('duplicate retained reference call ID')
        by_id[key] = ref
    if set(by_id) != {call['id'] for call in delivered}:
        gaps.append('missing or extra retained reference call IDs')
    for call in delivered:
        ref = by_id.get(call['id'], {})
        result = ref.get('result')
        if not isinstance(result, dict):
            gaps.append('missing retained Jail result')
            continue
        # Payload hash authenticates this exact code as a fresh, independent
        # reference request; never call Jail, import jail, or execute its payload.
        image = result.get('image') or {}
        provenance = (result.get('payload_sha256') == object_sha({'mode': 'reference', 'code': call.get('code')})
                      and result.get('worker_sha256') == worker_sha
                      and image.get('id') in IMAGE_IDS and image.get('revision') == REVISION)
        if not provenance:
            gaps.append('retained reference source/worker/image provenance gap')
        if ref.get('matches') is not True or not compare(call, result):
            gaps.append('retained reference does not match exact delivered status/output')
        if call.get('status') != 'ok' or result.get('status') != 'ok':
            gaps.append('delivered execution error retained as gap')
    return gaps


def numeric(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def metadata(row):
    """Explicit metadata only. No tokenization, subtraction, or timing inference."""
    usage, tf = row.get('usage') or {}, row.get('tensorfold') or {}
    details = usage.get('completion_tokens_details') or {}
    return dict(client_elapsed_seconds=numeric(row.get('elapsed_seconds')),
                prompt_tokens=numeric(usage.get('prompt_tokens')),
                reasoning_tokens=numeric(details.get('reasoning_tokens')),
                content_tokens=numeric(details.get('content_tokens')),
                server_seconds=numeric(tf.get('server_seconds')),
                tool_seconds=numeric(tf.get('tool_seconds')))


def costs(rows):
    result = {}
    for field in ('native_ids', 'completion_tokens', 'inserted_tokens', 'client_elapsed_seconds',
                  'prompt_tokens', 'reasoning_tokens', 'content_tokens', 'server_seconds', 'tool_seconds'):
        values = [row[field] for row in rows if row.get(field) is not None]
        result[field] = dict(observed_sum=sum(values) if values else None,
                             observed_rows=len(values), missing_rows=len(rows) - len(values))
    return result


def directory(root, boot):
    if boot == 'pilot-plain':
        turn = 'stdbench-20261105-v1-pilot'
    elif boot == 'pilot-lane':
        turn = 'stdbench-20261105-v2-pilot'
    else:
        turn = 'stdbench-20261105-v2-' + boot.split('-')[0]
    return root / turn / boot


def audit_boot(inputs, data_root, boot, apis):
    accounting, authenticate, compare, decoder, worker_sha = apis
    name, arm = boot['boot'], boot['arm']
    result = dict(boot=name, arm=arm, qualified=False, gaps=[], rows=[], http=None)
    folder = directory(data_root, name)
    observations, safeties = {}, {}
    for filename, templates in (('responses.jsonl', boot['requests']),
                                ('diagnostics.jsonl', boot['diagnostics'])):
        expected = indexed([dict(template, arm=arm, boot=name) for template in templates])
        try:
            actual = indexed(inputs.rows(folder / filename))
            if set(actual) != set(expected):
                result['gaps'].append(filename + ': missing/extra identities')
            observations.update({key: (row, expected.get(key), filename == 'diagnostics.jsonl')
                                 for key, row in actual.items()})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result['gaps'].append(filename + ': ' + type(exc).__name__)
    try:
        safeties = indexed(inputs.rows(folder / 'safety.jsonl'))
        if set(safeties) != set(observations):
            result['gaps'].append('missing/extra safety identities')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result['gaps'].append('safety.jsonl: ' + type(exc).__name__)
    diagnostic_successes = 0
    for key, (row, template, diagnostic) in observations.items():
        summary = dict(zip(('suite', 'id', 'seed', 'arm', 'boot'), key))
        summary.update(diagnostic=diagnostic, gaps=[], native_ids=None, completion_tokens=None,
                       inserted_tokens=None, calls=None, delivered_successes=None, **metadata(row))
        gate = 'frozen row identity/body/task'
        try:
            require(template is not None, 'unexpected row')
            require(row.get('diagnostic') is diagnostic, 'diagnostic identity flag drift')
            require(body_matches(row, template), 'request body/task SHA drift')
            gate = 'native accounting'
            counts = accounting(row)
            # Recompute and replace self-reported accounting before authentication.
            checked = dict(row, native_accounting=counts)
            binding = authenticate(checked, decoder)
            summary.update(native_ids=counts['returned_tokens'],
                           completion_tokens=numeric(counts['api_completion_tokens']),
                           inserted_tokens=(numeric(counts['inserted_tokens'])
                                            if counts['complete'] else None))
            require(counts['complete'], 'incomplete native accounting/error response')
            gate = 'exact native frame binding'
            require(binding['qualified'], 'native frame authentication failed')
            calls = ((row.get('tensorfold') or {}).get('compute') or {}).get('calls', [])
            summary.update(calls=len(calls), delivered_successes=binding['delivered_successes'])
            gate = 'retained safety/binding'
            safety = safeties.get(key, {})
            require(safety.get('qualified') is True, 'missing/unqualified retained safety receipt')
            require(safety.get('binding') == binding, 'retained binding differs from recomputed binding')
            summary['gaps'].extend(reference_audit(calls, safety, compare, worker_sha))
            if any(call.get('status') != 'ok' for call in calls):
                summary['gaps'].append('native execution error remains a gap')
            if arm == 'plain':
                gate = 'plain zero-call'
                require(not calls, 'plain native calls present')
            if diagnostic:
                gate = 'diagnostic accepted thinking/high and fixed successful output'
                tf = row.get('tensorfold') or {}
                require(tf.get('enable_thinking') is True and tf.get('reasoning_effort') == 'high',
                        'diagnostic thinking/high not accepted')
                if arm == 'lane':
                    reference = str(template['reference'])
                    require(reference in ('323', '55'), 'unexpected diagnostic reference')
                    good = [call for call in calls if call.get('inserted') is True
                            and call.get('status') == 'ok' and call.get('output') == reference]
                    successful = [call for call in calls if call.get('inserted') is True
                                  and call.get('status') == 'ok']
                    require(not successful or good, 'diagnostic successful output differs from fixed reference')
                    if good and not summary['gaps']:
                        diagnostic_successes += len(good)
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, UnicodeError):
            # Never emit exception text: malformed records can embed model text.
            summary['gaps'].append(gate + ' gate failed')
        result['rows'].append(summary)
    if arm == 'lane' and diagnostic_successes == 0:
        result['gaps'].append('no independently referenced successful delivered 323/55 diagnostic')
    try:
        result['http'] = http_audit(inputs.read(folder / 'rank0.log').decode('utf-8', 'strict'),
                                    len(boot['requests']) + 2)
    except (OSError, UnicodeError):
        result['http'] = dict(qualified=False, reasons=['missing/unreadable rank0.log coverage'])
    result['diagnostic_successes'] = diagnostic_successes
    result['expected_dataset_rows'] = len(boot['requests'])
    result['observed_dataset_rows'] = sum(not row['diagnostic'] for row in result['rows'])
    result['observed_diagnostic_rows'] = sum(row['diagnostic'] for row in result['rows'])
    result['safety_gap_rows'] = sum(bool(row['gaps']) for row in result['rows'])
    result['dataset_costs'] = costs([row for row in result['rows'] if not row['diagnostic']])
    result['diagnostic_costs'] = costs([row for row in result['rows'] if row['diagnostic']])
    result['qualified'] = (not result['gaps'] and not result['safety_gap_rows']
                           and result['http']['qualified'])
    return result


def run(args):
    inputs = Inputs()
    auditor_sha = digest(inputs.read(Path(__file__)))
    root, data_root = Path(args.freeze).resolve(), Path(args.data_root).resolve()
    manifest = inputs.json(root / 'manifest.json')
    require(manifest.get('spec_sha256') == SPEC_SHA, 'freeze does not bind required spec SHA')
    require(digest(frozen_read(inputs, root, manifest, 'sources/spec.bin')) == SPEC_SHA,
            'frozen spec SHA drift')
    inventory = inputs.json(root / 'file-hashes.json')
    require(digest(inputs.read(root / 'file-hashes.json')) ==
            inputs.read(root / 'file-hashes.sha256').decode().split()[0], 'freeze inventory drift')
    require(inventory.get('manifest.json') == digest(inputs.read(root / 'manifest.json')),
            'manifest/inventory mismatch')
    boots = []
    for entry in manifest['boots']:
        blob = frozen_read(inputs, root, manifest, entry['file'])
        require(digest(blob) == entry['sha256'], 'boot manifest SHA drift')
        boot = strict_json(blob)
        require(boot['boot'] == entry['boot'] and boot['arm'] == entry['arm'], 'boot identity drift')
        require(boot['arm'] in ('plain', 'lane') and len(boot['diagnostics']) == 2, 'boot contract drift')
        for template in boot['requests'] + boot['diagnostics']:
            require(object_sha(template['task']) == template['task_sha256']
                    and object_sha(template['body_template']) == template['body_sha256'],
                    'frozen template hash drift')
        boots.append(boot)
    require(len(boots) == 14 and {boot['boot'] for boot in boots} == set(BOOT_NAMES), 'boot coverage drift')
    expected = [dict(row, arm=boot['arm'], boot=boot['boot']) for boot in boots for row in boot['requests']]
    cohort_keys = [(row['suite'], row['id'], row['seed'], row['arm']) for row in expected]
    require(len(cohort_keys) == 368 and len(set(cohort_keys)) == 368, 'frozen cohort duplicate/count drift')
    require({row['seed'] for row in expected} == {101, 102}, 'frozen seed drift')
    for suite, n in (('gsm8k', 48), ('math500', 20), ('humaneval', 24)):
        subset = [key for key in cohort_keys if key[0] == suite]
        ids = {key[1] for key in subset}
        require(len(ids) == n and set(subset) == {(suite, task, seed, arm) for task in ids
                for seed in (101, 102) for arm in ('plain', 'lane')}, 'frozen suite grid drift')
    apis = load_apis(inputs, root, manifest, args.tokenizer, args.tokenizer_sha)
    verdicts = [audit_boot(inputs, data_root, boot, apis) for boot in boots]
    all_rows = [row for boot in verdicts for row in boot['rows']]
    dataset = [row for row in all_rows if not row['diagnostic']]
    diagnostics = [row for row in all_rows if row['diagnostic']]
    activation = []
    for suite in ('gsm8k', 'math500', 'humaneval'):
        for seed in (101, 102):
            selected = [row for row in dataset if row['suite'] == suite and row['seed'] == seed
                        and row['arm'] == 'lane']
            known = [row for row in selected if row['calls'] is not None]
            used = sum(row['calls'] > 0 for row in known)
            activation.append(dict(suite=suite, seed=seed, observed_rows=len(selected),
                call_coverage_rows=len(known), replies_using_tool=used,
                fraction_observed_using_tool=used / len(selected) if selected else None,
                calls=sum(row['calls'] for row in known) if known else None,
                successful_delivered=sum(row['delivered_successes'] for row in known) if known else None,
                executor='code tool' if known and used else None,
                label='DECLARED BUT UNUSED' if len(known) == len(selected) and selected and not used else None))
    inputs.stable()
    return dict(version=1, qualified=all(boot['qualified'] for boot in verdicts)
                and len(dataset) == 368 and len(diagnostics) == 28,
                auditor_sha256=auditor_sha, spec_sha256=SPEC_SHA,
                freeze_manifest_sha256=digest(inputs.read(root / 'manifest.json')),
                tokenizer_sha256=args.tokenizer_sha, input_file_sha256=inputs.hashes,
                expected_dataset_rows=368, observed_dataset_rows=len(dataset),
                expected_diagnostic_rows=28, observed_diagnostic_rows=len(diagnostics),
                dataset_costs=costs(dataset), diagnostic_costs=costs(diagnostics),
                activation=activation, boots=verdicts,
                limitations=['Offline retained-evidence audit; no candidate/reference execution or data requests.',
                            'Environment proof is handled separately by parent.',
                            'Freeze inventory is internally checked; externally authenticate its reported manifest SHA.',
                            'Missing metadata is null; observed sums are lower bounds, not full-cohort claims.'])


def selftest():
    """Synthetic fixtures only; parent runs this on constrained Spark, never dev."""
    line = '[tensorfold] 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" 200 -\n'
    require(http_audit(line * 2, 2)['qualified'], 'HTTP positive fixture')
    require(not http_audit('', 2)['qualified'], 'HTTP missing fixture')
    require(not http_audit(line.replace('127.0.0.1', '192.0.2.1'), 1)['qualified'], 'HTTP foreign fixture')
    require(not http_audit(line.replace('/v1/chat/completions', '/unexpected'), 1)['qualified'], 'HTTP path fixture')
    require(not http_audit(line * 2, 1)['qualified'], 'HTTP duplicate/count fixture')
    template = dict(task_sha256='task', body_sha256=object_sha({'seed': 101}))
    require(body_matches(dict(task_sha256='task', request={'seed': 101}), template), 'body positive fixture')
    require(not body_matches(dict(task_sha256='task', request={'seed': 102}), template), 'body drift fixture')
    require(reference_audit([dict(id='a', inserted=True)], {'references': []}, None, 'worker'),
            'missing reference fixture')
    require(costs([dict(native_ids=None)])['native_ids']['observed_sum'] is None, 'missing is not zero')
    return dict(selftest=True, synthetic=True, qualified=False, passed=True,
                tests=9, note='Helper fixtures only; no campaign qualification or native tokenizer test.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root')
    parser.add_argument('--freeze')
    parser.add_argument('--tokenizer')
    parser.add_argument('--tokenizer-sha')
    parser.add_argument('--out', required=True, help='new JSON file; never overwritten')
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if not args.selftest and not all((args.data_root, args.freeze, args.tokenizer, args.tokenizer_sha)):
        parser.error('audit requires --data-root --freeze --tokenizer --tokenizer-sha')
    out = Path(args.out).resolve()
    if out.exists():
        parser.error('--out must be a new file')
    if not args.selftest:
        for source in (Path(args.data_root).resolve(), Path(args.freeze).resolve()):
            if out.is_relative_to(source):
                parser.error('--out must be outside data-root and freeze')
        if out == Path(args.tokenizer).resolve():
            parser.error('--out aliases tokenizer')
    try:
        report = selftest() if args.selftest else run(args)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError, SyntaxError):
        report = dict(qualified=False, fatal='input/freeze/schema/API audit failure; no qualification',
                      note='No exception payload emitted to avoid exposing record content.')
    with out.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(canonical(dict(qualified=report.get('qualified', False), selftest=report.get('selftest', False),
                         passed=report.get('passed'), dataset=report.get('observed_dataset_rows'),
                         diagnostics=report.get('observed_diagnostic_rows'))))
    return 0 if report.get('qualified') or report.get('passed') else 1


if __name__ == '__main__':
    sys.exit(main())
