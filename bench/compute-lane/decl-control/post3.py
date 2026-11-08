#!/usr/bin/env python3
"""Postcollection I/O only. Parent runs on rank1-host after every G restoration.

prepare audits retained evidence, calls the sealed graders and seals all120 MATH
answers. finalize accepts math3.merge JSONL and changes only MATH review fields.
No transport, reference execution, source changes, or statistical implementation.
These wrappers were written after collection began and are not sealed business logic.
"""
import argparse
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'phase2'))
sys.path.insert(0, str(ROOT))
import analysis3
import audit_records
import client3
import grade3
import math3
import reconciliation
from generate_envs3 import IMAGE_IDS, PREFIX, ORDERS

BASE_SHA = analysis3.SPEC_SHA256
A1_SHA = analysis3.A1_SHA256
TOKENIZER_SHA = 'c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    math3.write_private(path, value)


def jsonl(path, rows):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(analysis3.canonical(row) + '\n')


def key(row):
    return analysis3.identity(row) + (row['boot'],)


def index(rows):
    result = {}
    for row in rows:
        identity = key(row)
        require(identity not in result, 'duplicate retained identity')
        result[identity] = row
    return result


def local_source(path):
    # Explicit root remap supports rank0-host copies on rank1-host/root2 without reading
    # arbitrary absolute paths from a receipt. Relative kit structure is preserved.
    parts = Path(path).parts
    require('stdbench' in parts, 'source path outside kit')
    return ROOT.parent.joinpath(*parts[parts.index('stdbench') + 1:])


def sources(inputs):
    receipt = inputs.json(ROOT / 'cpu-final-a1-v2/verification.json')
    require(receipt['passed'] is True and receipt['fresh_replies'] == 552,
            'final CPU prerequisite receipt required')
    require(receipt['base_spec_sha256'] == BASE_SHA and receipt['spec_sha256'] == A1_SHA,
            'CPU receipt protocol mismatch')
    for name, expected in (('T5-DECL-CONTROL-SPEC.md', BASE_SHA),
                           ('T5-DECL-CONTROL-SPEC-A1.md', A1_SHA)):
        require(audit_records.digest(inputs.read(ROOT / name)) == expected, 'spec drift')
    for path, expected in receipt['code_sha256'].items():
        require(audit_records.digest(inputs.read(local_source(path))) == expected,
                'precollection source/config drift')
    require(grade3.source_hashes() == receipt['grader_hashes'], 'grader source drift')
    frozen = ROOT / 'frozen-a1-v1'
    require(audit_records.digest(inputs.read(frozen / 'manifest.json')) == receipt['freeze_sha256'],
            'request freeze drift')
    manifest = inputs.json(frozen / 'manifest.json')
    require(manifest['base_spec_sha256'] == BASE_SHA and manifest['spec_sha256'] == A1_SHA,
            'freeze protocol mismatch')
    require(audit_records.digest(inputs.read(frozen / 'analysis-manifest.json')) ==
            manifest['analysis_manifest_sha256'], 'analysis manifest drift')
    analysis_manifest = inputs.json(frozen / 'analysis-manifest.json')
    require(audit_records.digest(inputs.read(ROOT / 'prompt-arrays.jsonl')) ==
            receipt['cpu_prompt_arrays_sha256'] == manifest['cpu_prompt_arrays_sha256'],
            'CPU arrays drift')
    prompts = inputs.rows(ROOT / 'prompt-arrays.jsonl')
    require(len(prompts) == 186 and len({tuple(p['key']) for p in prompts}) == 186,
            'full CPU prompt grid required')
    return receipt, manifest, analysis_manifest, {tuple(p['key']): p for p in prompts}


def turns(inputs, data, manifest):
    reports = []
    require(len(manifest['boots']) == 9 and manifest['fresh_replies'] == 552,
            'nine frozen boots required')
    require([t['name'] for t in manifest['turns']] == ['t01', 't02', 't03'], 'turn grid drift')
    for turn, order in ORDERS:
        layout = next(t for t in manifest['turns'] if t['name'] == turn)
        require(layout['boots'] == [turn + '-' + arm for arm in order], 'arm order drift')
        directory = data / (PREFIX + '-' + turn)
        verdict = inputs.json(directory / 'verdict.json')
        steps = inputs.rows(directory / 'steps.jsonl')
        require(verdict['verdict'] == 'QUALIFIED' and verdict['error'] is None
                and verdict['spec_sha256'] == A1_SHA, 'turn verdict unqualified')
        def event(step, value, boot=None):
            found = [e for e in steps if e['step'] == step and e['verdict'] == value
                     and (boot is None or e.get('boot') == boot)]
            require(len(found) == 1, 'missing/duplicate turn step evidence')
            return found[0]
        event('collection', 'COMPLETE')
        end = event('turn', 'QUALIFIED')
        released = event('gpu-locks', 'RELEASED')
        restored = event('G-restore', 'SERVING_VERIFIED')
        require(isinstance(restored.get('serving_run'), str)
                and restored['serving_run'].startswith('efast2-'), 'G restore identity missing')
        log = inputs.read(directory / 'G-restore.log').decode('utf-8')
        require('POPPED' in log, 'G restore log lacks completion')
        dt = lambda e: datetime.datetime.fromisoformat(e['at'])
        require(dt(released) <= dt(restored) <= dt(end), 'restoration step order invalid')
        start = event('gpu-locks', 'HELD')
        boot_times = []
        for boot in layout['boots']:
            ready = event('boot', 'READY_EFFECTIVE_ENV_VERIFIED', boot)
            event('diagnostics', 'QUALIFIED', boot)
            event('boot-stop', 'DRAINED', boot)
            drained = event('owned-containers', 'DRAIN_VERIFIED', boot)
            require(dt(start) <= dt(ready) <= dt(drained), 'boot step order invalid')
            boot_times.append({'boot': boot, 'admission_load_seconds': (dt(ready)-dt(start)).total_seconds()})
            start = drained
        reports.append({'turn': turn, 'qualified': True, 'boots': boot_times,
                        'G_restore_seconds': (dt(restored)-dt(released)).total_seconds(),
                        'serving_run': restored['serving_run'],
                        'evidence': 'controller verified actual G status and both image/running IDs; retained steps plus POPPED log'})
    return reports


def reference_matches(call, result):
    try:
        return (call.get('status') == result.get('status') and
                base64.b64decode(result['normalized_body_base64'], validate=True) ==
                call.get('output', '').encode())
    except (KeyError, ValueError, TypeError):
        return False


def audit_row(raw, safety, template, prompt, decoder, worker_sha, diagnostic):
    row = dict(raw)
    gaps = []
    def check(condition, reason):
        if not condition:
            gaps.append(reason)
    # Reassemble received fields from retained frames through the frozen parser.
    # DONE is a separate collector flag, not recoverable from its frame list.
    parsed = client3.old.parse_stream(['data: ' + client3.canonical(f)
                                      for f in row.get('sse_frames') or []])
    for field in ('usage', 'tensorfold', 'content', 'reasoning', 'finish_reason',
                  'external_tool_calls', 'live_compute_reports'):
        check(raw.get(field) == parsed[field], 'received SSE ' + field + ' mismatch')
        row[field] = parsed[field]
    check(row.get('diagnostic') is diagnostic, 'diagnostic flag mismatch')
    check(row.get('task_sha256') == template['task_sha256'], 'task hash mismatch')
    check(row.get('request') == template['body_template'], 'request object mismatch')
    check(row.get('execution_policy') == template['execution_policy'], 'execution policy mismatch')
    actual = row.get('actual_request_wire')
    check(isinstance(actual, str) and actual == template['wire'], 'actual wire bytes mismatch')
    if isinstance(actual, str):
        check(row.get('actual_request_wire_bytes') == len(actual.encode()), 'actual byte length mismatch')
        check(row.get('actual_request_wire_sha256') == reconciliation.digest(actual), 'actual wire hash mismatch')
    recon = reconciliation.reconcile(row, dict(template, body=template['body_template']), prompt)
    native = client3.accounting(row)
    for field, value in (('initial_prompt_reconciliation', recon), ('native_accounting', native)):
        # The frozen producer emits reported_counts as tuples; persisted JSON
        # necessarily reads them as arrays. Compare their exact JSON value,
        # preserving count values and list order rather than Python container type.
        check(client3.canonical(raw.get(field)) == client3.canonical(value),
              field + ' retained attestation mismatch')
        row[field] = value
    check(recon['qualified'], 'initial CPU/server count evidence missing or mismatched')
    check(native['complete'], 'native returned ID accounting incomplete')
    row['attempts'] = client3.assemble_attempts(row.get('sse_frames') or [])
    check(row['attempts'] == raw.get('attempts'), 'attempt assembly mismatch')
    binding = client3.authenticate_calls(row, decoder)
    check(binding == raw.get('native_binding'), 'native binding attestation mismatch')
    check(binding.get('qualified') is True, 'native authentication unqualified')
    row['native_binding'] = binding
    for field, value in (('input_accounting', client3.input_accounting(row, binding)),
                         ('call_accounting', client3.call_accounting(row, decoder))):
        check(raw.get(field) == value, field + ' retained attestation mismatch')
        row[field] = value
    check(safety is not None, 'missing safety join')
    safety = safety or {}
    check(safety.get('binding') == binding and safety.get('qualified') is True, 'safety binding unqualified')
    calls = ((row.get('tensorfold') or {}).get('compute') or {}).get('calls', [])
    gaps.extend(audit_records.reference_audit(calls, safety, reference_matches, worker_sha))
    if row['arm'] in ('plain', 'declared_off'):
        check(not calls, 'P/D native execution present')
    errors = client3.generation_errors(row)
    check(raw.get('generation_errors') == errors, 'generation error evidence mismatch')
    if row.get('transport_status') != 'ok' or errors:
        check(row['status'] != 'ok', 'generation failure labelled successful')
    if row['arm'] == 'declared_off' and (row.get('external_tool_calls') or row['attempts']['malformed']):
        check(row['status'] != 'ok' and row.get('answer_eligible') is False, 'D policy termination drift')
    verified = 0
    if diagnostic:
        tf = row.get('tensorfold') or {}
        check(tf.get('enable_thinking') is True and tf.get('reasoning_effort') == 'high', 'diagnostic settings mismatch')
        check(row['status'] in ('ok', 'policy_terminated') and row.get('transport_status') == 'ok',
              'diagnostic transport/generation failure')
        reference = str(template['reference'])
        check(reference in ('323', '55'), 'unexpected diagnostic reference')
        successful = [c for c in calls if c.get('inserted') is True and c.get('status') == 'ok']
        check(not successful or any(c.get('output') == reference for c in successful), 'diagnostic output mismatch')
        if not gaps:
            verified = sum(c.get('output') == reference for c in successful)
    return row, {'identity': list(key(row)), 'diagnostic': diagnostic, 'gaps': gaps,
                 'qualified': not gaps, 'initial_prompt_reconciliation': recon,
                 'native_accounting': native, 'input_accounting': row['input_accounting'],
                 'call_accounting': row['call_accounting'], 'verified_diagnostic_calls': verified}


def activation(rows):
    result = []
    fields = ('attempted', 'blocked_dispatches', 'successful_executions', 'failed_executions',
              'delivered_authenticated_results', 'reply_used_result')
    for suite in sorted({r['suite'] for r in rows}):
        for arm in analysis3.ARMS:
            for seed in analysis3.SEEDS:
                selected = [r for r in rows if (r['suite'], r['arm'], r['seed']) == (suite, arm, seed)]
                counts = {}
                for field in fields:
                    known = [r['call_accounting'][field] for r in selected
                             if r['call_accounting']['coverage'].get(field) is True]
                    counts[field] = {'known_sum': sum(known) if known else None,
                                     'known_rows': len(known), 'unknown_rows': len(selected)-len(known)}
                executed = counts['successful_executions']; failed = counts['failed_executions']
                full = all(c['unknown_rows'] == 0 for c in counts.values())
                label = ('DECLARED BUT UNUSED' if arm == 'lane' and selected and full
                         and counts['attempted']['known_sum'] == executed['known_sum'] == failed['known_sum'] == 0 else
                         'ACTIVATION COVERAGE UNKNOWN' if not full else 'OBSERVED')
                result.append({'suite': suite, 'arm': arm, 'seed': seed, 'scheduled': len(selected),
                               'counts': counts, 'label': label})
    return result


def prepare(data, out, environment_audit=None):
    inputs = audit_records.Inputs()
    receipt, frozen, manifest, prompts = sources(inputs)
    turn_reports = turns(inputs, data, frozen)
    decoder = client3.NativeDecoder(ROOT.parent / 'phase2/references/tokenizer.json', TOKENIZER_SHA)
    require(audit_records.digest(inputs.read(ROOT.parent / 'phase2/references/tokenizer.json')) == TOKENIZER_SHA,
            'tokenizer drift')
    worker_sha = sha(ROOT.parent / 'phase2/jail_worker.py')
    tasks = {(t['suite'], t['id']): t for t in analysis3.manifest_tasks(manifest)}
    raw_rows, checked_rows, diagnostics, raw_safety, audits, boot_reports = [], [], [], [], [], []
    raw_diagnostics = []
    pairs = {}
    for entry in frozen['boots']:
        path = ROOT / 'frozen-a1-v1' / entry['file']
        require(audit_records.digest(inputs.read(path)) == entry['sha256'], 'boot template drift')
        boot = inputs.json(path)
        require(boot['boot'] == entry['boot'] and boot['arm'] == entry['arm'], 'boot identity drift')
        directory = data / (PREFIX + '-' + boot['boot'].split('-')[0]) / boot['boot']
        safe_rows = inputs.rows(directory / 'safety.jsonl')
        safeties = index(safe_rows)
        observed_keys, boot_audits = set(), []
        for filename, templates, diagnostic in (('responses.jsonl', boot['requests'], False),
                                                 ('diagnostics.jsonl', boot['diagnostics'], True)):
            observed = inputs.rows(directory / filename)
            expected = [dict(t, arm=boot['arm'], boot=boot['boot']) for t in templates]
            require(list(map(key, observed)) == list(map(key, expected)), 'exact frozen request order/grid required')
            require(len(templates) == 2 if diagnostic else bool(templates), 'empty/invalid boot')
            for raw, template in zip(observed, templates):
                observed_keys.add(key(raw))
                require(analysis3.digest(template['task']) == template['task_sha256'], 'frozen task hash drift')
                if not diagnostic:
                    require(template['task'] == tasks[(raw['suite'], raw['id'])], 'analysis/request task drift')
                require(template['wire'] == reconciliation.wire(template['body_template'])
                        and reconciliation.digest(template['wire']) == template['wire_sha256'] == template['body_sha256'],
                        'frozen body/wire hash drift')
                prompt = prompts[tuple(template['key'])]
                require(prompt['declared_off_ids'] == prompt['declared_lane_ids'] == prompt['original_builtin_lane_ids'],
                        'CPU declared/builtin prompt inequality')
                row, evidence = audit_row(raw, safeties.get(key(raw)), template, prompt, decoder, worker_sha, diagnostic)
                pairs.setdefault((boot['boot'].split('-')[0] if diagnostic else 'dataset', tuple(template['key'])), {})[boot['arm']] = (template, row)
                audits.append(evidence); boot_audits.append(evidence)
                if diagnostic:
                    raw_diagnostics.append(raw)
                    diagnostics.append(row)
                else:
                    raw_rows.append(raw); checked_rows.append(row)
        require(set(safeties) == observed_keys, 'exact safety identity join required')
        raw_safety.extend(safe_rows)
        http = audit_records.http_audit(inputs.read(directory / 'rank0.log').decode('utf-8'), len(boot['requests'])+2)
        verified = sum(r['verified_diagnostic_calls'] for r in boot_audits)
        boot_reports.append({'boot': boot['boot'], 'arm': boot['arm'], 'http': http,
                             'verified_diagnostic_calls': verified,
                             'qualified': http['qualified'] and all(r['qualified'] for r in boot_audits)
                             and (verified > 0 if boot['arm'] == 'lane' else verified == 0)})
    pair_audit = []
    for pair, arms in sorted(pairs.items()):
        require(set(arms) == set(analysis3.ARMS), 'missing declared pair arm')
        d, l = arms['declared_off'], arms['lane']
        reconciliation.compare_declared_templates(dict(d[0], body=d[0]['body_template']),
                                                  dict(l[0], body=l[0]['body_template']), prompts[pair[1]])
        dr, lr = d[1]['initial_prompt_reconciliation'], l[1]['initial_prompt_reconciliation']
        okay = (d[1].get('actual_request_wire') == l[1].get('actual_request_wire') == d[0]['wire']
                and dr['qualified'] and lr['qualified']
                and dr['server_initial_prompt_tokens'] == lr['server_initial_prompt_tokens'])
        pair_audit.append({'scope': pair[0], 'key': list(pair[1]), 'qualified': bool(okay)})
    require(len(pair_audit) == 190 and len(raw_rows) == 552 and len(diagnostics) == 18, 'full552/18/190 grid required')
    grade3.strict.validate_responses(checked_rows, manifest)
    env = inputs.json(environment_audit) if environment_audit else {'qualified': False, 'reason': 'parent environment audit not supplied'}
    if environment_audit:
        require(env.get('rank_snapshots') == 18 and len(env.get('proofs', [])) == 18,
                'parent environment audit must cover18 rank snapshots')
        require({(p['boot'], p['rank']) for p in env['proofs']} ==
                {(b['boot'], r) for b in frozen['boots'] for r in (0, 1)}, 'environment boot/rank coverage drift')
        require(all(p.get('qualified') is True and p.get('image_id') == IMAGE_IDS[p['rank']]
                    for p in env['proofs']), 'environment proof qualification/image mismatch')
    inputs.stable()
    out.mkdir(exist_ok=False, mode=0o700)
    jsonl(out / 'raw-records.jsonl', raw_rows)
    jsonl(out / 'raw-diagnostics.jsonl', raw_diagnostics)
    jsonl(out / 'raw-safety.jsonl', raw_safety)
    audit = {'status': 'NEEDS-T5', 'spec_sha256': BASE_SHA, 'amendment_sha256': A1_SHA,
             'qualified': all(b['qualified'] for b in boot_reports) and all(p['qualified'] for p in pair_audit)
             and env.get('qualified') is True,
             'dataset_rows': 552, 'diagnostic_rows': 18, 'turns': turn_reports, 'boots': boot_reports,
             'rows': audits, 'declared_pairs': pair_audit, 'environment': env,
             'activation': activation(checked_rows), 'diagnostic_activation': activation(diagnostics),
             'freeze_sha256': receipt['freeze_sha256'], 'receipt_sha256': sha(ROOT / 'cpu-final-a1-v2/verification.json'),
             'analysis_manifest_sha256': analysis3.digest(manifest), 'input_file_sha256': inputs.hashes,
             'wrapper_sha256': sha(__file__), 'postcollection_wrapper': True,
             'limitations': ['Wrapper is postcollection I/O/reporting; frozen outcome mechanism unchanged.',
                             'No directly verified served-ID equality or execution-only causal isolation.',
                             'Missing input accounting is explicit; no inferred multi-round ledger.']}
    save(out / 'audit.json', audit)
    # No diagnostic or D tool attempt is executed here. Only the sealed grade3
    # HumanEval route can run candidate source, inside its actual pinned Jail.
    graded = grade3.grade_all(checked_rows, manifest, IMAGE_IDS[1], expected_hashes=receipt['grader_hashes'])
    jsonl(out / 'all-provisional.jsonl', graded)
    save(out / 'analysis-manifest.json', manifest)
    sealed = math3.seal(analysis3.manifest_tasks(manifest), graded, out / 'math-public.json',
                        out / 'math-private.json', preseal_path=ROOT / 'MATH-PRESEAL-A1-v1.json',
                        preseal_sha256=receipt['math_preseal_canonical_sha256'])
    inputs.stable()
    save(out / 'seal-receipt.json', sealed)
    save(out / 'prepare-receipt.json', {'status': 'NEEDS-T5', 'audit_sha256': sha(out / 'audit.json'),
         'provisional_sha256': sha(out / 'all-provisional.jsonl'), 'manifest_sha256': sha(out / 'analysis-manifest.json'),
         'seal_receipt_sha256': sha(out / 'seal-receipt.json'), 'wrapper_sha256': sha(__file__)})


def finalize(prepared, reviewed, review_audit, out):
    inputs = audit_records.Inputs()
    _, _, frozen_manifest, _ = sources(inputs)
    receipt = inputs.json(prepared / 'prepare-receipt.json')
    for name, field in (('audit.json', 'audit_sha256'), ('all-provisional.jsonl', 'provisional_sha256'),
                         ('analysis-manifest.json', 'manifest_sha256'), ('seal-receipt.json', 'seal_receipt_sha256')):
        require(audit_records.digest(inputs.read(prepared / name)) == receipt[field], 'prepared artifact drift')
    require(receipt['wrapper_sha256'] == sha(__file__), 'prepare/finalize wrapper drift')
    rows = inputs.rows(prepared / 'all-provisional.jsonl')
    manifest = inputs.json(prepared / 'analysis-manifest.json')
    require(manifest == frozen_manifest, 'prepared analysis manifest differs from freeze')
    original = {analysis3.identity(r): r for r in rows}
    replacements = inputs.rows(reviewed)
    require(len(replacements) == 120 and len({analysis3.identity(r) for r in replacements}) == 120,
            'all120 unique reviewed MATH rows required')
    evidence = inputs.json(review_audit)
    sealed = inputs.json(prepared / 'seal-receipt.json')
    require(evidence['packet_sha256'] == sealed['packet_sha256'] and
            evidence['private_key_sha256'] == sealed['private_key_sha256'], 'review seal mismatch')
    require(len(evidence['reviewer_ids']) >= 2 and len(set(evidence['reviewer_ids'])) == len(evidence['reviewer_ids']),
            'independent review identities required')
    require(len(evidence['labels']) == 120 and all(
        len(v['labels']) >= 2 and all(x.get('reviewer_type') in ('agent', 'human') for x in v['labels'])
        for v in evidence['labels']), 'two independent labelled reviews required')
    # Use the unchanged seal/merge functions to authenticate the full handoff;
    # do not invent another manual-label decision rule in this wrapper.
    private_path = prepared / 'math-private.json'
    require(private_path.stat().st_mode & 0o777 == 0o600, 'private key mode must be0600')
    packet = inputs.json(prepared / 'math-public.json')
    private = inputs.json(private_path)
    reviewer_ids = evidence['reviewer_ids']
    require(len(reviewer_ids) in (2, 3), 'two reviewers plus optional third required')
    def votes(reviewer):
        return [vote for item in evidence['labels'] for vote in item['labels']
                if vote['reviewer_id'] == reviewer]
    rebuilt, rebuilt_audit = math3.merge(
        packet, private, votes(reviewer_ids[0]), votes(reviewer_ids[1]),
        reviewer_ids[0], reviewer_ids[1], key_sha256=sealed['private_key_sha256'],
        third=votes(reviewer_ids[2]) if len(reviewer_ids) == 3 else None,
        third_id=reviewer_ids[2] if len(reviewer_ids) == 3 else None)
    require(sorted(replacements, key=analysis3.identity) == rebuilt, 'reviewed JSONL differs from sealed merge')
    # Review input list order can differ from candidate-sorted audit order;
    # original review hashes remain retained, without manufacturing new hashes.
    for field in ('labels', 'bounds', 'disputed_candidates', 'third_reviewed'):
        require(evidence[field] == rebuilt_audit[field], 'review audit differs from sealed merge')
    allowed = {'grade', 'math_review'}
    expected_math = {analysis3.identity(r) for r in rows if r['grading']['reason'] == 'all_MATH_requires_independent_blinded_review'}
    require({analysis3.identity(r) for r in replacements} == expected_math, 'reviewed cohort rekeyed')
    for row in replacements:
        old = original[analysis3.identity(row)]
        require({k: v for k, v in row.items() if k not in allowed} ==
                {k: v for k, v in old.items() if k not in allowed}, 'review altered non-review fields')
        review = row['math_review']
        require(review['sealed'] is True and review['packet_sha256'] == sealed['packet_sha256']
                and review['key_sha256'] == sealed['private_key_sha256'], 'unsealed MATH review')
        label = next((v for v in evidence['labels'] if v['candidate_id'] == review['candidate_id']), None)
        require(label is not None and row['grade'] == label['effective_grade']
                and review['label'] == label['decision'], 'review row/audit decision drift')
        original[analysis3.identity(row)] = row
    combined = [original[k] for k in sorted(original)]
    analysis3.validate_cohort(combined, manifest)
    result = analysis3.analyze(combined, manifest)
    inputs.stable()
    out.mkdir(exist_ok=False, mode=0o700)
    jsonl(out / 'all-final.jsonl', combined)
    save(out / 'analysis.json', result)
    save(out / 'audit.json', inputs.json(prepared / 'audit.json'))
    save(out / 'math-review-audit.json', evidence)
    save(out / 'finalize-receipt.json', {'status': 'NEEDS-T5', 'input_file_sha256': inputs.hashes,
         'analysis_sha256': sha(out / 'analysis.json'), 'audit_sha256': sha(out / 'audit.json'),
         'final_rows_sha256': sha(out / 'all-final.jsonl'), 'review_audit_sha256': sha(out / 'math-review-audit.json'),
         'wrapper_sha256': sha(__file__), 'business_logic_unchanged': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--data-root', required=True, type=Path, help='copied runs directory, read only')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--environment-audit', type=Path, help='parent18-snapshot audit; absent remains unqualified')
    p = sub.add_parser('finalize')
    p.add_argument('--prepared', required=True, type=Path)
    p.add_argument('--reviewed', required=True, type=Path, help='math3.merge output JSONL')
    p.add_argument('--review-audit', required=True, type=Path, help='math3.merge private audit JSON')
    p.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.data_root, args.out, args.environment_audit)
    else:
        finalize(args.prepared, args.reviewed, args.review_audit, args.out)
    # Deliberately no record, token, mapping, or private-key output.
    print(json.dumps({'status': 'NEEDS-T5', 'phase': args.command}))


if __name__ == '__main__':
    main()
