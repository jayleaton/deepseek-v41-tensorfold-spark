#!/usr/bin/env python3
"""Offline compute-DL artifact audit. No transport, execution, or task generation.

API: collect(campaign, data_root, dataset, out, evidence_root=None) -> report dict.
CLI: --campaign FILE --data-root DIR --dataset DIR --out NEW_DIR.
Paths in the campaign are used directly when present; retained Spark paths may
also resolve under this kit's stdbench root. Every consumed byte is hashed.
An absent observation is scheduled missingness, never a zero-cost observation.
"""
import argparse
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'phase2'))
sys.path.insert(0, str(ROOT.parent / 'decl-control'))
import audit_records
import client3
import reconciliation

def own_module(name):
    spec = importlib.util.spec_from_file_location('compute_dl_report_' + name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = own_module('analysis')
grade = own_module('grade')
SPEC_SHA256 = 'c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141'

ARMS = {'D': 'declared_off', 'L': 'lane'}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def resolve(path):
    path = Path(path)
    if path.exists():
        return path
    parts = path.parts
    if 'stdbench' in parts:
        return ROOT.parent.joinpath(*parts[parts.index('stdbench') + 1:])
    return path


def key(row):
    return audit_records.identity(row)


def indexed(rows):
    return audit_records.indexed(rows)


def reference_matches(call, result):
    try:
        return (call.get('status') == result.get('status') and
                base64.b64decode(result['normalized_body_base64'], validate=True)
                == call['output'].encode())
    except (KeyError, TypeError, ValueError):
        return False


def reference_audit(calls,safety,worker_sha):
    # Legacy A1 labelled every execution error as an integrity gap. Here an
    # authenticated error with exact independent replay agreement is an outcome.
    # All ID/source/image/worker/status/output mismatch checks remain mandatory.
    return [gap for gap in audit_records.reference_audit(calls,safety,reference_matches,worker_sha)
            if gap!='delivered execution error retained as gap']


def audit_row(raw, safety, template, prompt, decoder, worker_sha, diagnostic=False):
    """Recompute attestations from retained frames and join exact replay receipts."""
    row = copy.deepcopy(raw)
    gaps = []
    def check(ok, reason):
        if not ok:
            gaps.append(reason)
    parsed = client3.old.parse_stream(['data: ' + client3.canonical(f)
                                      for f in row.get('sse_frames') or []])
    for field in ('usage', 'tensorfold', 'content', 'reasoning', 'finish_reason',
                  'external_tool_calls', 'live_compute_reports'):
        check(raw.get(field) == parsed[field], 'SSE mismatch: ' + field)
        row[field] = parsed[field]
    check(raw.get('diagnostic') is diagnostic, 'diagnostic flag mismatch')
    check(raw.get('task_sha256') == template['task_sha256'], 'task hash mismatch')
    check(raw.get('request') == template['body_template'], 'request object mismatch')
    check(raw.get('execution_policy') == template['execution_policy'], 'execution policy mismatch')
    sent = raw.get('actual_request_wire')
    check(sent == template['wire'], 'actual sent wire mismatch/missing')
    if isinstance(sent, str):
        check(raw.get('actual_request_wire_sha256') == digest(sent.encode()), 'sent wire SHA mismatch')
        check(raw.get('actual_request_wire_bytes') == len(sent.encode()), 'sent byte count mismatch')
    recon = reconciliation.reconcile(row, dict(template, body=template['body_template']), prompt)
    values = {'initial_prompt_reconciliation': recon,
              'native_accounting': client3.accounting(row, 8192),
              'attempts': client3.assemble_attempts(row.get('sse_frames') or [])}
    row.update(values)
    binding = client3.authenticate_calls(row, decoder)
    row['native_binding'] = binding
    values['native_binding'] = binding
    values['input_accounting'] = client3.input_accounting(row, binding)
    values['call_accounting'] = client3.call_accounting(row, decoder)
    for field, value in values.items():
        check(client3.canonical(raw.get(field)) == client3.canonical(value), 'attestation mismatch: ' + field)
        row[field] = value
    check(recon['qualified'], 'initial count reconciliation unavailable/failed')
    check(row['native_accounting']['complete'], 'returned-ID accounting incomplete')
    check(binding.get('qualified') is True, 'native provenance unqualified')
    # A missing sidecar must not become an empty successful receipt.
    safety = copy.deepcopy(safety) if safety is not None else {}
    check(safety.get('binding') == binding and safety.get('qualified') is True, 'missing/unqualified safety binding')
    calls = ((row.get('tensorfold') or {}).get('compute') or {}).get('calls') or []
    gaps.extend(reference_audit(calls,safety,worker_sha))
    if template['arm'] == 'declared_off':
        check(not calls and binding.get('native_frame_count', 0) == 0, 'D EXECUTION: native call/frame receipt present')
        if row.get('external_tool_calls') or row['attempts']['malformed']:
            check(row.get('status') != 'ok' and row.get('answer_eligible') is False, 'D policy termination missing')
    errors = client3.generation_errors(row)
    if errors or row.get('transport_status') != 'ok':
        check(row.get('status') != 'ok', 'generation/transport failure labelled ok')
    row['safety'] = safety
    tf = row.get('tensorfold') or {}
    row['metadata_provenance'] = {'elapsed_seconds': 'retained client end-to-end observation'}
    for target, source in (('server_seconds', 'server_seconds'), ('tool_execution_seconds', 'tool_execution_seconds')):
        # Only explicit retained server metadata; no wall subtraction or timing guess.
        value = tf.get(source)
        row[target] = value if analysis.number(value) else None
        row['metadata_provenance'][target] = 'tensorfold.' + source if row[target] is not None else 'unknown'
    if gaps:
        # Keep every actual outcome, but do not let unverified cost attestations
        # or receipts qualify statistical endpoints or activation exposure.
        row['safety']['qualified'] = False
        if not recon['qualified'] or 'attestation mismatch: initial_prompt_reconciliation' in gaps:
            row['initial_prompt_reconciliation']['qualified'] = False
            row['initial_prompt_reconciliation']['qualified_under_proposed_count_contract'] = False
            row['input_accounting']['qualified'] = False
        if any(g.startswith('SSE mismatch:') or g == 'attestation mismatch: native_accounting' for g in gaps):
            row['native_accounting']['complete'] = False
        if 'attestation mismatch: input_accounting' in gaps:
            row['input_accounting']['qualified'] = False
    verified = 0
    if diagnostic:
        check(str(template.get('reference')) in ('323', '55'), 'diagnostic reference drift')
        check(row.get('status') in ('ok', 'policy_terminated') and row.get('transport_status') == 'ok', 'diagnostic failed')
        tf = row.get('tensorfold') or {}
        check(tf.get('enable_thinking') is True and tf.get('reasoning_effort') == 'high', 'diagnostic settings mismatch')
        successful = [c for c in calls if c.get('inserted') is True and c.get('status') == 'ok']
        check(not successful or any(c.get('output') == str(template['reference']) for c in successful), 'wrong diagnostic output')
        if not gaps:
            verified = sum(c.get('output') == str(template['reference']) for c in successful)
    if gaps:
        row['safety']['qualified'] = False
    return row, {'identity': list(key(raw)), 'diagnostic': diagnostic,
                 'qualified': not gaps, 'gaps': gaps, 'verified_diagnostic_calls': verified,
                 'initial_prompt_reconciliation': recon}


def render_results(report):
    result = report['analysis']; overall = result['overall']
    lines = ['# Compute-DL results', '', 'Status: **' + result['status'] + '**',
             'Spec SHA256: `' + SPEC_SHA256 + '`', '',
             'Integrity qualified: ' + str(report['integrity']['qualified']) + '.',
             'Observed: %s / 480; missing: %s.' % (result['observed'], len(result['missing_identities'])),
             'Activation: %s / 120 distinct tasks — %s.' % (result['activation']['distinct_tasks'], result['activation']['status']), '',
             'CPU-matched/count-checked prompts; served-array equality was not directly observed.',
             'Initial-prompt-plus-returned is an accounting subtotal. Unknown multi-round input costs are not zero.',
             'Published declaration-control/older study scores are historical context only; no matched synthetic published score is constructed.',
             'This is a fixed equal-weight mixture of six templates, not 120 independent task families.', '',
             '## Accuracy', '', '| Mixture | D solved / scheduled | L solved / scheduled | L−D | 95% CI | bounds |',
             '|---|---:|---:|---:|---|---|']
    def j(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    for name, item in [('overall', overall), *result['families'].items()]:
        d, l = item['arms']['D'], item['arms']['L']
        lines.append('| %s | %s/%s | %s/%s | %s | %s | %s |' % (name, d['solved'], d['scheduled'], l['solved'], l['scheduled'], j(item['solve_difference_L_minus_D']), j(item['solve_difference_ci95']), j(item['difference_bounds'])))
    lines += ['', 'Overall primary exact task sign-flip: ' + j(overall['sign_flip']),
              'Qualification: ' + j(result['qualification']), '', '## Costs and censoring', '']
    for metric, cost in overall['costs'].items():
        lines += ['### ' + metric, '', '```json', json.dumps(cost, indent=2, allow_nan=False), '```', '']
    for family, item in result['families'].items():
        lines += ['### Family: ' + family, '', '```json',
                  json.dumps({'arms': item['arms'], 'costs': item['costs'],
                              'uncensored_sensitivity': item['complete_both_uncensored_sensitivity']}, indent=2, allow_nan=False), '```', '']
    lines += ['## Paired gains and losses', '', 'Gains: ' + j(overall['gains']),
              'Losses: ' + j(overall['losses']),
              'Net observed paired gains minus losses: ' + str(len(overall['gains']) - len(overall['losses'])), '',
              '## Rates, exposure and sensitivities', '', '```json',
              json.dumps({'arms': overall['arms'], 'activation': result['activation'],
                          'uncensored_sensitivity': overall['complete_both_uncensored_sensitivity'],
                          'budget': report['budget'], 'integrity_gaps': report['integrity']['gaps']}, indent=2, allow_nan=False), '```', '',
              result['interpretation'], '',
              report.get('budget_evidence_limitation', ''), '',
              'INCOMPLETE cohorts support bounds, not a completed-study efficacy claim. Missing replies have unknown costs.',
              'Family, cost and subgroup results are descriptive. No serving-default promotion or total-token-saving claim.', '']
    return '\n'.join(lines)


def collect(campaign, data_root, dataset, out, evidence_root=None):
    inputs = audit_records.Inputs()
    cfg = inputs.json(campaign)
    data_root, dataset, out = Path(data_root), Path(dataset), Path(out)
    gaps = []
    def check(ok, reason):
        if not ok:
            gaps.append(reason)
    def pinned(path, expected):
        path = resolve(path)
        require(digest(inputs.read(path)) == expected, 'sealed file drift: ' + str(path))
        return path
    require(cfg.get('spec_sha256') == SPEC_SHA256, 'unqualified spec identity')
    pinned(cfg['spec'], SPEC_SHA256)
    seal = {str(resolve(p).resolve()): h for p, h in cfg['code_sha256'].items()}
    require(len(seal) >= 25, 'incomplete precollection source/data seal')
    for p, h in seal.items():
        pinned(p, h)
    for name in ('collect_report.py', 'report_check.py', 'analysis.py', 'grade.py', 'protocol.py', 'reference_a.py', 'reference_b.py'):
        require(str((ROOT / name).resolve()) in seal, 'parent must seal report/analysis/grader before first request: ' + name)
    dispatch_root = ROOT.parent / 'decl-control/dispatch-high'
    dispatch_path = pinned(dispatch_root / 'dispatch-proof.json',
        '77e43ebc13afdc16a8304836869953f8ceadfc8c6775b8f551aceb3b15af0436')
    dispatch = inputs.json(dispatch_path)
    require(dispatch.get('verdict') == 'PASS' and dispatch.get('observed_fixtures') == 8
            and dispatch.get('whole_stream_noncall_and_wrong_arm_controls') is True
            and dispatch.get('guard_mutant_compiled') is True
            and dispatch.get('guard_mutant_behaviorally_caught') is True
            and dispatch.get('deployed_binary_instrumentation') is False, 'reused dispatch proof failed')
    for relative, expected in dispatch['sha256'].items():
        pinned(audit_records.inside(dispatch_root.resolve(), relative), expected)
    production = inputs.json(dispatch_root / 'unmodified-source-sha256.json')
    require(len(production) == dispatch.get('production_sources_unchanged') == 3436, 'production source inventory differs')
    for relative, expected in production.items():
        pinned(audit_records.inside(dispatch_root.resolve(), relative), expected)
    proof = inputs.json(pinned(cfg['cpu_receipt'], cfg['cpu_receipt_sha256']))
    for p,h in proof.get('preparation_file_sha256',{}).items():
        pinned(p,h)
    # Validate the exact gate set without live GO reads or rank-local path policy.
    # The protocol's predicate is the authority for the frozen gate dictionary.
    import ast
    tree = ast.parse(inputs.read(ROOT / 'protocol.py'))
    gates = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'required' for t in n.targets)]
    require(len(gates) == 1 and proof.get('gates') == ast.literal_eval(gates[0]), 'CPU gate set differs')
    require(proof.get('passed') is True and proof.get('spec_sha256') == SPEC_SHA256
            and proof.get('code_sha256') == cfg['code_sha256'], 'CPU qualification/source seal differs')
    freeze = resolve(cfg['freeze'])
    manifest = inputs.json(pinned(freeze / 'manifest.json', cfg['freeze_sha256']))
    require(proof.get('freeze_sha256') == cfg['freeze_sha256'], 'CPU freeze binding differs')
    require(manifest['spec_sha256'] == SPEC_SHA256 and manifest['prefix'] == cfg['prefix'], 'freeze identity differs')
    require(manifest['turns'] == cfg['layout']['turns'] and manifest['boots'] == cfg['layout']['boots'], 'layout drift')
    require(manifest['turns'] == [{'name': 't01', 'boots': ['t01-D', 't01-L']}, {'name': 't02', 'boots': ['t02-L', 't02-D']}], 'frozen turn order differs')
    require([(b['boot'], b['arm']) for b in manifest['boots']] == [('t01-D','D'),('t01-L','L'),('t02-L','L'),('t02-D','D')], 'boot grid differs')
    prompt_path = pinned(cfg['cpu_prompt_arrays'], cfg['cpu_prompt_arrays_sha256'])
    require(proof.get('cpu_prompt_arrays_sha256') == cfg['cpu_prompt_arrays_sha256'], 'CPU prompt seal mismatch')
    prompts = inputs.rows(prompt_path)
    prompt_map = {tuple(p['key']): p for p in prompts}
    require(len(prompts) == len(prompt_map) == 242, '242 unique CPU prompt keys required')
    for p in prompts:
        ids = p['declared_off_ids']
        require(ids == p['declared_lane_ids'] == p['original_builtin_lane_ids'] and ids
                and all(type(n) is int and n >= 0 for n in ids), 'CPU prompt equality/array failed')
    def sealed_dataset(name):
        p = (dataset / name).resolve()
        require(str(p) in seal, 'dataset/reference receipt not precollection sealed: ' + name)
        return p
    tasks_path = sealed_dataset('tasks-public.jsonl')
    tasks = inputs.rows(tasks_path)
    analysis.validate_cohort([], tasks)
    require(digest(inputs.read(tasks_path)) == manifest['tasks_public_sha256'], 'public dataset freeze drift')
    answers = []
    for side in ('a', 'b'):
        p = sealed_dataset('answers-' + side + '.json')
        receipt = inputs.json(sealed_dataset(p.name + '.receipt.json'))
        source = ROOT / ('reference_' + side + '.py')
        require(receipt['status'] == 'PASS' and receipt['count'] == 120
                and receipt['input_sha256'] == digest(inputs.read(tasks_path))
                and receipt['output_sha256'] == digest(inputs.read(p))
                and receipt['source_sha256'] == digest(inputs.read(source)), 'reference receipt mismatch')
        answers.append(inputs.json(p))
    require(set(answers[0]) == set(answers[1]) == {t['id'] for t in tasks}, 'reference coverage differs')
    for t in tasks:
        require(grade.valid_value(answers[0][t['id']], t['family'])
                and grade.valid_value(answers[1][t['id']], t['family'])
                and answers[0][t['id']] == answers[1][t['id']], 'typed reference disagreement')
    verification = inputs.json(sealed_dataset('reference-verification.json'))
    require(verification.get('passed') is True and verification.get('tasks') == 120
            and verification.get('independent_processes') is True and verification.get('spec_sha256') == SPEC_SHA256
            and verification.get('public_sha256') == digest(inputs.read(tasks_path))
            and verification.get('reference_a_sha256') == digest(inputs.read(dataset / 'answers-a.json'))
            and verification.get('reference_b_sha256') == digest(inputs.read(dataset / 'answers-b.json')), 'independent-reference verification seal differs')
    require(all(inputs.json(dataset / ('answers-' + side + '.json.receipt.json')).get('python_version')
                == verification.get('python_version') for side in ('a', 'b')), 'reference Python pin differs')
    tokenizer = pinned(cfg['tokenizer'], cfg['tokenizer_sha256'])
    decoder = client3.NativeDecoder(tokenizer, cfg['tokenizer_sha256'])
    # Imported pure APIs must be members of the same precollection seal.
    import native_frames
    import generate_envs3
    for module in (client3, client3.old, reconciliation, native_frames, audit_records, generate_envs3):
        require(str(Path(module.__file__).resolve()) in seal, 'unsealed audit dependency: ' + str(module.__file__))
    require(str((ROOT.parent / 'phase2/jail_worker.py').resolve()) in seal, 'unsealed replay worker')
    worker_sha = digest(inputs.read(ROOT.parent / 'phase2/jail_worker.py'))
    def optional_rows(p):
        if not p.exists():
            gaps.append('missing artifact: ' + str(p))
            return []
        return inputs.rows(p)
    rows, diagnostics, audits, pairs, boots = [], [], [], {}, []
    by_id = {t['id']: t for t in tasks}
    for entry in manifest['boots']:
        boot = inputs.json(pinned(audit_records.inside(freeze.resolve(), entry['file']), entry['sha256']))
        name, arm = entry['boot'], entry['arm']
        require((boot['boot'], boot['arm']) == (name, arm), 'boot identity drift')
        directory = data_root / (cfg['prefix'] + '-' + name.split('-')[0]) / name
        raw_transport = optional_rows(directory / 'raw-observations.jsonl')
        safeties = indexed(optional_rows(directory / 'safety.jsonl'))
        seen = set(); boot_audits = []; retained_by_key = {}
        require(len(boot['requests']) == 120 and len(boot['diagnostics']) == 2, 'boot request count differs')
        require([(t['key'], str(t['reference'])) for t in boot['diagnostics']] == [
            (['diagnostic', 'python-17x19', 101], '323'), (['diagnostic', 'python-sum11', 101], '55')], 'fixed diagnostic order/reference differs')
        for filename, templates, diagnostic in [('responses.jsonl', boot['requests'], False), ('diagnostics.jsonl', boot['diagnostics'], True)]:
            expected = [dict(t, arm=ARMS[arm], boot=name) for t in templates]
            actual = optional_rows(directory / filename)
            actual_map = indexed(actual)
            expected_map = indexed(expected)
            require(set(actual_map) <= set(expected_map), 'foreign or extra retained observation')
            check(list(actual_map) == list(expected_map)[:len(actual_map)], 'retained request prefix/order differs: ' + name)
            check(list(actual_map) == list(expected_map), 'missing replies: ' + name + '/' + filename)
            for template in expected:
                k = key(template)
                require(template['arm'] == ARMS[arm] and template['execution_policy'] == {'TF_DSV41_COMPUTE_LANE': str(int(arm == 'L'))}, 'arm policy drift')
                require(reconciliation.wire(template['body_template']) == template['wire']
                        and digest(template['wire'].encode()) == template['wire_sha256'] == template['body_sha256'], 'template wire drift')
                require(audit_records.object_sha(template['task']) == template['task_sha256'], 'template task hash drift')
                if not diagnostic:
                    require(template['task'] == by_id[template['id']], 'public/request task mismatch')
                scope = name.split('-')[0] if diagnostic else 'dataset'
                pair = pairs.setdefault((scope, tuple(template['key'])), {})
                require(arm not in pair, 'duplicate planned pair')
                pair[arm] = {'template': template, 'row': None, 'audit': None}
                if k not in actual_map:
                    continue
                raw = actual_map[k]; seen.add(k); retained_by_key[(tuple(template['key']), diagnostic)] = raw
                row, audit = audit_row(raw, safeties.get(k), template, prompt_map[tuple(template['key'])], decoder, worker_sha, diagnostic)
                pair[arm].update(row=row, audit=audit)
                audits.append(audit); boot_audits.append(audit)
                (diagnostics if diagnostic else rows).append(row)
        check(set(safeties) == seen, 'safety identity coverage differs: ' + name)
        transport_keys = [(tuple(r['template_key']), r['diagnostic']) for r in raw_transport]
        check(len(transport_keys) == len(set(transport_keys)) == len(seen), 'raw transport/response count differs: ' + name)
        check(set(transport_keys) == {((k[0], k[1], k[2]), k[0] == 'diagnostic') for k in seen}, 'raw transport/response identity differs: ' + name)
        for transport in raw_transport:
            retained = retained_by_key.get((tuple(transport['template_key']), transport['diagnostic']))
            if retained is not None:
                check(all(transport['response'].get(f) == retained.get(f) for f in (
                    'sse_frames', 'sse_done', 'actual_request_wire', 'elapsed_seconds', 'transport_status',
                    'content', 'reasoning', 'usage', 'tensorfold')), 'raw transport bytes/metadata mismatch: ' + name)
        verified = sum(a['verified_diagnostic_calls'] for a in boot_audits)
        check(verified > 0 if arm == 'L' else verified == 0, 'diagnostic activation failed: ' + name)
        http = audit_records.http_audit(inputs.read(directory / 'rank0.log').decode(), len(seen)) if (directory / 'rank0.log').exists() else {'qualified': False, 'reasons': ['missing rank0.log']}
        check(http['qualified'], 'HTTP access coverage failed: ' + name)
        boots.append({'boot': name, 'http': http, 'verified_diagnostic_calls': verified})
        for rank in (0, 1):
            mp = pinned(cfg['rank_manifests'][name][str(rank)], cfg['rank_manifest_sha256'][name][str(rank)])
            rank_manifest = inputs.json(mp)
            snapshot = directory / ('snapshot-r%s.log' % rank)
            if not snapshot.exists():
                check(False, 'missing actual native snapshot: %s rank%s' % (name, rank)); continue
            records = [audit_records.strict_json(line) for line in inputs.read(snapshot).splitlines() if line.startswith(b'{')]
            records = [r for r in records if r.get('command') == 'snapshot']
            check(len(records) == 1, 'one actual snapshot required: ' + name)
            if len(records) != 1:
                continue
            pointer = records[0]
            proof_path = resolve(pointer['proof'])
            if not proof_path.exists() and evidence_root is not None:
                # Explicit retained-copy contract; ranks never share a proof namespace.
                proof_path = Path(evidence_root) / ('rank%s' % rank) / 'proof' / name / Path(pointer['proof']).name
            if not proof_path.exists():
                check(False, 'missing pointed-to native proof: ' + str(proof_path)); continue
            r = inputs.json(proof_path)
            check(all(r.get(k) == pointer.get(k) for k in ('command', 'rank', 'boot', 'dry_run')), 'snapshot pointer identity differs')
            native = r.get('native_process') or {}; env = native.get('env') or {}
            from generate_envs3 import reject_banned, IMAGE_IDS, ZERO_FLAGS
            require(cfg['image_ids'] == list(IMAGE_IDS), 'qualified image pins differ')
            expected_env = rank_manifest['phase2']['ranks'][rank]['effective_env']
            try:
                reject_banned(env)
            except ValueError:
                check(False, 'banned effective environment: ' + name)
            check(all(env.get('TF_DSV41_' + flag) == '0' for flag in ZERO_FLAGS), 'expected-off flag drift: ' + name)
            check(all(k in expected_env for k in env if any(word in k for word in ('RUNNER', 'TRACE', 'GUIDE', 'S3', 'STATED', 'COMMIT', 'SYNC_GRAPH'))), 'unsealed optional native flag: ' + name)
            check(r.get('dry_run') is False and r.get('present') is True and r.get('boot') == name
                  and r.get('rank') == rank and r.get('manifest_sha256') == digest(inputs.read(mp))
                  and r.get('image_id') == cfg['image_ids'][rank] and native.get('pid_namespace', 0) > 1,
                  'native snapshot identity/image failed: ' + name)
            check(all(env.get(k) == v for k, v in expected_env.items() if k.startswith(('TF_', 'GLM53_TF', 'NCCL')))
                  and env.get('TF_DSV41_COMPUTE_LANE') == str(int(arm == 'L')),
                  'effective environment drift: ' + name)
    require(len(pairs) == 244, '244 planned pair comparisons including per-turn diagnostics required')
    pair_audits = []
    for (scope, pk), sides in sorted(pairs.items()):
        require(set(sides) == {'D', 'L'}, 'planned pair missing arm')
        d, l = sides['D'], sides['L']
        reconciliation.compare_declared_templates(dict(d['template'], body=d['template']['body_template']), dict(l['template'], body=l['template']['body_template']), prompt_map[pk])
        okay = bool(d['row'] and l['row'] and d['audit']['qualified'] and l['audit']['qualified']
                    and d['row'].get('actual_request_wire') == l['row'].get('actual_request_wire')
                    and d['audit']['initial_prompt_reconciliation']['server_initial_prompt_tokens'] == l['audit']['initial_prompt_reconciliation']['server_initial_prompt_tokens'])
        pair_audits.append({'scope': scope, 'key': list(pk), 'qualified': okay})
    check(all(p['qualified'] for p in pair_audits), 'wire/count pair coverage incomplete/failed')
    check(all(a['qualified'] for a in audits), 'retained row provenance/safety/accounting failed')
    budget_path = resolve(cfg['budget_ledger'])
    budget = inputs.json(budget_path) if budget_path.exists() else None
    check(bool(budget) and budget.get('spec_sha256') == SPEC_SHA256 and budget.get('active') is None
          and budget.get('limit_seconds') == 18000 and budget.get('admission_seconds') == 15900, 'budget ledger missing/unrestored/invalid')
    if budget:
        intervals = budget.get('turns', [])
        check(analysis.number(budget.get('held_seconds')) and all(analysis.number(t.get('held_seconds')) for t in intervals)
              and abs(sum(t['held_seconds'] for t in intervals) - budget['held_seconds']) < .01, 'held-time reconciliation failed')
        budget['reported_overrun_seconds'] = max(0, budget['held_seconds'] - 18000) if analysis.number(budget.get('held_seconds')) else None
        check(len({t.get('turn') for t in intervals}) == len(intervals), 'duplicate held interval turn')
    for turn in manifest['turns']:
        td = data_root / (cfg['prefix'] + '-' + turn['name'])
        steps = optional_rows(td / 'steps.jsonl')
        verdict_path = td / 'verdict.json'
        verdict = inputs.json(verdict_path) if verdict_path.exists() else {}
        check(verdict.get('spec_sha256') == SPEC_SHA256 and verdict.get('verdict') == 'QUALIFIED'
              and verdict.get('error') is None, 'turn verdict missing/unqualified: ' + turn['name'])
        def event(step, verdict):
            return [e for e in steps if e.get('step') == step and e.get('verdict') == verdict]
        check(len(event('collection', 'COMPLETE')) == 1, 'turn incomplete: ' + turn['name'])
        check(len(event('G-restore', 'SERVING_VERIFIED')) == 1 and len(event('gpu-locks', 'RELEASED')) == 1, 'G restoration/release missing: ' + turn['name'])
        restored = event('G-restore', 'SERVING_VERIFIED')
        restore_log = td / 'G-restore.log'
        if len(restored) == 1 and restore_log.exists():
            check(restored[0].get('actual_step_sha256') == digest(inputs.read(restore_log)), 'restore log hash differs: ' + turn['name'])
        check(restore_log.exists() and b'POPPED' in inputs.read(restore_log), 'G POPPED receipt missing: ' + turn['name'])
    graded = grade.grade_all(rows, tasks, answers[0])
    observed = {analysis.identity(r): r for r in graded}
    planned = []
    for task in tasks:
        for seed in (101, 102):
            for arm in ('D', 'L'):
                r = observed.get((task['id'], seed, arm))
                planned.append({'id': task['id'], 'family': task['family'], 'seed': seed, 'arm': arm,
                                'classification': 'missing' if r is None else ('solved' if r['grade'] == 'correct' else 'incorrect'),
                                'status': r['status'] if r else 'missing', 'costs_unknown': r is None or any(v is None for v in analysis.metric_values(r))})
    check(len(rows) == 480 and len(diagnostics) == 8, '480 study / 8 diagnostic coverage incomplete')
    inputs.stable()
    integrity = {'qualified': not gaps, 'spec_sha256': SPEC_SHA256, 'gaps': gaps,
                 'provenance': {'input_file_sha256': dict(inputs.hashes), 'row_audits': len(audits),
                                'planned_pair_comparisons': 244, 'cpu_prompt_keys': 242}}
    result = analysis.analyze(graded, tasks, integrity=integrity)
    report = {'schema': 1, 'spec_sha256': SPEC_SHA256, 'integrity': integrity, 'budget': budget,
              'planned_observations': planned, 'row_audits': audits, 'pair_audits': pair_audits,
              'boots': boots, 'analysis': result, 'historical_context': 'Earlier published studies are historical only; no matched synthetic published score.',
              'prompt_limitation': 'CPU-matched/count-checked; served-array equality not directly observed.',
              'budget_evidence_limitation': 'Cumulative operator-held ledger and restored-G receipts audited. Individual request starts have no independent timestamp; cutoff enforcement relies on the sealed controller.',
              'historical_dispatch_proof': {'sha256': '77e43ebc13afdc16a8304836869953f8ceadfc8c6775b8f551aceb3b15af0436',
                                            'freshly_rerun': False, 'source_identity_rechecked': True}}
    inputs.stable()
    out.mkdir(exist_ok=False, mode=0o700)
    for name, value in [('report.json', report), ('analysis.json', result), ('audit.json', integrity)]:
        with (out / name).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    for name, values in [('graded.jsonl', graded), ('diagnostics.jsonl', diagnostics)]:
        with (out / name).open('x') as stream:
            for value in values:
                stream.write(client3.canonical(value) + '\n')
    with (out / 'RESULTS.md').open('x') as stream:
        stream.write(render_results(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('campaign', 'data-root', 'dataset', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--evidence-root', type=Path, help='optional copied rankN/proof/BOOT/*.json tree')
    args = parser.parse_args()
    report = collect(args.campaign, args.data_root, args.dataset, args.out, args.evidence_root)
    print(json.dumps({'status': report['analysis']['status'], 'integrity_qualified': report['integrity']['qualified'], 'out': str(args.out)}))
    if not report['integrity']['qualified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
