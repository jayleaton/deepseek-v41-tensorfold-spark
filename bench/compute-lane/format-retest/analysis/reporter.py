#!/usr/bin/env python3
"""Frozen, offline format-retest reporter. No model execution or output repair.

See HANDOFF.md for the raw-row and independent parent-audit contracts.
Only the CLI writes artifacts; analyze() is a pure reporting API.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import uuid

SPEC_SHA256 = 'b47ede697eed3a38d244ad5ff7447690732bd12945ff4592554d45bc2d8527ec'
GRADER_SHA256 = '1a2249a1b5e0b06f03a4300995d0c320e5a21630d261b3791501f7c83740fb9f'
FAMILIES = tuple(sorted(('integer', 'modular', 'division', 'dp', 'graph', 'aggregation')))
P_FAMILIES = ('aggregation', 'dp', 'graph')
ARMS = ('L-old', 'L-new', 'P-old', 'P-new')
SEEDS = (101, 102)
DRAWS = 20000
PASS_RULE = ('PASS only if all are true: collection/provenance complete; L-new has≤2/120 format failures; '
             'L-new strict solves≥L-old strict solves; zero P paired losses '
             '(P-old correct/P-new incorrect) across its60 pairs; safety passes.')
CALL_FIELDS = ('attempted', 'blocked_dispatches', 'successful_executions',
               'failed_executions', 'delivered_authenticated_results', 'reply_used_result')
METRICS = ('returned_ids', 'initial_prompt_ids', 'initial_prompt_plus_returned_SUBTOTAL',
           'full_wall_seconds', 'full_input_ids', 'full_input_plus_returned_ids')
LIMITS = ('Development cohort: six generated templates, 10 tasks per family and two seeds. '
          'No old compute-DL outcomes are pooled. Prompt length is part of the intervention; '
          'cold-boot/prompt-cache state does not establish KV-state equality. P arithmetic '
          'was omitted: no new P-arithmetic efficacy claim. The observed ≤2 target does not '
          'prove population failure ≤1.67%; zero observed P losses does not prove zero '
          'population risk. A PASS supports clearer formatting on this limited cohort, '
          'not a proven turn-boundary cause, benchmark superiority, or serving-default change.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_grader(path):
    """Load the pinned legacy module under an isolated name, never sys.modules."""
    if digest(path) != GRADER_SHA256:
        raise ValueError('strict grader SHA256 mismatch')
    spec = importlib.util.spec_from_file_location('_format_retest_grade_' + uuid.uuid4().hex, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def count(value):
    return type(value) is int and value >= 0


def sha256_value(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def native_receipt_verified(receipt, call):
    """Validate parent A1 proof metadata, not execute code or emulate a codec.

    Full-reference verification does not prove invisible native tail bytes.
    Only the landed presentation can be verified for a clipped result.
    """
    landed = call['output'].encode()
    clipped = receipt.get('clipped')
    if not (receipt.get('qualified') is True and receipt.get('provenance')
            and receipt.get('presentation_verified') is True
            and receipt.get('full_reference_verified') is True
            and receipt.get('policy_transform_verified') is True
            and receipt.get('frame_binding_verified') is True
            and receipt.get('source_binding_verified') is True
            and receipt.get('native_prefix_exact') is True
            and receipt.get('matches_bytes') is True and receipt.get('in_text') is True
            and isinstance(call.get('status'), str)
            and receipt.get('native_status') == call['status']
            and receipt.get('reference_status') == call['status']
            and sha256_value(receipt.get('native_frame_sha256'))
            and receipt.get('source_sha256') == hashlib.sha256(call['code'].encode()).hexdigest()
            and receipt.get('landed_output_sha256') == hashlib.sha256(landed).hexdigest()
            and receipt.get('expected_presentation_sha256') == receipt['landed_output_sha256']
            and sha256_value(receipt.get('full_reference_sha256'))
            and receipt.get('landed_output_bytes') == len(landed)
            and count(receipt.get('full_reference_bytes'))
            and count(receipt.get('full_reference_body_tokens'))
            and count(receipt.get('retained_prefix_bytes'))
            and count(receipt.get('retained_prefix_tokens'))):
        return False
    prefix_bytes = receipt['retained_prefix_bytes']
    prefix_tokens = receipt['retained_prefix_tokens']
    if clipped is True:
        marker = ('\n' if prefix_bytes else '') + '[output truncated at 512 tokens]'
        return (receipt.get('complete_value') is False
                and receipt['full_reference_body_tokens'] > 512
                and prefix_tokens <= 512
                and prefix_bytes <= receipt['full_reference_bytes']
                and receipt.get('marker_bytes') == marker
                and landed.endswith(marker.encode())
                and len(landed) == prefix_bytes + len(marker.encode()))
    if clipped is False:
        return (receipt.get('complete_value') is True
                and receipt['full_reference_body_tokens'] <= 512
                and prefix_tokens == receipt['full_reference_body_tokens']
                and prefix_bytes == receipt['full_reference_bytes'] == len(landed)
                and receipt.get('marker_bytes') == ''
                and receipt['full_reference_sha256'] == receipt['landed_output_sha256'])
    return False


def key_json(key):
    return {'id': key[0], 'seed': key[1], 'study_arm': key[2]}


def canonical_key(row):
    task_id = row.get('taskid', row.get('id'))
    if 'taskid' in row and 'id' in row and row['taskid'] != row['id']:
        raise ValueError('conflicting taskid/id')
    return task_id, row.get('seed'), row.get('study_arm')


def expected_grid(tasks):
    return sorted((t['id'], seed, arm) for t in tasks for seed in SEEDS for arm in ARMS
                  if arm.startswith('L') or t['family'] in P_FAMILIES)


def validate_tasks(tasks, answers, grader):
    if len(tasks) != 60 or len({t['id'] for t in tasks}) != 60:
        raise ValueError('exactly 60 distinct public tasks required')
    if set(answers) != {t['id'] for t in tasks}:
        raise ValueError('private answer coverage mismatch')
    for family in FAMILIES:
        expected = {f'compute-format-v1-{family}-{slot:03d}' for slot in range(1, 11)}
        if {t['id'] for t in tasks if t['family'] == family} != expected:
            raise ValueError('frozen family/slot grid mismatch')
    for task in tasks:
        if task.get('suite') != 'compute_dl' or not grader.valid_value(answers[task['id']], task['family']):
            raise ValueError('invalid public task suite or private typed answer')
        if any(k in task for k in ('answer', 'reference', 'reference_a', 'reference_b')):
            raise ValueError('private answers in public tasks')


def classify(row, task, answer, grader):
    """Strict verdict and secondary syntax classification remain independent."""
    graded = grader.grade(row, task, answer)
    value, source, error = grader.extract(row.get('content'))
    typed = error is None and grader.valid_value(value, task['family'])
    eligible = (row.get('status') == 'ok' and row.get('answer_eligible') is not False
                and not row.get('external_tool_calls')
                and row.get('finish_reason') not in
                ('tool_calls', 'function_call', 'error', 'failed', 'cancelled'))
    selected = ('absent' if error == 'missing_final_content' else
                'invalid_json' if error else 'wrong_shape' if not typed else
                'typed_wrong_value' if value != answer else 'typed_exact_value')
    return {
        'strict_correct': graded['grade'] == 'correct',
        'strict_reason': graded['grader']['reason'],
        'format_failure': not eligible or not typed,
        'typed_wrong_value': typed and value != answer,
        'eligible': eligible,
        'selected_span_classification': selected,
        'selected_span_source': source,
        'selected_span_error': error,
        # Never publish private references or selected answer values.
    }


def normalize(rows, tasks, answers, grader):
    by_id = {t['id']: t for t in tasks}
    grid = set(expected_grid(tasks))
    indexed = {}
    physical = {'lane': 'L', 'L': 'L', 'plain': 'P', 'P': 'P'}
    for raw in rows:
        key = canonical_key(raw)
        if (type(key[1]) is not int or key not in grid or key in indexed
                or raw.get('diagnostic') or raw.get('included_in_cohort') is False):
            raise ValueError('unexpected, duplicate, diagnostic or excluded raw observation')
        task = by_id[key[0]]
        if raw.get('family') != task['family']:
            raise ValueError('row family mismatch')
        # study_arm is the intervention; arm is physical collection configuration.
        if 'arm' in raw and physical.get(raw['arm']) != key[2][0]:
            raise ValueError('physical arm disagrees with study_arm')
        if not isinstance(raw.get('status'), str) or not raw['status']:
            raise ValueError('raw observation requires explicit status')
        row = dict(raw, id=key[0], study_arm=key[2])
        row['analysis'] = classify(row, task, answers[key[0]], grader)
        indexed[key] = row
    return indexed


def metrics(row):
    if row is None:
        return dict.fromkeys(METRICS)
    native = row.get('native_accounting') or {}
    initial = row.get('initial_prompt_reconciliation') or {}
    full = row.get('input_accounting') or {}
    returned = native.get('returned_tokens')
    if native.get('complete') is not True or not count(returned):
        returned = None
    prompt = initial.get('initial_prompt_tokens', initial.get('server_initial_prompt_tokens'))
    if not (initial.get('qualified') is True or initial.get('qualified_under_proposed_count_contract') is True) or not count(prompt):
        prompt = None
    processed = full.get('processed_input_tokens')
    if not (full.get('qualified') is True and full.get('complete') is True
            and full.get('provenance') and count(processed)):
        processed = None
    wall = row.get('elapsed_seconds')
    if not numeric(wall):
        wall = None
    return dict(zip(METRICS, (returned, prompt,
        prompt + returned if prompt is not None and returned is not None else None,
        wall, processed,
        processed + returned if processed is not None and returned is not None else None)))


def coverage(values, keys):
    known = [v for v in values if v is not None]
    return {'scheduled': len(keys), 'known': len(known),
            'known_sum': sum(known) if known else None,
            'sum': sum(known) if len(known) == len(keys) else None,
            'unknown_keys': [key_json(k) for k, v in zip(keys, values) if v is None]}


def calls(row, field):
    evidence = (row or {}).get('call_accounting') or {}
    value = evidence.get(field)
    if ((evidence.get('complete') is True or (evidence.get('coverage') or {}).get(field) is True)
            and count(value)):
        return value
    return None


def summarize(keys, indexed):
    observed = [indexed[k] for k in keys if k in indexed]
    return {
        'scheduled': len(keys), 'observed': len(observed),
        'missing': len(keys) - len(observed),
        'strict_solves': sum(r['analysis']['strict_correct'] for r in observed),
        'format_failures': len(keys) - len(observed) + sum(r['analysis']['format_failure'] for r in observed),
        'typed_wrong_values': sum(r['analysis']['typed_wrong_value'] for r in observed),
        'caps': sum(r.get('finish_reason') in ('length', 'max_tokens', 'max_output_tokens') for r in observed),
        'deadlines': sum(r.get('status') in ('timeout', 'deadline') or r.get('finish_reason') in ('timeout', 'deadline') for r in observed),
        'costs': {m: coverage([metrics(indexed.get(k))[m] for k in keys], keys) for m in METRICS},
        'calls': {f: coverage([calls(indexed.get(k), f) for k in keys], keys) for f in CALL_FIELDS},
    }


def paired(indexed, tasks, left, right, field):
    gains, losses, missing = [], [], []
    for task in sorted(tasks, key=lambda t: t['id']):
        for seed in SEEDS:
            a, b = (task['id'], seed, left), (task['id'], seed, right)
            identity = {'id': task['id'], 'family': task['family'], 'seed': seed}
            if a not in indexed or b not in indexed:
                missing.append(identity)
                continue
            old, new = indexed[a]['analysis'][field], indexed[b]['analysis'][field]
            # Gains always mean improvement, including format failure reductions.
            gain = old and not new if field == 'format_failure' else new and not old
            loss = new and not old if field == 'format_failure' else old and not new
            if gain:
                gains.append(identity)
            if loss:
                losses.append(identity)
    return {'gains': gains, 'losses': losses, 'missing_pairs': missing}


def qualification(audit, indexed, tasks):
    """Only a separately supplied parent audit can qualify safety/provenance.

    This validates the receipt contract; it does not independently perform the
    parent's r7 replay, source audit, or native clipping verification.
    """
    reasons = []
    integrity = audit.get('integrity') or {}
    if not (audit.get('kind') == 'qualified-real-collection'
            and audit.get('spec_sha256') == SPEC_SHA256
            and audit.get('grader_sha256') == GRADER_SHA256
            and integrity.get('qualified') is True and integrity.get('provenance')):
        reasons.append('qualified independent real-collection integrity provenance absent')
    for gate in ('sealed_inputs', 'disjointness', 'prompt_neutrality', 'configuration',
                 'dispatch_boundary', 'source_image_rank_environment', 'post_run_bindings'):
        evidence = integrity.get(gate) or {}
        if evidence.get('qualified') is not True or not evidence.get('provenance'):
            reasons.append('integrity gate: ' + gate)
    if audit.get('scheduled_keys') != [key_json(k) for k in expected_grid(tasks)]:
        reasons.append('audit scheduled grid mismatch')
    r7 = audit.get('r7') or {}
    if not (r7.get('qualified') is True and r7.get('provenance')
            and r7.get('scope') == 'owned_runner_cells'
            and r7.get('all_optional_runners_off') is True
            and r7.get('oracle_pinned') is True and sha256_value(r7.get('oracle_sha256'))
            and all(type(r7.get(f)) is int and r7[f] == 0 for f in
                    ('owned_cell_count', 'checked_owned_cell_count', 'wrong_values',
                     'wrong_bytes', 'not_in_text', 'cuts', 'unknowns'))):
        reasons.append('r7 pinned owned-runner-cell audit/zero-owned coverage absent or failed')
    replay = audit.get('native_replay') or {}
    if not (replay.get('qualified') is True and replay.get('provenance')
            and replay.get('independently_isolated_cpython') is True
            and replay.get('presentation_token_limit') == 512
            and replay.get('clipping_policy_qualified') is True
            and sha256_value(replay.get('policy_source_sha256'))
            and sha256_value(replay.get('tokenizer_sha256'))
            and all(type(replay.get(f)) is int and replay[f] == 0 for f in
                    ('wrong_values', 'wrong_bytes', 'not_in_text', 'unresolved_cuts', 'unknowns'))):
        reasons.append('independent clipping-aware native replay absent or failed')
    expected = {}
    for key, row in indexed.items():
        if not key[2].startswith('L'):
            continue
        compute = ((row.get('tensorfold') or {}).get('compute') or {})
        delivered = [c for c in compute.get('calls', []) if c.get('inserted') is True]
        if calls(row, 'delivered_authenticated_results') != len(delivered):
            reasons.append('delivered result census missing/mismatched: ' + str(key))
        for call in delivered:
            call_key = (*key, call.get('id'))
            if (not isinstance(call.get('id'), str) or not call['id'] or call_key in expected
                    or not isinstance(call.get('code'), str) or not isinstance(call.get('output'), str)):
                reasons.append('invalid delivered native record: ' + str(key))
                continue
            expected[call_key] = call
    receipts = {}
    for receipt in replay.get('receipts', []):
        k = (*canonical_key(receipt), receipt.get('call_id'))
        if k in receipts:
            reasons.append('duplicate replay receipt: ' + str(k))
        receipts[k] = receipt
    if set(expected) != set(receipts):
        reasons.append('native replay receipt coverage mismatch')
    native_results = []
    for k, call in expected.items():
        receipt = receipts.get(k, {})
        verified = native_receipt_verified(receipt, call)
        if not verified:
            reasons.append('unqualified delivered replay: ' + str(k))
        native_results.append(dict(key_json(k[:3]), call_id=k[3],
            presentation_verified=verified,
            complete_value=verified and receipt.get('complete_value') is True,
            clipped=receipt.get('clipped'),
            full_reference_verified=receipt.get('full_reference_verified') is True,
            classification=('presentation-verified clipped' if receipt.get('clipped') is True else
                            'verified complete') if verified else 'unresolved',
            proof=receipt))
    policy_clips = sum(x['presentation_verified'] and x['clipped'] is True for x in native_results)
    if not count(replay.get('policy_clipped_results')) or replay['policy_clipped_results'] != policy_clips:
        reasons.append('policy-clipped result census missing/mismatched')
    return {'qualified': not reasons, 'blocking_reasons': reasons,
            'delivered_results': len(expected), 'replay_receipts': len(receipts),
            'presentation_verified_policy_clips': policy_clips,
            'verified_complete_values': sum(x['complete_value'] for x in native_results),
            'unresolved_native_results': sum(not x['presentation_verified'] for x in native_results),
            'reported_unresolved_cuts': replay.get('unresolved_cuts'),
            'native_results': native_results,
            'r7_scope': 'Owned runner cells only; all optional runners OFF implies zero owned cells. '
                        'r7 does not verify native compute code.',
            'r7_owned_cells': r7.get('owned_cell_count'),
            'r7_checked_owned_cells': r7.get('checked_owned_cell_count'),
            'basis': 'Separate parent audit; native compute uses independent isolated Jail/CPython '
                     'plus native frames and pinned clipping proof, never raw safety self-report. '
                     'A verified clipped presentation is not a complete value or proof of invisible native tail bytes.'}


def acceptance(complete, qualified, arms, p_losses):
    if not qualified:
        return 'BLOCKED'
    if not complete:
        return 'INCOMPLETE'
    if p_losses or arms['L-new']['strict_solves'] < arms['L-old']['strict_solves']:
        return 'FAIL'
    failures = arms['L-new']['format_failures']
    return 'PASS' if failures <= 2 else 'INCONCLUSIVE' if failures <= 6 else 'FAIL'


def interval(values):
    if len(values) != DRAWS or any(v is None for v in values):
        return None
    values = sorted(values)
    return [values[500], values[19499]]


def bootstrap(indexed, tasks):
    """Shared family/task draws; keep both seeds/wordings; bounded O(draws) RAM."""
    groups = {f: sorted(t['id'] for t in tasks if t['family'] == f) for f in FAMILIES}
    contrasts = [('primary_L_wording', FAMILIES, 'L-old', 'L-new', 'format_failure'),
                 ('secondary_P_wording', P_FAMILIES, 'P-old', 'P-new', 'strict_correct'),
                 ('secondary_Lnew_minus_Pnew', P_FAMILIES, 'P-new', 'L-new', 'strict_correct')]
    contrasts += [('primary_family_' + f, (f,), 'L-old', 'L-new', 'format_failure')
                  for f in FAMILIES]
    # Preaggregate each task's two seeds once, never copy raw replies in a draw.
    totals = {}
    for task in tasks:
        for arm in ARMS:
            rows = [indexed.get((task['id'], s, arm)) for s in SEEDS]
            if any(r is None for r in rows):
                continue
            totals[task['id'], arm] = {
                field: sum(r['analysis'][field] for r in rows)
                for field in ('format_failure', 'strict_correct')}
            for m in METRICS:
                vals = [metrics(r)[m] for r in rows]
                totals[task['id'], arm][m] = None if any(v is None for v in vals) else sum(vals)
    records = {}
    for name, families, left, right, field in contrasts:
        ids = [i for f in families for i in groups[f]]
        complete = all((i, a) in totals for i in ids for a in (left, right))
        def cost_ratio(sample, metric):
            vals = [[totals[i, a][metric] for i in sample] for a in (left, right)]
            if any(v is None for side in vals for v in side) or sum(vals[0]) == 0:
                return None
            return sum(vals[1]) / sum(vals[0])
        records[name] = {
            'families': list(families), 'left': left, 'right': right,
            'rate_field': field, 'complete': complete,
            'difference_right_minus_left': (sum(totals[i, right][field] - totals[i, left][field] for i in ids) / (2 * len(ids))) if complete else None,
            'ci95': None,
            'cost_ratios_right_over_left': {m: {'ratio_of_sums': cost_ratio(ids, m) if complete else None, 'ci95': None} for m in METRICS},
        }
    samples = {name: [] for name, *_ in contrasts}
    costs = {name: {m: [] for m in METRICS} for name, *_ in contrasts}
    cost_complete = {
        (name, m): records[name]['complete'] and all(
            totals[i, a][m] is not None
            for f in families for i in groups[f] for a in (left, right))
        for name, families, left, right, _ in contrasts for m in METRICS}
    rng = random.Random(20261107)
    for _ in range(DRAWS):
        draw = {f: [groups[f][rng.randrange(10)] for _ in range(10)] for f in FAMILIES}
        for name, families, left, right, field in contrasts:
            if not records[name]['complete']:
                continue
            ids = [i for f in families for i in draw[f]]
            samples[name].append(sum(totals[i, right][field] - totals[i, left][field] for i in ids) / (2 * len(ids)))
            for m in METRICS:
                if not cost_complete[name, m]:
                    continue
                vals = [[totals[i, a][m] for i in ids] for a in (left, right)]
                ratio = None if any(v is None for side in vals for v in side) or sum(vals[0]) == 0 else sum(vals[1]) / sum(vals[0])
                costs[name][m].append(ratio)
    for name in records:
        records[name]['ci95'] = interval(samples[name])
        for m in METRICS:
            records[name]['cost_ratios_right_over_left'][m]['ci95'] = interval(costs[name][m])
    records['primary_family_ci95'] = {
        f: records['primary_family_' + f]['ci95'] for f in FAMILIES}
    records['method'] = {'draws': DRAWS, 'seed': 20261107, 'family_order': list(FAMILIES),
                         'task_order': groups, 'tasks_per_family_per_draw': 10,
                         'percentile_indices': [500, 19499], 'unit': 'task cluster with both seeds and wordings',
                         'costs': 'paired ratios of sums, shared draws; any unknown input withholds that interval'}
    return records


def observation_ledger(tasks, indexed):
    by_id = {t['id']: t for t in tasks}
    ledger = []
    for key in expected_grid(tasks):
        row = indexed.get(key)
        ledger.append(dict(key_json(key), family=by_id[key[0]]['family'],
                           observed=row is not None,
                           **(row['analysis'] if row else {'strict_correct': False, 'format_failure': True,
                               'strict_reason': 'scheduled_missing', 'typed_wrong_value': False}),
                           **metrics(row)))
    return ledger


def analyze(rows, tasks, answers, audit, grader):
    validate_tasks(tasks, answers, grader)
    indexed = normalize(rows, tasks, answers, grader)
    grid = expected_grid(tasks)
    arms = {a: summarize([k for k in grid if k[2] == a], indexed) for a in ARMS}
    families = {f: {a: summarize([k for k in grid if k[2] == a and k[0].startswith('compute-format-v1-' + f + '-')], indexed)
                   for a in ARMS if a.startswith('L') or f in P_FAMILIES} for f in FAMILIES}
    p_tasks = [t for t in tasks if t['family'] in P_FAMILIES]
    pairs = {'L_strict': paired(indexed, tasks, 'L-old', 'L-new', 'strict_correct'),
             'L_format': paired(indexed, tasks, 'L-old', 'L-new', 'format_failure'),
             'P_strict': paired(indexed, p_tasks, 'P-old', 'P-new', 'strict_correct'),
             'secondary_Lnew_Pnew_strict': paired(indexed, p_tasks, 'P-new', 'L-new', 'strict_correct')}
    qualified = qualification(audit, indexed, tasks)
    missing = [key_json(k) for k in grid if k not in indexed]
    complete = not missing
    status = acceptance(complete, qualified['qualified'], arms, pairs['P_strict']['losses'])
    return {'spec_sha256': SPEC_SHA256, 'grader_sha256': GRADER_SHA256,
            'status': status, 'collection_status': 'COMPLETE' if complete else 'INCOMPLETE',
            'scheduled': 360, 'observed': len(indexed), 'missing_keys': missing,
            'pass_rule': PASS_RULE, 'qualification': qualified,
            'arms': arms, 'families': families, 'pairs': pairs,
            'bootstrap': bootstrap(indexed, tasks), 'ledger': observation_ledger(tasks, indexed),
            'interpretation': LIMITS, 'parent_audit': audit}


def markdown(report):
    lines = ['# Format retest results', '', f"Status: **{report['status']}**; collection: **{report['collection_status']}**.",
             f"Observed {report['observed']}/360; missing {len(report['missing_keys'])}.", '',
             f'Spec SHA256: `{SPEC_SHA256}`', f'Grader SHA256: `{GRADER_SHA256}`', '',
             'NEEDS-T5: real collection and independent parent safety/provenance qualification are required.', '',
             'Frozen PASS rule: ' + PASS_RULE, '',
             '| Arm/family | Observed/scheduled | Strict solves | Format failures | Typed wrong | Caps | Deadlines |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for label, summaries in [('all', report['arms']), *report['families'].items()]:
        for arm, s in summaries.items():
            lines.append(f"| {arm}/{label} | {s['observed']}/{s['scheduled']} | {s['strict_solves']} | {s['format_failures']} | {s['typed_wrong_values']} | {s['caps']} | {s['deadlines']} |")
    lines += ['', 'Missing replies count against the format gate. Typed wrong values include typed spans in ineligible replies; strict verdicts remain unchanged.', '',
              '## Paired changes', '']
    for name, p in report['pairs'].items():
        lines += [f"{name}: {len(p['gains'])} gains; {len(p['losses'])} losses; {len(p['missing_pairs'])} incomplete pairs.",
                  'Gains: ' + json.dumps(p['gains'], sort_keys=True),
                  'Losses: ' + json.dumps(p['losses'], sort_keys=True), '']
    lines += ['## Contrasts and cost accounting', '']
    for name, contrast in report['bootstrap'].items():
        if isinstance(contrast, dict) and 'difference_right_minus_left' in contrast:
            lines.append(f"{name}: right-minus-left rate {contrast['difference_right_minus_left']}; CI95 {contrast['ci95']}. Cost ratios: " + json.dumps(contrast['cost_ratios_right_over_left'], sort_keys=True))
    lines += ['', 'Primary family-stratum CI95: ' + json.dumps(report['bootstrap']['primary_family_ci95'], sort_keys=True), '',
              'Costs and calls below show known/scheduled coverage. Unknown totals are withheld; initial+returned is a SUBTOTAL, never full multi-round input cost.', '']
    for label, summaries in [('all', report['arms']), *report['families'].items()]:
        for arm, s in summaries.items():
            for kind in ('costs', 'calls'):
                for name, c in s[kind].items():
                    lines.append(f"- {arm}/{label} {name}: sum={c['sum']}; known sum={c['known_sum']}; coverage={c['known']}/{c['scheduled']}.")
    lines += ['', '## Qualification', '', json.dumps(report['qualification'], indent=2), '',
              '## Scheduled missing keys', '', json.dumps(report['missing_keys'], indent=2), '',
              '## Interpretation', '', LIMITS, '', 'No post-hoc repair, unwrapping, answer-driven fallback, or grader changes.', '']
    return '\n'.join(lines)


def write_report(report, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (out / 'RESULTS.md').write_text(markdown(report))
    fields = list(dict.fromkeys(k for row in report['ledger'] for k in row))
    with (out / 'per-observation.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(report['ledger'])


def read_json(path):
    def unique(items):
        value = {}
        for k, v in items:
            if k in value:
                raise ValueError('duplicate JSON key: ' + k)
            value[k] = v
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('rows', 'tasks', 'answers', 'audit', 'grader', 'spec', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    if digest(args.spec) != SPEC_SHA256:
        raise ValueError('format spec SHA256 mismatch')
    grader = load_grader(args.grader)
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    audit = read_json(args.audit)
    # Bind independently qualified audit to the exact files read by this run.
    bindings = audit.get('input_sha256') or {}
    for name in ('rows', 'tasks', 'answers'):
        if bindings.get(name) != digest(getattr(args, name)):
            raise ValueError('parent audit input hash mismatch: ' + name)
    report = analyze(rows, tasks, read_json(args.answers), audit, grader)
    report['input_sha256'] = bindings
    write_report(report, args.out)


if __name__ == '__main__':
    main()
