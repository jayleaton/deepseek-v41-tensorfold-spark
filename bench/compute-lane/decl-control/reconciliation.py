#!/usr/bin/env python3
"""Per-request CPU-prompt/server-count reconciliation, without network access.

This supports the proposed amendment; it does not amend or qualify the spec.
The future collector must persist this result before deciding whether to continue.
"""
import hashlib
import json


def wire(body):
    # Native template JSON serialization preserves schema insertion order.
    return json.dumps(body, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def reconcile(record, template, cpu_prompt):
    """Retain missing evidence as null; do not infer served-array equality.

    `actual_request_wire` must be the UTF-8 request bytes captured at send time,
    not a later reserialization of the response's request object.
    """
    arm = template['arm']
    field = {'plain': 'plain_ids', 'declared_off': 'declared_off_ids',
             'lane': 'declared_lane_ids'}[arm]
    ids = cpu_prompt[field]
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError('invalid CPU prompt ID array')
    if list(template['key']) != list(cpu_prompt['key']):
        raise ValueError('CPU prompt belongs to another task/seed')
    frozen_wire = template['wire']
    if digest(frozen_wire) != template['wire_sha256'] or frozen_wire != wire(template['body']):
        raise ValueError('frozen wire/body drift')
    actual_wire = record.get('actual_request_wire')
    wire_ok = isinstance(actual_wire, str) and actual_wire == frozen_wire
    values = []
    missing = []
    usage = record.get('usage')
    if not isinstance(usage, dict) or 'prompt_tokens' not in usage:
        missing.append('final usage.prompt_tokens')
    else:
        values.append(('final', usage['prompt_tokens']))
    for index, frame in enumerate(record.get('sse_frames') or []):
        if not isinstance(frame, dict):
            missing.append('malformed SSE frame')
            continue
        item = frame.get('usage')
        if isinstance(item, dict) and 'prompt_tokens' in item:
            values.append((f'sse:{index}', item['prompt_tokens']))
    valid = bool(values) and all(type(n) is int and n > 0 for _, n in values)
    consistent = valid and len({n for _, n in values}) == 1
    count = values[0][1] if consistent else None
    count_ok = count == len(ids) if count is not None else False
    complete = bool(wire_ok and count_ok and consistent and not missing)
    return {'qualified_under_proposed_count_contract': complete,
            'qualified': complete, 'initial_prompt_tokens': count,
            'cpu_initial_prompt_tokens': len(ids), 'server_initial_prompt_tokens': count,
            'count_matches': count_ok, 'wire_matches_frozen': wire_ok,
            'actual_wire_sha256': digest(actual_wire) if isinstance(actual_wire, str) else None,
            'frozen_wire_sha256': template['wire_sha256'], 'reported_counts': values,
            'missing': missing, 'served_array_equality_proven': False,
            'all_round_prompt_tokens': None,
            'all_round_reason': 'requires separately qualified multi-round ledger or single-round evidence'}


def compare_declared_templates(d, l, cpu_prompt):
    if d['arm'] != 'declared_off' or l['arm'] != 'lane' or d['key'] != l['key']:
        raise ValueError('not a matched declared-arm pair')
    if d['wire'] != l['wire'] or d['wire_sha256'] != l['wire_sha256']:
        raise ValueError('declared-arm wire mismatch')
    if cpu_prompt['declared_off_ids'] != cpu_prompt['declared_lane_ids']:
        raise ValueError('declared-arm CPU array mismatch')
    if d['execution_policy'] != {'TF_DSV41_COMPUTE_LANE': '0'} or l['execution_policy'] != {'TF_DSV41_COMPUTE_LANE': '1'}:
        raise ValueError('execution-policy mismatch')
    return True
