#!/usr/bin/env python3
"""Frozen compute-DL exact grader. Never executes model content.

grade(row, task, answer) returns a copied observation with grade/grader fields.
grade_all(rows, tasks, answer_map) grades observed rows only; analysis preserves
the unobserved scheduled grid. Answers are private typed values keyed by task ID.
"""
import copy
import json
import re

SPEC_SHA256 = 'c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141'
FAMILIES = ('integer', 'modular', 'division', 'dp', 'graph', 'aggregation')


def valid_value(value, family):
    if family in ('integer', 'modular', 'dp', 'graph'):
        return type(value) is int
    if family == 'division':
        return type(value) is list and len(value) == 2 and all(type(x) is int for x in value)
    if family == 'aggregation':
        return (type(value) is dict and set(value) == set('ABCDEF')
                and all(type(x) is int for x in value.values()))
    return False


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate_key')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('nonfinite_constant')


def _float(value):
    raise ValueError('json_float')


def extract(content):
    """Select syntactic JSON before type/answer validation; never rescue it.

    The first permissive parse is ONLY a selection predicate, including duplicate
    keys and nonstandard constants so these cannot trigger a later rescue.
    The selected span is then parsed with strict duplicate/constant rejection.
    """
    if not isinstance(content, str) or not content.strip():
        return None, None, 'missing_final_content'
    trimmed = content.strip()
    candidates = [('whole', trimmed)]
    # Consume other tagged blocks too, so their closing fences cannot become
    # spurious empty-tag openers that swallow a later valid JSON block.
    fences = [match for match in re.finditer(
        r'^```([^\r\n`]*)\r?\n(.*?)^```[ \t]*(?=\r?$)', trimmed, re.M | re.S)
        if match.group(1).strip() in ('', 'json')]
    if fences and not trimmed[fences[-1].end():].strip():
        candidates.append(('last_fence', fences[-1].group(2).strip()))
    line = trimmed.splitlines()[-1].strip()
    candidates.append(('last_line', re.sub(r'^Final answer:\s*', '', line, flags=re.I)))
    for source, span in candidates:
        try:
            json.loads(span)
        except (ValueError, RecursionError):
            continue
        try:
            value = json.loads(span, object_pairs_hook=_pairs, parse_constant=_constant,
                               parse_float=_float)
        except (ValueError, RecursionError) as error:
            return None, source, str(error)
        return value, source, None
    return None, None, 'no_complete_json'


def grade(row, task, answer):
    family = task.get('family')
    if (task.get('suite') != 'compute_dl' or family not in FAMILIES
            or not valid_value(answer, family)):
        raise ValueError('invalid public task or private typed reference')
    result = copy.deepcopy(row)
    reason, source, value = None, None, None
    if row.get('status') != 'ok' or row.get('answer_eligible') is False:
        reason = 'generation_' + str(row.get('status', 'missing'))
    elif row.get('external_tool_calls') or row.get('finish_reason') in ('tool_calls', 'function_call', 'error', 'failed', 'cancelled'):
        reason = 'tool_call_or_error_final'
    else:
        value, source, reason = extract(row.get('content'))
        if reason is None and not valid_value(value, family):
            reason = 'wrong_json_type_or_shape'
        if reason is None and value != answer:
            reason = 'wrong_answer'
    result.update(grade='correct' if reason is None else 'incorrect', family=family)
    result['grader'] = {'spec_sha256': SPEC_SHA256, 'extraction': source,
                        'reason': reason or 'exact_match', 'parsed_value': value,
                        'format_rejection': reason in ('no_complete_json', 'duplicate_key',
                            'nonfinite_constant', 'json_float', 'wrong_json_type_or_shape')}
    return result


def grade_all(rows, tasks, answer_map):
    tasks = tasks['tasks'] if isinstance(tasks, dict) else tasks
    by_id = {t['id']: t for t in tasks}
    if len(by_id) != len(tasks) or set(by_id) != set(answer_map):
        raise ValueError('duplicate tasks or private answer coverage mismatch')
    if any(t.get('suite') != 'compute_dl' or t.get('family') not in FAMILIES
           or not valid_value(answer_map[t['id']], t['family']) for t in tasks):
        raise ValueError('invalid public task or private typed reference')
    output, seen = [], set()
    for row in rows:
        key = (row.get('id'), row.get('seed'), row.get('arm'))
        if (key in seen or row.get('suite') != 'compute_dl' or key[0] not in by_id
                or type(key[1]) is not int or key[1] not in (101, 102)
                or key[2] not in ('D', 'L', 'declared_off', 'lane') or row.get('diagnostic')):
            raise ValueError('unexpected or duplicate observation')
        normalized = (key[0], key[1], {'declared_off': 'D', 'lane': 'L'}.get(key[2], key[2]))
        if normalized in seen:
            raise ValueError('duplicate observation arm alias')
        seen.add(normalized)
        output.append(grade(row, by_id[key[0]], answer_map[key[0]]))
    return output
