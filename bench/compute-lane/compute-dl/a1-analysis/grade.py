"""A1 strict API adapter plus sealed, explicitly secondary final-span sensitivity."""
import copy
from fractions import Fraction
import re

from pins import load_selector, load_strict, SELECTOR_SHA256, STRICT_SHA256, SPEC_SHA256

strict = load_strict()
select_final = load_selector()
SCALARS = ('integer', 'modular', 'dp', 'graph')
ALIASES = {'declared_off': 'D', 'lane': 'L', 'plain': 'P'}
NUM = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:\s*/\s*[+-]?\d+)?"
PATTERN = r"\s*[A-Za-z\s'’]*\$?\s*(" + NUM + r")[A-Za-z\s\-'’]*[.!]?\s*"
BLOCKED = r"\b(or|about|approximately|million|billion|thousand|hundred|percent)\b"


def identity(row):
    return row['id'], row['seed'], ALIASES.get(row['arm'], row['arm'])


def eligible(row):
    return (row.get('status') == 'ok' and row.get('answer_eligible') is not False
            and not row.get('external_tool_calls') and row.get('finish_reason') not in
            ('tool_calls', 'function_call', 'error', 'failed', 'cancelled'))


def grade(row, task, answer):
    # P is supported by the pure frozen grade() API without changing its bytes.
    return strict.grade(row, task, answer)


def grade_all(rows, tasks, answer_map):
    tasks = tasks['tasks'] if isinstance(tasks, dict) else tasks
    by_id = {t['id']: t for t in tasks}
    if len(by_id) != len(tasks) or set(by_id) != set(answer_map):
        raise ValueError('duplicate tasks or private answer coverage mismatch')
    for task in tasks:
        if task.get('suite') != 'compute_dl' or not strict.valid_value(answer_map[task['id']], task.get('family')):
            raise ValueError('invalid public task or private typed reference')
    output, seen = [], set()
    for row in rows:
        key = identity(row)
        if (key in seen or key[0] not in by_id or type(key[1]) is not int or key[1] not in (101, 102)
                or key[2] not in ('D', 'L', 'P') or row.get('suite') != 'compute_dl'
                or row.get('diagnostic') or row.get('included_in_cohort') is False):
            raise ValueError('unexpected or duplicate observation')
        seen.add(key)
        output.append(grade(row, by_id[key[0]], answer_map[key[0]]))
    return output


def sensitivity(row, task, answer):
    result = copy.deepcopy(row)
    # Always recompute strict first, ignoring caller-supplied grades.
    primary = grade(row, task, answer)
    reason, span, fraction = 'strict_preserved', None, None
    solved = primary['grade'] == 'correct'
    if not solved and task['family'] in SCALARS and eligible(row):
        span = select_final(row.get('content'))
        reason = 'ambiguous_or_missing_final'
        if isinstance(span, str) and not re.search(BLOCKED, span, re.I):
            match = re.fullmatch(PATTERN, span)
            if match:
                try:
                    fraction = Fraction(match.group(1).replace(',', '').replace(' ', ''))
                except (ValueError, ZeroDivisionError):
                    reason = 'fraction_parse_failure'
                else:
                    solved = fraction == answer
                    reason = 'exact_final_span_match' if solved else 'wrong_answer'
    elif not solved:
        reason = 'non_scalar_strict_preserved' if task['family'] not in SCALARS else 'ineligible_final'
    result.update(primary)
    result['grade'] = 'correct' if solved else 'incorrect'
    result['strict_grade'] = primary['grade']
    result['sensitivity'] = {'spec_sha256': SPEC_SHA256, 'selector_sha256': SELECTOR_SHA256,
        'strict_source_sha256': STRICT_SHA256, 'grade': result['grade'], 'reason': reason,
        'selected_final': span, 'fraction': str(fraction) if fraction is not None else None,
        'changed': result['grade'] != primary['grade'], 'scalar': task['family'] in SCALARS,
        'registration': 'prospectively fixed for new collection; post hoc for retained outcomes'}
    return result


def sensitivity_all(rows, tasks, answer_map):
    primary = grade_all(rows, tasks, answer_map)
    tasks = tasks['tasks'] if isinstance(tasks, dict) else tasks
    by_id = {t['id']: t for t in tasks}
    return [sensitivity(row, by_id[row['id']], answer_map[row['id']]) for row in primary]
