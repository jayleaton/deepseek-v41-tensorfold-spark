"""Offline report assembly, never collection, inference, or receipt revalidation.

collect(rows,tasks,answer_map,*,metadata) accepts only parent-audited merged rows.
metadata['integrity'] is the parent's independent uniform A1 qualification.
No files are written. Parent owns audit/source seals, publishing and budget gates.
"""
import copy
import hashlib
import json
from pathlib import Path

from pins import load_local, verify_protocols, SPEC_SHA256, STRICT_SHA256, SELECTOR_SHA256

analysis = load_local('analysis')
grade = load_local('grade')


def _row_detail(row):
    return {'identity': list(grade.identity(row)), 'family': row['family'],
            'retention_batch': row.get('retention_batch', 'unknown'), 'status': row['status'],
            'finish_reason': row.get('finish_reason'), 'answer_eligible': row.get('answer_eligible'),
            'content': row.get('content'), 'strict_grade': row['strict_grade'],
            'strict_grader': row['grader'], 'sensitivity': row['sensitivity']}


def collect(rows, tasks, answer_map, *, metadata):
    verify_protocols()
    if not isinstance(metadata, dict):
        raise ValueError('parent audit metadata required')
    rows = list(rows)
    strict_rows = grade.grade_all(rows, tasks, answer_map)
    analysis.validate_cohort(strict_rows, tasks)
    secondary_rows = grade.sensitivity_all(rows, tasks, answer_map)
    integrity = metadata.get('integrity', metadata)
    primary = analysis.analyze(strict_rows, tasks, integrity=integrity)
    secondary = analysis.analyze(secondary_rows, tasks, integrity=integrity)
    # Sensitivity can never supply another primary success or rescue a strict claim.
    secondary['qualification'] = {'efficacy_claim': False, 'total_token_saving_claim': False,
                                  'serving_default_promotion': False, 'descriptive_only': True}
    for contrast in secondary['contrasts'].values():
        contrast['primary'] = False
        contrast['interpretation'] = 'secondary extraction sensitivity; post hoc for retained outcomes'
        for item in [contrast['overall'], contrast['scalar_only'], *contrast['families'].values()]:
            if item['sign_flip']:
                item['sign_flip']['primary'] = False
    changed = [_row_detail(r) for r in secondary_rows if r['sensitivity']['changed']]
    residual = [_row_detail(r) for r in secondary_rows if r['grade'] == 'incorrect']
    coverage = {}
    for arm in analysis.ARMS:
        selected = [r for r in secondary_rows if grade.identity(r)[2] == arm]
        scalar = [r for r in selected if r['sensitivity']['scalar']]
        coverage[arm] = {'combined_scheduled': 240, 'combined_observed': len(selected),
                         'combined_solved': sum(r['grade'] == 'correct' for r in selected),
                         'scalar_scheduled': 160, 'scalar_observed': len(scalar),
                         'scalar_solved': sum(r['grade'] == 'correct' for r in scalar),
                         'non_scalar_strict_unchanged': all(r['grade'] == r['strict_grade'] for r in selected if not r['sensitivity']['scalar'])}
    presentations = []
    for row in strict_rows:
        for receipt in (row.get('safety') or {}).get('references', []):
            if receipt.get('presentation') is not None:
                presentations.append({'identity': list(grade.identity(row)), 'call_id': receipt.get('call_id'),
                                      **copy.deepcopy(receipt['presentation'])})
    return {'spec_sha256': SPEC_SHA256, 'strict_grader_sha256': STRICT_SHA256,
            'selector_sha256': SELECTOR_SHA256, 'analysis': primary, 'secondary_analysis': secondary,
            'secondary': {'coverage': coverage, 'changed_rows': changed, 'residual_rows': residual,
                'registration': 'post hoc for original342; fixed before new collection',
                'rule': 'all eligible rows/all arms; four scalar families only; division/aggregation strict unchanged'},
            'strict_rows': strict_rows, 'secondary_rows': secondary_rows,
            'presentation_receipts': presentations,
            'presentation_summary': {'retained_receipts': len(presentations),
                'presentation_verified_clipped': sum(p.get('qualified') is True and p.get('classification') == 'presentation-verified clipped' for p in presentations),
                'limitation': 'landed presentation authenticated; invisible tail is not authenticated; clipped fragments are not complete answers'},
            'metadata': copy.deepcopy(metadata), 'source_sha256': source_hashes()}


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).resolve().parent.glob('*.py'))}


def render_results(report):
    result = report['analysis']
    lines = ['# Compute amendment A1', '', result['status'] + ': ' + str(result['observed']) + '/720 observed.', '',
             result['interpretation'], '', result['historical'], '',
             'CPU-matched/count-checked prompts; served-array equality was not directly observed.',
             'Missing replies have unknown costs. Incomplete cohorts support bounds only.',
             'Tool policy terminations are separate from natural caps; D is an unavailable-executor mechanism control.', '',
             '| Strict arm | Solved / scheduled | Observed | Solve count bounds |',
             '|---|---:|---:|---|']
    for arm, data in result['arms'].items():
        lines.append('| %s | %s/240 | %s | %s |' % (arm, data['solved'], data['observed'], data['solved_count_bounds']))
    for name, contrast in result['contrasts'].items():
        lines += ['', '## ' + name + (' — amended strict primary' if contrast['primary'] else ' — descriptive'), '',
                  '| Mixture | Difference | 95% CI | Worst/best bounds |', '|---|---:|---|---|']
        for label, data in [('overall', contrast['overall']), *contrast['families'].items()]:
            point = data['solve_difference_' + contrast['treatment'] + '_minus_' + contrast['baseline']]
            lines.append('| %s | %s | %s | %s |' % (label, point, data['solve_difference_ci95'], data['difference_bounds']))
        lines += ['', 'Exact task sign flip: ' + json.dumps(contrast['overall']['sign_flip']), '',
                  'Paired gains: ' + json.dumps(contrast['overall']['gains']),
                  'Paired losses: ' + json.dumps(contrast['overall']['losses'])]
    lines += ['', '## Sensitivity (descriptive)', '',
              'Four scalar families:160 scheduled per arm. Combined:240 per arm; arrays/maps remain strict.',
              'Changed/residual row details and scalar/family/combined estimates are retained in the JSON report.', '',
              '```json', json.dumps(report['secondary']['coverage'], indent=2, allow_nan=False), '```', '',
              '## Qualification, costs, censoring and receipt classifications', '', '```json',
              json.dumps({'qualification': result['qualification'], 'activation': result['activation'],
                  'arms': result['arms'], 'contrasts': result['contrasts'],
                  'presentations': report['presentation_summary'], 'metadata': report['metadata']}, indent=2, allow_nan=False), '```', '',
              'No serving-default promotion. No full-token-saving claim from an initial-plus-returned subtotal.', '']
    return '\n'.join(lines)
