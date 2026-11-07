#!/usr/bin/env python3
"""Verify delivered report/CSV values against the actual sealed final records."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import post3
from audit_records import Inputs

root=Path(__file__).resolve().parent
inputs=Inputs()
cpu,_,_,_=post3.sources(inputs)
final=root/'finalized-a1-v1'
analysis=inputs.json(final/'analysis.json')
audit=inputs.json(final/'audit.json')
receipt=inputs.json(final/'finalize-receipt.json')
for name,field in (('analysis.json','analysis_sha256'),('audit.json','audit_sha256'),
                   ('all-final.jsonl','final_rows_sha256'),('math-review-audit.json','review_audit_sha256')):
    assert hashlib.sha256(inputs.read(final/name)).hexdigest()==receipt[field]
rows=inputs.rows(final/'all-final.jsonl')
assert len(rows)==552 and audit['qualified'] is True
assert all(r['status']=='ok' for r in rows)
doc=inputs.read(root.parent/'RESULTS.md').decode()
assert post3.BASE_SHA in doc and post3.A1_SHA in doc
assert inputs.read(root.parent/'RESULTS-DECL-CONTROL-A1.md')==doc.encode()
names={'GSM8K':'gsm8k','MATH-500':'math500','HumanEval':'humaneval'}
arms={'P':'plain','D':'declared_off','L':'lane'}
found=set(); primary=set()
for line in doc.splitlines():
    if not line.startswith('| '): continue
    cells=[v.strip().replace('−','-') for v in line.strip('|').split('|')]
    if not cells or cells[0] not in names: continue
    suite=names[cells[0]]
    if len(cells)==9 and cells[1] in arms:
        arm=arms[cells[1]]; key=(suite,arm)
        assert key not in found; found.add(key)
        selected=[r for r in rows if (r['suite'],r['arm'])==key]
        a=analysis['suites'][suite]['arms'][arm]
        solved=sum(r['grade']=='correct' for r in selected)
        assert cells[2]==f'{solved}/{len(selected)}'
        assert cells[3]==f'{100*solved/len(selected):.2f}%'
        assert cells[4]==f"{statistics.mean(len(r['tensorfold']['token_ids']) for r in selected):.4f}"
        assert cells[5]==f"{statistics.mean(r['elapsed_seconds'] for r in selected):.4f}"
        assert cells[6:]==list(map(str,(a['timeouts'],a['caps'],a['errors'])))
    if len(cells)==6 and cells[1] in ('D-P','L-D'):
        key=(suite,cells[1]); assert key not in primary; primary.add(key)
        c=analysis['suites'][suite]['contrasts'][cells[1]]
        assert cells[2]==f"{100*c['accuracy_difference_comparison_minus_base']:.4f}"
        assert cells[3]=='['+', '.join(f'{100*v:.4f}' for v in c['accuracy_ci95'])+']'
        assert cells[4]==f"{c['sign_flip']['p_two_sided']:.4f}"
        assert cells[5]==f"{c['sign_flip']['holm_adjusted_p_family6']:.4f}"
assert len(found)==9 and len(primary)==6
csv_path=final/'per-request.csv'
csv_rows=list(csv.DictReader(csv_path.open()))
assert len(csv_rows)==552
identity=lambda r:(r['suite'],r['id'],int(r['seed']),r['arm'])
by_key={identity(r):r for r in rows}
assert len(by_key)==552 and len({identity(r) for r in csv_rows})==552
for r in csv_rows:
    original=by_key[identity(r)]
    assert r['strict_grade']==original['grade'] and r['status']==original['status']
    assert int(r['returned_ids'])==len(original['tensorfold']['token_ids'])
    assert float(r['wall_seconds'])==original['elapsed_seconds']
    assert r['actual_wire_sha256']==original['actual_request_wire_sha256']
renderer=inputs.json(root/'RESULTS.md.receipt.json')
assert renderer['report_sha256']==post3.sha(root/'RESULTS.md')
assert audit['environment']['rank_snapshots']==18
assert len(audit['declared_pairs'])==190 and all(p['qualified'] for p in audit['declared_pairs'])
assert all(p['counts']['successful_executions']['known_sum']==0 for p in audit['activation'])
assert sum(b['verified_diagnostic_calls'] for b in audit['boots'])==6
inputs.stable()
proof={'passed':True,'main_report_rows_verified':len(found),'primary_tests_verified':len(primary),
       'per_request_rows_verified':len(rows),'sealed_sources_unchanged':len(cpu['code_sha256']),
       'both_spec_hashes_verified':True,'native_snapshots':18,'declared_pairs':190,
       'main_report_sha256':post3.sha(root.parent/'RESULTS.md'),
       'detailed_report_sha256':post3.sha(root/'RESULTS.md'),
       'per_request_csv_sha256':post3.sha(csv_path),'input_sha256':inputs.hashes,
       'verification_source_sha256':post3.sha(__file__)}
with (root/'exports-a1-v1.json').open('x') as output:
    json.dump(proof,output,indent=2);output.write('\n')
print(json.dumps({k:v for k,v in proof.items() if k not in ('input_sha256',)}))
