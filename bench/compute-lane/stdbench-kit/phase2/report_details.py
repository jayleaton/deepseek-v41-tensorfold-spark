#!/usr/bin/env python3
"""Export descriptive evidence; never changes frozen grades or endpoints."""
from collections import Counter
import csv
import json
from pathlib import Path
root=Path(__file__).resolve().parent
out=root/'post-v1'
load=lambda p: json.loads(Path(p).read_text())
rows=[json.loads(x) for x in (out/'all-provisional.jsonl').read_text().splitlines()]
report=load(out/'analysis.json')
details={}
with (out/'gsm8k-extractions.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['id','seed','arm','grade','reason','extracted_final','reference_final','final_content'])
    writer.writeheader()
    for row in rows:
        if row['suite']=='gsm8k':
            grade=row['grading']
            writer.writerow({**{k:row[k] for k in ('id','seed','arm','grade')},'reason':grade['reason'],'extracted_final':grade.get('extracted_final'),'reference_final':grade.get('reference_final'),'final_content':row['content']})
for suite,value in report['suites'].items():
    selected=[r for r in rows if r['suite']==suite]
    details[suite]={'scheduled_per_arm':value['scheduled_per_arm'],'accuracy_ci95':value['accuracy_ci95'],'sign_flip':value['sign_flip'],'losses':value['losses'],'gains':value['gains'],'tokens':value['tokens'],'time':value['time'],'per_arm':{a:{'solved':value[a]['solved'],'per_seed':value[a]['per_seed'],'grading_reasons':dict(Counter(r['grading']['reason'] for r in selected if r['arm']==a)),'incorrect_grading_reasons':dict(Counter(r['grading']['reason'] for r in selected if r['arm']==a and r['grade']=='incorrect'))} for a in ('plain','lane')}}
with (out/'report-details.json').open('x') as f:json.dump(details,f,indent=2);f.write('\n')
allrows=[json.loads(x) for p in (root/'post-data-v1').glob('*/*/diagnostics.jsonl') for x in p.read_text().splitlines()]+rows
schema={'tensorfold_keys':sorted({k for r in allrows for k in (r.get('tensorfold') or {})}),'usage_keys':sorted({k for r in allrows for k in (r.get('usage') or {})}),'completion_detail_keys':sorted({k for r in allrows for k in ((r.get('usage') or {}).get('completion_tokens_details') or {})}),'compute_keys':sorted({k for r in allrows for k in ((r.get('tensorfold') or {}).get('compute') or {})}),'call_keys':sorted({k for r in allrows for c in (((r.get('tensorfold') or {}).get('compute') or {}).get('calls') or []) for k in c})}
with (out/'telemetry-schema.json').open('x') as f:json.dump(schema,f,indent=2);f.write('\n')
print(json.dumps({'report_details_exported':True,'GSM_incorrect_reasons':{a:details['gsm8k']['per_arm'][a]['incorrect_grading_reasons'] for a in ('plain','lane')},'telemetry_schema':schema}))
