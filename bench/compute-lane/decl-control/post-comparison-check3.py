#!/usr/bin/env python3
"""One-off reporting persistence regression, run only in the guarded Spark scope."""
import json
from pathlib import Path
import client3
import post3
import reconciliation
from audit_records import Inputs

root=Path(__file__).resolve().parent
inputs=Inputs()
receipt,manifest,analysis,prompts=post3.sources(inputs)
checked=0
for entry in manifest['boots']:
    boot=json.loads((root/'frozen-a1-v1'/entry['file']).read_text())
    directory=root/'runs'/(post3.PREFIX+'-'+boot['boot'].split('-')[0])/boot['boot']
    for filename,templates in (('responses.jsonl',boot['requests']),('diagnostics.jsonl',boot['diagnostics'])):
        rows=[json.loads(line) for line in (directory/filename).read_text().splitlines()]
        for row,template in zip(rows,templates):
            fresh=reconciliation.reconcile(row,dict(template,body=template['body_template']),prompts[tuple(template['key'])])
            persisted=row['initial_prompt_reconciliation']
            assert client3.canonical(fresh)==client3.canonical(persisted)
            changed=json.loads(client3.canonical(persisted))
            changed['server_initial_prompt_tokens']+=1
            assert client3.canonical(fresh)!=client3.canonical(changed)
            changed=json.loads(client3.canonical(persisted))
            changed['reported_counts'].reverse()
            if len(changed['reported_counts'])>1:
                assert client3.canonical(fresh)!=client3.canonical(changed)
            checked+=1
assert checked==570
inputs.stable()
with (root/'post-comparison-a1-v2.json').open('x') as output:
    json.dump({'passed':True,'stored_live_attestations':checked,
               'changed_count_rejected':True,'array_order_preserved':True,
               'post_wrapper_sha256':post3.sha(root/'post3.py'),
               'sealed_sources_unchanged':len(receipt['code_sha256'])},output,indent=2)
    output.write('\n')
print(json.dumps({'persisted_attestations':'PASS','rows':checked,'negative_controls':'PASS'}))
