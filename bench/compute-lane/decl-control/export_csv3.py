#!/usr/bin/env python3
"""Project sealed final-row metrics to CSV; no grading or inference changes."""

import sys as _archive_sys
import csv
import json
from pathlib import Path
import sys
import analysis3

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

source=Path(sys.argv[1]); target=Path(sys.argv[2])
rows=[json.loads(line) for line in source.read_text().splitlines() if line]
assert len(rows)==552
fields=['suite','id','seed','arm','strict_solved','strict_grade','gsm_secondary_grade','status',
        'finish_reason','wall_seconds','returned_ids','native_accounting_complete','initial_prompt_ids',
        'prompt_count_reconciled','processed_input_ids','prompt_plus_returned','input_classification',
        'attempted_calls','blocked_dispatches','successful_executions','failed_executions',
        'delivered_authenticated_results','reply_received_result','actual_wire_sha256']
with target.open('x',newline='') as f:
    output=csv.DictWriter(f,fieldnames=fields);output.writeheader()
    for row in sorted(rows,key=analysis3.identity):
        native=row.get('native_accounting') or {}; recon=row.get('initial_prompt_reconciliation') or {}
        inputs=row.get('input_accounting') or {}; calls=row.get('call_accounting') or {}
        output.writerow(dict(suite=row['suite'],id=row['id'],seed=row['seed'],arm=row['arm'],
            strict_solved=None if analysis3.unresolved(row) else analysis3.correct(row),strict_grade=row['grade'],
            gsm_secondary_grade=(row.get('gsm_sensitivity') or {}).get('grade'),status=row['status'],
            finish_reason=row.get('finish_reason'),wall_seconds=row['elapsed_seconds'],
            returned_ids=analysis3.phase2.native_tokens(row),native_accounting_complete=native.get('complete'),
            initial_prompt_ids=analysis3.initial_tokens(row),prompt_count_reconciled=recon.get('qualified'),
            processed_input_ids=analysis3.input_tokens(row),prompt_plus_returned=analysis3.total_tokens(row),
            input_classification=inputs.get('classification'),attempted_calls=calls.get('attempted'),
            blocked_dispatches=calls.get('blocked_dispatches'),successful_executions=calls.get('successful_executions'),
            failed_executions=calls.get('failed_executions'),
            delivered_authenticated_results=calls.get('delivered_authenticated_results'),
            reply_received_result=calls.get('reply_used_result'),actual_wire_sha256=row['actual_request_wire_sha256']))
print(json.dumps({'per_request_CSV':'PASS','scheduled_rows':len(rows),'unknown_cells':'blank, never zero'}))
