#!/bin/bash
set -uo pipefail
out=$T5_OUT;mkdir -p "$out/audit";ps -o sid= -p $$ | tr -d ' ' > "$out/sid"
python3 -B - "$out/audit" <<'PY'
import pathlib,json,sys
B=pathlib.Path.home()/'work/stdbench/format-retest';J=lambda p:json.loads(p.read_text())
a=J(B/'audit-v2/raw-audit.json');print('AUDIT',list(a));print('RAWPATHS',[p for p in a['provenance']['input_file_sha256'] if p.endswith('responses.jsonl')]);r=json.loads(next((B/'audit-v2/rows.jsonl').open()));print('ROWKEYS',list(r));print('ROW',json.dumps({k:v for k,v in r.items() if k in ('study_arm','arm','grade','grader','native_accounting','calls','call_accounting','family','tokens')}));t=json.loads(next((B/'dataset-v1/tasks.jsonl').open()));print('TASK',json.dumps(t)[:1800]);print('REF',str(J(B/'dataset-v1/reference-a.json'))[:450]);print('PRESENT',next((B/'audit-v2/presentations.jsonl').open())[:1800]);print('REPORTEVAL',list(J(B/'results-v1/report.json')))
PY
rc=$?;echo "$rc" > "$out/job-exit";exit "$rc"
