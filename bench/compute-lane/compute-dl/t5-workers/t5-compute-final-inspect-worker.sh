#!/bin/bash
set -uo pipefail
out=$T5_OUT;mkdir -p "$out/audit"
ps -o sid= -p $$ | tr -d ' ' > "$out/sid"
python3 -B - "$out/audit" <<'PY'
import pathlib,json,sys,collections
B=pathlib.Path.home()/'work/stdbench/compute-dl';O=pathlib.Path(sys.argv[1]);J=lambda p:json.loads(p.read_text())
A=J(B/'a2-report/final-audit-v1/audit.json');print('AUDIT_KEYS',list(A));print('PROV_KEYS',list(A.get('provenance',{})));print('BUDGET',A.get('budget'));files=A.get('provenance',{}).get('input_file_sha256',{});paths=[p for p in files if p.endswith('responses.jsonl')];print('RAWFILES',paths)
R=json.loads(next((B/'a2-results-v1/graded.jsonl').open()));print('GRADED_KEYS',list(R));print('NATIVE',R.get('native_accounting'));print('RAWFIELD',[(k,type(v).__name__) for k,v in R.items() if 'raw' in k or 'token' in k])
r=json.loads(next((B/'a1-continuation/final-audit-v3/audited-rows.jsonl').open()));print('AUDITED_KEYS',list(r));print('IDENTITY',r['id'],r['arm'],r.get('boot'));print('SAFETY_KEYS',list(r.get('safety',{})));print('SSE0',r.get('sse_frames',[None])[0]);print('DETAIL',r.get('call_accounting'),r.get('input_accounting'))
report=J(B/'a2-results-v1/report.json');print('REPORT_KEYS',list(report));print('SECONDARY_KEYS',list(report['secondary']));res=[x for x in report['secondary']['residual_rows'] if x['identity'][-1]=='L'];print('L_RESIDUALS',len(res));
for x in res[:4]:print('EXAMPLE',json.dumps(x))
print('BREAK',report['P_break_batches'])
(O/'inspect.json').write_text(json.dumps({'raw_paths':paths,'auditkeys':list(A)},indent=2))
PY
rc=$?;echo "$rc" > "$out/job-exit";exit "$rc"
