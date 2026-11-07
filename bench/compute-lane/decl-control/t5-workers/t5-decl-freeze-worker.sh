#!/bin/bash
set -uo pipefail
out=$T5_OUT;mkdir -p "$out/audit"
ps -o sid= -p $$ | tr -d ' ' > "$out/sid"
python3 -B - "$out/audit" <<'PY'
import pathlib,sys,hashlib,json,datetime,re
O=pathlib.Path(sys.argv[1]);P=pathlib.Path.home()/'work/imr-t5-s1/transport/T5-DECL-CONTROL-SPEC.md';b=P.read_bytes();s=b.decode();assert 3*2*(48+20+24)==552
for x in ['552 fresh replies','Recollect all three arms','Prompt P intentionally differs','all120 fresh final answers','Holm across six primary accuracy tests','ONE-NUMERAL FINAL-SPAN','maximum four turns']:assert x in s,x
num=r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:\s*/\s*[+-]?\d+)?'
a=re.compile(r"\s*[A-Za-z\s'’]*\$?\s*("+num+r")[A-Za-z\s\-'’]*[.!]?\s*")
z=re.compile(r"^\s*[A-Za-z\s'’]*\$?\s*("+num+r")\s*[A-Za-z\s\-\'’]*[.!]?\s*$")
count=0
for l in (pathlib.Path.home()/'work/stdbench/phase2/post-v1/nonmath-graded.jsonl').read_text().splitlines():
 r=json.loads(l)
 if r['suite']!='gsm8k':continue
 text=r['grading']['extracted_final'] or '';m=a.fullmatch(text);n=z.fullmatch(text);assert bool(m)==bool(n) and (not m or m[1]==n[1]);count+=1
assert count==192
result={'spec_sha256':hashlib.sha256(b).hexdigest(),'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bytes':len(b),'scheduled_replies':552,'scheduled_turns':3,'reserve_turns':1,'secondary_grammar_old_span_equivalence':count,'new_outcomes_read':False}
(O/'freeze.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
PY
rc=$?;echo "$rc" > "$out/job-exit";exit "$rc"
