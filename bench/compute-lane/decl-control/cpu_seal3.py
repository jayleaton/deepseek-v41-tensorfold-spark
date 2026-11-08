#!/usr/bin/env python3
"""Final A1 precollection gate; constrained Spark CPU scope only."""

import sys as _archive_sys
import hashlib
import json
from pathlib import Path
import py_compile
import shutil
import sys
import tempfile

root = Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'phase2'))
import native_frames
from jail import Jail, qualify
sys.path.insert(0,str(root))
import analysis3_check
import grade3
import math3
import reserve3
from dispatch_audit3 import audit
from freeze3 import BASE_SHA, A1_SHA, sha, load, save
from generate_envs3 import PREFIX, IMAGE_IDS

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

out = root/sys.argv[1]
out.mkdir(exist_ok=False)
assert sha(root/'T5-DECL-CONTROL-SPEC.md')==BASE_SHA
assert sha(root/'T5-DECL-CONTROL-SPEC-A1.md')==A1_SHA
proof = load(root/'dispatch-high/dispatch-proof.json')
assert proof['verdict']=='PASS' and proof['guard_mutant_compiled'] and proof['guard_mutant_behaviorally_caught']
for path,digest in proof['sha256'].items(): assert sha(root/'dispatch-high'/path)==digest
dispatch_receipts = audit(root/'dispatch-high')
cpu_prompt = load(root/'cpu-proof.json')
assert cpu_prompt['cpu_prompt_arrays_equal'] and cpu_prompt['dataset_pairs']==184
assert sha(root/'prompt-arrays.jsonl')==cpu_prompt['prompt_arrays_sha256']
mapping = {'compute_turns.zig':'server/compute_turns.zig','json.zig':'server/json.zig',
           'json_parse.zig':'server/json_parse.zig','json_write.zig':'server/json_write.zig',
           'template.zig':'families/deepseek_v41/serve/template.zig',
           'tokenizer.zig':'families/deepseek_v41/serve/tokenizer.zig',
           'encoder.zig':'families/deepseek_v41/serve/encoder.zig',
           'unicode.zig':'core/tokenizer/unicode.zig','unicode_data.zig':'core/tokenizer/unicode_data.zig'}
for name,path in mapping.items():
    assert sha(root/'dispatch-high/zig/src'/path)==cpu_prompt['native_source_sha256'][name]
for rank in (0,1):
    assert load(root/f'rank-preflight-r{rank}.json')['verdict']=='PASS'
    assert load(root/f'identity-r{rank}.json')['verdict']=='PASS'
client = load(root/'client3-check.json')
assert client['verdict']=='PASS' and client['fake_sends']==4
for name,digest in client['sha256'].items(): assert sha(root/name)==digest
with tempfile.TemporaryDirectory(prefix='decl-a1-bytecode-',dir=root) as scratch:
    for p in sorted(root.glob('*.py')):
        py_compile.compile(str(p),cfile=str(Path(scratch)/(p.name+'c')),doraise=True)
checks = analysis3_check.selfcheck()
assert checks['fake_checks']=='passed'
checks['reserve']=reserve3.selfcheck()
native_frames.selftest(out/'native-frames.json')
manifest = load(root/'frozen-a1-v1/analysis-manifest.json')
private = root/'MATH-PRESEAL-A1-v1.json'
expected = math3.build_preseal(manifest['tasks'])
if not private.exists(): math3.write_private(private,expected)
assert load(private)==expected and len(expected['mapping'])==120
hashes = grade3.verify_sources()
save(out/'grader-hashes.json',hashes)
jail_path=out/'jail-qualification/qualification.json'
if len(sys.argv)>2:
    jail_path=root/sys.argv[2]/'jail-qualification/qualification.json'
    jail_receipt=load(jail_path)
    assert jail_receipt['image']['id']==IMAGE_IDS[1]
    for name,digest in jail_receipt['sources_sha256'].items(): assert sha(root.parent/'phase2'/name)==digest
    assert sha(root.parent/'kit/sources/humaneval.jsonl.gz')==jail_receipt['original_source_gz_sha256']
    assert sha(root.parent/'humaneval-selected.jsonl')==jail_receipt['selected_sha256']
else:
    jail_receipt = qualify(Jail(IMAGE_IDS[1]),root.parent/'kit/sources/humaneval.jsonl.gz',
                            root.parent/'humaneval-selected.jsonl',out/'jail-qualification')
assert jail_receipt['qualified'] and len(jail_receipt['results'])==102
code = {str(p):sha(p) for pattern in ('*.py','*.sh') for p in sorted(root.glob(pattern))}
index=load(root/PREFIX/'manifest.json')
for relative,digest in index['files'].items():
    p=root/PREFIX/relative
    assert sha(p)==digest
    code[str(p)]=digest
for rank in (0,1):
    for p in (root/PREFIX/f'rank{rank}/configs').glob('*.env'):
        actual=root/PREFIX/'configs'/p.name
        if not actual.exists(): shutil.copy2(p,actual)
        assert sha(actual)==sha(p)
        code[str(actual)]=sha(actual)
code[str(root/PREFIX/'manifest.json')]=sha(root/PREFIX/'manifest.json')
for name in ('build.zig','build.zig.zon','dispatch_spy.c','run_dispatch.py','prepare_guard_mutant.sh',
             'zig/src/server/decl_control_probe.zig','zig/src/server/decl_control_tools.json'):
    p=root/'dispatch-high'/name; code[str(p)]=sha(p)
for name in ('client.py','native_frames.py','analysis.py','grade_all.py','jail.py','jail_worker.py','jail.md'):
    p=root.parent/'phase2'/name; code[str(p)]=sha(p)
for name in ('bench.py','grader_worker.py','datasets.lock.json'):
    p=root.parent/'kit'/name; code[str(p)]=sha(p)
receipt = {'passed':True,'base_spec_sha256':BASE_SHA,'spec_sha256':A1_SHA,
           'fresh_replies':552,'checks':checks,'code_sha256':code,
           'dispatch_proof_sha256':sha(root/'dispatch-high/dispatch-proof.json'),
           'dispatch_receipt_audit':dispatch_receipts,
           'cpu_prompt_arrays_sha256':sha(root/'prompt-arrays.jsonl'),
           'freeze_sha256':sha(root/'frozen-a1-v1/manifest.json'),
           'rank_preflight_sha256':{str(r):sha(root/f'rank-preflight-r{r}.json') for r in (0,1)},
           'client_receipt_sha256':sha(root/'client3-check.json'),
           'math_preseal_canonical_sha256':math3.digest(expected),'math_preseal_file_sha256':sha(private),
           'grader_hashes':hashes,'jail_cases':102,
           'jail_receipt_sha256':sha(jail_path),'jail_receipt_path':str(jail_path),
           'prompt_claim':'CPU-matched declaration prompts, identical request bodies and matching served prompt counts; no direct served-ID proof',
           'multi_round_input_totals':'WITHHELD','GPU':False}
save(out/'verification.json',receipt)
print(json.dumps({'A1_CPU_gate':'PASS','fresh_replies':552,'jail_cases':102,'math_presealed':120,
                  'dispatch':'real routing whole/SSE + noncall + behavioral mutant','GPU':False}))
