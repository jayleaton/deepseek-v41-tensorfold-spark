#!/usr/bin/env python3
"""Read-only A1 reuse qualification; no new compilation or GPU request."""

import sys as _archive_sys
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
legacy=root.parent/'decl-control'
sys.path.insert(0,str(legacy))
import generate_envs3
import post3
from audit_records import Inputs

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

rank=int(sys.argv[1]); assert rank in (0,1)
inputs=Inputs()
receipt,_,_,_=post3.sources(inputs)
proof_path=legacy/'dispatch-high/dispatch-proof.json'
proof=inputs.json(proof_path)
assert proof['verdict']=='PASS' and proof['observed_fixtures']==8
assert proof['whole_stream_noncall_and_wrong_arm_controls']
assert proof['guard_mutant_compiled'] and proof['guard_mutant_behaviorally_caught']
assert proof['deployed_binary_instrumentation'] is False
source_hashes=inputs.json(legacy/'dispatch-high/unmodified-source-sha256.json')
assert len(source_hashes)==proof['production_sources_unchanged']==3436
for relative,expected in source_hashes.items():
    path=legacy/'dispatch-high'/relative
    assert path.resolve().is_relative_to((legacy/'dispatch-high').resolve())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
for relative,expected in proof['sha256'].items():
    path=legacy/'dispatch-high'/relative
    assert path.resolve().is_relative_to((legacy/'dispatch-high').resolve())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
item=json.loads(subprocess.check_output(['docker','image','inspect',generate_envs3.IMAGE],text=True,timeout=30))[0]
assert item['Id']==generate_envs3.IMAGE_IDS[rank]
assert item['Config']['Labels']['org.tensorfold.revision']==generate_envs3.REVISION
inputs.stable()
result={'passed':True,'rank':rank,'status':'REUSED_A1_EVIDENCE_ONLY',
        'old_sealed_files_unchanged':len(receipt['code_sha256']),
        'production_source_hashes_rechecked':len(source_hashes),'image_id':item['Id'],
        'dispatch_proof_sha256':hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        'dispatch_fixture_freshly_rerun':False,'collection_authorized':False,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (root/f'baseline-r{rank}.json').open('x') as output:
    json.dump(result,output,indent=2);output.write('\n')
print(json.dumps(result))
