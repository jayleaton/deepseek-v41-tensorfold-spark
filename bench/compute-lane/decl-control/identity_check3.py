#!/usr/bin/env python3
"""Read actual node model/tokenizer and immutable Docker metadata; no boot."""

import sys as _archive_sys
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from generate_envs3 import IMAGE, IMAGE_IDS, REVISION, read_reference

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

root=Path(__file__).resolve().parent
rank=int(sys.argv[1]); assert rank in (0,1)
config=read_reference(root.parent/f'phase2/references/base-r{rank}.env')
tokenizer=Path(config['MODEL_DIR'])/'tokenizer.json'
actual=hashlib.sha256(tokenizer.read_bytes()).hexdigest()
assert actual=='c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
image=subprocess.check_output(['docker','image','inspect','--format','{{.Id}} {{index .Config.Labels "org.tensorfold.revision"}}',IMAGE],text=True,timeout=7200).strip()
assert image==IMAGE_IDS[rank]+' '+REVISION
receipt={'verdict':'PASS','rank':rank,'model_dir':config['MODEL_DIR'],'tokenizer_sha256':actual,
         'image_id':IMAGE_IDS[rank],'revision':REVISION,'GPU':False}
with (root/f'identity-r{rank}.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({'identity':'PASS','rank':rank,'GPU':False}))
