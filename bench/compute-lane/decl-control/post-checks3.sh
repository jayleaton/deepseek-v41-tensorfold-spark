#!/usr/bin/env bash
# CPU-only syntax check, invoked through the existing guarded 2G CPU scope.
# preflight-requires: <rank0-work>/stdbench/decl-control/post3.py
# preflight-requires: <rank0-work>/stdbench/decl-control/render_results3.py
# preflight-requires: <rank0-work>/stdbench/decl-control/environment_audit3.py
# preflight-requires: <rank0-work>/stdbench/decl-control/export_csv3.py
# preflight-requires: <rank0-work>/stdbench/decl-control/cleanup_compiler_caches3.py
# preflight-requires: <rank0-work>/stdbench/decl-control/merge_reviews3.py
# preflight-requires: <rank0-work>/stdbench/decl-control/post-comparison-check3.py
# preflight-requires: <rank0-work>/stdbench/decl-control/idle_audit3.py
# preflight-requires: <rank0-work>/stdbench/decl-control/verify_exports3.py
# preflight-requires: <rank0-work>/stdbench/decl-control/restored_native3.py
set -euo pipefail
exec nice -n 19 ionice -c3 timeout 7200 python3 -B -c '
import ast, hashlib, json
from pathlib import Path
root=Path("<rank0-work>/stdbench/decl-control")
names=("post3.py","render_results3.py","environment_audit3.py","export_csv3.py","cleanup_compiler_caches3.py","merge_reviews3.py","post-comparison-check3.py","idle_audit3.py","verify_exports3.py","restored_native3.py")
checks={}
for name in names:
    data=(root/name).read_bytes()
    ast.parse(data, filename=name)
    checks[name]=hashlib.sha256(data).hexdigest()
with (root/"post-syntax-a1-v4.json").open("x") as output:
    json.dump({"passed":True,"python_source_sha256":checks},output,indent=2)
    output.write("\n")
print(json.dumps({"postcollection_syntax":"PASS","files":len(checks)}))
'
