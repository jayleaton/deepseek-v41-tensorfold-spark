#!/usr/bin/env python3
"""Bounded CPU build: verify extraction before and after compiling, then seal."""
import json
from pathlib import Path
import subprocess
from prepare_native import main as prepare,sha,HERE,SOURCE

def main():
    prepare()
    files=['build.zig','cap_main.zig','cap_function.zig','body_function.zig','prepare_native.py','build_native.py']
    before={name:sha(HERE/name) for name in files}
    receipt=json.loads((HERE/'native-source-receipt.json').read_text())
    assert before['cap_function.zig']==receipt['cap_function_sha256']
    zig=Path('<rank0-home>/lane-tools/zig-aarch64-linux-0.17.0/zig')
    subprocess.run([str(zig),'build','-j1','-Dsource='+str(SOURCE)],cwd=HERE,check=True,timeout=7200)
    assert before=={name:sha(HERE/name) for name in files},'harness changed during build'
    for rel,digest in receipt['source_pins'].items(): assert sha(SOURCE/rel)==digest
    receipt.update(binary_sha256=sha(HERE/'zig-out/bin/native-cap'),harness_sha256=before)
    (HERE/'native-build-receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({'native_build':'PASS','source_files':len(receipt['source_pins']),'GPU_requests':0}))
if __name__=='__main__': main()
