#!/usr/bin/env python3
"""Run pinned native routing controls inside the parent's constrained scope."""

import sys as _archive_sys
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)

root = Path(__file__).resolve().parent
zig = '<rank1-home>/lane-tools/zig-aarch64-linux-0.17.0/zig'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
resource.setrlimit(resource.RLIMIT_CORE, (0,0))
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void',
           ZIG_GLOBAL_CACHE_DIR=str(root/'.zig-global-cache'), MAKEFLAGS='-j1', OMP_NUM_THREADS='1')
assert subprocess.check_output([zig,'version'], text=True, timeout=7200).strip() == '0.17.0'
source_hashes = json.loads((root/'unmodified-source-sha256.json').read_text())
def verify_source():
    assert all(sha(root/p)==h for p,h in source_hashes.items()), 'pinned production source drift'
def run(args,cwd,stdout,stderr=None):
    with (cwd/stdout).open('x') as output:
        if stderr:
            with (cwd/stderr).open('x') as errors:
                return subprocess.run(args,cwd=cwd,env=env,stdout=output,stderr=errors,timeout=7200).returncode
        return subprocess.run(args,cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=7200).returncode
verify_source()
assert run([zig,'build','compile','-j1'],root,'compile.log') == 0, 'baseline compile failed; compile.log'
assert run([zig,'build','check','-j1'],root,'observed.jsonl','check.log') == 0, 'baseline behavioral check failed; check.log'
assert run(['bash','prepare_guard_mutant.sh'],root,'mutant-preparation.log') == 0
mutant = root/'guard-mutant'
assert run([zig,'build','compile','-j1'],mutant,'compile.log') == 0, 'mutant VOID: compile failure'
rc = run([zig,'build','check','-j1'],mutant,'observed.jsonl','check.log')
failure = (mutant/'check.log').read_text(errors='replace')
assert rc != 0 and ('attempt to use null value' in failure or 'TestUnexpectedResult' in failure), 'mutant not caught behaviorally'
verify_source()
rows = [json.loads(p.read_text()) for p in sorted(root.glob('fixture-*.json'))]
assert len(rows) == 8, 'missing dispatch fixture receipts'
with (root/'fixture-receipts.jsonl').open('x') as f:
    for row in rows: f.write(json.dumps(row)+'\n')
proof = {'verdict':'PASS','base_spec_sha256':'adb05167fff21469ee7251555cb243f1f109c4d8e3de237fdcaa59ada8eb55e6',
         'spec_sha256':'e7308621c645aef4402924a1b47c1a52e05baabe00239f915011f8431bbd0e89',
         'whole_stream_noncall_and_wrong_arm_controls':True,'guard_mutant_compiled':True,
         'guard_mutant_behaviorally_caught':True,'production_sources_unchanged':len(source_hashes),
         'observed_fixtures':len(rows),'GPU':False,'deployed_binary_instrumentation':False,
         'sha256':{str(p.relative_to(root)):sha(p) for p in [root/'build.zig',root/'dispatch_spy.c',
             root/'zig/src/server/decl_control_probe.zig',root/'zig/src/server/decl_control_tools.json',
             root/'fixture-receipts.jsonl',root/'prepare_guard_mutant.sh',root/'unmodified-source-sha256.json',
             mutant/'source.patch',mutant/'compile.log',mutant/'check.log']},
         'compiler_sha256':sha(Path(zig))}
with (root/'dispatch-proof.json').open('x') as f: json.dump(proof,f,indent=2); f.write('\n')
print(json.dumps({'dispatch_proof':'PASS','fixtures':8,'guard_mutant':'CAUGHT','GPU':False}))
