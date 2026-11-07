#!/usr/bin/env python3
"""Host-only native prompt feasibility; not server/dispatch/collection qualification."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
root=Path(__file__).resolve().parent
phase=root.parent/'phase2'
spec_sha='adb05167fff21469ee7251555cb243f1f109c4d8e3de237fdcaa59ada8eb55e6'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def event(step,verdict,**kwargs):
    with (root/'cpu-steps.jsonl').open('a') as f:f.write(json.dumps({'step':step,'verdict':verdict,**kwargs})+'\n')
assert sha(root/'T5-DECL-CONTROL-SPEC.md')==spec_sha
image=subprocess.check_output(['docker','image','inspect','--format','{{.Id}} {{index .Config.Labels "org.tensorfold.revision"}}','tensorfold-dsv41-imr:r-3a429d6f122d'],text=True,timeout=7200).strip()
assert image=='sha256:80beaba732d1ed9439e012b007d8dbe19ec2f8962f50fbac29a18c8cd6fda306 3a429d6f122d9eb2900c43e622933d311e4be44f'
assert sha(phase/'references/tokenizer.json')=='c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
manifest=json.loads((phase/'frozen-v3/manifest.json').read_text())
records=[];bodies={};seen=set();diagnostics=None
for entry in manifest['boots']:
    if entry['arm']!='lane':continue
    boot=json.loads((phase/'frozen-v3'/entry['file']).read_text())
    diagnostics=diagnostics or boot['diagnostics']
    for row in boot['requests']:
        key=(row['suite'],row['id'],row['seed'])
        assert key not in seen;seen.add(key)
        body=row['body_template']
        assert body['tools']==[{'type':'code_interpreter'}]
        assert all(m['role']=='user' and isinstance(m['content'],str) and set(m)=={'role','content'} for m in body['messages'])
        records.append({'key':list(key),'messages':body['messages']});bodies[key]=body
assert len(records)==184
records += [{'key':['diagnostic',r['id'],r['seed']],'messages':r['body_template']['messages']} for r in diagnostics]
save(root/'prompt-inputs.json',records)
zig='<rank1-home>/lane-tools/zig-aarch64-linux-0.17.0/zig'
version=subprocess.check_output([zig,'version'],text=True,timeout=7200).strip()
assert version=='0.17.0'
scratch=root/'scratch'
scratch.mkdir(exist_ok=False)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void',OMP_NUM_THREADS='1',MAKEFLAGS='-j1',ZIG_GLOBAL_CACHE_DIR=str(scratch/'global-cache'))
try:
    with (root/'build.log').open('x') as log:
        run=subprocess.run([zig,'build','-j1','--cache-dir',str(scratch/'local-cache'),'--prefix',str(scratch/'install')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
    event('native-host-build','PASS' if run.returncode==0 else 'FAIL')
    if run.returncode:raise RuntimeError('native host harness build failed; retained build.log')
    binary=scratch/'install/bin/prompt-gate'
    with (root/'prompt-arrays.jsonl').open('x') as output, (root/'render.log').open('x') as log:
        run=subprocess.run([str(binary),str(root/'prompt-inputs.json'),str(phase/'references/tokenizer.json')],stdout=output,stderr=log,env=env,timeout=7200)
    event('native-render','PASS' if run.returncode==0 else 'FAIL')
    if run.returncode:raise RuntimeError('native render failed; retained render.log')
    rendered=[json.loads(line) for line in (root/'prompt-arrays.jsonl').read_text().splitlines()]
    assert len(rendered)==186
    for row in rendered:
        assert row['declared_off_ids']==row['declared_lane_ids']==row['original_builtin_lane_ids']
        assert row['held_off'] is False and row['held_lane'] is True
    tools=rendered[0]['explicit_tools']
    save(root/'canonical-python-tools.json',tools)
    # Same API body in D/L; only the separately pinned rank execution policy differs.
    # Wire must preserve object insertion order, unlike the old client.canonical(sorted_keys=True).
    layouts=[];requests=[]
    import random
    for turn in (1,2,3):
        order={1:['plain','declared_off','lane'],2:['declared_off','lane','plain'],3:['lane','plain','declared_off']}[turn]
        pairs=[]
        for suite,sizes in [('gsm8k',[16,16,16]),('math500',[7,7,6]),('humaneval',[8,8,8])]:
            ids=sorted({k[1] for k in bodies if k[0]==suite});start=sum(sizes[:turn-1]);chosen=ids[start:start+sizes[turn-1]]
            units=sorted((task,seed) for task in chosen for seed in (101,102));random.Random(20261105).shuffle(units)
            pairs += [(suite,task,seed) for task,seed in units]
        layouts.append({'turn':turn,'arms':order,'pair_order':[list(k) for k in pairs]})
        for arm in order:
            for key in pairs:
                body=json.loads(json.dumps(bodies[key]));body.pop('tools',None)
                if arm!='plain':body['tools']=tools
                wire=json.dumps(body,ensure_ascii=False,separators=(',',':'))
                requests.append({'turn':turn,'arm':arm,'key':list(key),'execution_policy':{'TF_DSV41_COMPUTE_LANE':str(int(arm=='lane'))},'body':body,'wire':wire,'wire_sha256':hashlib.sha256(wire.encode()).hexdigest()})
    assert len(requests)==552
    save(root/'DRAFT-requests-NOT-QUALIFIED.json',{'spec_sha256':spec_sha,'status':'CPU_FEASIBILITY_ONLY_NOT_COLLECTABLE','layouts':layouts,'requests':requests})
    save(root/'cpu-proof.json',{'cpu_prompt_arrays_equal':True,'dataset_pairs':184,'diagnostic_pairs':2,'arrays_retained':'prompt-arrays.jsonl','prompt_arrays_sha256':sha(root/'prompt-arrays.jsonl'),'spec_sha256':spec_sha,'tokenizer_sha256':sha(phase/'references/tokenizer.json'),'zig_version':version,'binary_sha256':sha(binary),'native_source_sha256':{p.name:sha(p) for p in sorted((root/'native').glob('*.zig'))},'harness_sha256':{p.name:sha(p) for p in [root/'build.zig',root/'prompt_gate.zig',root/'cpu_probe.py']},'request_side_arming_only':True,'dispatch_negative_control_qualified':False,'actual_served_prompt_array_capture_available':False,'collection_qualified':False,'limitations':['Inert Server/API/chat types only exercise unchanged tools/arming functions, not actual server dispatch.','No actual served initial prompt arrays are exposed on the pinned chat route.','All-round actual prompt-processing accounting is not emitted by compute_segments merge.','No GPU, endpoint, queue, lock or engine edit was used.']})
    shutil.copy2(binary,root/'prompt-gate')
    event('CPU-full-array-comparison','PASS',dataset_pairs=184,diagnostic_pairs=2)
    event('collection-feasibility','BLOCKED',reason='actual served initial prompt IDs and all-round prompt accounting unavailable')
    print(json.dumps({'native_CPU_prompt_equality':'PASS','dataset_pairs':184,'diagnostic_pairs':2,'draft_requests':552,'collection':'BLOCKED','no_GPU_or_endpoint_requests':True}))
finally:
    shutil.rmtree(scratch) # Exact owned temporary compiler/install directory only.
