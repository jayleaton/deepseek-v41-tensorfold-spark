#!/usr/bin/env python3
"""Prospectively pinned generation and exclusions; never a model request."""
import argparse
import datetime
import hashlib
import json
import platform
from pathlib import Path
import sys
import generate_tasks
from disjointness import CollisionIndex
from exclusion_sources import load_exclusions

ROOT=Path(__file__).resolve().parent
SPEC=ROOT/'T5-COMPUTE-DL-SPEC.md'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)


def save(path,value,private=False):
    with Path(path).open('x') as output:
        if private: Path(path).chmod(0o600)
        output.write(canonical(value)+'\n')


def inventory(config,out):
    records,audit=load_exclusions(json.loads(config.read_text()))
    out.mkdir(exist_ok=False)
    save(out/'records.json',records)
    save(out/'audit.json',audit)
    pending=[row for row in audit['semantic_review_inventory'] if row['classification']=='unresolved-original']
    save(out/'unresolved.json',pending)
    print(json.dumps({'exclusion_sources':'PINNED','counts':audit['counts'],'unresolved_originals':len(pending)}))


def pin(out):
    if sha(SPEC)!=generate_tasks.SPEC_SHA256: raise ValueError('spec SHA drift')
    names=['generate_tasks.py','reference_a.py','reference_b.py','disjointness.py','exclusion_sources.py']
    record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'spec_sha256':sha(SPEC),'python_version':platform.python_version(),
            'python_full':sys.version,'source_sha256':{name:sha(ROOT/name) for name in names},
            'study_candidates_generated_before_this_pin':0}
    save(out,record)
    print(json.dumps({'pre_generation_pin':'WRITTEN','python_version':record['python_version'],
                      'generator_sha256':record['source_sha256']['generate_tasks.py']}))


def generate(pin_path,config,resolutions_path,out):
    pin=json.loads(pin_path.read_text())
    generate_tasks.verify_generation_pin(python_version=pin['python_version'],
        generator_sha256=pin['source_sha256']['generate_tasks.py'],spec_path=SPEC)
    for name,expected in pin['source_sha256'].items():
        if sha(ROOT/name)!=expected: raise ValueError('pinned preparation source drift: '+name)
    records,audit=load_exclusions(json.loads(config.read_text()))
    resolutions=json.loads(resolutions_path.read_text())
    prior_by_id={record['id']:record for record in records}
    for ident,resolution in resolutions.items():
        record=prior_by_id[ident]
        evidence=resolution['original_evidence']
        if (evidence['path']!=record['source_path']
                or evidence['sha256']!=record['source_sha256']
                or sha(evidence['path'])!=evidence['sha256']
                or ident not in evidence['location']):
            raise ValueError('manual original evidence binding differs: '+ident)
    index=CollisionIndex(records,resolutions)
    out.mkdir(exist_ok=False)
    accepted=[];collisions=[]
    with (out/'candidates.jsonl').open('x') as candidate_stream:
        for family in generate_tasks.FAMILIES:
            for slot in range(20):
                for attempt in range(1000):
                    candidate=generate_tasks.generate_candidate(family,slot,attempt)
                    hit=index.check(candidate)
                    candidate_stream.write(canonical({'candidate':candidate,'collision':hit})+'\n')
                    candidate_stream.flush()
                    if hit:
                        collisions.append(hit)
                        continue
                    index.add(candidate);accepted.append(candidate)
                    break
                else: raise RuntimeError('1000 disjointness collisions: preparation stopped')
    if len(accepted)!=120: raise RuntimeError('fixed task count differs')
    public=out/'tasks-public.jsonl'
    with public.open('x') as output:
        for task in accepted: output.write(canonical(task)+'\n')
    save(out/'collisions.json',collisions)
    save(out/'exclusion-audit.json',audit)
    save(out/'manual-resolutions.json',resolutions)
    proof={'passed':True,'spec_sha256':sha(SPEC),'pre_generation_pin_sha256':sha(pin_path),
           'public_sha256':sha(public),'task_count':120,'families':list(generate_tasks.FAMILIES),
           'collision_count':len(collisions),'exclusion_config_sha256':sha(config),
           'manual_resolutions_sha256':sha(resolutions_path),'reference_verification':'PENDING',
           'model_requests':0,'generation_source_sha256':pin['source_sha256']['generate_tasks.py'],
           'python_version':platform.python_version(),'generator_attempts':'only documented collisions'}
    save(out/'generation.json',proof)
    print(json.dumps({'generated_tasks':120,'task_manifest_sha256':sha(public),
                      'collisions':len(collisions),'references':'PENDING','model_requests':0}))


def verify(pin_path,dataset):
    pin=json.loads(pin_path.read_text())
    public=dataset/'tasks-public.jsonl';raw=public.read_bytes()
    tasks=[json.loads(line) for line in raw.splitlines()]
    ids={t['id'] for t in tasks}
    if len(tasks)!=120 or len(ids)!=120: raise ValueError('public grid differs')
    answers=[]
    for name in ('a','b'):
        source='reference_'+name+'.py'
        if sha(ROOT/source)!=pin['source_sha256'][source]: raise ValueError('reference source drift')
        result=dataset/('answers-'+name+'.json');receipt=json.loads(Path(str(result)+'.receipt.json').read_text())
        if (receipt['status']!='PASS' or receipt['count']!=120 or receipt['input_sha256']!=sha(public)
                or receipt['source_sha256']!=sha(ROOT/source) or receipt['output_sha256']!=sha(result)
                or receipt['python_version']!=pin['python_version']): raise ValueError('independent reference receipt mismatch')
        values=json.loads(result.read_text())
        if set(values)!=ids: raise ValueError('reference ID coverage mismatch')
        answers.append(values)
    import grade
    for task in tasks:
        a,b=(v[task['id']] for v in answers)
        if not grade.valid_value(a,task['family']) or not grade.valid_value(b,task['family']) or a!=b:
            raise ValueError('reference disagreement or malformed typed answer; never regenerate')
    save(dataset/'reference-verification.json',{'passed':True,'tasks':120,'independent_processes':True,
        'public_sha256':sha(public),'reference_a_sha256':sha(dataset/'answers-a.json'),
        'reference_b_sha256':sha(dataset/'answers-b.json'),'source_sha256':pin['source_sha256'],
        'spec_sha256':sha(SPEC),'python_version':pin['python_version']})
    print(json.dumps({'independent_reference_agreement':'PASS','tasks':120,'model_requests':0}))


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('inventory');a.add_argument('--config',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=s.add_parser('pin');a.add_argument('--out',type=Path,required=True)
    a=s.add_parser('generate');a.add_argument('--pin',type=Path,required=True);a.add_argument('--config',type=Path,required=True)
    a.add_argument('--resolutions',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=s.add_parser('verify');a.add_argument('--pin',type=Path,required=True);a.add_argument('--dataset',type=Path,required=True)
    a=p.parse_args()
    if a.action=='inventory': inventory(a.config,a.out)
    elif a.action=='pin': pin(a.out)
    elif a.action=='generate': generate(a.pin,a.config,a.resolutions,a.out)
    else: verify(a.pin,a.dataset)


if __name__=='__main__': main()
