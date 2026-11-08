#!/usr/bin/env python3
"""CPU-only diagnosis of the retained stop; never changes grades or resumes collection."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PREFIX='stdbench-compute-dl-20261105-v1'
SPEC='c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest=lambda b:hashlib.sha256(b).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]

def save(path,value):
    with path.open('x') as output:
        path.chmod(0o600);json.dump(value,output,indent=2,ensure_ascii=False,allow_nan=False)

def typed_map(body):
    try:
        value=ast.literal_eval(body.decode('utf-8').strip())
        if type(value) is dict and set(value)==set('ABCDEF') and all(type(v) is int for v in value.values()):
            return value
    except (SyntaxError,ValueError,UnicodeError,TypeError):pass
    return None

def main():
    cfg=load(HERE/PREFIX/'campaign-v1.json')
    if cfg['spec_sha256']!=SPEC or sha(cfg['spec'])!=SPEC:raise ValueError('spec drift')
    for path,expected in cfg['code_sha256'].items():
        if sha(path)!=expected:raise ValueError('sealed runtime source/data drift')
    if sha(cfg['cpu_receipt'])!=cfg['cpu_receipt_sha256']:raise ValueError('CPU seal drift')
    failures=[]
    for boot in cfg['layout']['boots']:
        name=boot['boot'];directory=HERE/'runs'/(PREFIX+'-'+name[:3])/name
        if not (directory/'safety.jsonl').exists():continue
        observations={(r['suite'],r['id'],r['seed']):r for r in rows(directory/'responses.jsonl')}
        for safety in rows(directory/'safety.jsonl'):
            if safety.get('qualified') is False:
                key=(safety['suite'],safety['id'],safety['seed'])
                failures.append((directory,safety,observations[key]))
    if len(failures)!=1:raise ValueError('expected the one retained first safety stop')
    directory,safety,row=failures[0]
    out=HERE/'diagnosis-v1';out.mkdir(exist_ok=False)
    save(out/'retained-case.json',{'identity':{k:row[k] for k in ('suite','id','seed','arm','boot')},
        'response_file':str(directory/'responses.jsonl'),'response_file_sha256':sha(directory/'responses.jsonl'),
        'safety_file_sha256':sha(directory/'safety.jsonl'),'safety':safety,
        'calls':((row.get('tensorfold') or {}).get('compute') or {}).get('calls',[])})
    calls=((row.get('tensorfold') or {}).get('compute') or {}).get('calls',[])
    refs={r['call_id']:r for r in safety.get('references',[])}
    candidates=[c for c in calls if c.get('inserted') is True and refs.get(c.get('id'),{}).get('matches') is not True]
    result={'spec_sha256':SPEC,'frozen_gate_still_failed':True,'collection_must_remain_stopped':True,
            'source_sha256':sha(__file__),'case_id':row['id'],'seed':row['seed'],
            'binding_qualified':safety['binding'].get('qualified'),'calls':[],
            'diagnosis_only_not_regrading':True,'GPU_requests':0}
    answers=load(HERE/'dataset-v1/answers-a.json')
    if safety['binding'].get('qualified') is True:
        sys.path.insert(0,str(HERE.parent/'phase2'))
        from jail import Jail
        for call in candidates:
            original=refs[call['id']]['result']
            native=call['output'].encode()
            reference=base64.b64decode(original['normalized_body_base64'],validate=True)
            record={'call_id':call['id'],'code_sha256':digest(call['code'].encode()),
                    'native_status':call['status'],'original_reference_status':original['status'],
                    'native_body_sha256':digest(native),'original_reference_body_sha256':digest(reference),
                    'native_typed_map':typed_map(native),'original_reference_typed_map':typed_map(reference),
                    'replays':[]}
            for index in range(3):
                replay=Jail(cfg['image_ids'][0]).run_reference(call['code'])
                identifier=digest(str(call['id']).encode())[:16]
                save(out/('replay-'+identifier+'-'+str(index)+'.json'),replay)
                body=base64.b64decode(replay['normalized_body_base64'],validate=True)
                record['replays'].append({'status':replay['status'],'body_sha256':digest(body),'typed_map':typed_map(body)})
            values=[record['native_typed_map'],record['original_reference_typed_map']]+[r['typed_map'] for r in record['replays']]
            record['all_typed_maps_equal_task_reference']=all(v is not None and v==answers[row['id']] for v in values)
            record['distinct_body_variants']=len({record['native_body_sha256'],record['original_reference_body_sha256'],
                                                *(r['body_sha256'] for r in record['replays'])})
            record['CPU_replays_vary']=len({r['body_sha256'] for r in record['replays']})>1
            record['classification']=('BYTE_REPLAY_NONDETERMINISM_EQUAL_TYPED_VALUES'
                if record['CPU_replays_vary'] and record['all_typed_maps_equal_task_reference']
                else 'REPLAY_MISMATCH_REQUIRES_ADJUDICATION')
            result['calls'].append(record)
    save(out/'diagnosis.json',result)
    print(json.dumps({'case_id':result['case_id'],'seed':result['seed'],
        'classifications':[c['classification'] for c in result['calls']],
        'frozen_gate_still_failed':True,'collection_must_remain_stopped':True,'GPU_requests':0}))

if __name__=='__main__':main()
