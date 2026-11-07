"""Validate prepared explicit arrays; never generate tasks or choose observations."""
import hashlib
import json
import random
from protocol import require

def wire(body):return json.dumps(body,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def validate_boots(boots):
    require([(b['boot'],b['arm']) for b in boots]==[('format-L','L'),('format-P','P')],'two physical cold boots')
    schedule=[]
    for boot in boots:
        physical=boot['arm'];rows=boot['requests']
        require(len(rows)==(240 if physical=='L' else 120),'boot request count')
        pairs=[];orientations=[];seen=set()
        for first,second in zip(rows[::2],rows[1::2]):
            require(first['key']==second['key'],'wording pair must be consecutive')
            require({first['study_arm'],second['study_arm']}=={physical+'-old',physical+'-new'},'exact old/new pair')
            pairs.append((first['id'],first['seed']));orientations.append(first['study_arm'].endswith('-new'))
        require(len(set(pairs))==len(pairs),'duplicate task/seed pair')
        expected=sorted(pairs);random.Random(20261107).shuffle(expected)
        order=[False]*(len(pairs)//2)+[True]*(len(pairs)//2);random.Random(20261108).shuffle(order)
        require(pairs==expected and orientations==order,'prepared order differs from frozen RNG order')
        for row in rows+boot['diagnostics']:
            require(row['arm']==('lane' if physical=='L' else 'plain'),'native arm differs')
            require(row['execution_policy']=={'TF_DSV41_COMPUTE_LANE':str(int(physical=='L'))},'lane differs')
            require(wire(row['body_template'])==row['wire'] and hashlib.sha256(row['wire'].encode()).hexdigest()==row['wire_sha256'],'request bytes differ')
            require(('tools' in row['body_template'])==(physical=='L'),'declaration mismatch')
        require([(r['id'],str(r['reference'])) for r in boot['diagnostics']]==[('python-17x19','323'),('python-sum11','55')],'fixed diagnostics')
        for row in rows:
            require(row['seed'] in (101,102) and row['suite']=='compute_dl','suite/seed')
            key=[row['study_arm'],*row['key']]
            require(tuple(key) not in seen,'duplicate logical key');seen.add(tuple(key))
            schedule.append(dict(key=key,boot=boot['boot'],wire_sha256=row['wire_sha256'],task_sha256=row['task_sha256']))
    require(len(schedule)==360 and len({tuple(r['key']) for r in schedule})==360,'exact360 keys')
    ids={r['id'] for r in boots[0]['requests']}
    require(ids=={f'compute-format-v1-{f}-{i:03d}' for f in ('integer','modular','division','dp','graph','aggregation') for i in range(1,11)},'sixty public tasks')
    require({r['id'] for r in boots[1]['requests']}=={i for i in ids if i.split('-')[3] in ('dp','graph','aggregation')},'P family grid')
    return schedule
