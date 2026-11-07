"""Read prep's sealed stage result without importing mutable running modules."""
import copy
import hashlib
import json
from pathlib import Path
from protocol import bound, require, sha, local_path, QUALIFIED_SPEC_SHA256
from manifest import wire
HERE=Path(__file__).resolve().parent

def prepared(path,expected,config_path,config_sha):
    bound(path,expected);bound(config_path,config_sha)
    prep=json.loads(local_path(path).read_text());config=json.loads(local_path(config_path).read_text())
    require(prep['spec_sha256']==config['spec_sha256']==QUALIFIED_SPEC_SHA256,'prep protocol')
    pins={str(Path(path).resolve()):expected,str(Path(config_path).resolve()):config_sha}
    for field in ('tasks','generation_audit','settings','tools'):
        p=config[field+'_path'];h=config[field+'_sha256'];bound(p,h);pins[p]=h
    require(prep['tasks_sha256']==config['tasks_sha256'],'prepared tasks differ')
    audit=json.loads(local_path(config['generation_audit_path']).read_text())
    for name,p in config['references'].items():
        h=audit['files_sha256'][name];bound(p,h);pins[p]=h
    tasks=[json.loads(l) for l in local_path(config['tasks_path']).read_text().splitlines()]
    byid={t['id']:t for t in tasks};require(len(byid)==len(tasks)==60,'public60 grid')
    # Original qualified fixed diagnostics, only declaration removed for P.
    compute=HERE.parent.parent/'compute-dl'
    old=compute/'stdbench-compute-dl-20261105-v1/frozen-v1'
    index=json.loads((old/'manifest.json').read_text());pins[str(old/'manifest.json')]=sha(old/'manifest.json')
    source=next(b for b in index['boots'] if b['boot']=='t01-L')
    source_path=old/source['file'];bound(source_path,source['sha256']);pins[str(source_path)]=source['sha256']
    diags=json.loads(source_path.read_text())['diagnostics']
    tools=json.loads(local_path(config['tools_path']).read_text())
    require(all(r['body_template']['tools']==tools for r in diags),'canonical declaration differs from qualified baseline')
    settings=json.loads(local_path(config['settings_path']).read_text())
    boots=[];inputs={}
    for physical in ('L','P'):
        rows=[]
        for original in prep['boots'][physical]:
            row=copy.deepcopy(original);task=byid[row['key'][1]];arm=row['logical_arm']
            require(row['request_identity']==[*row['key'],arm] and arm in (physical+'-old',physical+'-new'),'prepared logical identity')
            body=row['body_template'];base={k:v for k,v in body.items() if k not in ('messages','seed','tools')}
            require(base==settings and body['seed']==row['key'][2] and row['key'][0]==task['suite'],'sampling differs')
            require((body.get('tools')==tools) if physical=='L' else 'tools' not in body,'canonical tools differ')
            row.update(suite=task['suite'],id=task['id'],seed=row['key'][2],study_arm=arm,
                       prompt_key=list(row['request_identity']),task_sha256=hashlib.sha256(json.dumps(task,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest())
            rows.append(row)
            inputs[tuple(row['prompt_key'])]=dict(key=row['prompt_key'],messages=body['messages'])
        diagnostics=copy.deepcopy(diags)
        for row in diagnostics:
            row['study_arm']=None;row['prompt_key']=[*row['key'],None]
            if physical=='P':
                row['arm']='plain';row['body_template'].pop('tools')
                row['execution_policy']={'TF_DSV41_COMPUTE_LANE':'0'}
                row['wire']=wire(row['body_template']);row['wire_sha256']=hashlib.sha256(row['wire'].encode()).hexdigest()
            row.update(suite=row['key'][0],id=row['key'][1],seed=row['key'][2])
            inputs[tuple(row['prompt_key'])]=dict(key=row['prompt_key'],messages=row['body_template']['messages'])
        boots.append(dict(boot='format-'+physical,arm=physical,requests=rows,diagnostics=diagnostics))
    # Exact suffix/neutrality checks use prep's frozen wording constants, with no imports.
    import importlib.util
    spec=importlib.util.spec_from_file_location('_format_prep_wording',HERE.parent/'prep/common.py')
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    for boot in boots:
        for row in boot['requests']:
            task=byid[row['id']];old=common.OLD[task['family']];new=common.NEW[task['family']]
            prompt=task['prompt'];require(prompt.endswith(old),'old suffix differs')
            if row['study_arm'].endswith('-new'):prompt=prompt[:-len(old)]+new
            require(row['body_template']['messages']==[dict(role='user',content=prompt)],'task message neutrality differs')
    for p in (HERE.parent/'prep').glob('*.py'):pins[str(p)]=sha(p)
    require(len(inputs)==362,'360 logical prompts plus2 diagnostics')
    return prep,boots,list(inputs.values()),pins
