"""Exclusive derivative rank/config staging from prep's frozen request arrays."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parent.parent/'decl-control'))
from generate_envs3 import IMAGE, IMAGE_IDS, REVISION, STRIP, ZERO_FLAGS, env_text, filtered, read_reference, reject_banned
from protocol import PREFIX, QUALIFIED_SPEC_SHA256 as SPEC_SHA256, BASE_SHA, A1_TURNS, require, sha
from manifest import validate_boots, wire
TOKENIZER_SHA256='c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
CLIENT_ARM={'L':'lane','P':'plain'}
def digest(data):return hashlib.sha256(data).hexdigest()
def json_text(value):return json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
def load_text(text):return json.loads(text)
def spark_root(rank):
    require(type(rank) is int and rank in (0,1),'invalid rank')
    return Path(f'<rank{rank}-work>/stdbench/format-retest')
def prefix_root(rank):return spark_root(rank)/PREFIX
def new_directory(path):
    path=Path(path).absolute()
    require('..' not in path.parts and not any(p.is_symlink() for p in (path,*path.parents)),'unsafe output path')
    path.mkdir(exist_ok=False);return path
def write_new(root,name,data):
    name=Path(name);require(not name.is_absolute() and '..' not in name.parts,'unsafe relative output')
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as out:out.write(data)
def rank_artifacts(base, references, boot, arm):
    manifest = copy.deepcopy(base)
    record = copy.deepcopy(base['arms']['D'])
    record.update(engine_sha=REVISION, description=f'compute D/L {boot}; lane={int(arm == "L")}')
    record.pop('alias_of', None)
    manifest.update(active_arms=['D'], arms={'D': record})
    manifest.pop('decl_control_a1', None)
    metas, settings, configs = [], [], {}
    off = {f'TF_DSV41_{key}': '0' for key in ZERO_FLAGS}
    for rank in (0, 1):
        root = prefix_root(rank)
        env = {k: v for k, v in references[rank].items()
               if not any(word in k for word in STRIP)
               and not k.startswith(('GUARD', 'SPARK'))}
        env.update(IMAGE=IMAGE, ZIG_COMMIT=REVISION, CONTAINER_PREFIX=f'{PREFIX}-{boot}',
                   STATE_DIR=str(root / 'state' / boot), CACHE_DIR=str(root / 'cache' / boot))
        # These are already set by the pinned native launch adapter. Make its
        # rank identity/kernel defaults explicit in the complete effective seal.
        env.update(TF_TP_WORLD='2',TF_TP_RANK=str(rank),TF_TP_DEVICE='0',
                   TF_TP_KERNEL_IMAGE='/opt/tensorfold/share/tp_mailbox.fatbin')
        env.update(off)
        env.update(TF_DSV41_COMPUTE_LANE=str(int(arm == 'L')), TF_DSV41_RUNNER_FORMS='fence,repl,inline')
        env.setdefault('NCCL_CUMEM_ENABLE', '0')
        env.setdefault('NCCL_CUMEM_HOST_ENABLE', '0')
        reject_banned(env)
        if 'IMR_PUBLISH' in str(env):
            raise ValueError('publication flag forbidden')
        cfg = root / 'configs' / f'{boot}-r{rank}.env'
        stage_cfg = f'rank{rank}/configs/{boot}-r{rank}.env'
        configs[stage_cfg] = env_text(env).encode()
        values = {k: v for k, v in env.items() if k.startswith(('TF_', 'GLM53_TF'))}
        settings.append(values)
        record['ranks'][rank].update(config=str(cfg), image_id=IMAGE_IDS[rank], settings=values)
        metas.append({'rank': rank, 'root': str(root), 'config': str(cfg), 'stage_config': stage_cfg,
                      'container_name': f'{PREFIX}-{boot}-r{rank}', 'state_dir': env['STATE_DIR'],
                      'cache_dir': env['CACHE_DIR'], 'effective_env': filtered(env),
                      'table_source': str(Path(record['ranks'][rank]['recipe_dir']) / 'config/pfdense-table.json'),
                      'expected_off': dict(off), 'guard_source': '/etc/spark/thermal.env'})
    record['settings'] = {k: v for k, v in settings[0].items() if settings[1].get(k) == v}
    record['native_settings'] = {}
    manifest['phase2'] = {'schema': 1, 'prefix': PREFIX, 'boot': boot, 'arm': arm,
                          'lane': int(arm == 'L'), 'runner_forms': 'fence,repl,inline',
                          'guard_source': '/etc/spark/thermal.env', 'ranks': metas}
    for rank in (0, 1):
        local = copy.deepcopy(manifest)
        local['compute_dl'] = {'status': 'STAGED_CPU_GATES_PENDING',
                               'campaign': str(prefix_root(rank) / 'campaign-v1.json'),
                               'spec_sha256': SPEC_SHA256, 'boot_min_gib': 114}
        configs[f'rank{rank}/boots/{boot}/manifest.json'] = json_text(local).encode()
    return configs


def stage(prep_manifest,prep_sha256,out=None,*,prep_config,prep_config_sha256,references=None,qualification=None):
    """Parent CPU only. Consumes prep's manifest and never creates tasks or GO."""
    from prep_bridge import prepared
    prep,boots,inputs,prep_pins=prepared(prep_manifest,prep_sha256,prep_config,prep_config_sha256)
    schedule=validate_boots(boots)
    references=Path(references or HERE.parent.parent/'phase2/references')
    base=load_text((references/'base-arms.json').read_text())
    refs=[read_reference(references/f'base-r{r}.env') for r in (0,1)]
    payloads={};entries=[]
    for b in boots:
        data=json_text(b).encode();name='boots/'+b['boot']+'.json'
        payloads['frozen-v1/'+name]=data
        entries.append(dict(boot=b['boot'],arm=b['arm'],file=name,sha256=digest(data)))
        payloads.update(rank_artifacts(base,refs,b['boot'],b['arm']))
    index=dict(schema=1,prefix=PREFIX,spec_sha256=SPEC_SHA256,turns=A1_TURNS,boots=entries,
               scheduled_replies=360,schedule=schedule,prep_manifest_sha256=prep_sha256,collection_qualified=False)
    payloads['frozen-v1/manifest.json']=json_text(index).encode()
    payloads['prompt-inputs.json']=json_text(inputs).encode()
    target=prefix_root(0);compute=HERE.parent.parent/'compute-dl'
    original=load_text((compute/'stdbench-compute-dl-20261105-v1/campaign-v1.json').read_text())
    cfg={k:original[k] for k in ('model_alias','boot_min_gib','image_ids','diagnostic_tensorfold','tokenizer','tokenizer_sha256','transport_contract','g_restore_step')}
    cfg.update(prefix=PREFIX,spec=str(spark_root(0)/'runtime/T5-FORMAT-RETEST-SPEC.md'),spec_sha256=SPEC_SHA256,
        base_spec=str(compute/'T5-COMPUTE-DL-SPEC.md'),base_spec_sha256=BASE_SHA,arms=['L','P'],
        layout=dict(turns=A1_TURNS,boots=entries),freeze=str(target/'frozen-v1'),freeze_sha256=digest(payloads['frozen-v1/manifest.json']),
        cpu_receipt=str(target/'cpu-seal/verification.json'),cpu_receipt_sha256=None,
        cpu_prompt_arrays=str(target/'prompt-arrays.jsonl'),cpu_prompt_arrays_sha256=None,
        prep_manifest=str(Path(prep_manifest).resolve()),prep_manifest_sha256=prep_sha256,
        prep_config=str(Path(prep_config).resolve()),prep_config_sha256=prep_config_sha256,
        qualification=qualification or {},go_file=str(target/'GO'),budget_ledger=str(target/'budget.json'),
        budget=dict(limit_seconds=6300,admission_seconds=5880),timeouts=dict(client_wait_s=360,gpu_parent_s=7200,remote_control_s=7200,turn_s=14400),
        cap_policy=dict(binary=str(compute/'a1-prep/zig-out/bin/native-cap'),tokenizer=str(HERE.parent.parent/'phase2/references/tokenizer.json'),source=str(HERE.parent.parent/'decl-control/dispatch-high'),receipt=str(compute/'a1-prep/native-build-receipt.json')),status='CPU_GATES_PENDING',collection_qualified=False)
    # Seal references and only operational dependencies, never transplant A2 history.
    sources=[*HERE.glob('*.py'),*HERE.glob('*.sh'),HERE/'T5-FORMAT-RETEST-SPEC.md',references/'base-arms.json',references/'base-r0.env',references/'base-r1.env']
    sources += [HERE.parent.parent/'decl-control'/n for n in ('client3.py','bench.py','reconciliation.py','generate_envs3.py','jail.py') if (HERE.parent.parent/'decl-control'/n).exists()]
    sources += list((compute/'a1-prep').glob('reference_policy.py'))
    sources += [compute/'a1-prep/audit_lifecycle.py',compute/'collect_report.py',compute/'analysis.py',compute/'grade.py']
    sources += [HERE.parent.parent/'phase2'/n for n in ('client.py','native_frames.py','jail.py','jail_worker.py','audit_records.py')]
    sources += [HERE.parent.parent/'kit'/n for n in ('bench.py',)]
    sources += [Path('<rank0-work>/dsv41-bench/ops')/n for n in ('rank.py','guard.py','mem_guard.py','node.sh')]
    cfg['code_sha256']={str(p):sha(p) for p in sources};cfg['code_sha256'].update(prep_pins)
    cfg['rank_manifests']={b['boot']:{str(r):str(prefix_root(r)/f'rank{r}/boots'/b['boot']/'manifest.json') for r in (0,1)} for b in entries}
    cfg['rank_manifest_sha256']={b['boot']:{str(r):digest(payloads[f'rank{r}/boots/{b["boot"]}/manifest.json']) for r in (0,1)} for b in entries}
    payloads['stage-campaign.json']=json_text(cfg).encode()
    result=dict(prefix=PREFIX,spec_sha256=SPEC_SHA256,files={n:digest(v) for n,v in payloads.items()})
    destination=new_directory(out or HERE/'runtime-stage-v1')
    for n,v in payloads.items():write_new(destination,n,v)
    write_new(destination,'stage-manifest.json',json_text(result).encode())
    return result

def materialize(stage_root,rank,destination=None):
    spark_root(rank);stage_root=Path(stage_root);index=load_text((stage_root/'stage-manifest.json').read_text())
    require(index['prefix']==PREFIX and index['spec_sha256']==SPEC_SHA256,'stage identity')
    contents={}
    for name,h in index['files'].items():
        p=stage_root/name;p.resolve().relative_to(stage_root.resolve());data=p.read_bytes();require(digest(data)==h,'stage drift')
        if name.startswith(f'rank{rank}/configs/'):target=name.split('/',1)[1]
        elif name.startswith('rank') and '/boots/' in name:
            if rank==1 and not name.startswith('rank1/'):continue
            target=name
        elif name.startswith('frozen-v1/') or name in ('stage-campaign.json','prompt-inputs.json'):target=name
        else:continue
        contents[target]=data
    dest=new_directory(destination or prefix_root(rank))
    for n,v in contents.items():write_new(dest,n,v)
    receipt=dict(rank=rank,spec_sha256=SPEC_SHA256,stage_manifest_sha256=sha(stage_root/'stage-manifest.json'),files={n:digest(v) for n,v in contents.items()})
    write_new(dest,'materialized.json',json_text(receipt).encode());return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('stage');a.add_argument('prep_manifest');a.add_argument('prep_sha256');a.add_argument('--out');a.add_argument('--prep-config',required=True);a.add_argument('--prep-config-sha256',required=True);a.add_argument('--qualification',help='JSON mapping of actual gate receipt paths/hashes')
    a=s.add_parser('materialize');a.add_argument('stage_root');a.add_argument('rank',type=int);a.add_argument('--destination')
    a=p.parse_args()
    print(json_text(stage(a.prep_manifest,a.prep_sha256,a.out,prep_config=a.prep_config,prep_config_sha256=a.prep_config_sha256,qualification=json.loads(Path(a.qualification).read_text()) if a.qualification else None) if a.command=='stage' else materialize(a.stage_root,a.rank,a.destination)))
