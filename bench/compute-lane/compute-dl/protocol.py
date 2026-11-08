#!/usr/bin/env python3
"""Prospective protocol boundary. No study generation/collection before freeze."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
# Intentionally unset. Adopt only the SHA steered by JJ after reading that spec,
# implementing its gates and qualifying their actual CPU evidence.
QUALIFIED_SPEC_SHA256='c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(value):
    config=dict(value) if isinstance(value,dict) else json.loads(Path(value).read_text())
    if not isinstance(config,dict): raise ValueError('configuration must be an object')
    return config


def require_bound_spec(value):
    config=read_config(value)
    expected=config.get('spec_sha256')
    if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):
        raise ValueError('SPEC_PENDING: externally frozen spec SHA required before task generation')
    if not config.get('spec') or sha(local_path(config['spec']))!=expected:
        raise ValueError('spec file absent or differs from supplied SHA')
    return config


def local_path(path):
    if str(ROOT).startswith('<rank1-home>/'):
        return Path(str(path).replace('<rank0-home>/','<rank1-home>/',1))
    return Path(path)


def require_staging_ready(value):
    """Read-only rank verification only; never sufficient for a launch."""
    config=require_bound_spec(value)
    if config['spec_sha256']!=QUALIFIED_SPEC_SHA256:
        raise ValueError('spec has not been qualified for this collector')
    if not re.fullmatch(r'stdbench-compute-dl-[A-Za-z0-9_.-]+',config.get('prefix','')):
        raise ValueError('invalid frozen study namespace')
    if config.get('arms')!=['D','L']:
        raise ValueError('frozen spec is D/L only')
    if config.get('transport_contract')!={'max_tokens':8192,'deadline_s':270}:
        raise ValueError('frozen transport contract differs')
    if config.get('g_restore_step')!={'format':'g-restore-v1','events':['SERVING','MODELS200','POPPED']}:
        raise ValueError('restoration evidence contract differs')
    turns=config.get('layout',{}).get('turns',[])
    if turns!=[{'name':'t01','boots':['t01-D','t01-L']},{'name':'t02','boots':['t02-L','t02-D']}]:
        raise ValueError('frozen two-turn arm order differs')
    if config.get('budget')!={'limit_seconds':18000,'admission_seconds':15900}:
        raise ValueError('frozen cumulative five-hour budget differs')
    ledger=Path(config['budget_ledger'])
    if ledger!=Path('<rank0-work>/stdbench/compute-dl')/config['prefix']/'budget.json':
        raise ValueError('budget ledger must be the single rank0-host study ledger')
    for name,expected in config.get('code_sha256',{}).items():
        if sha(local_path(name))!=expected: raise ValueError('sealed source/data drift')
    return config


def require_collection_ready(value):
    config=require_staging_ready(value)
    path=local_path(config['cpu_receipt'])
    if sha(path)!=config['cpu_receipt_sha256']:
        raise ValueError('CPU qualification receipt drift')
    proof=json.loads(path.read_text())
    if proof.get('passed') is not True or proof.get('spec_sha256')!=QUALIFIED_SPEC_SHA256:
        raise ValueError('CPU_GATES_PENDING: qualified frozen spec evidence required')
    if str(ROOT).startswith('<rank0-home>/'):
        for name,expected in proof.get('preparation_file_sha256',{}).items():
            if sha(name)!=expected:raise ValueError('preparation source/evidence drift')
    required={'generated_tasks':120,'scheduled_replies':480,'disjointness':True,
              'independent_references':True,'prompt_pairs':242,'prior_dispatch':True,
              'rank_checks':8,'baseline_rank_checks':2,'preflight':True,'fake_checks':True,
              'grader_analysis_pinned':True,'pre_generation_pin':True}
    if proof.get('gates')!=required: raise ValueError('CPU_GATES_PENDING: incomplete evidence gate set')
    if proof.get('code_sha256')!=config['code_sha256'] or len(config['code_sha256'])<25:
        raise ValueError('source/data seal differs from CPU qualification')
    if (proof.get('freeze_sha256')!=config['freeze_sha256']
            or sha(local_path(config['freeze'])/'manifest.json')!=config['freeze_sha256']
            or proof.get('cpu_prompt_arrays_sha256')!=config['cpu_prompt_arrays_sha256']
            or sha(local_path(config['cpu_prompt_arrays']))!=config['cpu_prompt_arrays_sha256']):
        raise ValueError('request/prompt seal differs from CPU qualification')
    if local_path(config['go_file']).read_text().strip()!=QUALIFIED_SPEC_SHA256:
        raise ValueError('external GO binding absent or changed')
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign')
    p.add_argument('--check',action='store_true')
    a=p.parse_args()
    require_collection_ready(a.campaign)


if __name__=='__main__': main()
