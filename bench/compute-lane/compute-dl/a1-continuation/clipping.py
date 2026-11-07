#!/usr/bin/env python3
"""Thin A1 adapter: native cap plus unchanged native/Jail provenance audit."""
import importlib.util
from pathlib import Path
from protocol import local_path

def policy():
    path=Path(__file__).resolve().parent.parent/'a1-prep/reference_policy.py'
    spec=importlib.util.spec_from_file_location('_a1_reference_policy',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def make_capper(config):
    p=policy(); values=config['cap_policy']
    return p.NativeCap(*(local_path(values[k]) for k in ('binary','tokenizer','source','receipt')))

def compare(config,call,result):
    p=policy(); capper=make_capper(config)
    matched=p.compare_pairs(capper,[(call,result)])[0]
    metadata=p.presentation(capper,call,result)
    if metadata['qualified'] is not matched: raise ValueError('native policy comparison inconsistency')
    return matched,metadata
