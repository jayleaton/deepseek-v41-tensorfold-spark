#!/usr/bin/env python3
"""Prospective turn boundary. No transport/queue/engine APIs before A1 freeze."""
import argparse
import json
from pathlib import Path
import p_check
HERE=Path(__file__).resolve().parent
QUALIFIED_A1_SHA256=None

def preflight():
    p_check.check_stage(HERE/'staged-P')
    clip=json.loads((HERE/'clip-check-receipt.json').read_text())
    assert clip['passed'] is True and clip['GPU_requests']==0
    ranks=[json.loads((HERE/f'env-check-r{rank}.json').read_text()) for rank in (0,1)]
    assert all(r['passed'] is True and r['new_namespace_clear'] is True for r in ranks)
    return {'passed':True,'stage':'CPU_PREP_ONLY','spec_sha256':QUALIFIED_A1_SHA256,
            'collection_authorized':False,'P_requests':240,'remaining_original':138,
            'layout':'ORDER_NOT_FROZEN','GPU_requests':0}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    if not args.preflight_only:
        raise SystemExit('SPEC_PENDING: A1 SHA, final completion/P layout and qualified collector required; no turn started')
    print(json.dumps(preflight()))
if __name__=='__main__': main()
