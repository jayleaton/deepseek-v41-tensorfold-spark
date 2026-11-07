#!/usr/bin/env python3
"""Pre-GO versioned seal update for reviewed reporting-only fixes."""
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'a1-runtime'))
import protocol
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())

def main():
    target=ROOT/protocol.PREFIX
    assert not (target/'GO').exists()
    cfg=load(target/'campaign-v1.json');old=load(cfg['cpu_receipt'])
    assert sha(cfg['cpu_receipt'])==cfg['cpu_receipt_sha256']
    assert old['campaign_sha256']==protocol.campaign_digest(cfg)
    updated={str(HERE/'audit_collection.py'),str(HERE/'audit_lifecycle.py'),str(HERE/'lifecycle_check.py'),str(HERE/'reseal_cpu.py')}
    for p,h in cfg['code_sha256'].items():
        if p not in updated:assert sha(p)==h,'unreviewed source drift: '+p
    lifecycle=load(HERE/'lifecycle-check-receipt.json');assert lifecycle['passed'] is True
    for p,h in lifecycle['source_sha256'].items():assert sha(p)==h
    for p in updated:cfg['code_sha256'][p]=sha(p)
    old['code_sha256']=cfg['code_sha256']
    old['preparation_file_sha256'][str(HERE/'lifecycle-check-receipt.json')]=sha(HERE/'lifecycle-check-receipt.json')
    old['budget_stop_audit_controls']=True
    old['supersedes_pre_GO_receipt_sha256']=cfg['cpu_receipt_sha256']
    cfg['cpu_receipt']=str(target/'cpu-seal/verification-v2.json')
    old['campaign_sha256']=protocol.campaign_digest(cfg)
    with Path(cfg['cpu_receipt']).open('x') as f:json.dump(old,f,indent=2)
    cfg['cpu_receipt_sha256']=sha(cfg['cpu_receipt'])
    with (target/'campaign-before-reporting-fix.json').open('x') as f:f.write((target/'campaign-v1.json').read_text())
    with (target/'campaign-resealed.json').open('x') as f:json.dump(cfg,f,indent=2)
    (target/'campaign-resealed.json').replace(target/'campaign-v1.json')
    protocol.require_collection_seal(target/'campaign-v1.json')
    print(json.dumps({'passed':True,'sealed_sources':len(cfg['code_sha256']),'GO_created':False,'GPU_requests':0,
                      'campaign_sha256':sha(target/'campaign-v1.json')}))
if __name__=='__main__':main()
