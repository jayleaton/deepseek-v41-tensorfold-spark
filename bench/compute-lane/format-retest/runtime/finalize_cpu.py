"""Seal completed CPU evidence and initialize fresh zero budget; never create GO."""
import argparse
import json
from pathlib import Path
import protocol
from budget import initialize
from qualification import validate
HERE=Path(__file__).resolve().parent

def finalize(candidate):
    cfg=protocol.require_staging_ready(candidate);target=protocol.local_path(Path(cfg['budget_ledger']).parent)
    protocol.require(not (target/'GO').exists(),'GO must remain absent')
    cfg['cpu_prompt_arrays_sha256']=protocol.sha(protocol.local_path(cfg['cpu_prompt_arrays']))
    for name in protocol.GATES:protocol.receipt(cfg,name)
    validate(cfg)
    cfg.update(status='CPU_QUALIFIED_PARENT_GO_PENDING',collection_qualified=True)
    proof=dict(passed=True,spec_sha256=protocol.QUALIFIED_SPEC_SHA256,campaign_sha256=protocol.campaign_digest(cfg),
               code_sha256=cfg['code_sha256'],qualification=cfg['qualification'],freeze_sha256=cfg['freeze_sha256'],
               cpu_prompt_arrays_sha256=cfg['cpu_prompt_arrays_sha256'],prep_manifest_sha256=cfg['prep_manifest_sha256'])
    receipt=protocol.local_path(cfg['cpu_receipt']);receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x') as out:json.dump(proof,out,indent=2)
    cfg['cpu_receipt_sha256']=protocol.sha(receipt)
    initialize(target/'budget.json')
    with (target/'sealed-campaign.json').open('x') as out:json.dump(cfg,out,indent=2)
    # Parent installs this exact campaign to both candidate paths, then verifies
    # both seals with GO absent. No live or candidate configuration is overwritten.
    return cfg
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('candidate');a=p.parse_args();finalize(a.candidate)
