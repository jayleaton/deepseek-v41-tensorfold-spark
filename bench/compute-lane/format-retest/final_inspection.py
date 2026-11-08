#!/usr/bin/env python3
"""Read the actual published artifact; verify ledger coverage and sealed inputs."""
import csv
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def main():
    result=load(HERE/'results-v1/report.json');raw=load(HERE/'audit-v2/raw-audit.json')
    cfg=load(HERE/'stdbench-format-20261107-v1/campaign-v1.json')
    assert result['spec_sha256']==raw['spec_sha256']==cfg['spec_sha256']=='b47ede697eed3a38d244ad5ff7447690732bd12945ff4592554d45bc2d8527ec'
    assert all(sha(p)==h for p,h in cfg['code_sha256'].items())
    assert raw['qualified'] and not raw['gaps'] and result['qualification']['qualified']
    assert result['observed']==len(result['ledger'])==360 and not result['missing_keys']
    with (HERE/'results-v1/per-observation.csv').open() as f:assert len(list(csv.DictReader(f)))==360
    arms={}
    for arm,data in result['arms'].items():
        ledger=[r for r in result['ledger'] if r['study_arm']==arm]
        assert len(ledger)==data['observed']==data['scheduled'] and all(r['observed'] for r in ledger)
        assert sum(r['strict_correct'] for r in ledger)==data['strict_solves']
        assert sum(r['format_failure'] for r in ledger)==data['format_failures']
        compact={k:data[k] for k in ('scheduled','strict_solves','format_failures','typed_wrong_values','caps','deadlines')}
        compact['means']={m:None if c['sum'] is None else c['sum']/data['scheduled'] for m,c in data['costs'].items()}
        compact['calls']={m:c['sum'] for m,c in data['calls'].items()}
        arms[arm]=compact
    ledger=load(cfg['budget_ledger']);assert ledger['active'] is None and ledger['overrun_seconds']==0
    assert not Path(cfg['go_file']).exists()
    output=dict(passed=True,status=result['status'],collection=result['collection_status'],observed=360,
        arms=arms,pairs={k:{f:len(v) for f,v in p.items()} for k,p in result['pairs'].items()},
        primary=result['bootstrap']['primary_L_wording'],secondary=result['bootstrap']['secondary_Lnew_minus_Pnew'],
        native_presentations=raw['native_presentations'],budget_seconds=ledger['held_seconds'],
        source_sha256=sha(__file__),report_sha256=sha(HERE/'results-v1/report.json'),
        original_auditor_sha256=sha(HERE/'audit_collection.py'),post_auditor_sha256=sha(HERE/'audit_collection_post_v2.py'))
    with (HERE/'final-inspection.json').open('x') as f:json.dump(output,f,indent=2)
    print(json.dumps(output))
if __name__=='__main__':main()
