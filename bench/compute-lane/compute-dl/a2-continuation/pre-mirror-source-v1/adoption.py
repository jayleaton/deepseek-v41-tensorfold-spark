"""Read-only, hash-bound A1 settlement and exact original P suffix contract."""
import hashlib
import json
from pathlib import Path
import a1_budget

A1_PREFIX=a1_budget.RUN_PREFIX
A2_SHA='8eaa9575592f519b504d08022d400602e4b552874c8424b02c59c34f4af174e2'
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
local=a1_budget.local
canonical=lambda p:str(p).replace("<rank1-home>/","<rank0-home>/",1)
require=a1_budget.require

def validate(root, budget_path, budget_sha, audit_path, audit_sha):
    root=Path(root)
    require(local(budget_path)==root/A1_PREFIX/'budget.json','must use actual A1v3 ledger')
    require(local(audit_path)==root/'a1-continuation/final-audit-v3/audit.json','must use final A1 audit')
    prior=a1_budget.bound_json(budget_path,budget_sha)
    cfg=load(root/A1_PREFIX/'campaign-v1.json')
    # Old schema3 validation MUST retain old 18000/15900 constants.
    old=a1_budget.HeldBudget(local(budget_path),a1_budget.A1_SHA,'A1-plain',**a1_budget.campaign_kwargs(cfg))
    require(old.data==prior and prior['active'] is None,'A1 not closed')
    a=a1_budget.bound_json(audit_path,audit_sha)
    require(a['qualified'] is True and a['gaps']==[] and a['spec_sha256']==a1_budget.A1_SHA
            and a['parent_spec_sha256']==a1_budget.BASE_SHA and a['scheduled']==720
            and a['budget']==prior,'qualified settled A1 audit required')
    pins={str(local(p)):h for p,h in a["provenance"]["input_file_sha256"].items()};require(bool(pins),"audit provenance missing")
    for path,digest in pins.items():a1_budget.bound_json(path,digest) if str(path).endswith('.json') else require(sha(local(path))==digest,'A1 audit input drift: '+path)
    require(pins.get(str(local(budget_path)))==budget_sha,'audit must bind actual closed ledger')
    require(pins.get(str(root/A1_PREFIX/'campaign-v1.json'))==sha(root/A1_PREFIX/'campaign-v1.json'),'A1 campaign audit binding missing')
    files={canonical(local(audit_path)):audit_sha}
    directory=local(audit_path).parent
    values={}
    for name in ('audited-rows.jsonl','diagnostics.jsonl','presentations.jsonl'):
        p=directory/name;files[canonical(p)]=sha(p)
        values[name]=[json.loads(l) for l in p.read_text().splitlines() if l]
    rows=values['audited-rows.jsonl']
    require(a['observed']==len(rows) and a['missing']==720-len(rows),'audit rows/count mismatch')
    # Settlement requires both actual turn logs and normal verified restoration.
    require([t['turn'] for t in a['turns']]==['A1-resume','A1-plain'],'both A1 turns must settle')
    for turn in a['turns']:
        require(turn['verdict'] in ('QUALIFIED','INCOMPLETE') and turn['held'] is True
                and turn.get('budget_event') is not None,'unsettled A1 turn')
        sp=root/'a1-continuation/runs'/(A1_PREFIX+'-'+turn['turn'])/'steps.jsonl'
        require(pins.get(str(sp))==sha(sp),'settled steps not audit bound')
        require(pins.get(str(local(turn['restore_log'])))==turn['restore_log_sha256']==sha(local(turn['restore_log'])),'actual restore not audit bound')
    # Audit must cover every extant attempt, including policy/error outcomes.
    attempt_dir=root/'a1-continuation/attempted'/A1_PREFIX
    require(attempt_dir.is_dir(),'A1 attempt inventory absent')
    actual={str(p):sha(p) for p in attempt_dir.glob('*.json')}
    audited={p:h for p,h in pins.items() if Path(p).parent==attempt_dir}
    require(actual==audited,'unreconciled/new A1 attempt markers')
    return prior,a,rows,files

def suffix(originals,rows):
    """No outcome selection, inferred counts, retries, reordering or new keys."""
    from runtime_stage import plain, CLIENT_ARM
    dl={(b['arm'],*r['key']) for b in originals.values() for r in b['requests']}
    p=[plain(r) for name in ('t01-D','t02-D') for r in originals[name]['requests']]
    inverse={v:k for k,v in CLIENT_ARM.items()}
    keys=[(inverse[r['arm']],r['suite'],r['id'],r['seed']) for r in rows]
    require(len(keys)==len(set(keys)),'duplicate audited keys')
    observed_dl={k for k in keys if k[0]!='P'}
    observed_p=[k for k in keys if k[0]=='P']
    require(observed_dl==dl and len(dl)==480,'all original D/L must be observed')
    full=[('P',*r['key']) for r in p]
    require(len(full)==len(set(full))==240 and observed_p==full[:len(observed_p)],'durable P is not exact original prefix')
    require(set(keys)<=dl|set(full),'foreign audit key')
    require(len(observed_p)<len(p),'no P suffix remains')
    return p[len(observed_p):],set(keys),p
