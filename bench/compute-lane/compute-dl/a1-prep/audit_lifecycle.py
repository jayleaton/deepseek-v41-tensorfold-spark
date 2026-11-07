#!/usr/bin/env python3
"""Pure classification of settled turn evidence, including honest budget refusal."""
def require(ok,why):
    if not ok:raise ValueError(why)
def classify(steps):
    terminal=[s for s in steps if s['step']=='turn']
    require(len(terminal)==1,'settled turn log missing')
    verdict=terminal[0]['verdict']
    incomplete=[s for s in steps if s['step']=='collection' and s['verdict']=='INCOMPLETE']
    budget_stop=verdict=='INCOMPLETE' and bool(incomplete) and all('cutoff' in s.get('reason','').lower() for s in incomplete)
    ready={s['boot'] for s in steps if s['step']=='boot' and s['verdict']=='READY_EFFECTIVE_ENV_VERIFIED'}
    diagnostic={s['boot'] for s in steps if s['step']=='diagnostics' and s['verdict']=='QUALIFIED'}
    drained={s['boot'] for s in steps if s['step']=='owned-containers' and s['verdict']=='DRAIN_VERIFIED'}
    charges=[s for s in steps if s['step']=='budget' and s['verdict']=='RECORDED']
    restores=[s for s in steps if s['step']=='G-restore' and s['verdict']=='SERVING_VERIFIED']
    stops=[s for s in steps if s['step']=='G-stop' and s['verdict']=='REQUESTED']
    if not charges:
        require(budget_stop and not ready and not diagnostic and not restores and not stops,
                'unheld refusal lacks budget-only lifecycle evidence')
        require(not any(s['step'] in ('response','gpu-locks') or s['step']=='queue' and s['verdict']=='APPENDED_AT_TAIL' for s in steps),
                'unheld refusal has attempted GPU/queue work')
        return dict(verdict=verdict,budget_stop=True,held=False,ready=ready,diagnostic=diagnostic,drained=drained,budget_event=None,restore=None)
    require(len(charges)==1,'settled held-time log ambiguous')
    if stops:require(len(stops)==len(restores)==1,'stopped G lacks unique restoration proof')
    else:
        require(not restores and not ready and not diagnostic and not any(s['step']=='response' for s in steps),
                'G-unchanged interval contains serving activity')
        require(any(s['step']=='queue' and s['verdict']=='OWN_ENTRY_WITHDRAWN_G_UNCHANGED' for s in steps),
                'G-unchanged interval lacks withdrawal evidence')
    return dict(verdict=verdict,budget_stop=budget_stop,held=True,ready=ready,diagnostic=diagnostic,
                drained=drained,budget_event=charges[0],restore=restores[0] if restores else None)

def allow_partial_boot(state,boot,observed,diagnostics,attempted):
    if boot not in state['ready']:
        require(state['budget_stop'] and observed==diagnostics==attempted==0 and boot in state['drained'],
                'unready boot contains attempts or lacks budget/drain evidence')
        return True
    if observed or attempted:
        require(diagnostics==2 and boot in state['diagnostic'],'dataset attempted before both diagnostics qualified')
        return False
    if diagnostics<2:
        require(state['budget_stop'],'partial boot diagnostics without explicit budget cutoff')
        return True
    require(boot in state['diagnostic'],'complete diagnostics lack qualification event')
    return False
