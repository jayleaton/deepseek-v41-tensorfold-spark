#!/usr/bin/env python3
"""Render actual frozen-analysis/evidence artifacts; never regrade or rerun inference."""

import sys as _archive_sys
from collections import Counter
import datetime
import hashlib
import json
import math
import py_compile
import statistics
import tempfile
from pathlib import Path

# Export adapter: help must not execute this historical host script.
if __name__ == "__main__" and "--help" in _archive_sys.argv[1:]:
    print(__doc__ or "Historical study script; requires its original inputs and host prerequisites.")
    raise SystemExit(0)
root=Path(__file__).resolve().parent
out=root/'post-v1'
export=root/'report-v2'
export.mkdir(exist_ok=False)
load=lambda p: json.loads(Path(p).read_text())
readrows=lambda p: [json.loads(x) for x in Path(p).read_text().splitlines()]
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=load(out/'analysis.json');activation=load(out/'activation.json');audit=load(out/'native-audit.json');env=load(out/'environment-audit.json')
assert audit['qualified'] and env['qualified']
assert report['spec_sha256']==audit['spec_sha256']
freeze=load(root/'frozen-v3/manifest.json')
assert audit['freeze_manifest_sha256']==sha(root/'frozen-v3/manifest.json')==load(root/'cpu-final-v3/verification.json')['freeze_sha256']
with tempfile.TemporaryDirectory(prefix='stdbench-export-syntax-') as temp:
    for p in sorted(root.glob('*.py')):py_compile.compile(str(p),cfile=str(Path(temp)/(p.name+'c')),doraise=True)
rows=readrows(out/'all-provisional.jsonl')
boots=[];diags=[];inputs={}
for entry in freeze['boots']:
    boot=load(root/'frozen-v3'/entry['file']);name=boot['boot'];turn=name.split('-')[0]
    version='v1' if name=='pilot-plain' else 'v2'
    directory=root/'post-data-v1'/f'stdbench-20261105-{version}-{turn}'/name
    observations=readrows(directory/'responses.jsonl');diagnostics=readrows(directory/'diagnostics.jsonl')
    identity=lambda r:(r['suite'],r['id'],r['seed'])
    assert list(map(identity,observations))==list(map(identity,boot['requests'])), 'chronological request order drift'
    assert list(map(identity,diagnostics))==list(map(identity,boot['diagnostics'])), 'diagnostic order drift'
    boots.append((name,observations,diagnostics));diags+=diagnostics
    inputs[str(directory/'responses.jsonl')]=sha(directory/'responses.jsonl')
assert sum(len(b[1]) for b in boots)==368 and len(diags)==28
numeric=lambda x:type(x) in (int,float) and math.isfinite(x)
def stats(values):
    good=[x for x in values if numeric(x)]
    return {'observed_rows':len(good),'missing_rows':len(values)-len(good),'mean':statistics.mean(good) if good else None,'median':statistics.median(good) if good else None,'sum':sum(good) if good else None}
telemetry={}
for suite in ('gsm8k','math500','humaneval'):
    telemetry[suite]={}
    for arm in ('plain','lane'):
        selected=[r for r in rows if r['suite']==suite and r['arm']==arm]
        telemetry[suite][arm]={k:stats([(r.get('tensorfold') or {}).get(field) for r in selected]) for k,field in [('server_prefill_seconds','prefill_seconds'),('server_decode_seconds','seconds'),('ttft_seconds','time_to_first_token')]}
        telemetry[suite][arm]['prompt_tokens']=stats([(r.get('usage') or {}).get('prompt_tokens') for r in selected])
        telemetry[suite][arm]['reasoning_tokens']=stats([((r.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens') for r in selected])
        telemetry[suite][arm]['content_tokens']=None
        telemetry[suite][arm]['native_tool_calls']=sum(len(((r.get('tensorfold') or {}).get('compute') or {}).get('calls',[])) for r in selected)
diagnostic={}
for arm in ('plain','lane'):
    selected=[r for r in diags if r['arm']==arm]
    calls=[c for r in selected for c in (((r.get('tensorfold') or {}).get('compute') or {}).get('calls') or [])]
    diagnostic[arm]={'replies':len(selected),'native_ids':sum(r['native_accounting']['returned_tokens'] for r in selected),'wall_seconds':sum(r['elapsed_seconds'] for r in selected),'calls':len(calls),'native_exec_ms':stats([c.get('exec_ms') for c in calls]),'successful_delivered':sum(c.get('inserted') is True and c.get('status')=='ok' for c in calls)}
telemetry['_diagnostics']=diagnostic
telemetry['_definition']='prefill_seconds and seconds are native server prefill/decode telemetry. No all-round server total inferred. Diagnostics can be multi-round; native exec_ms is separately reported. Client wall is the end-to-end endpoint.'
with (export/'telemetry.json').open('x') as f:json.dump(telemetry,f,indent=2);f.write('\n')
with (export/'request-order-audit.json').open('x') as f:json.dump({'qualified':True,'boots':14,'dataset_rows':368,'diagnostic_rows':28,'files_sha256':inputs},f,indent=2);f.write('\n')
fmt=lambda x,n=3:'unavailable' if x is None else f'{x:.{n}f}'
ci=lambda x,scale=1:'unavailable' if x is None else f'[{x[0]*scale:.3f}, {x[1]*scale:.3f}]'
names={'gsm8k':'GSM8K test 48 (strict extraction)','math500':'MATH-500 test 20','humaneval':'HumanEval 24 (original)'}
lines=['# Standard benchmark results — NEEDS-T5','',f'Rendered {datetime.datetime.now(datetime.timezone.utc).isoformat()}. Frozen T5 spec SHA256: `{report["spec_sha256"]}`.','',
'**All 368 replies were collected; all six main turns qualified and G was restored after every turn. All three suites are DECLARED BUT UNUSED: activation diagnostics fired, but no dataset reply made a Python call. These are declaration/configuration comparisons, not evidence of a compute-execution benefit.**','',
'GSM8K has a net strict-grader solve decrease. HumanEval savings are dominated by one capped plain reply. MATH meets the registered local numerical efficiency conditions on this fixed subset, with no observed solve decrease; its unused lane prevents attributing that result to execution. No suite has a Holm-significant accuracy gain. No pooled benchmark score or population non-inferiority claim is made.','',
'Exact served weights: **deepseek-ai/DeepSeek-V4.1-Flash-1017**, revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, deployed flash17 EXL3 2.80bpw pack. API alias `DeepSeek-V4.1-Flash-TF`; engine `3a429d6f122d9eb2900c43e622933d311e4be44f`, per-SHA image `tensorfold-dsv41-imr:r-3a429d6f122d`. Rank image IDs are pinned in the [CPU/source receipt](proof/phase2/cpu-final-v3/verification.json). This does not measure unquantized upstream weights.','',
'Both arms: thinking/high, temperature 1, top_p=0.95, top_k=20, seeds 101/102, concurrency 1, 8,192 aggregate native returned IDs, 270s all-round server deadline and 360s client wait. Plain has lane=0 and no tool declaration; lane-only has lane=1 and built-in code_interpreter. Runner, guide, trace, S3, COMMIT/STATED/CHAIN/MARK/SYNC_GRAPH were OFF on both actual rank processes. Native telemetry was identical. Fixed selected IDs, task-block shuffle, cold starts and fresh per-boot caches followed the [frozen spec](../in-model-runner/T5-STDBENCH-SPEC.md). Pilot replies remain in the final denominators.','',
'## Per-suite observed outcomes','',
'Native IDs include all generated/tool/protocol/forced output IDs once, prompt excluded. Coverage is 100% for all 368 dataset replies and 28 diagnostics. Wall time is client monotonic end-to-end, including every native round; boot, restoration, diagnostics and independent reference checks are excluded from per-problem costs. Timeouts/errors are 0/0 throughout.','',
'| Suite | Arm | Solved / scheduled | Accuracy | Native IDs/problem mean / median | Wall s/problem mean / median | Capped |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for suite in ('gsm8k','math500','humaneval'):
    s=report['suites'][suite]
    for arm in ('plain','lane'):
        lines.append(f'| {names[suite]} | {arm} | {s[arm]["solved"]}/{s["scheduled_per_arm"]} | {s[arm]["rate"]*100:.2f}% | {s["tokens"][arm]["mean"]:.2f} / {s["tokens"][arm]["median"]:.2f} | {s["time"][arm]["mean"]:.3f} / {s["time"][arm]["median"]:.3f} | {s[arm]["censored"]} |')
lines+=['','GSM8K uses the frozen strict final-answer extractor. Of its incorrect grades, 41 plain and 44 lane replies were `ambiguous_or_missing_final`; one additional plain reply had a numeric mismatch. For example, `#### 20 cookies` is rejected against reference `20`. This includes formatting/extraction rejection and must not be interpreted as a general mathematical capability score. No extractor was loosened after outcomes. Every extracted/reference pair and actual final text is published in [gsm8k-extractions.csv](proof/phase2/post-v1/gsm8k-extractions.csv).','',
'MATH: all 80 final answers were independently labelled by two blinded reviewer agent reviewers. Both saw only the presealed public packet; neither saw arm/seed/pairing, the other labels, aggregate scores, or queried the serving endpoint. Labels were sealed before merge. There were 0 disagreements and 0 unresolved answers. Adverse/favorable paired bounds are therefore both 0 pp; per-arm bounds are [100%, 100%]. These are agent adjudications, not human adjudications. The tied [0, 0] bootstrap interval does not establish population non-inferiority. [Public packet and labels](review/public.json), [review audit](proof/phase2/post-v1/math-review-audit.json).','',
'HumanEval uses the original official tests plus check(entry_point), in a network-disabled/unprivileged/read-only CPU jail with a required harness completion receipt. The plain `HumanEval/163`, seed 101 reply reached the 8,192-ID cap and remained incorrect; it was retained. Capped replies were eligible to pass if a complete correct final solution had landed. This is not HumanEval+.','',
'## Registered paired tests','',
'Each independent task retains both generation seeds and both arms. Accuracy uses exact task-cluster sign flips, including ties, then Holm across the three suites. CIs use 20,000 paired task-cluster bootstrap draws, RNG 20261104, sorted task IDs, endpoints 500/19,499; no seed resampling. Token/time intervals are descriptive.','',
'| Suite | Lane−plain accuracy, pp [CI95] | Paired gains / losses | Exact two-sided p | Holm p |','| --- | ---: | ---: | ---: | ---: |']
for suite in ('gsm8k','math500','humaneval'):
    s=report['suites'][suite]
    lines.append(f'| {names[suite]} | {s["accuracy_difference_lane_minus_plain"]*100:+.3f} {ci(s["accuracy_ci95"],100)} | {len(s["gains"])} / {len(s["losses"])} | {s["sign_flip"]["p_two_sided"]:.6f} | {s["sign_flip"]["holm_adjusted_p"]:.6f} |')
lines+=['','| Suite | Native-ID paired sums L/P [CI95] | Native-ID mean Δ L−P [CI95] | Wall paired sums L/P [CI95] | Wall mean Δs L−P [CI95] |','| --- | ---: | ---: | ---: | ---: |']
for suite in ('gsm8k','math500','humaneval'):
    s=report['suites'][suite];t=s['tokens'];w=s['time']
    lines.append(f'| {names[suite]} | {t["paired_sums_ratio_lane_over_plain"]:.3f} {ci(t["ratio_ci95"])} | {t["paired_mean_difference_lane_minus_plain"]:+.3f} {ci(t["difference_ci95"])} | {w["paired_sums_ratio_lane_over_plain"]:.3f} {ci(w["ratio_ci95"])} | {w["paired_mean_difference_lane_minus_plain"]:+.3f} {ci(w["difference_ci95"])} |')
lines+=['','All metric pairs are complete. GSM/MATH have no censored pairs, so complete-both sensitivities equal the primary cost estimates and censor savings contributions are 0%. HumanEval has 47 uncensored complete-both pairs: native-ID ratio 0.961, wall ratio 0.928. Its one censored pair contributes 89.50% of net native-ID savings  / 50.77% of positive native-ID savings and 82.96%  / 50.80% of wall savings. These exceed 25%; no uncensored speedup claim is made. There is no point-ratio censoring reversal. Missing accounting was never assigned zero.','',
'Every isolated GSM strict-grader loss (task ID/seed): '+', '.join(f'`{r["id"]}/{r["seed"]}`' for r in report['suites']['gsm8k']['losses'])+'. HumanEval and MATH have no paired solve losses. Full gain/loss identities, per-task differences, exact DP numerators/denominators, missingness, censoring and bounds are in [analysis.json](proof/phase2/post-v1/analysis.json).','',
'| Suite | Arm | Seed 101 solved / tasks | Seed 102 solved / tasks |','| --- | --- | ---: | ---: |']
for suite in ('gsm8k','math500','humaneval'):
    s=report['suites'][suite]
    for arm in ('plain','lane'):
        x=s[arm]['per_seed'];lines.append(f'| {names[suite]} | {arm} | {x["101"]["solved"]}/{x["101"]["scheduled"]} | {x["102"]["solved"]}/{x["102"]["scheduled"]} |')
lines+=['','## Activation, server telemetry and diagnostics','',
'| Suite | Lane seed 101 calls / replies using tool | Lane seed 102 calls / replies using tool | Plain calls | Label |','| --- | ---: | ---: | ---: | --- |']
for suite in ('gsm8k','math500','humaneval'):
    s=activation['suites'][suite];a=s['lane']['101'];b=s['lane']['102']
    lines.append(f'| {names[suite]} | {a["calls"]} / {a["replies_using_tool"]}/{a["scheduled_replies"]} | {b["calls"]} / {b["replies_using_tool"]}/{b["scheduled_replies"]} | 0 | **{s["lane_dataset_activation"]}** |')
lines+=['','Dataset successful/delivered/reference-verified call counts and executor identities are empty in every suite/seed because there were zero calls. Diagnostics across seven lane boots delivered 14 successful Python results, all native-frame authenticated and independently reference-executed; seven plain boots made zero calls. The executor is the code tool under the pinned S3-OFF contract (the native call report omits executor when S3 is absent). [Per-seed activation](proof/phase2/post-v1/activation.json).','',
'| Suite | Arm | Server prefill s mean | Server decode s mean | TTFT s mean | Prompt IDs mean | Reasoning IDs mean |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for suite in ('gsm8k','math500','humaneval'):
    for arm in ('plain','lane'):
        t=telemetry[suite][arm]
        lines.append('| '+names[suite]+' | '+arm+' | '+' | '.join(fmt(t[k]['mean']) for k in ('server_prefill_seconds','server_decode_seconds','ttft_seconds','prompt_tokens','reasoning_tokens'))+' |')
lines+=['','Server values are native `prefill_seconds`, `seconds` (decode from prefill end), and `time_to_first_token`, not a fabricated server end-to-end total. Dataset replies had no tool rounds. Content-token counts are not separately exposed; their text is retained and no subtraction/character estimate substitutes for IDs. Diagnostic multi-round server runtime retains final-segment timing, so only client wall is used as its all-round total. Native call `exec_ms` is reported separately below; no reference-container time is substituted.','',
'| Excluded diagnostics | Replies | Native IDs total | Client wall s total | Successful delivered calls | Native exec_ms mean / median |','| --- | ---: | ---: | ---: | ---: | ---: |']
for arm in ('plain','lane'):
    d=diagnostic[arm];e=d['native_exec_ms'];lines.append(f'| {arm} | {d["replies"]} | {d["native_ids"]} | {d["wall_seconds"]:.3f} | {d["successful_delivered"]} | {fmt(e["mean"])} / {fmt(e["median"])} |')
lines+=['','## Boot, recovery and integrity','',
'| Successful turn | Boot admission/load to readiness, s | G restore interval, s |','| --- | ---: | ---: |']
for turn in ('pilot','t01','t02','t03','t04','t05','t06'):
    directory=root/'post-data-v1'/f'stdbench-20261105-v2-{turn}'
    events=readrows(directory/'steps.jsonl')
    dt=lambda event:datetime.datetime.fromisoformat(event['at'])
    start=next(e for e in events if e['step']=='gpu-locks' and e['verdict']=='HELD')
    intervals=[]
    for e in events:
        if e['step']=='boot' and e['verdict']=='READY_EFFECTIVE_ENV_VERIFIED':intervals.append((dt(e)-dt(start)).total_seconds())
        if e['step']=='owned-containers' and e['verdict']=='DRAIN_VERIFIED':start=e
    release=next(e for e in events if e['step']=='gpu-locks' and e['verdict']=='RELEASED')
    restored=next(e for e in events if e['step']=='G-restore' and e['verdict']=='SERVING_VERIFIED')
    lines.append(f'| {turn} | '+ ' / '.join(f'{x:.3f}' for x in intervals)+f' | {(dt(restored)-dt(release)).total_seconds():.3f} |')
lines+=['','Startup intervals include admission, thermal waiting and loading; per-boot diagnostics are separate above. The resumed pilot row contains lane only. Its retained plain boot also cold-started (about 321.45s to readiness). The first pilot plain block collected 12 replies and drained, then a case-sensitive Docker absence check misread lowercase `no such object`. Both exact owned containers/PIDs were independently confirmed gone, the owned controller was settled, and G restored through its queue entry. The general check and retained-complete-boot/no-duplicate gate passed CPU regression; only the unmeasured lane side resumed in fresh namespace v2. All 12 original plain observations were retained. This added a separate G restoration, documented in the [ledger](LEDGER.md), without changing N, caps, graders or completed outcomes.','',
'All 14 boot request orders matched frozen chronological order. The [native audit](proof/phase2/post-v1/native-audit.json) qualified 368 dataset + 28 diagnostic replies, exact ID/hash reconciliation, native source/result frames, every retained independent tool reference, and HTTP access-log coverage with no unexpected/foreign requests. The [environment audit](proof/phase2/post-v1/environment-audit.json) verified 28 actual rank-process snapshots against manifests and approved image IDs. No runner/trace/S3 execution or inherited guide was observed.','',
'Collection freeze manifest SHA256: `'+audit['freeze_manifest_sha256']+'`. Analysis graded-row SHA256: `'+report['graded_rows_sha256']+'`. Grader/extraction/statistics hashes are in the [precollection CPU receipt](proof/phase2/cpu-final-v3/verification.json); no grading rules changed after outcomes. The 102 isolated canonical/wrong/timeout/early-exit/reference controls remain hash-qualified. CPU work used 2 GiB/no-swap user scopes, nice 19, ionice-c3, one worker, timeout 7200, QUIET/admission guards. No <other-host>, engine build, banned engine, publication or push was used. Temporary compilation output and CPU containers were cleaned.','',
'## DeepSeek published references — historical only','',
'Exact Flash-1017 instruct scores for these three suites are **not published in the reviewed exact-revision card**. Published token/time baselines are unavailable. The following rows are different BASE checkpoints and protocols; MATH is the full MATH benchmark, not MATH-500. They are not matched local baselines, replications, retention ratios, or evidence of full-model quality loss.','',
'| Historical reference | GSM8K exact match, 8-shot | MATH exact match, 4-shot | HumanEval pass@1, 0-shot |','| --- | ---: | ---: | ---: |',
'| September V4.1-Flash-Base | 93.0 | 61.1 | 79.4 |','| V4-Pro-Base historical/preview card | 92.6 | 64.5 | 76.8 |','',
'Sources: [Flash card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md), [Pro card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/README.md), retrieved 2026-11-04 UTC. Exact evaluation run dates, sampling, generation counts and output caps are undisclosed; base evaluation is an internal framework, not instruct thinking/high. The Pro reference is not verified as an August-GA evaluation or V4.1-Pro release. The [exact 1017 card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/README.md) is the October 17 release, retrieved November 4. [PUBLISHED.md](PUBLISHED.md) retains all sourced release/mode/protocol caveats and full-model GA references.','',
'Runnable kit and raw chronological replies: `rank0-host:~/work/stdbench/phase2/` (campaign-v2, frozen-v3, runs, post-v1). CPU outputs are also retained on rank1-host. Local report evidence is under `proof/phase2/post-v1/`; [reproduction API](phase2/analysis.md). Status: **NEEDS-T5**.','']
text='\n'.join(lines)
with (export/'RESULTS.md').open('x') as f:f.write(text)
validation={'passed':True,'spec_sha256':report['spec_sha256'],'results_sha256':sha(export/'RESULTS.md'),'renderer_sha256':sha(__file__),'analysis_sha256':sha(out/'analysis.json'),'native_audit_sha256':sha(out/'native-audit.json'),'environment_audit_sha256':sha(out/'environment-audit.json'),'request_order_audit_sha256':sha(export/'request-order-audit.json'),'rows':368,'diagnostics':28,'pooled_scores':False,'dataset_calls':0,'status':'NEEDS-T5'}
with (export/'report-validation.json').open('x') as f:json.dump(validation,f,indent=2);f.write('\n')
print(json.dumps({'report_rendered':True,'dataset_calls':0,'rows':368,'status':'NEEDS-T5','results_sha256':validation['results_sha256']}))
