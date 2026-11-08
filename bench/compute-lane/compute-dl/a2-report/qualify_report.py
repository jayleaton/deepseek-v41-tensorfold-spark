#!/usr/bin/env python3
"""CPU-only source/projection/full report-format controls; never grants collection."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'a2-continuation'))
import protocol
import adoption
import publish
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,why):
    if not ok:raise ValueError(why)
def source_hashes():return {str(p):sha(p) for p in HERE.rglob('*') if p.suffix in ('.py','.sh')}
def main():
    cfg_path=ROOT/protocol.PREFIX/'stage-campaign.json';cfg=load(cfg_path)
    require(not Path(cfg['go_file']).exists(),'CPU report qualification requires absent GO')
    protocol.require_staging_ready(cfg)
    before=source_hashes()
    for path in before:
        if Path(path).suffix=='.py':ast.parse(Path(path).read_text(),filename=path)
    prior,integrity,retained,files=adoption.validate(ROOT,cfg['prior_budget'],cfg['prior_budget_sha256'],
                                                   cfg['a1_audit'],cfg['a1_audit_sha256'])
    require(all(sha(path)==digest for path,digest in files.items()),'A1 test input drift')
    require(cfg['a1_audit_files_sha256']==files,'candidate audit output bindings differ')
    tasks=[json.loads(l) for l in (ROOT/'dataset-v1/tasks-public.jsonl').read_text().splitlines()]
    answers=load(ROOT/'dataset-v1/answers-a.json');byid={t['id']:t for t in tasks}
    frozen=publish.collect_report;kernel=frozen.analysis.kernel;activated=set()
    for raw in retained:
        projected=publish.compact(raw);t=byid[raw['id']];answer=answers[raw['id']]
        require(publish.compact(frozen.grade.grade(raw,t,answer))==frozen.grade.grade(projected,t,answer),'strict projection changed')
        require(kernel.metric_values(raw)==kernel.metric_values(projected),'cost projection changed')
        require(kernel._row_safe(raw)==kernel._row_safe(projected),'safety projection changed')
        arm=frozen.grade.identity(raw)[2]
        active=kernel._verified_activation(dict(raw,arm=arm))
        require(active==kernel._verified_activation(dict(projected,arm=arm)),'activation projection changed')
        if arm=='L' and active:activated.add(raw['id'])
    require(len(activated)>=30,'activation gate failed on retained cohort')
    # Only the formatting harness receives explicit CPU fixture authority. It
    # uses actual qualified retained rows, with no new observations or GO. The
    # production auditor is exercised AFTER the real final seal, before GO.
    test_integrity=dict(integrity,spec_sha256=publish.A2_SHA,a1_spec_sha256=publish.A1_SHA,
                        CPU_REPORT_FORMAT_FIXTURE_NOT_A2_COLLECTION_AUDIT=True)
    with tempfile.TemporaryDirectory(prefix='a2-report-format-') as name:
        temp=Path(name);audit=temp/'audit';audit.mkdir()
        for filename in ('audited-rows.jsonl','diagnostics.jsonl','presentations.jsonl'):
            (audit/filename).symlink_to(Path(cfg['a1_audit']).parent/filename)
        (audit/'audit.json').write_text(json.dumps(test_integrity))
        # Add descriptive batch labels only to a fixture projection. Original
        # rows, native frames, correctness and cost fields stay immutable.
        (audit/'audited-rows.jsonl').unlink()
        with (audit/'audited-rows.jsonl').open('x') as out:
            for row in retained:
                r=dict(row)
                if r['arm']=='plain':r['retention_batch']='P-before-break'
                out.write(json.dumps(r,separators=(',',':'))+'\n')
        out=temp/'rendered';oldargv=sys.argv
        try:
            sys.argv=[str(HERE/'publish.py'),'--audit',str(audit),'--campaign',str(cfg_path),'--out',str(out)]
            try:publish.main()
            except ValueError as error:require('authorities must remain distinct' in str(error),'wrong fixture refusal')
            else:raise AssertionError('CPU fixture accepted as production audit')
            require(not out.exists(),'fixture refusal created a report')
            sys.argv.append('--cpu-format-fixture')
            with contextlib.redirect_stdout(io.StringIO()):publish.main()
        finally:sys.argv=oldargv
        result=load(out/'analysis.json')
        require(result['observed']==len(retained) and result['status']=='INCOMPLETE','partial study formatting changed denominator')
        require(result['contrasts']['L_minus_P']['overall']['sign_flip'] is None,'partial primary test must be withheld')
        require(len((out/'per-problem.csv').read_text().splitlines())==721,'full scheduled CSV coverage missing')
        require(all(s in (out/'RESULTS.md').read_text() for s in (publish.A1_SHA,publish.A2_SHA,cfg['base_spec_sha256'])),'three protocol hashes missing')
        require('CPU FORMAT FIXTURE' in (out/'RESULTS.md').read_text()
                and not (out/'NEEDS-T5.md').exists(),'fixture was presented as a collection result')
    require(before==source_hashes(),'report source changed during qualification')
    require(all(sha(path)==digest for path,digest in files.items()),'A1 input changed during report checks')
    result={'passed':True,'spec_sha256':publish.A2_SHA,'source_sha256':before,
            'retained_rows':len(retained),'activation_distinct_L':len(activated),'strict_cost_safety_projection':True,
            'full_20000_draw_report_format':True,'scheduled_CSV_rows':720,'partial_primary_withheld':True,
            'CPU_fixture_production_publication_refused':True,
            'CPU_fixture_outputs_removed':True,'sealed_initial_raw_audit_required_before_GO':True,
            'GPU_requests':0,'GO_created':False}
    with (HERE/'qualification.json').open('x') as out:json.dump(result,out,indent=2)
    print(json.dumps({k:result[k] for k in ('passed','retained_rows','activation_distinct_L','GPU_requests')}))
if __name__=='__main__':main()
