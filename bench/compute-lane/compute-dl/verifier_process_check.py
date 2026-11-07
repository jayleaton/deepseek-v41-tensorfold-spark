#!/usr/bin/env python3
"""Execute two tiny handwritten verifier fixtures in the actual stdlib runner."""
import json
from pathlib import Path
import tempfile
import intake


def main():
    with tempfile.TemporaryDirectory(prefix='FAKE_NOT_COLLECTION-') as directory:
        root=Path(directory)
        spec=root/'spec.txt';spec.write_text('FAKE_NOT_COLLECTION; no study generation rule')
        spec_sha=intake.digest(spec.read_bytes())
        tasks={'schema_version':1,'spec_sha256':spec_sha,'tasks':[
            {'id':'TEST-ONLY-A','suite':'TEST_ONLY','family':'toy','prompt':'Add these small numbers.',
             'payload':{'values':[1,2,3]},'answer':{'kind':'integer','value':6}}]}
        task_file=root/'tasks.json';task_file.write_bytes(intake.canonical(tasks))
        source_a="import json,sys\nd=json.load(sys.stdin)\na=[{'id':t['id'],'answer':{'kind':'integer','value':sum(t['payload']['values'])}} for t in d['tasks']]\nprint(json.dumps({'schema_version':1,'answers':a}))\n"
        source_b="import json,sys\nd=json.load(sys.stdin)\na=[]\nfor t in d['tasks']:\n total=0\n for value in reversed(t['payload']['values']):\n  total=total+value\n a.append({'id':t['id'],'answer':{'kind':'rational','numerator':total,'denominator':1}})\nprint(json.dumps({'schema_version':1,'answers':a}))\n"
        verifiers=[]
        for name,source in (('toy-sum',source_a),('toy-reverse-fold',source_b)):
            path=root/(name+'.py');path.write_text(source)
            verifiers.append({'name':name,'path':str(path),'sha256':intake.digest(path.read_bytes())})
        task_sha=intake.digest(task_file.read_bytes())
        cfg={'schema_version':1,'spec_sha256':spec_sha,'taskset_sha256':task_sha,
             'expected_task_ids':['TEST-ONLY-A'],'verifier_timeout_seconds':7200,'verifiers':verifiers,
             'review':{'operator':'TEST_ONLY','spec_sha256':spec_sha,'taskset_sha256':task_sha,
                       'source_sha256':[v['sha256'] for v in verifiers],
                       'independence_basis':'Tiny plumbing fixtures only; not production algorithm qualification',
                       'public_input_review':'Handwritten fixture lacks references in public view',
                       'spec_source_review_complete':False}}
        receipt=intake.intake(task_file,spec_file=spec,config=cfg,test_only=True)
        assert receipt['status']=='FAKE_NOT_COLLECTION' and not receipt['collection_authorized']
        assert all(v['runner']=='isolated_python' for v in receipt['verifiers'])
        bad=Path(verifiers[1]['path']);bad.write_text(source_b.replace('total=0','total=1'))
        verifiers[1]['sha256']=intake.digest(bad.read_bytes())
        cfg['review']['source_sha256']=[v['sha256'] for v in verifiers]
        try: intake.intake(task_file,spec_file=spec,config=cfg,test_only=True)
        except intake.IntakeError: pass
        else: raise AssertionError('actual independent verifier disagreement accepted')
    result={'passed':True,'status':'FAKE_NOT_COLLECTION','actual_serial_verifier_processes':4,
            'mismatched_answer_rejected':True,'real_tasks_generated':0,'collection_authorized':False}
    with (Path(__file__).parent/'verifier-process-check.json').open('x') as output:
        json.dump(result,output,indent=2);output.write('\n')
    print(json.dumps(result))


if __name__=='__main__': main()
