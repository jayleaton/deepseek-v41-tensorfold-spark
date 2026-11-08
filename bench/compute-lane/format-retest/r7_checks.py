#!/usr/bin/env python3
"""Handwritten controls through unchanged r7 and the actual CPU enclosure adapter."""
import hashlib
import json
from pathlib import Path
import tempfile
import r7_audit
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    oracle=Path('<rank0-work>/owned-oracle/tool/oracle.py')
    results=[]
    with tempfile.TemporaryDirectory(prefix='format-r7-fixture-',dir=HERE) as folder:
        root=Path(folder)
        for label,number,present in [('correct',323,True),('wrong',324,True),('not-in-text',323,False)]:
            value=f'[exact, computed by the engine: 17*19 = {number}]\n'
            rec={'cell':0,'form':'trace','route':'graph','status':'ok','lang':'python','field':'reasoning',
                 'at':0,'token':3,'drafted':0,'path':'held','span':'17*19','display':'17*19','value':value,'ids':30,'landed':30,'forced':30}
            row={'reasoning':value if present else 'No engine value was placed here.','content':'','finish':'stop',
                 'tensorfold':{'finish':'done','runner':{'trace':{'owned':1},'cells':[rec]}}}
            source=root/(label+'.jsonl');source.write_text(json.dumps(row)+'\n')
            proof=r7_audit.run(source,oracle,root/label)
            if label=='correct':assert proof['qualified'] and proof['runner_owned_count']==1
            elif label=='wrong':assert not proof['qualified'] and proof['wrong_values']>0
            else:assert not proof['qualified'] and proof['not_in_text']>0
            results.append({'fixture':label,'observed_gate':proof['verdict'],'input_sha256':proof['input_rows_sha256']})
        empty=root/'empty.jsonl';empty.write_bytes(b'')
        proof=r7_audit.run(empty,oracle,root/'empty')
        assert proof['qualified'] is False and proof['executed'] is False and proof['observations']==0
        assert proof['verdict']['verdict']=='UNEXERCISED' and proof['wrong_values'] is None
        results.append({'fixture':'empty observations','observed_gate':proof['verdict'],'input_sha256':proof['input_rows_sha256']})
    receipt={'passed':True,'spec_sha256':'b47ede697eed3a38d244ad5ff7447690732bd12945ff4592554d45bc2d8527ec',
             'controls':results,'oracle_sha256':r7_audit.ORACLE_SHA,'fixture_outputs_cleaned':True,'model_requests':0,
             'source_sha256':{str(p):sha(p) for p in (Path(__file__),HERE/'r7_audit.py',oracle)}}
    with (HERE/'r7-adapter-check.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({'passed':True,'controls':len(results),'GPU_requests':0}))
if __name__=='__main__':main()
