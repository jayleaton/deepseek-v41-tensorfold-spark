#!/usr/bin/env python3
"""Fake-only regression of reviewed stop/admission boundaries; Spark scope only."""
import base64
import json
from pathlib import Path
import tempfile
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch
import controller
spec=importlib.util.spec_from_file_location('compute_rank_ops_checked',Path(__file__).with_name('rank_ops.py'))
rank_ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(rank_ops)
assert Path(rank_ops.__file__).resolve()==Path(__file__).with_name('rank_ops.py').resolve()
import request_adapter
from reconciliation import wire,digest


def cleanup_checks(root):
    """Real stop dispatch + inspect/drain logic; only Docker transport is faked."""
    prefix='stdbench-compute-dl-FAKE-cleanup'
    boot='t01-D';rank=0;name=f'{prefix}-{boot}-r{rank}'
    canonical=Path('<rank0-work>/stdbench/compute-dl')/prefix/'rank0/boots'/boot/'manifest.json'
    identity=rank_ops.cleanup_identity(canonical,rank)
    assert identity==(canonical.parents[3],prefix,boot,name)
    for wrong_path,wrong_rank in ((canonical,1),
        (Path(str(canonical).replace('t01-D','foreign-boot')),0),
        (Path(str(canonical).replace('stdbench-compute-dl-','foreign-')),0),
        (canonical.parent/'../manifest.json',0), (canonical,True)):
        try: rank_ops.cleanup_identity(wrong_path,wrong_rank)
        except (ValueError,TypeError): pass
        else: raise AssertionError('unsafe cleanup identity accepted')

    for number,reason in enumerate(('revoked GO binding','sealed source/data drift')):
        rank=number;name=f'{prefix}-{boot}-r{rank}'
        local=root/f'cleanup-{number}';local.mkdir()
        path=local/'manifest.json'
        manifest={'compute_dl':{'campaign':str(local/'campaign-v1.json')},'phase2':{'boot':boot}}
        path.write_text(json.dumps(manifest))
        (local/'started').mkdir();(local/'started'/f'{boot}-r{rank}').write_text(name+'\n')
        exact_identity=(local,prefix,boot,name)
        with patch.object(rank_ops,'cleanup_identity',return_value=exact_identity):
            receipt=rank_ops.launch_ownership(path,rank,local,manifest,name)
        if number==1:
            manifest['FAKE_source_drift']=True
            path.write_text(json.dumps(manifest))
        calls=[];state={'running':True,'foreign_running':True}

        def fake_docker(args,**kwargs):
            calls.append(args)
            if args==['docker','container','inspect',name]:
                if state['running']:
                    return SimpleNamespace(returncode=0,stdout=json.dumps([{'Name':'/'+name,
                        'Image':rank_ops.IMAGE_IDS[rank],'State':{'Running':True}}]),stderr='')
                return SimpleNamespace(returncode=1,stdout='',stderr='No such container: '+name)
            if args==['docker','stop','--time','120',name]:
                state['running']=False
                return SimpleNamespace(returncode=0)
            raise AssertionError('non-owned/unsupported Docker action: '+repr(args))

        # Exercise the public operation: it must never enter qualified context.
        with patch.object(rank_ops,'cleanup_identity',return_value=exact_identity), \
             patch.object(rank_ops,'require_collection_ready',side_effect=ValueError(reason)), \
             patch.object(rank_ops,'context',side_effect=AssertionError('stop entered qualified context')), \
             patch.object(rank_ops,'load_module',side_effect=AssertionError('cleanup loaded engine/config code')), \
             patch.object(rank_ops.subprocess,'run',side_effect=fake_docker):
            assert rank_ops.operate('stop',path,rank)==0
        assert calls==[['docker','container','inspect',name],
                       ['docker','stop','--time','120',name],['docker','container','inspect',name]]
        assert not state['running'] and state['foreign_running']
        assert (local/'cancelled'/f'{boot}-r{rank}').exists()
        proofs=[json.loads(p.read_text()) for p in (local/'proof'/boot).glob('stop-*.json')]
        assert len(proofs)==1 and proofs[0]['verdict']=='DRAINED'
        assert proofs[0]['collection_qualification']=={'passed':False,'error':'ValueError: '+reason}
        assert proofs[0]['retained_launch']['image_id']==rank_ops.IMAGE_IDS[rank]

        # Starts/snapshots still call the collection gate and fail on revocation.
        with patch.object(rank_ops,'require_collection_ready',side_effect=ValueError(reason)), \
             patch.object(rank_ops.subprocess,'run',side_effect=AssertionError('qualified action executed')):
            for command in ('start','snapshot'):
                try: rank_ops.operate(command,path,rank)
                except ValueError as error: assert str(error)==reason
                else: raise AssertionError('revoked action passed: '+command)

        for foreign_name,foreign_image in (('FOREIGN-container',rank_ops.IMAGE_IDS[rank]),
                                            (name,rank_ops.IMAGE_IDS[1-rank])):
            def wrong_container(args,**kwargs):
                assert args==['docker','container','inspect',name], 'foreign identity reached Docker stop'
                return SimpleNamespace(returncode=0,stdout=json.dumps([{'Name':'/'+foreign_name,
                    'Image':foreign_image,'State':{'Running':True}}]),stderr='')
            with patch.object(rank_ops,'cleanup_identity',return_value=exact_identity), \
                 patch.object(rank_ops,'require_collection_ready',side_effect=ValueError(reason)), \
                 patch.object(rank_ops.subprocess,'run',side_effect=wrong_container):
                try: rank_ops.operate('stop',path,rank)
                except ValueError as error: assert 'name/image differs' in str(error)
                else: raise AssertionError('foreign Docker identity accepted')

        # A forged retained image or a symlink cannot broaden cleanup ownership.
        value=json.loads(receipt.read_text());value['image_id']=rank_ops.IMAGE_IDS[1-rank]
        receipt.write_text(json.dumps(value))
        with patch.object(rank_ops,'cleanup_identity',return_value=exact_identity), \
             patch.object(rank_ops.subprocess,'run',side_effect=AssertionError('forged ownership touched Docker')):
            try: rank_ops.operate('stop',path,rank)
            except ValueError as error: assert 'ownership/pins' in str(error)
            else: raise AssertionError('wrong retained image accepted')
        receipt.unlink();receipt.symlink_to(path)
        with patch.object(rank_ops,'cleanup_identity',return_value=exact_identity), \
             patch.object(rank_ops.subprocess,'run',side_effect=AssertionError('symlink touched Docker')):
            try: rank_ops.operate('stop',path,rank)
            except ValueError as error: assert 'symlink' in str(error)
            else: raise AssertionError('symlinked ownership accepted')

    pending=root/'pending-cleanup';pending.mkdir()
    path=pending/'manifest.json'
    name=f'{prefix}-{boot}-r0'
    with patch.object(rank_ops,'cleanup_identity',return_value=(pending,prefix,boot,name)), \
         patch.object(rank_ops,'require_collection_ready',side_effect=ValueError('revoked before launch')), \
         patch.object(rank_ops.subprocess,'run',return_value=SimpleNamespace(
             returncode=1,stdout='',stderr='No such container: '+name)) as fake_run:
        assert rank_ops.operate('stop',path,0)==0
        fake_run.assert_called_once_with(['docker','container','inspect',name],text=True,capture_output=True)
    assert (pending/'cancelled'/f'{boot}-r0').exists()


def main():
    with tempfile.TemporaryDirectory(prefix='FAKE_CONTROLLER-') as directory:
        root=Path(directory);out=root/'measured';(out/'FAKE-L').mkdir(parents=True)
        task=controller.Turn.__new__(controller.Turn)
        task.me='stdbench-compute-dl-FAKE';task.out=out
        task.health=lambda:None;task.budget=SimpleNamespace(ensure_can_start=lambda:None)
        task.cfg={'transport_contract':{'max_tokens':8192,'deadline_s':270},'timeouts':{'client_wait_s':360},
                  'go_file':'FAKE','spec':'FAKE','tokenizer':'FAKE','tokenizer_sha256':'FAKE','image_ids':['sha256:'+'0'*64]}
        task.spec_sha='f'*64;task.prompts={('TEST_ONLY','one',101):{}}
        body={'max_tokens':8192,'deadline_s':270};raw=wire(body)
        template={'key':['TEST_ONLY','one',101],'suite':'TEST_ONLY','id':'one','seed':101,
                  'task_sha256':'FAKE','body_template':body,'wire':raw,'wire_sha256':digest(raw),
                  'execution_policy':{'TF_DSV41_COMPUTE_LANE':'1'}}
        call={'id':'FAKE-call','inserted':True,'status':'ok','code':'print(1)','output':'1'}
        row={'status':'error','transport_status':'ok','content':'','elapsed_seconds':1,
             'usage':{},'tensorfold':{'compute':{'calls':[call]}},
             'native_binding':{'qualified':True,'native_frame_count':1},'native_accounting':{'complete':True},
             'initial_prompt_reconciliation':{'qualified':True},'suite':'TEST_ONLY','id':'one','seed':101,
             'arm':'lane','boot':'FAKE-L'}
        fake_transport=lambda *args,**kwargs:lambda body:dict(row)
        import jail
        with patch.object(controller,'QUIET',root/'QUIET'), \
             patch.object(controller.client,'Transport',side_effect=fake_transport), \
             patch.object(controller.client,'NativeDecoder',return_value=None), \
             patch.object(request_adapter,'classify_observation',side_effect=lambda *a:dict(row)), \
             patch.object(jail,'Jail',return_value=SimpleNamespace(run_reference=lambda code:
                 {'status':'ok','normalized_body_base64':base64.b64encode(b'WRONG').decode()})):
            try: task.measure_one({'boot':'FAKE-L','arm':'lane'},template)
            except RuntimeError as error: assert 'safety/reference' in str(error)
            else: raise AssertionError('delivered mismatch on error reply did not stop')
        saved=json.loads((out/'FAKE-L/safety.jsonl').read_text())
        assert saved['qualified'] is False and saved['references'][0]['matches'] is False
        assert (out/'FAKE-L/raw-observations.jsonl').exists()
        assert not (root/'QUIET').exists()

        # Cooling crosses the cutoff: no thermal/rank process may be launched.
        boot_task=controller.Turn.__new__(controller.Turn);boot_task.out=root/'cool';boot_task.out.mkdir()
        state={'calls':0}
        def admission():
            state['calls']+=1
            if state['calls']>=2: raise controller.BudgetExhausted('FAKE cutoff')
        boot_task.budget=SimpleNamespace(ensure_can_start=admission)
        boot_task.cool=lambda:None
        with patch.object(controller.subprocess,'Popen',side_effect=AssertionError('rank started after cooling cutoff')):
            try: boot_task.boot({'boot':'FAKE'})
            except controller.BudgetExhausted: pass
            else: raise AssertionError('post-cooling budget check absent')

        # Real cooling loop and a stalled sensor both drain at the held cutoff.
        tick=[0.0]
        def cutoff():
            if tick[0]>=1:raise controller.BudgetExhausted('FAKE held cutoff')
        waiting=controller.Turn.__new__(controller.Turn)
        waiting.cfg={'timeouts':{'gpu_parent_s':43200,'remote_control_s':43200},'boot_min_gib':114}
        waiting.budget=SimpleNamespace(ensure_can_start=cutoff,elapsed=lambda:15899+tick[0])
        limits=[]
        def hot_sensor(*args,**kwargs):
            limits.append(kwargs['timeout'])
            return '99,hot,120,85,70,100,20'
        def advance(seconds):tick[0]+=seconds
        with patch.object(controller.subprocess,'check_output',side_effect=hot_sensor),patch.object(controller.time,'sleep',side_effect=advance):
            try:waiting.cool()
            except controller.BudgetExhausted:pass
            else:raise AssertionError('cooling ignored held cutoff')
        assert limits and all(0<value<=1 for value in limits)
        tick[0]=0
        def stalled_sensor(command,**kwargs):
            tick[0]=1
            raise controller.subprocess.TimeoutExpired(command,kwargs['timeout'])
        with patch.object(controller.subprocess,'check_output',side_effect=stalled_sensor):
            try:waiting.cool()
            except controller.BudgetExhausted:pass
            else:raise AssertionError('stalled sensor ignored held cutoff')

        # Live-looking fake ranks never become ready; no post-cutoff snapshot.
        tick[0]=0;waiting.out=root/'fake-readiness';waiting.out.mkdir();waiting.ranks=[]
        waiting.cool=lambda:None;waiting.health=lambda:None;waiting.event=lambda *a,**k:None
        waiting.admission_output=lambda *a,**k:str(controller.time.time())
        waiting.rank_action=lambda *a,**k:(_ for _ in ()).throw(AssertionError('late snapshot'))
        live=SimpleNamespace(poll=lambda:None)
        timeouts=[]
        def never_ready(*args,**kwargs):
            timeouts.append(kwargs['timeout']);raise OSError('FAKE not ready')
        with patch.object(controller.subprocess,'Popen',return_value=live), \
             patch.object(controller.urllib.request,'urlopen',side_effect=never_ready), \
             patch.object(controller.time,'sleep',side_effect=advance):
            try:waiting.boot({'boot':'FAKE-readiness','rank_manifest':{'0':'FAKE0','1':'FAKE1'}})
            except controller.BudgetExhausted:pass
            else:raise AssertionError('readiness ignored held cutoff')
        assert timeouts and all(0<value<=1 for value in timeouts)

        # A rank waits for memory past deadline: Docker must never start.
        owned=root/'rank';owned.mkdir();state_path=owned/'state';(state_path/'sessions').mkdir(parents=True)
        manifest={'compute_dl':{'boot_min_gib':114},'phase2':{'boot':'FAKE'}}
        (root/'FAKE.json').write_text(json.dumps(manifest))
        env={'STATE_DIR':str(state_path)}
        tick=[0]
        def wait_past_cutoff(*args): tick[0]=10
        with patch.object(rank_ops,'context',return_value=(manifest,{}, {},env,owned,root,'FAKE-container')), \
             patch.object(rank_ops,'prepare'),patch.object(rank_ops,'verify_original',return_value='FAKE'), \
             patch.object(rank_ops,'docker_command',return_value=(['DOCKER_MUST_NOT_START'],{})), \
             patch.object(rank_ops,'inspect_container',return_value=None), \
             patch.object(rank_ops,'load_module',return_value=SimpleNamespace(before_boot=wait_past_cutoff)), \
             patch.object(rank_ops.time,'time',side_effect=lambda:tick[0]), \
             patch.object(rank_ops.subprocess,'run',side_effect=AssertionError('Docker started past admission')):
            try: rank_ops.operate('start',root/'FAKE.json',0,False,5)
            except RuntimeError as error: assert 'cancelled after admission' in str(error)
            else: raise AssertionError('post-memory admission check absent')

        # Refusal before requesting STOP leaves G unchanged and withdraws own head.
        queue=root/'ORDER';turn=controller.Turn.__new__(controller.Turn)
        turn.path=root/'campaign.json';turn.spec_sha='f'*64;turn.turn='FAKE-t01'
        turn.me='stdbench-compute-dl-FAKE-t01';turn.out=root/'unchanged-G'
        turn.cfg={'budget_ledger':str(root/'FAKE-budget.json')};turn.error=None;turn.stopped_g=False
        turn.validate=lambda:None;turn.preflight=lambda:None
        turn.queue=lambda:queue.write_text(turn.me+'\nFOREIGN_JOB\n')
        turn.stop_g=lambda:(_ for _ in ()).throw(ValueError('FAKE wrong G identity'))
        turn.release=lambda:[]
        with patch.object(controller,'QUEUE',queue),patch.object(controller,'require_collection_ready',return_value={}):
            try: turn.execute()
            except RuntimeError as error: assert 'wrong G identity' in str(error)
            else: raise AssertionError('expected pre-STOP refusal missing')
        assert queue.read_text()=='FOREIGN_JOB\n' and json.loads((turn.out/'verdict.json').read_text())['verdict']=='STOPPED'
        with patch('builtins.print'):
            cleanup_checks(root)
    result={'passed':True,'status':'FAKE_NOT_COLLECTION','delivered_error_mismatch_stops':True,
            'post_cooling_and_memory_cutoffs':True,'pre_STOP_own_queue_withdrawal':True,
            'cooling_stalled_sensor_and_readiness_cutoffs':True,
            'revoked_GO_and_source_drift_cleanup':True,'cleanup_exact_owned_identity':True,
            'start_snapshot_still_qualified':True,'real_network':False}
    import time
    with (Path(__file__).parent/('controller-safety-check-'+str(time.time_ns())+'.json')).open('x') as output:
        json.dump(result,output,indent=2);output.write('\n')
    print(json.dumps(result))


if __name__=='__main__': main()
