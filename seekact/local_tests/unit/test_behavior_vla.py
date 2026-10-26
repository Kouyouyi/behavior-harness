"""CPU tests of learned-policy ownership, prefix provenance and async waits."""
from contextlib import nullcontext
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError, HTTPError

import numpy as np
from services.behavior_ap.vla import VlaTools, RemotePolicy, render_ap, AP_CHECKPOINT, CHAMPION_REVISION
from services.behavior_ap.schema import tools, validate


class VlaJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.owner = threading.get_ident();self.captured=[];self.applied=[]
        self.env=SimpleNamespace(steps=0,done=False,success=False)
        def capture(path):
            self.assertEqual(threading.get_ident(),self.owner)
            Path(path).mkdir();self.captured.append(self.env.steps)
            return {'state':[0.]*61,'image_paths':[],'step':self.env.steps}
        def step(action):
            self.assertEqual(threading.get_ident(),self.owner)
            self.applied.append(action.copy());self.env.steps+=1;return True
        self.env.capture=capture
        self.worker=SimpleNamespace(output=Path(self.temp.name),env=self.env,np=np,
            ledger=SimpleNamespace(span=lambda *a,**kw:nullcontext()),step=step,
            control=SimpleNamespace(adopt_external_action=Mock()),observe=lambda:({'step':self.env.steps},[]))
        self.options={'enabled':True,'ap_url':'http://localhost:1','champion_url':'http://localhost:2',
            'champion_skills':{'turning_on_radio':0}}
        self.v=VlaTools(self.worker,self.options);self.addCleanup(self.v.close)
        self.gate=threading.Event();self.addCleanup(self.gate.set)
        def prediction(kind):
            def run(obs,args,epoch,directory,stop):
                self.assertNotEqual(threading.get_ident(),self.owner)
                self.gate.wait(3)
                return {'actions':np.tile(np.arange(23,dtype=np.float32),(32 if kind=='ap' else 20,1)).tolist()}
            return run
        for k in ('ap','champion'):self.v.clients[k].predict=Mock(side_effect=prediction(k))

    def start(self,kind='ap',chunks=1):
        args={'motion_tags':['forward'],'target_description':'收音机','chunks':chunks}if kind=='ap'else{'skill':'turning_on_radio','chunks':chunks}
        return self.v.start(kind,args)[0]['job_id']

    def test_http_wait_never_advances_physics_or_blocks_native_transport(self):
        ident=self.start();value,images=self.v.continue_job(ident,wait_seconds=0)
        self.assertEqual(value['status'],'waiting_for_model');self.assertEqual(self.env.steps,0)
        self.assertTrue(self.v.pending());self.assertEqual(images,[])
        with self.assertRaisesRegex(RuntimeError,'pending'):self.start()

    def test_only_saved_ap_prefix_is_applied_once_and_each_chunk_observes_current_state(self):
        self.gate.set();ident=self.start(chunks=2)
        self.v.continue_job(ident,wait_seconds=1);self.assertEqual(self.env.steps,8)
        value,_=self.v.continue_job(ident,wait_seconds=1)
        self.assertEqual(value['vla_job']['status'],'completed');self.assertEqual(self.env.steps,16)
        self.assertEqual(self.captured,[0,8]);self.assertEqual(self.worker.control.adopt_external_action.call_count,16)
        self.assertTrue(np.array_equal(np.stack(self.applied),np.tile(np.arange(23,dtype=np.float32),(16,1))))
        self.v.continue_job(ident,wait_seconds=0);self.assertEqual(self.env.steps,16)
        prefix=json.loads((Path(self.temp.name)/'vla/job_00001/chunk_0000/execution_prefix.json').read_text())
        self.assertEqual((prefix['executed_prefix'],prefix['discarded_suffix']),(8,24))

    def test_champion_original_20_actions_and_explicit_skill(self):
        self.gate.set();ident=self.start('champion',chunks=2)
        self.v.continue_job(ident,1);self.v.continue_job(ident,1)
        self.assertEqual(self.env.steps,40);self.assertFalse(self.v.pending())
        with self.assertRaisesRegex(ValueError,'authorized'):
            self.v.start('champion',{'skill':'putting_away_toys'})

    def test_nonfinite_actions_are_rejected_before_any_step(self):
        self.v.clients['ap'].predict=Mock(return_value={'actions':np.full((32,23),np.nan).tolist()})
        ident=self.start()
        with self.assertRaisesRegex(ValueError,'Nonfinite'):self.v.continue_job(ident,1)
        self.assertEqual(self.env.steps,0);self.assertTrue(self.v.infrastructure_errors)
        with self.assertRaises(RuntimeError):self.start()

    def test_stale_observation_is_not_executed(self):
        ident=self.start();self.env.steps=1;self.gate.set()
        with self.assertRaisesRegex(RuntimeError,'Physics advanced'):self.v.continue_job(ident,1)
        self.assertEqual(self.applied,[])

    def test_partial_step_error_cannot_launch_another_prediction_or_replay(self):
        def partial(a):self.env.steps+=1;raise RuntimeError('uncertain physical advance')
        self.worker.step=partial;self.gate.set();ident=self.start()
        with self.assertRaisesRegex(RuntimeError,'uncertain'):self.v.continue_job(ident,1)
        self.assertEqual(self.env.steps,1);self.assertEqual(self.v.clients['ap'].predict.call_count,1)
        self.v.continue_job(ident,0);self.assertEqual(self.env.steps,1)


class RemoteBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.d=Path(self.temp.name);self.image=self.d/'image';self.image.write_bytes(b'png')
        self.obs={'state':[0.]*61,'image_paths':[str(self.image)]*3,'step':0}
        self.options={'ap_url':'http://localhost','champion_url':'http://localhost','champion_skills':{'turning_on_radio':0}}

    def test_network_retry_keeps_identical_post_bytes_and_no_physics(self):
        client=RemotePolicy('ap',self.options,self.d);calls=[]
        def open_(req,timeout):
            calls.append(req.data)
            if len(calls)==1:raise URLError('temporary loss')
            return io.BytesIO(b'{"ok":true}')
        client.opener.open=open_;stop=Mock();stop.is_set.return_value=False
        result=client.fetch('/predict',b'{"immutable":1}',stop,self.d)
        self.assertEqual(result,{'ok':True});self.assertEqual(calls,[b'{"immutable":1}']*2)

    def test_stateful_inference_error_is_not_retried_or_hidden(self):
        c=RemotePolicy('champion',self.options,self.d)
        c.opener.open=Mock(side_effect=HTTPError('url',500,'failed',{},io.BytesIO(b'{"requires_restart":true}')))
        with self.assertRaisesRegex(RuntimeError,'protocol/state'):c.fetch('/predict',b'{}',threading.Event(),self.d)
        self.assertEqual(c.opener.open.call_count,1)

    def test_pinned_checkpoint_and_champion_language_absence(self):
        c=RemotePolicy('champion',self.options,self.d,episode='e')
        health={'role':'seekact_champion_tools','ready':True,'revision':CHAMPION_REVISION,
                'checkpoint':'checkpoint_2','supported_skills':{'turning_on_radio':0},'boot_id':'b'}
        result={'representation':'published_champion_wrapper_R1Pro_controller_inputs','boot_id':'b',
                'seq':0,'episode':'e','skill':'turning_on_radio','context_epoch':3,'actions':[]}
        c.fetch=Mock(side_effect=[health,result]);c.predict(self.obs,{'skill':'turning_on_radio'},3,self.d,threading.Event())
        body=json.loads((self.d/'request.json').read_text())
        self.assertNotIn('instruction',body);self.assertEqual(body['context_epoch'],3);self.assertEqual(c.seq,1)
        c.fetch=Mock(return_value={**health,'boot_id':'changed'})
        with self.assertRaisesRegex(ValueError,'identity changed'):c.predict(self.obs,{'skill':'turning_on_radio'},3,self.d,threading.Event())

    def test_ap_contract_rejects_unsupported_skills_metrics_and_duplicate_tags(self):
        self.assertEqual(render_ap(['turn_left','forward'],'收音机'),'向左转并向前移动，观察并接近收音机。')
        for tags,target in [(['grasp'],'玩具'),(['turn_left','turn_right'],'门'),(['forward','forward'],'门'),(['forward'],'前方1米')]:
            with self.assertRaises(ValueError):render_ap(tags,target)

    def test_tools_are_opt_in_and_bounds_enforced(self):
        self.assertNotIn('behavior_champion_vla',[x['name']for x in tools()])
        self.assertIn('behavior_champion_vla',[x['name']for x in tools(True)])
        validate('behavior_active_perception_vla',{'motion_tags':['forward'],'target_description':'桌子','chunks':15})
        for name,args in [('behavior_active_perception_vla',{'motion_tags':['forward'],'target_description':'x','chunks':16}),
                          ('behavior_champion_vla',{'skill':'x','chunks':True}),
                          ('behavior_vla_continue',{'job_id':'job','wait_seconds':21})]:
            with self.assertRaises(ValueError):validate(name,args)

    def test_native_tool_handback_preserves_raw_gripper_commands_without_physics(self):
        from services.behavior_ap.robot import R1ProControl, ACTION_INDICES
        c=R1ProControl.__new__(R1ProControl);c.action_idx=ACTION_INDICES;c._held_grippers={}
        a=np.zeros(23);a[14]=-.73;a[22]=.64;c.adopt_external_action(a)
        self.assertEqual(c._held_grippers,{'left':-.73,'right':.64})


class WorkerIntegrationTests(unittest.TestCase):
    def test_pending_job_blocks_ordinary_motion_and_false_failure_but_not_state(self):
        from services.behavior_ap.worker import Worker
        w=Worker.__new__(Worker);w.calls=0;w.config={'task':'x','prompt':'x','max_steps':10000}
        w.env=SimpleNamespace(done=False,success=False,steps=0,termination_reason=None)
        w.finish_decision=w.uncertain=None;w.acknowledged=0
        w.vla=SimpleNamespace(pending=lambda:True,infrastructure_errors=[],state=lambda:{'job':'pending'})
        w.control=SimpleNamespace(read_state=lambda:{},move_base=Mock())
        with self.assertRaisesRegex(RuntimeError,'pending'):
            w.call('behavior_move_base',{'vx':.1,'vy':0,'yaw_rate':0})
        v,_=w.call('behavior_finish',{'outcome':'failure','reason':'model waiting'})
        self.assertFalse(v['accepted']);self.assertEqual(w.call('behavior_state',{})[0]['step'],0)
        w.control.move_base.assert_not_called()

    def test_launcher_opt_in_records_capabilities_and_exposes_only_extra_vla_tools(self):
        import os
        from services.controller.config import Configuration
        from harness.codex_cli import behavior
        with tempfile.TemporaryDirectory() as t,patch.dict(os.environ,{'RAX_ARTIFACT_ROOT':t}):
            c=Configuration(root=Path(t)/'episode',task='radio',task_description='Turn on radio',
                robot='behavior_ap',agent_provider='codex',backend_options={'vla_tools':{'enabled':True,'champion_skills':{'turning_on_radio':0}}})
            c.root.mkdir();p=c.root/'config.json';p.write_text(json.dumps(c.to_dict()))
            plan=behavior.prepare(c,p,'/python','/codex')
            allowed=next(x.split('=',1)[1]for x in plan['command']if x.startswith('mcp_servers.rax_robot.enabled_tools='))
            self.assertIn('behavior_vla_continue',json.loads(allowed))
            self.assertTrue(plan['vla_tools']['enabled'])
            self.assertIn('not the earlier native-only baseline',(c.root/'agent-workspace/AGENTS.md').read_text())


if __name__=='__main__':unittest.main()
