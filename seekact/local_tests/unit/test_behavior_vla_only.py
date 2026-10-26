"""Hard control boundaries: retained native implementations cannot move strict trials."""
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock,patch
import numpy as np
from services.behavior_ap.schema import tools,NATIVE_MOTION_TOOLS,VLA_HANDOFF
from services.behavior_ap.vla import VlaTools
from services.behavior_ap.worker import Worker
from services.behavior_ap.backend import BehaviorBackend
from harness.codex_cli.behavior import allowed_tools

class ControlBoundaryTests(unittest.TestCase):
 def setUp(self):
  self.options={'enabled':True,'vla_only':True,'ap_url':'http://localhost:1','champion_url':'http://localhost:2','champion_skills':{'turning_on_radio':0}}
  self.config=SimpleNamespace(backend_options={'vla_tools':self.options})
 def test_schema_and_frontend_hide_native_controls_but_legacy_retains_them(self):
  visible={t['name']for t in tools(True,True)}
  self.assertFalse(visible & NATIVE_MOTION_TOOLS);self.assertIn(VLA_HANDOFF,visible)
  self.assertFalse(set(allowed_tools(self.config)) & NATIVE_MOTION_TOOLS)
  self.assertTrue(NATIVE_MOTION_TOOLS <= {t['name']for t in tools()})
 def test_backend_rejects_direct_native_rpc_before_transport(self):
  backend=BehaviorBackend.__new__(BehaviorBackend);backend.config=self.config
  with patch('services.behavior_ap.backend.request')as transport:
   for name in NATIVE_MOTION_TOOLS:
    with self.assertRaisesRegex(ValueError,'disabled'):backend.call(name,{})
   transport.assert_not_called()
 def test_worker_rejects_valid_native_requests_before_controller_or_physics(self):
  worker=Worker.__new__(Worker);worker.config={'vla_tools':self.options};worker.control=Mock();worker.calls=0
  args={'behavior_move_base':{'vx':0,'vy':0,'yaw_rate':0},'behavior_move_arm':{'side':'left','position':[0,0,0]},'behavior_gripper':{'side':'left','opening':1},'behavior_torso':{'joints':[0,0,0,0]}}
  for name in NATIVE_MOTION_TOOLS:
   with self.assertRaisesRegex(ValueError,'disabled'):worker.call(name,args[name])
  self.assertEqual(worker.calls,0);self.assertFalse(worker.control.mock_calls)
 def test_low_level_step_rejects_native_owner_before_journaling(self):
  worker=Worker.__new__(Worker);worker.config={'vla_tools':self.options};worker.action_owner={'controller':'native_tools'}
  with self.assertRaisesRegex(RuntimeError,'non-policy'):worker.step(np.zeros(23))
 def test_vla_only_requires_enabled_tools(self):
  with self.assertRaises(ValueError):tools(False,True)

class SearchHandoffTests(unittest.TestCase):
 def setUp(self):
  self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.env=SimpleNamespace(steps=0,done=False,success=False)
  self.worker=SimpleNamespace(output=Path(self.tmp.name),env=self.env,np=np,frames={},ledger=SimpleNamespace(span=lambda *a,**kw:nullcontext()),control=SimpleNamespace(adopt_external_action=Mock()),observe=lambda:({'step':self.env.steps},[]))
  def capture(p):Path(p).mkdir();return {'step':self.env.steps,'state':[0]*61,'image_paths':[]}
  self.env.capture=capture
  def step(a):self.env.steps+=1
  self.worker.step=step
  self.options={'enabled':True,'vla_only':True,'ap_url':'http://localhost:1','champion_url':'http://localhost:2','champion_skills':{'turning_on_radio':0}}
  self.v=VlaTools(self.worker,self.options);self.addCleanup(self.v.close)
  self.gate=threading.Event();self.addCleanup(self.gate.set)
  for kind in ('ap','champion'):
   def predict(obs,args,epoch,directory,stop,kind=kind):
    self.gate.wait(3);return {'actions':np.zeros((32 if kind=='ap' else 20,23)).tolist()}
   self.v.clients[kind].predict=Mock(side_effect=predict)
 def ap(self):return self.v.start('ap',{'motion_tags':['turn_left'],'target_description':'radio','chunks':1})[0]['job_id']
 def frame(self):self.worker.frames['f']={'step':self.env.steps}
 def test_champion_cannot_bypass_initial_search(self):
  with self.assertRaisesRegex(ValueError,'AP search'):self.v.start('champion',{'skill':'turning_on_radio'})
  self.assertEqual(self.v.count,0);self.v.clients['champion'].predict.assert_not_called()
 def test_handoff_without_executed_ap_rejected(self):
  self.frame()
  with self.assertRaisesRegex(ValueError,'AP must execute'):self.v.authorize_champion('f','radio','Visible target')
 def test_pending_and_stale_handoff_rejected(self):
  ident=self.ap();self.frame()
  with self.assertRaisesRegex(RuntimeError,'pending'):self.v.authorize_champion('f','radio','Visible target')
  self.gate.set();self.v.continue_job(ident,1)
  with self.assertRaisesRegex(ValueError,'fresh'):self.v.authorize_champion('f','radio','Visible target')
 def test_handoff_adds_zero_steps_and_ap_return_requires_fresh_authorization(self):
  self.gate.set();ident=self.ap();self.v.continue_job(ident,1);self.frame()
  result=self.v.authorize_champion('f','radio','Radio visible on the table');self.assertEqual(result['physical_steps_added'],0);self.assertEqual(self.env.steps,8)
  ident=self.v.start('champion',{'skill':'turning_on_radio','chunks':1})[0]['job_id'];self.v.continue_job(ident,1);self.assertEqual(self.env.steps,28)
  ident=self.ap();self.v.continue_job(ident,1)
  with self.assertRaisesRegex(ValueError,'AP search'):self.v.start('champion',{'skill':'turning_on_radio'})
  self.assertEqual(self.v.state()['phase'],'search');self.assertFalse(self.v.state()['native_motion_enabled'])

if __name__=='__main__':unittest.main()
