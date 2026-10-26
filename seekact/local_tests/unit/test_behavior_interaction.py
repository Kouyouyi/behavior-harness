"""Interaction ownership and finite wall time without a physical step budget."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from local_tests.unit import test_behavior_closed_loop as helpers
from local_tests.unit.test_behavior_experiment import closed
from services.behavior_ap.worker import Worker
from scripts.run_behavior_experiment import classify_episode

class InteractionTests(unittest.TestCase):
 setUp=helpers.ClosedLoopTests.setUp
 frame=helpers.ClosedLoopTests.frame
 execute=helpers.ClosedLoopTests.execute
 authorize=helpers.ClosedLoopTests.authorize
 def args(self,kind,budget=13,progress='initial'):
  args=helpers.ClosedLoopTests.args(self,kind,budget,progress)
  if kind=='champion':args['interaction_goal']='Lift current radio only'
  return args
 def prepare(self):
  self.options['interaction_only']=True;self.worker.config['max_steps']=None
  self.execute('ap',8);self.authorize()
 def assess(self,outcome):
  self.frame();return self.v.assess_interaction('f',outcome,'Current wrist view shows the local interaction result')
 def test_every_champion_segment_needs_fresh_gpt_assessment(self):
  self.prepare();self.execute('champion',20)
  with self.assertRaisesRegex(ValueError,'assess_interaction'):self.execute('champion',20)
  self.worker.frames['f']['step']-=1
  with self.assertRaisesRegex(ValueError,'fresh'):self.v.assess_interaction('f','completed','Object visibly lifted')
  self.assertEqual(self.env.steps,28)
 def test_completed_interaction_forces_ap_before_next_champion(self):
  self.prepare();self.execute('champion',20);self.assess('completed')
  with self.assertRaisesRegex(ValueError,'AP search'):self.execute('champion',20)
  with self.assertRaisesRegex(ValueError,'executed AP'):self.authorize()
  self.execute('ap',8);self.authorize();self.execute('champion',20)
  self.assertEqual(self.env.steps,56)
 def test_in_progress_only_allows_same_interaction(self):
  self.prepare();self.execute('champion',20);self.assess('in_progress')
  args=self.args('champion');args['interaction_goal']='Place another object'
  with self.assertRaisesRegex(ValueError,'different interaction'):self.v.start('champion',args)
  self.execute('champion',20);self.assertEqual(self.env.steps,48)
 def test_null_total_budget_does_not_clip_requested_steps(self):
  self.options['interaction_only']=True;self.worker.config['max_steps']=None
  job,_=self.v.start('ap',self.args('ap',20000))
  self.assertEqual(job['step_budget'],20000);self.assertEqual(self.env.steps,0)
 def test_deadline_seals_pending_job_without_any_action(self):
  self.prepare();job,_=self.v.start('champion',self.args('champion',2000))
  self.worker.expire_if_due=lambda:True
  result,_=self.v.continue_job(job['job_id'],0)
  self.assertTrue(result['vla_job']['done']);self.assertEqual(self.env.steps,8)
 def test_unsafe_assessment_rejects_further_motion(self):
  self.prepare();self.execute('champion',20);self.assess('unsafe')
  with self.assertRaisesRegex(ValueError,'unsafe'):self.execute('ap',8)

class WallTimeTests(unittest.TestCase):
 def test_wall_expiration_sets_authoritative_terminal_without_physics(self):
  w=Worker.__new__(Worker);w.config={'deadline_monotonic':100}
  backend=SimpleNamespace(done=False,truncated=False,termination_reason=None)
  class Env:
   @property
   def done(self):return backend.done
  w.env=Env();w.env.backend=backend
  with patch('services.behavior_ap.worker.time.monotonic',return_value=101):self.assertTrue(w.expire_if_due())
  self.assertEqual(backend.termination_reason,'wall_time_limit');self.assertTrue(backend.done)
 def test_wall_timeout_above_old_budget_is_valid_timeout_not_infrastructure(self):
  c=closed(steps=12000);c.update(max_steps=None,wall_limit_seconds=3600,termination_reason='wall_time_limit')
  a={'status':'agent_timeout','exit_code':-15,'timed_out':True}
  r=classify_episode(a,c,agent_exit=-15,worker_exit=0,agent_started=True)
  self.assertTrue(r['evaluation_valid']);self.assertEqual(r['status'],'task_timeout')

class WorkerEnvironmentPropagationTests(unittest.TestCase):
 def test_display_coexistence_survives_privilege_boundary(self):
  import os
  from scripts.run_behavior_experiment import worker_command
  with patch.dict(os.environ,{'BEHAVIOR_ALLOW_TODESK_DISPLAY':'1'}):
   self.assertIn('BEHAVIOR_ALLOW_TODESK_DISPLAY=1',worker_command('config.json'))
