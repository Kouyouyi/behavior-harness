"""Bounded re-observation, exact prefix execution and policy switching admission."""
import unittest
from local_tests.unit import test_behavior_vla_only as fixtures
import numpy as np

class ClosedLoopTests(unittest.TestCase):
 frame=fixtures.SearchHandoffTests.frame
 def setUp(self):
  fixtures.SearchHandoffTests.setUp(self);self.options['closed_loop']=True;self.worker.config={'max_steps':10000};self.gate.set()
 def args(self,kind,budget=13,progress='initial'):
  self.frame()
  return dict(frame_id='f',step_budget=budget,visual_progress=progress,
   reason='Target remains visible; wrist has not reached it',next_goal='Inspect approach',
   **({'motion_tags':['turn_left'],'target_description':'radio'}if kind=='ap'else{'skill':'turning_on_radio'}))
 def ap(self):return self.v.start('ap',self.args('ap',8))[0]['job_id']
 def execute(self,kind,budget=13,progress='initial'):
  j=self.v.start(kind,self.args(kind,budget,progress))[0]
  while not j['done']:
   r,_=self.v.continue_job(j['job_id'],1);j=r.get('vla_job',r)
  return j
 def authorize(self):
  self.frame();self.v.authorize_champion('f','radio','Radio visible')
 def test_exact_nonmultiple_budgets_and_discarded_tail_history(self):
  j=self.execute('ap',13);self.assertEqual(j['executed_steps'],13);self.assertEqual(j['chunks_completed'],2)
  self.authorize();epoch=self.v.epoch
  j=self.execute('champion',27);self.assertEqual(j['executed_steps'],27);self.assertEqual(self.env.steps,40)
  self.assertEqual(self.v.epoch,epoch+1)
  self.assertEqual(len(self.v.decisions),2)
  self.assertEqual(self.v.decisions[-1]['step'],13)
 def test_stale_decision_cannot_reuse_prior_observation(self):
  args=self.args('ap');self.env.steps+=1
  with self.assertRaisesRegex(ValueError,'fresh'):self.v.start('ap',args)
  self.assertEqual(self.v.count,0);self.assertEqual(self.v.epoch,0)
 def test_no_implicit_long_job_or_missing_decision(self):
  with self.assertRaisesRegex(ValueError,'Closed-loop'):self.v.start('ap',{'motion_tags':['turn_left'],'target_description':'radio'})
  args=self.args('ap');args['chunks']=1
  with self.assertRaisesRegex(ValueError,'omit chunks'):self.v.start('ap',args)
 def test_stagnation_requires_ap_then_fresh_champion_handoff(self):
  self.execute('ap',8);self.authorize()
  for _ in range(3):self.execute('champion',20,'no_progress')
  with self.assertRaisesRegex(ValueError,'stagnation'):self.execute('champion',20,'no_progress')
  self.execute('ap',8,'lost_target')
  with self.assertRaisesRegex(ValueError,'AP search'):self.execute('champion',20,'progress')
  self.authorize();self.execute('champion',20,'progress');self.assertEqual(self.env.steps,96)
 def test_unsafe_assessment_rejects_without_physics(self):
  args=self.args('ap',13,'unsafe')
  with self.assertRaisesRegex(ValueError,'Unsafe'):self.v.start('ap',args)
  self.assertEqual(self.env.steps,0)

class ClosedLoopFinishTests(unittest.TestCase):
 def test_temporary_stagnation_does_not_seal_failure_before_budget(self):
  from services.behavior_ap.worker import Worker
  from types import SimpleNamespace
  w=Worker.__new__(Worker);w.config={'vla_tools':{'closed_loop':True,'vla_only':True},'max_steps':10000}
  w.env=SimpleNamespace(steps=360,done=False,success=False);w.calls=0;w.finish_decision=None
  result,_=w.call('behavior_finish',{'outcome':'failure','reason':'Three champion segments stalled'})
  self.assertFalse(result['accepted']);self.assertIsNone(w.finish_decision)

class UncappedBudgetTests(unittest.TestCase):
 setUp=ClosedLoopTests.setUp
 frame=ClosedLoopTests.frame
 args=ClosedLoopTests.args
 execute=ClosedLoopTests.execute
 authorize=ClosedLoopTests.authorize
 def test_invalid_ap_combination_can_be_corrected_without_replay(self):
  args=self.args('ap',150);args['motion_tags']=['turn_left','lower_view']
  with self.assertRaisesRegex(ValueError,'No policy request'):self.v.start('ap',args)
  self.assertEqual(self.env.steps,0);self.assertEqual(self.v.count,0);self.assertFalse(self.v.infrastructure_errors)
  job=self.execute('ap',150);self.assertEqual(job['executed_steps'],150)
 def test_longer_than_120_and_episode_budget_are_distinct(self):
  self.worker.config['max_steps']=333
  job=self.execute('ap',137);self.assertEqual(job['executed_steps'],137)
  self.authorize();job=self.execute('champion',100000)
  self.assertEqual(job['step_budget'],196);self.assertEqual(job['executed_steps'],196)
  self.assertEqual(self.env.steps,333)
