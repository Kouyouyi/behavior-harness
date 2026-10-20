import tempfile
import unittest
from pathlib import Path
import numpy as np
from ap_harness.paths import ROOT
from ap_harness.control import run_loop
from planner_contract import render

def decision(status='execute', tags=None):
    command = {'status': status, 'motion_tags': ['turn_left'] if tags is None else tags,
               'target_description': '收音机', 'goal_kind': 'inspect_target',
               'visual_evidence': '左侧有可见通路', 'target_visible': status == 'reached'}
    return {'command': command, 'instruction': render(command)}

class Sim:
    steps=0; done=False; success=False; termination_reason=None
    def __init__(self): self.received=[]
    def capture(self, path): return {'state': [0.]*61, 'image_paths': [], 'step': self.steps}
    def refresh(self): pass
    def verify_approach(self, directory): return {'find_and_approach_success': False}
    def step(self, action): self.received.append(action.copy()); self.steps += 1; return True

class Policy:
    def health(self): return {'ready':True, 'checkpoint':'fixed', 'state_dim':61, 'action_dim':23, 'prediction_steps':32}
    def predict(self, obs, text): return np.arange(32*23, dtype=np.float32).reshape(32,23)

class Tests(unittest.TestCase):
    def run_case(self, sim, planner, policy=None, rounds=1):
        with tempfile.TemporaryDirectory() as tmp:
            return run_loop(sim, planner, policy or Policy(), '收音机', Path(tmp)/'run', max_rounds=rounds)
    def test_only_eight_raw_actions(self):
        sim=Sim(); self.run_case(sim, lambda *a:decision())
        self.assertEqual(sim.steps,8)
        np.testing.assert_array_equal(sim.received[-1], np.arange(161,184))
    def test_reached_is_not_success(self):
        sim=Sim(); reason=self.run_case(sim,lambda *a:decision('reached',[]))
        self.assertEqual(reason,'arrival_verification_failed'); self.assertFalse(sim.success); self.assertEqual(sim.steps,0)
    def test_verified_arrival_stops_without_native_success(self):
        sim=Sim()
        sim.verify_approach=lambda directory: {'find_and_approach_success': True}
        self.assertEqual(self.run_case(sim,lambda *a:decision('reached',[])), 'find_and_approach_success')
        self.assertFalse(sim.success)
        self.assertEqual(sim.steps,0)
    def test_visibility_and_distance_both_required(self):
        from ap_harness.scoring import evaluate_approach
        evidence={'base_position':[0,0,0], 'target_position':[0.8,0,1], 'target_visible':True}
        self.assertTrue(evaluate_approach(evidence)['find_and_approach_success'])
        self.assertFalse(evaluate_approach({**evidence,'target_visible':None})['find_and_approach_success'])
        self.assertFalse(evaluate_approach({**evidence,'target_position':[2,0,0]})['find_and_approach_success'])
    def test_reuse_keeps_vla_observations_fresh(self):
        calls=[]; observations=[]; sim=Sim(); policy=Policy()
        original=policy.predict
        def predict(obs,text): observations.append(obs['step']); return original(obs,text)
        policy.predict=predict
        def planner(*args): calls.append(True); return decision()
        with tempfile.TemporaryDirectory() as tmp:
            run_loop(sim,planner,policy,'收音机',Path(tmp)/'run',max_rounds=5,planner_max_chunks=4)
        self.assertEqual(len(calls),2)
        self.assertEqual(observations,[0,8,16,24,32])
        self.assertEqual(sim.steps,40)
    def test_opening_commit_is_explicit_and_bounded(self):
        sim=Sim(); calls=[]
        def planner(*args):
            calls.append(True)
            c=decision()['command'];c.update(motion_tags=['open_drawer'],goal_kind='inspect_container_interior')
            return {'command':c,'instruction':render(c)}
        with tempfile.TemporaryDirectory() as tmp:
            run_loop(sim,planner,Policy(),'收音机',Path(tmp)/'run',max_rounds=13,opening_commit_chunks=12)
        self.assertEqual(len(calls),2)
        self.assertEqual(sim.steps,104)
    def test_open_is_never_reused(self):
        sim=Sim(); calls=[]
        def planner(*args):
            calls.append(True)
            c=decision()['command']
            c.update(motion_tags=['open_drawer'],goal_kind='inspect_container_interior')
            return {'command':c,'instruction':render(c)}
        with tempfile.TemporaryDirectory() as tmp:
            run_loop(sim,planner,Policy(),'收音机',Path(tmp)/'run',max_rounds=3,planner_max_chunks=4)
        self.assertEqual(len(calls),3)
    def test_observation_request_bounded(self):
        sim=Sim(); reason=self.run_case(sim,lambda *a:decision('need_observation',[]),rounds=10)
        self.assertEqual(reason,'observation_request_limit'); self.assertEqual(sim.steps,0)
    def test_partial_failure_no_replay(self):
        sim=Sim()
        def fail(action): sim.steps+=1; raise RuntimeError('partial physics step')
        sim.step=fail
        with self.assertRaises(RuntimeError): self.run_case(sim,lambda *a:decision())
        self.assertEqual(sim.steps,1)
    def test_reject_stale_inference(self):
        sim=Sim(); policy=Policy()
        def stale(obs,text): sim.steps+=1; return np.ones((32,23))
        policy.predict=stale
        with self.assertRaises(RuntimeError): self.run_case(sim,lambda *a:decision(),policy)
        self.assertEqual(len(sim.received),0)
    def test_reject_invalid_chunk(self):
        policy=Policy(); policy.predict=lambda *a:np.zeros((8,23))
        with self.assertRaises(ValueError): self.run_case(Sim(),lambda *a:decision(),policy)
    def test_reject_untrained_mixed_open_move(self):
        with self.assertRaises(ValueError): decision(tags=['open_door','forward'])
    def test_open_container_template(self):
        command=decision()['command']
        command.update(motion_tags=['open_drawer'],target_description='抽屉',goal_kind='inspect_container_interior')
        self.assertEqual(render(command),'拉开抽屉，观察抽屉内部。')
    def test_prompt_cannot_truncate_state(self):
        from ap_harness.preprocessing import prepare_text
        with self.assertRaises(ValueError): prepare_text([0.]*61,'观察收音机。'*300)
    def test_changed_remote_identity_stops(self):
        policy=Policy(); original=policy.health; count=[0]
        def health():
            count[0]+=1
            result=original()
            if count[0]>1: result['checkpoint']='changed'
            return result
        policy.health=health
        sim=Sim()
        with self.assertRaises(RuntimeError): self.run_case(sim,lambda *a:decision(),policy)
        self.assertEqual(sim.steps,0)

if __name__ == '__main__': unittest.main()
