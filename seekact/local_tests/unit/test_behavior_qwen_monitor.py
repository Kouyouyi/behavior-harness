"""Two distinct visual votes interrupt champion without replay or extra actions."""
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch, Mock
from local_tests.unit.test_behavior_interaction import InteractionTests
from services.behavior_ap.qwen_monitor import CompletionVotes, parse_verdict, QwenMonitor

class VerdictTests(unittest.TestCase):
    def test_exact_output_and_consecutive_distinct_frames(self):
        v=CompletionVotes()
        self.assertFalse(v.add('已完成',20))
        with self.assertRaises(ValueError):v.add('已完成',20)
        self.assertFalse(v.add('继续',40))
        self.assertFalse(v.add('已完成',60))
        self.assertTrue(v.add('已完成',80))
        v.reset();self.assertFalse(v.add('已完成',100))
        for text in ('完成','已完成。','先继续','<think>yes</think>已完成',''):
            with self.assertRaises(ValueError):parse_verdict(text)

class SwitchingTests(unittest.TestCase):
    setUp=InteractionTests.setUp
    frame=InteractionTests.frame
    args=InteractionTests.args
    execute=InteractionTests.execute
    authorize=InteractionTests.authorize
    prepare=InteractionTests.prepare
    def monitor(self, answers):
        self.qwen_gate=threading.Event();self.qwen_gate.set()
        self.addCleanup(self.qwen_gate.set)
        it=iter(answers)
        def check(obs, goal, directory, stop):
            self.qwen_gate.wait(2)
            return dict(verdict=next(it),observation_step=obs['step'],interaction_goal=goal)
        self.v.qwen=SimpleNamespace(check=Mock(side_effect=check))
    def drive(self,job):
        while not job['done']:
            r,_=self.v.continue_job(job['job_id'],1);job=r.get('vla_job',r)
        return job
    def test_two_positives_interrupt_before_third_chunk_and_require_ap(self):
        self.prepare();self.monitor(['已完成','已完成'])
        job=self.execute('champion',200)
        self.assertEqual(job['executed_steps'],40)
        self.assertEqual(self.v.clients['champion'].predict.call_count,2)
        self.assertEqual(self.env.steps,48)
        self.assertEqual(self.v.phase,'search');self.assertFalse(self.v.awaiting_interaction_assessment)
        self.assertEqual(self.v.interaction_assessments[-1]['source'],'qwen_two_consecutive_current_frames')
        with self.assertRaisesRegex(ValueError,'AP search'):self.execute('champion',20)
        with self.assertRaisesRegex(ValueError,'executed AP'):self.authorize()
        self.execute('ap',8);self.authorize()
        self.assertEqual(self.v.qwen_votes.streak,0)
    def test_negative_resets_streak_and_final_short_prefix_is_judged(self):
        self.prepare();self.monitor(['已完成','继续','已完成','已完成'])
        job=self.execute('champion',67)
        self.assertEqual(job['executed_steps'],67)
        self.assertEqual([x['observation_step']for x in self.v.qwen_checks],[28,48,68,75])
        self.assertEqual(self.v.phase,'search')
    def test_waiting_for_qwen_freezes_physics_and_inference(self):
        self.prepare();self.monitor(['继续']);self.qwen_gate.clear()
        job=self.v.start('champion',self.args('champion',200))[0]
        self.v.continue_job(job['job_id'],1)
        self.assertEqual(self.env.steps,28)
        self.v.continue_job(job['job_id'],0)
        self.assertEqual(self.env.steps,28)
        self.assertEqual(self.v.clients['champion'].predict.call_count,1)
        self.qwen_gate.set()
        self.v.current['event'].wait(1)
        self.v.continue_job(job['job_id'],0)
        self.assertEqual(self.env.steps,28) # next inference only, no physical replay
    def test_budget_return_does_not_allow_gpt_to_fabricate_completion(self):
        self.prepare();self.monitor(['继续'])
        self.execute('champion',20);self.frame()
        with self.assertRaisesRegex(ValueError,'two consecutive'):self.v.assess_interaction('f','completed','Looks done')
        self.v.assess_interaction('f','need_ap','Need another view')
        self.assertEqual(self.v.phase,'search')

class RequestIsolationTests(unittest.TestCase):
    def test_one_current_composite_and_goal_only_no_secret_in_artifacts(self):
        from tempfile import TemporaryDirectory
        from PIL import Image
        import io
        with TemporaryDirectory() as tmp:
            root=Path(tmp);key=root/'key';key.write_text('test-secret')
            paths=[]
            for i in range(3):
                p=root/f'{i}.png';Image.new('RGB',(8,8),'red').save(p);paths.append(str(p))
            output=root/'check';output.mkdir()
            m=QwenMonitor(dict(api_key_file=str(key),base_url='http://example.invalid/v1',model='test'))
            response=io.BytesIO(json.dumps({'choices':[{'message':{'content':'已完成'}}]}).encode())
            with patch.object(m.opener,'open',return_value=response):
                result=m.check(dict(step=20,image_paths=paths,state=['PRIVATE_STATE']), 'Pick up red block',output,threading.Event())
            request=json.loads((output/'request.json').read_text())
            self.assertEqual(len(request['messages']),1)
            self.assertEqual(len(request['messages'][0]['content']),2)
            self.assertNotIn('PRIVATE_STATE',json.dumps(request))
            self.assertNotIn('test-secret',''.join(p.read_text()for p in output.glob('*.json')))
            self.assertEqual(result['verdict'],'已完成')
