"""Read-only live wallboard, path containment and real artifact provenance."""
import json,tempfile,threading,unittest
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from services.behavior_ap.wallboard import Artifacts,make_server

class WallboardTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  root=Path(self.tmp.name);self.root=root/'study';self.root.mkdir()
  self.pointer=root/'pointer.json';self.write(self.pointer,{'output':str(self.root)})
  self.write(self.root/'protocol.json',dict(max_steps=None,wall_limit_seconds=None,
   tasks=[dict(task_name='turning_on_radio',task_index=0)],vla_tools=dict(qwen_monitor=dict(model='Qwen',api_key='DO_NOT_LEAK'))))
  self.write(self.root/'study_status.json',dict(state='suite'))
  self.write(self.root/'suite_status.json',dict(state='running',task='turning_on_radio'))
  self.trial=self.root/'turning_on_radio/zero_shot_test';self.trial.mkdir(parents=True)
  self.write(self.trial/'sim/progress.json',dict(step=40,success=False))
  images=self.trial/'observations/o';images.mkdir(parents=True)
  paths=[]
  for role in ('head','left_wrist','right_wrist'):
   p=images/(role+'.png');p.write_bytes(b'fixture image bytes');paths.append(str(p))
  self.write(self.trial/'live_frame.json',dict(step=40,image_paths=paths))
  self.write(self.trial/'vla/live_state.json',dict(phase='champion',qwen_completion_streak=1,
   job=dict(job_id='job_00001',kind='champion',executed_steps=40,start_step=0,step_budget=200,status='waiting_for_qwen'),
   active_interaction_goal='Press radio power switch',recent_decisions=[]))
  (self.trial/'execution.jsonl').write_text(json.dumps(dict(event='submitted',sequence=39,owner=dict(controller='champion_vla')))+'\n'+json.dumps(dict(event='ack',sequence=39,step_after=40))+'\n')
  self.artifacts=Artifacts(self.pointer)
 @staticmethod
 def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
 def test_only_real_current_artifacts_are_exposed(self):
  state=self.artifacts.snapshot();a=state['active']
  self.assertEqual(a['steps'],40);self.assertEqual(a['controller'],'champion_vla')
  self.assertEqual(a['frame_step'],40);self.assertEqual(a['qwen_completion_streak'],1)
  self.assertFalse(a['official_success']);self.assertTrue(state['read_only'])
  self.assertNotIn('DO_NOT_LEAK',json.dumps(state));self.assertEqual(state['physics_commands'],0)
 def test_images_are_whitelisted_and_cannot_escape_trial(self):
  self.assertEqual(self.artifacts.image('head'),b'fixture image bytes')
  with self.assertRaises(ValueError):self.artifacts.image('../../secret')
  outside=Path(self.tmp.name)/'head.png';outside.write_bytes(b'PRIVATE')
  self.write(self.trial/'live_frame.json',dict(step=40,image_paths=[str(outside)]))
  with self.assertRaises(FileNotFoundError):self.artifacts.image('head')
 def test_http_is_loopback_readonly_and_correct_mime(self):
  server=make_server(self.pointer,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
  url='http://127.0.0.1:'+str(server.server_address[1])
  with urlopen(url+'/')as response:self.assertIn('text/html',response.headers['Content-Type'])
  with urlopen(url+'/api/state')as response:self.assertEqual(json.load(response)['active']['steps'],40)
  for request in (Request(url+'/api/state',data=b'{}'),Request(url+'/api/state',headers={'Host':'evil.example'}),Request(url+'/arbitrary-file')):
   with self.assertRaises(HTTPError)as error:urlopen(request)
   self.assertIn(error.exception.code,(403,404,405))
