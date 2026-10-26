"""Public repository entrypoint is portable and does not copy private credentials."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]

class ReleaseTests(unittest.TestCase):
 def test_prepare_uses_new_output_and_external_secret_path_only(self):
  with tempfile.TemporaryDirectory()as temporary:
   root=Path(temporary);key=root/'secret';key.write_text('PRIVATE_TEST_KEY');key.chmod(0o600)
   out=root/'experiment'
   command=[sys.executable,str(REPO/'scripts/seekact/cli.py'),'prepare','--output',str(out),
       '--qwen-base-url','http://example.invalid:8000/v1','--qwen-key-file',str(key),
       '--ap-checkpoint','/other/machine/models/checkpoint_024000']
   env=dict(os.environ,PYTHONPATH=str(REPO/'seekact'))
   result=subprocess.run(command,env=env,capture_output=True,text=True)
   self.assertEqual(result.returncode,0,result.stderr)
   protocol=json.loads((out/'protocol.json').read_text())
   self.assertIsNone(protocol['wall_limit_seconds']);self.assertIsNone(protocol['max_steps'])
   self.assertEqual([x['task_index']for x in protocol['tasks']],[22,0,21])
   self.assertEqual(protocol['vla_tools']['qwen_monitor']['api_key_file'],str(key))
   for p in out.iterdir():self.assertNotIn('PRIVATE_TEST_KEY',p.read_text())
   second=subprocess.run(command,env=env,capture_output=True,text=True)
   self.assertNotEqual(second.returncode,0)
 def test_private_key_permissions_are_enforced_before_output_creation(self):
  with tempfile.TemporaryDirectory()as temporary:
   root=Path(temporary);key=root/'secret';key.write_text('PRIVATE_TEST_KEY');key.chmod(0o644)
   out=root/'experiment'
   result=subprocess.run([sys.executable,str(REPO/'scripts/seekact/cli.py'),'prepare','--output',str(out),
       '--qwen-base-url','http://example.invalid/v1','--qwen-key-file',str(key),'--ap-checkpoint','/models/ap'],
       env=dict(os.environ,PYTHONPATH=str(REPO/'seekact')),capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0);self.assertFalse(out.exists())
