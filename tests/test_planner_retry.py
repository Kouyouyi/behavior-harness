import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ap_harness.planner import plan

class RetryTests(unittest.TestCase):
    def run_attempts(self, forbidden=False):
        calls=[]
        command={'status':'execute','motion_tags':['turn_left'],'target_description':'柜子',
                 'goal_kind':'inspect_target','visual_evidence':'左侧有柜子','target_visible':False}
        def popen(args, **kwargs):
            calls.append(True)
            first=len(calls)==1
            events=[{'type':'turn.started'}]
            if first:
                events.append({'type':'item.completed','item':{'type':'command_execution' if forbidden else 'error'}})
            else:
                events.append({'type':'turn.completed'})
                Path(args[args.index('--output-last-message')+1]).write_text(json.dumps(command))
            kwargs['stdout'].write(''.join(json.dumps(e)+'\n' for e in events))
            class Proc:
                stdin=io.StringIO()
                returncode=1 if first else 0
                def poll(self): return self.returncode
            return Proc()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);images=[]
            for i in range(3):
                p=root/f'{i}.png';p.write_bytes(b'fixture');images.append(str(p))
            with patch('ap_harness.planner.shutil.which',return_value='codex'), patch('ap_harness.planner._default_preferences',return_value={}), patch('ap_harness.planner.subprocess.Popen',side_effect=popen):
                if forbidden:
                    with self.assertRaises(ValueError):plan(images,'收音机',[],root/'out')
                    self.assertEqual(len(calls),1)
                else:
                    result=plan(images,'收音机',[],root/'out')
                    self.assertEqual(result['command'],command)
                    self.assertEqual(len(calls),2)
                    self.assertTrue((root/'out/transport_failures.json').exists())
    def test_incomplete_failed_attempt_can_retry(self): self.run_attempts()
    def test_forbidden_tool_attempt_cannot_retry(self): self.run_attempts(True)
