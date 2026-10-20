import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from ap_harness.task_execution import run_task

class TaskExecutionTests(unittest.TestCase):
    def test_official_success_stops_mid_chunk_and_preserves_ack(self):
        class Sim:
            steps=0
            done=False
            termination_reason='none'
            @property
            def success(self):return self.steps==3
            def capture(self,directory):return {'step':self.steps}
            def step(self,action):self.steps+=1;return True
        class Policy:
            def __init__(self,output):pass
            def health(self):return {'role':'original_task_execution','revision':'b036d3c31e541efecb266c3e76f97513f99aa803','tasks':['task']}
            def predict(self,obs,task):return np.ones((32,23),dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp,patch('ap_harness.task_execution.TaskPolicy',Policy):
            out=Path(temp)/'control';sim=Sim()
            self.assertEqual(run_task(sim,'task',out),'official_task_success')
            events=[json.loads(x) for x in (out/'round_00000/execution.jsonl').read_text().splitlines()]
            self.assertEqual(len([e for e in events if e['event']=='submitted']),3)
            self.assertEqual(len([e for e in events if e['event']=='ack']),3)
            self.assertTrue(json.loads((out/'summary.json').read_text())['official_task_success'])
