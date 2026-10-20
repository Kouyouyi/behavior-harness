import json
import unittest
from ap_harness.planner import completed_answer

class CompletedTests(unittest.TestCase):
    def events(self, complete=True, tool=False):
        command={'status':'execute','motion_tags':['turn_left'],'target_description':'柜子','goal_kind':'inspect_target','visual_evidence':'左侧有柜子','target_visible':False}
        rows=[{'type':'turn.started'},{'type':'item.completed','item':{'type':'agent_message','text':json.dumps(command)}}]
        if tool:rows.append({'type':'item.completed','item':{'type':'command_execution'}})
        if complete:rows.append({'type':'turn.completed'})
        return '\n'.join(map(json.dumps,rows))

    def test_completed_stream_without_output_file(self):
        self.assertEqual(completed_answer(self.events())['motion_tags'],['turn_left'])
    def test_incomplete_answer_not_executable(self):
        self.assertIsNone(completed_answer(self.events(False)))
    def test_completed_tool_turn_rejected(self):
        with self.assertRaises(ValueError):completed_answer(self.events(tool=True))
    def test_completed_after_waiting_for_network(self):
        error=json.dumps({'type':'error','message':'Reconnecting... waiting for network (Connection failed: error sending request)'})
        lines=self.events().splitlines();lines.insert(1,error)
        self.assertEqual(completed_answer('\n'.join(lines))['motion_tags'],['turn_left'])
        self.assertIsNone(completed_answer('\n'.join(lines[:-1])))
    def test_unknown_error_still_rejected(self):
        lines=self.events().splitlines();lines.insert(1,json.dumps({'type':'error','message':'unexpected failure'}))
        with self.assertRaises(ValueError):completed_answer('\n'.join(lines))
