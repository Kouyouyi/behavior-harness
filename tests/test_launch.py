import json
from pathlib import Path
import tempfile
import unittest
from ap_harness.launch import validate_exit

class Tests(unittest.TestCase):
    def test_native_zero_does_not_mask_error(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);(root/'error.json').write_text('{"type":"RuntimeError"}')
            self.assertEqual(validate_exit(root,0),1)
    def test_missing_terminal_record_is_error(self):
        with tempfile.TemporaryDirectory() as p:self.assertEqual(validate_exit(p,0),1)
    def test_clean_unsuccessful_episode_is_valid_execution(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);(root/'sim').mkdir()
            (root/'sim/result.json').write_text(json.dumps({'success':False,'errors':[]}))
            self.assertEqual(validate_exit(root,0),0)
    def test_native_abort_keeps_partial_step_evidence(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);(root/'sim').mkdir()
            (root/'sim/progress.json').write_text('{"step":1}')
            self.assertEqual(validate_exit(root,-11),-11)
            failure=json.loads((root/'launcher_failure.json').read_text())
            self.assertEqual(failure['last_persisted_progress']['step'],1)
            self.assertFalse(failure['complete_episode'])
            self.assertFalse(failure['automatic_replay'])
