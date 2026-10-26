import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import wait_behavior_expert_exports as mod


class ExpertWaitTests(unittest.TestCase):
    def test_gate_calls_existing_validator_first(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            proof = {'record': 'sealed.json'}
            with patch.object(mod, 'valid_failed_zero_shot', return_value=proof) as validator:
                self.assertEqual(mod.gate_task(root, mod.TASKS[0], root / 'protocol.json'), ('eligible', proof))
                validator.assert_called_once_with(root, mod.TASKS[0], root / 'protocol.json')

    def test_unsealed_stays_waiting(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / mod.TASKS[0] / 'zero_shot_v2').mkdir(parents=True)
            with patch.object(mod, 'valid_failed_zero_shot', side_effect=ValueError('unsealed')):
                self.assertEqual(mod.gate_task(root, mod.TASKS[0], root / 'protocol.json')[0], 'waiting')

    def test_existing_partial_or_complete_never_reexport(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); task = mod.TASKS[0]; dest = root / task / 'expert_episode_10800'
            dest.mkdir(parents=True)
            self.assertEqual(mod.existing_export(root, task, 10800)['status'], 'existing_export_requires_manual_review')
            (dest / 'export_complete.json').write_text(json.dumps({'status': 'export_verified', 'source_count': 1,
                'task_name': task, 'fixed_expert': {'episode_index': 10800}}))
            self.assertEqual(mod.existing_export(root, task, 10800)['status'], 'existing_complete_no_reexport')

    def test_success_starts_no_export_and_existing_state_refuses_restart(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'protocol.json').write_text('{}')
            rows = [{'task_name': name, 'task_index': mod._fixed_task(name)['task_index'],
                     'fixed_expert': mod._fixed_task(name)} for name in mod.TASKS]
            with patch.object(mod, 'protocol_tasks', return_value=({}, rows)), \
                    patch.object(mod, 'relevant_processes', return_value=[]), \
                    patch.object(mod, 'gate_task', return_value=('not_required', {})), \
                    patch.object(mod.subprocess, 'Popen') as popen, \
                    patch.object(mod, 'existing_export') as existing:
                self.assertEqual(mod.run(root, 1), 0)
                popen.assert_not_called(); existing.assert_not_called()
                record = json.loads((root / 'expert_export_supervisor/status.json').read_text())
                self.assertTrue(all(x['status'] == 'not_required' for x in record['tasks'].values()))
                with self.assertRaises(FileExistsError):
                    mod.run(root, 1)


if __name__ == '__main__':
    unittest.main()
