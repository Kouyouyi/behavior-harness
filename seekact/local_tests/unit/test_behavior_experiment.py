"""CPU-only supervisor tests. No models, demonstrations or simulator are started."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import run_behavior_experiment as run


def closed(success=False, steps=2):
    videos = {'passed': True, 'status': 'verified' if steps else 'no_frames',
              'expected_frames': steps, 'playable_video_available': bool(steps), 'videos': {}}
    if steps:
        videos['videos'] = {role: {'full_decode_passed': True, 'actual_frames': steps}
                           for role in ('robot_cameras', 'third_person')}
    return {'status': 'closed', 'steps': steps, 'submitted': steps, 'acknowledged': steps,
            'execution_integrity': True, 'official_success': success, 'uncertain_action': None,
            'errors': [], 'render_profile': 'fast', 'videos': videos}


AGENT = {'status': 'agent_completed', 'exit_code': 0}

# A tiny test socket server returns metadata fixtures, not actual video validation.
FAKE_WORKER = r'''
import json, pathlib, socket, sys
c=json.loads(pathlib.Path(sys.argv[1]).read_text()); out=pathlib.Path(c['output'])
sock=socket.socket(socket.AF_UNIX); sock.bind(c['socket_path']); sock.listen(2)
(out/'worker_ready.json').write_text(json.dumps(dict(status='ready',task=c['task'],render_profile='fast',split='public_test',instance_id=301,step=0)))
while True:
 conn,_=sock.accept()
 with conn:
  with conn.makefile('rb') as f: r=json.loads(f.readline())
  closing=r['name']=='_close'
  value={'closing':True} if closing else {'task':c['task'],'step':0,'official_success':False}
  conn.sendall((json.dumps({'id':r['id'],'value':value,'images':[]})+'\n').encode())
 if closing: break
sock.close(); pathlib.Path(c['socket_path']).unlink()
record=json.loads(sys.argv[2]); record['task']=c['task']
(out/'episode_closed.json').write_text(json.dumps(record))
'''

FAKE_AGENT = r'''
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); c=json.loads(p.read_text()); root=pathlib.Path(c['root'])
(root/'agent_timing.json').write_text(json.dumps({'status':'agent_completed','exit_code':0}))
'''


class BehaviorExperimentTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.dict(os.environ, {'RAX_ARTIFACT_ROOT': str(self.root)}))
        self.tasks = [{'task_index': n, 'task_name': name, 'prompt': 'Complete ' + name}
                      for n, name in enumerate(('task_a', 'task_b', 'task_c'))]
        self.protocol = {'render_profile': 'fast', 'variant': 'original', 'split': 'public_test',
                         'instance_id': 301, 'seed': 20261020, 'tasks': self.tasks}
        self.path = self.root / 'protocol.json'
        self.path.write_text(json.dumps(self.protocol))

    def classify(self, agent=AGENT, episode=None, **kwargs):
        return run.classify_episode(agent, episode if episode is not None else closed(),
            agent_exit=kwargs.pop('agent_exit', 0), worker_exit=kwargs.pop('worker_exit', 0),
            agent_started=kwargs.pop('agent_started', True), **kwargs)

    def test_vla_only_attribution_and_handoff_are_required(self):
        e=closed();e['vla_evidence']={'vla_only':True,'completed_jobs':[{'kind':'champion','executed_steps':2}],'handoffs':[]}
        self.assertIn('champion_executed_without_visual_handoff',self.classify(episode=e)['reasons'])
        e['vla_evidence']['handoffs']=[{'source':'planner_visual_assessment'}]
        self.assertTrue(self.classify(episode=e)['evaluation_valid'])
        e['vla_evidence']['completed_jobs'][0]['executed_steps']=1
        self.assertIn('vla_only_action_attribution_incomplete',self.classify(episode=e)['reasons'])

    def test_vla_only_search_failure_does_not_require_fake_champion_execution(self):
        e=closed();e['vla_evidence']={'vla_only':True,'completed_jobs':[{'kind':'ap','executed_steps':2}],'handoffs':[]}
        self.assertEqual(self.classify(episode=e)['status'],'task_failure')

    def test_protocol_task_order_aliases_and_original_constraints(self):
        self.assertEqual(run.protocol_tasks(self.path)[1], self.tasks)
        self.assertEqual(run.protocol_tasks(self.path, '1')[1], [self.tasks[1]])
        self.assertEqual(run.protocol_tasks(self.path, 'task_c')[1], [self.tasks[2]])
        with self.assertRaises(ValueError):
            run.protocol_tasks(self.path, 'unknown')
        self.path.write_text(json.dumps({**self.protocol, 'variant': 'ood'}))
        with self.assertRaisesRegex(ValueError, 'variant'):
            run.protocol_tasks(self.path)

    def test_success_failure_and_infrastructure_are_separate(self):
        self.assertEqual(self.classify()['status'], 'task_failure')
        self.assertEqual(self.classify(episode=closed(True))['status'], 'task_success')
        self.assertEqual(self.classify(episode=closed(steps=0))['status'], 'task_failure')
        for status in ('walltime_budget_exhausted', 'infrastructure_error', 'interrupted'):
            result = self.classify(agent={'status': status, 'exit_code': 124}, agent_exit=124,
                                   episode=closed(True))
            self.assertEqual(result['status'], 'agent_infrastructure_error')
            self.assertIs(result['official_success'], True)
            self.assertFalse(result['task_success'])

    def test_vla_model_errors_unfinished_jobs_and_unused_tools_are_not_capability_failures(self):
        good={'completed_jobs':[{'executed_steps':2}], 'last_job':{'done':True}, 'infrastructure_errors':[]}
        e=closed();e['vla_evidence']=good
        self.assertTrue(self.classify(episode=e)['evaluation_valid'])
        for change in ({'infrastructure_errors':[{'error':'state changed'}]},
                       {'last_job':{'done':False}}, {'completed_jobs':[]}):
            e=closed();e['vla_evidence']=dict(good,**change)
            result=self.classify(episode=e)
            self.assertFalse(result['evaluation_valid']);self.assertEqual(result['status'],'infrastructure_error')
        self.assertEqual(self.classify(worker_exit=1)['status'], 'infrastructure_error')
        self.assertFalse(self.classify(worker_error={'error': 'initialization'})['evaluation_valid'])

    def test_missing_ack_or_video_cannot_become_valid_task_failure(self):
        for modify in (
            lambda d: d.update(acknowledged=1),
            lambda d: d.update(execution_integrity=False),
            lambda d: d.update(uncertain_action='unknown prefix'),
            lambda d: d['videos'].update(passed=False),
            lambda d: d['videos']['videos'].pop('third_person'),
        ):
            record = closed()
            modify(record)
            self.assertEqual(self.classify(episode=record)['status'], 'infrastructure_error')

    def seal_zero(self, name='zero_shot_v1', *, status='task_failure', success=False, stamp='2026-10-23T01:00:00Z'):
        d = self.root / 'task_a' / name
        (d / 'agent').mkdir(parents=True)
        (d / 'agent/agent_timing.json').write_text(json.dumps(AGENT))
        (d / 'episode_closed.json').write_text(json.dumps(closed(success)))
        record = {'task': 'task_a', 'phase': 'zero_shot', 'status': status,
                  'agent_started': True, 'agent_exit_code': 0, 'worker_exit_code': 0,
                  'started_at_utc': stamp, 'protocol_sha256': run.sha256(self.path)}
        (d / 'trial_record.json').write_text(json.dumps(record))
        return d

    def test_experience_gate_rechecks_sealed_failure_and_rejects_success_or_unsealed(self):
        with self.assertRaisesRegex(ValueError, 'existing sealed'):
            run.valid_failed_zero_shot(self.root, 'task_a')
        d = self.seal_zero()
        proof = run.valid_failed_zero_shot(self.root, 'task_a')
        self.assertEqual(len(proof['record_sha256']), 64)
        self.path.write_text(json.dumps({**self.protocol, 'seed': 1}))
        with self.assertRaisesRegex(ValueError, 'protocol differs'):
            run.valid_failed_zero_shot(self.root, 'task_a', self.path)
        episode = closed(); episode['acknowledged'] = 1
        (d / 'episode_closed.json').write_text(json.dumps(episode))
        with self.assertRaisesRegex(ValueError, 'latest valid'):
            run.valid_failed_zero_shot(self.root, 'task_a')
        (d / 'episode_closed.json').write_text(json.dumps(closed()))
        self.seal_zero('zero_shot_v2', status='task_success', success=True, stamp='2026-10-23T02:00:00Z')
        with self.assertRaisesRegex(ValueError, 'latest valid'):
            run.valid_failed_zero_shot(self.root, 'task_a')
        (self.root / 'task_a/zero_shot_v3').mkdir()
        with self.assertRaisesRegex(ValueError, 'Unsealed'):
            run.valid_failed_zero_shot(self.root, 'task_a')

    def test_zero_shot_never_reads_experience_and_retry_cannot_read_early(self):
        absent = self.root / 'DO_NOT_READ.md'
        with patch.object(run, 'prepare_experience', side_effect=AssertionError('must not read')) as reader:
            with self.assertRaises(ValueError):
                run.run_trial(self.root, self.path, self.protocol, self.tasks[0], 'zero_shot', 'v1', experience=absent)
            with self.assertRaises(ValueError):
                run.run_trial(self.root, self.path, self.protocol, self.tasks[0], 'one_expert_retry', 'v1', experience=absent)
            reader.assert_not_called()
        self.assertFalse((self.root / 'task_a').exists())

    def test_single_experience_snapshot_hash_and_new_attempt_boundary(self):
        self.seal_zero()
        proof = run.valid_failed_zero_shot(self.root, 'task_a')
        source = self.root / 'expert.md'; source.write_text('One observed example with uncertainty.')
        snapshot = self.root / 'snapshot.md'
        result = run.prepare_experience(source, snapshot, proof)
        self.assertEqual(result['sha256'], run.sha256(source))
        self.assertEqual(snapshot.read_text(), source.read_text())
        with self.assertRaises(ValueError):
            run.prepare_experience(source, snapshot, None)
        with self.assertRaises(FileExistsError):
            run.run_trial(self.root, self.path, self.protocol, self.tasks[0], 'zero_shot', 'v1')

    def test_command_passes_argv_without_shell_or_credentials(self):
        command = run.worker_command(self.root / 'a space.json')
        expected=['bash',str(run.SUDO_HELPER),'env'] if run.SUDO_HELPER else ['env']
        self.assertEqual(command[:len(expected)],expected)
        self.assertIn('PYTHONPATH=' + str(run.ROOT) + os.pathsep + str(run.SIM_ROOT), command)
        self.assertIn('BEHAVIOR_GPU_LOCK_PATH=' + str(run.GPU_LOCK), command)
        self.assertEqual(command[-1], str(self.root / 'a space.json'))
        self.assertFalse(any('password' in arg.lower() or 'credential' in arg.lower() for arg in command))

    def test_wait_for_runner_selects_blocking_lock_only_when_requested(self):
        for extra, expected in (([], run.fcntl.LOCK_EX | run.fcntl.LOCK_NB),
                                (['--wait-for-runner'], run.fcntl.LOCK_EX)):
            with self.subTest(extra=extra), \
                    patch.object(run.fcntl, 'flock', side_effect=RuntimeError('stop at lock')) as flock, \
                    patch.object(run, 'run_trial') as trial:
                with self.assertRaisesRegex(RuntimeError, 'stop at lock'):
                    run.main(['--protocol', str(self.path), '--task', 'task_a', *extra])
                self.assertEqual(flock.call_args.args[1], expected)
                trial.assert_not_called()

    def test_wait_for_runner_rechecks_fresh_directory_after_acquiring_lock(self):
        def acquired(_lock, mode):
            self.assertEqual(mode, run.fcntl.LOCK_EX)
            (self.root / 'task_a/zero_shot_v1').mkdir(parents=True)
        with patch.object(run.fcntl, 'flock', side_effect=acquired), \
                patch.object(run, 'run_trial') as trial:
            with self.assertRaisesRegex(FileExistsError, 'Existing attempt'):
                run.main(['--protocol', str(self.path), '--task', 'task_a', '--wait-for-runner'])
            trial.assert_not_called()

    def test_wait_for_runner_validates_zero_shot_after_lock_before_reading_expert(self):
        acquired = []
        def gate(*_):
            self.assertEqual(acquired, [run.fcntl.LOCK_EX])
            raise ValueError('retry eligibility rechecked after lock')
        with patch.object(run.fcntl, 'flock', side_effect=lambda _lock, mode: acquired.append(mode)), \
                patch.object(run, 'valid_failed_zero_shot', side_effect=gate), \
                patch.object(run, 'prepare_experience') as reader:
            with self.assertRaisesRegex(ValueError, 'retry eligibility rechecked after lock'):
                run.main(['--protocol', str(self.path), '--task', 'task_a',
                          '--phase', 'one_expert_retry', '--experience', str(self.root / 'DO_NOT_READ.md'),
                          '--wait-for-runner'])
            reader.assert_not_called()
        self.assertFalse((self.root / 'task_a').exists())

    def fake_processes(self, *, cancelled=None):
        original_popen = subprocess.Popen
        def popen(command, **kwargs):
            if 'harness.codex_cli.behavior' in command:
                self.assertFalse(cancelled and cancelled['requested'])
                command = [sys.executable, '-c', FAKE_AGENT, command[command.index('--config') + 1]]
            return original_popen(command, **kwargs)
        self.stack.enter_context(patch.object(run, 'worker_command', side_effect=lambda p:
            [sys.executable, '-c', FAKE_WORKER, str(p), json.dumps(closed())]))
        self.stack.enter_context(patch.object(run.subprocess, 'Popen', side_effect=popen))
        sleep = time.sleep
        self.stack.enter_context(patch.object(run.time, 'sleep', side_effect=lambda n: sleep(min(n, .01))))

    def test_real_cpu_socket_lifecycle_records_all_phases_and_summary(self):
        self.fake_processes()
        record = run.run_trial(self.root, self.path, self.protocol, self.tasks[0], 'zero_shot', 'v1', session_seconds=60)
        self.assertEqual(record['status'], 'task_failure')
        self.assertTrue(record['evaluation_valid'])
        self.assertEqual(record['worker_exit_code'], 0)
        self.assertFalse(Path(record['socket_path']).exists())
        timing = record['supervisor_timing']
        self.assertTrue(timing['accounting']['exclusive_plus_idle_equals_total'])
        self.assertEqual(set(timing['by_name']), {'worker_startup', 'agent_session', 'worker_finalize'})
        self.assertFalse(timing['diagnostics']['pure_model_inference_time_measured'])
        summary = run.update_summary(self.root, self.path)
        self.assertEqual(summary['counts'], {'task_failure': 1})
        self.assertEqual(summary['attempts'], 1)

    def test_cancellation_waits_ready_then_closes_worker_without_agent(self):
        flag = {'requested': True}
        self.fake_processes(cancelled=flag)
        record = run.run_trial(self.root, self.path, self.protocol, self.tasks[0], 'zero_shot', 'v1', cancelled=flag)
        self.assertFalse(record['agent_started'])
        self.assertEqual(record['worker_exit_code'], 0)
        self.assertEqual(record['status'], 'infrastructure_error')
        self.assertFalse(Path(record['socket_path']).exists())

    def test_retry_uses_one_snapshot_after_valid_zero_shot(self):
        self.seal_zero()
        source = self.root / 'only_expert.md'; source.write_text('One expert summary fixture.')
        self.fake_processes()
        record = run.run_trial(self.root, self.path, self.protocol, self.tasks[0],
                               'one_expert_retry', 'v1', experience=source)
        self.assertEqual(record['status'], 'task_failure')
        self.assertEqual(record['experience']['sha256'], run.sha256(source))
        self.assertEqual(record['experience']['sources_permitted'], 1)
        commands = run.read_json(Path(record['trial']) / 'launch_commands.json')
        self.assertIn('--experience', commands['agent'])
        self.assertEqual(commands['agent'][-1], record['experience']['snapshot'])

    def test_corrupt_agent_evidence_is_sealed_as_infrastructure(self):
        self.fake_processes()
        code = "import json,pathlib,sys; c=json.loads(pathlib.Path(sys.argv[1]).read_text()); (pathlib.Path(c['root'])/'agent_timing.json').write_text('broken JSON')"
        with patch.object(sys.modules[__name__], 'FAKE_AGENT', code):
            record = run.run_trial(self.root, self.path, self.protocol, self.tasks[0], 'zero_shot', 'v1')
        self.assertEqual(record['status'], 'agent_infrastructure_error')
        self.assertEqual(record['supervisor_error']['type'], 'EvidenceReadError')
        self.assertTrue((Path(record['trial']) / 'trial_record.json').is_file())


if __name__ == '__main__':
    unittest.main()
