"""CPU-only worker boundary tests; no Isaac import, GPU, or recorded episode data."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from services.behavior_ap.worker import Worker, serve


class FakeLedger:
    def __init__(self):
        self.events = []

    def span(self, *args, **kwargs):
        return nullcontext()

    def emit(self, name, **values):
        self.events.append((name, values))


class FakeEnvironment:
    def __init__(self):
        self.steps = 0
        self.done = False
        self.success = False
        self.termination_reason = None
        self.actions = []
        self.backend = SimpleNamespace(action_owner=None)

    def step(self, action):
        self.actions.append(action.copy())
        self.steps += 1
        return True

    def close(self, reason):
        return dict(steps=self.steps, success=self.success,
                    termination_reason=reason, errors=[], videos={})


class AckWriteFailure:
    """Keep a real file descriptor so the submitted-action fsync is exercised."""
    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        if json.loads(text).get('event') == 'ack':
            raise OSError('injected ACK storage failure')
        return self.stream.write(text)

    def __getattr__(self, name):
        return getattr(self.stream, name)


class WorkerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='behavior-worker-test-')
        self.addCleanup(self.directory.cleanup)
        self.worker = worker = Worker.__new__(Worker)
        worker.np = np
        worker.output = Path(self.directory.name)
        worker.config = {'task': 'turning_on_radio', 'prompt': 'Test prompt', 'max_steps': 100}
        worker.ledger = FakeLedger()
        worker.env = FakeEnvironment()
        worker.action_owner={'controller':'native_tools'}
        worker.control = SimpleNamespace(read_state=Mock(return_value={'joints': [0.]}),
            move_base=Mock(), move_arm=Mock(), gripper=Mock(), torso=Mock())
        worker.frames = {}
        worker.observations = worker.submitted = worker.acknowledged = worker.calls = 0
        worker.actions = []
        worker.uncertain = worker.finish_decision = None
        worker.execution = (worker.output / 'execution.jsonl').open('x', buffering=1)
        self.addCleanup(worker.execution.close)

    def frame(self, ident='frame-0', step=0):
        values = dict(depth=np.ones((3, 3)), K=np.eye(3),
                      camera_position=np.zeros(3), camera_quat=np.array([0., 0., 0., 1.]),
                      robot_base_position=np.zeros(3), robot_base_quat=np.array([0., 0., 0., 1.]))
        self.worker.frames = {ident: {'step': step, 'cameras': {'head': values}}}
        return dict(frame_id=ident, camera='head', x=0, y=0, radius=0)

    def test_recording_owner_matches_action_journal_before_physics(self):
        worker=self.worker
        owner={'controller':'champion_vla','job_id':'job_00012','chunk':2,'offset':0}
        worker.action_owner=owner.copy()
        original=worker.env.step
        def physical_step(action):
            self.assertEqual(worker.env.backend.action_owner,owner)
            return original(action)
        worker.env.step=physical_step
        worker.step(np.zeros(23))
        submitted=json.loads((worker.output/'execution.jsonl').read_text().splitlines()[0])
        self.assertEqual(submitted['owner'],owner)

    def test_one_physical_step_has_one_ordered_ack(self):
        worker = self.worker
        self.assertTrue(worker.step(np.zeros(23)))
        records = [json.loads(line) for line in (worker.output / 'execution.jsonl').read_text().splitlines()]
        self.assertEqual([r['event'] for r in records], ['submitted', 'ack'])
        self.assertEqual([r['sequence'] for r in records], [0, 0])
        self.assertEqual(records[0]['step_before'], 0)
        self.assertEqual(records[1]['step_after'], 1)
        self.assertEqual((worker.submitted, worker.acknowledged, worker.env.steps), (1, 1, 1))
        self.assertEqual(worker.env.actions[0].dtype, np.float32)

    def test_physical_error_after_advance_seals_execution_and_forbids_replay(self):
        worker = self.worker
        def partial(action):
            worker.env.actions.append(action.copy())
            worker.env.steps += 1
            raise RuntimeError('injected failure after physical advance')
        worker.env.step = partial
        with self.assertRaisesRegex(RuntimeError, 'physical advance'):
            worker.step(np.zeros(23))
        self.assertIsNotNone(worker.uncertain)
        self.assertEqual((worker.submitted, worker.acknowledged), (1, 0))
        evidence = json.loads((worker.output / 'uncertain_action.json').read_text())
        self.assertTrue(evidence['replay_forbidden'])
        self.assertEqual((evidence['step_before'], evidence['step_after']), (0, 1))
        with self.assertRaises(RuntimeError):
            worker.step(np.zeros(23))
        self.assertEqual(len(worker.env.actions), 1)

    def test_wrong_step_count_is_not_acknowledged(self):
        worker = self.worker
        worker.env.step = lambda action: True
        with self.assertRaisesRegex(RuntimeError, 'acknowledgement'):
            worker.step(np.zeros(23))
        self.assertIsNotNone(worker.uncertain)
        self.assertEqual(worker.acknowledged, 0)

    def test_ack_storage_error_is_uncertain_and_does_not_count_as_ack(self):
        worker = self.worker
        worker.execution = AckWriteFailure(worker.execution)
        with self.assertRaisesRegex(OSError, 'ACK storage failure'):
            worker.step(np.zeros(23))
        self.assertEqual(worker.env.steps, 1)
        self.assertIsNotNone(worker.uncertain)
        self.assertEqual(worker.acknowledged, 0)
        with self.assertRaises(RuntimeError):
            worker.step(np.zeros(23))
        self.assertEqual(len(worker.env.actions), 1)

    def test_pixel_rejects_old_step_and_unknown_frame(self):
        args = self.frame()
        self.worker.env.steps = 1
        with self.assertRaisesRegex(ValueError, 'Stale frame'):
            self.worker.call('behavior_pixel_to_position', args)
        args['frame_id'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'Stale frame'):
            self.worker.call('behavior_pixel_to_position', args)

    def test_current_pixel_projects_without_privileged_object_state(self):
        args = self.frame()
        value, images = self.worker.call('behavior_pixel_to_position', args)
        self.assertEqual(images, [])
        self.assertTrue(np.allclose(value['point'], [0., 0., -1.]))
        self.assertNotIn('object', json.dumps(value).lower())

    def test_uncertain_action_invalidates_frame_even_without_step_increment(self):
        args = self.frame()
        worker = self.worker
        def partial(action):
            worker.env.actions.append(action.copy())
            raise RuntimeError('physical advance before bookkeeping failed')
        worker.env.step = partial
        with self.assertRaises(RuntimeError):
            worker.step(np.zeros(23))
        self.assertEqual(worker.env.steps, 0)
        with self.assertRaises((ValueError, RuntimeError)):
            worker.call('behavior_pixel_to_position', args)

    def test_finish_does_not_accept_unverified_success(self):
        worker = self.worker
        value, _ = worker.call('behavior_finish', {'outcome': 'success', 'reason': 'looks complete'})
        self.assertFalse(value['accepted'])
        self.assertIsNone(worker.finish_decision)

    def test_accepted_finish_blocks_further_physical_tools(self):
        worker = self.worker
        value, _ = worker.call('behavior_finish', {'outcome': 'blocked', 'reason': 'cannot proceed'})
        self.assertTrue(value['accepted'])
        with self.assertRaises(RuntimeError):
            worker.call('behavior_move_base', {'vx': .1, 'vy': 0, 'yaw_rate': 0, 'steps': 2})
        worker.control.move_base.assert_not_called()
        with self.assertRaises(RuntimeError):
            worker.step(np.zeros(23))
        self.assertEqual(worker.env.steps, 0)

    def test_finish_decision_cannot_be_overwritten(self):
        worker = self.worker
        worker.call('behavior_finish', {'outcome': 'blocked', 'reason': 'original reason'})
        original = dict(worker.finish_decision)
        try:
            worker.call('behavior_finish', {'outcome': 'failure', 'reason': 'replacement reason'})
        except (ValueError, RuntimeError):
            pass
        self.assertEqual(worker.finish_decision, original)
        self.assertEqual(json.loads((worker.output / 'agent_finish.json').read_text()), original)

    def test_finalization_does_not_claim_verified_video_after_verifier_error(self):
        with patch('services.behavior_ap.video.verify_videos', side_effect=RuntimeError('video invalid')):
            with self.assertRaisesRegex(RuntimeError, 'video invalid'):
                self.worker.finalize('test_finished')
        self.assertFalse((self.worker.output / 'episode_closed.json').exists())

    def test_finalization_preserves_partial_video_error_evidence(self):
        from services.behavior_ap.video import VideoVerificationError
        evidence = {'passed': False, 'failed_role': 'third_person', 'failed_stage': 'decode',
                    'stderr': 'injected damaged packet', 'videos': {'robot_cameras': {'passed': True}}}
        with patch('services.behavior_ap.video.verify_videos',
                   side_effect=VideoVerificationError('video invalid', evidence)):
            with self.assertRaises(VideoVerificationError):
                self.worker.finalize('test_finished')
        self.assertEqual(json.loads((self.worker.output / 'video_verification_error.json').read_text()), evidence)
        self.assertFalse((self.worker.output / 'episode_closed.json').exists())


class SocketCacheTests(unittest.TestCase):
    """A real local UNIX socket, but a fake worker and no simulation imports."""
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='bap-socket-')
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.path = self.output / 'worker.sock'
        self.worker = SimpleNamespace(output=self.output, ledger=FakeLedger(),
                                      call=Mock(return_value=({'step': 1}, [])))
        self.errors = []
        def run():
            try:
                serve(self.worker, {'socket_path': str(self.path), 'uid': os.getuid(), 'gid': os.getgid()})
            except BaseException as exc:
                self.errors.append(exc)
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 3
        while not self.path.exists() and self.thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertTrue(self.path.exists(), f'server did not bind: {self.errors}')
        self.addCleanup(self.stop)

    def stop(self):
        if self.thread.is_alive():
            try:
                self.request({'id': 'test-cleanup-close', 'name': '_close'})
            except (OSError, AssertionError):
                pass
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive(), 'socket server did not stop')

    def raw_request(self, raw):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(2)
            client.connect(str(self.path))
            client.sendall(raw)
            with client.makefile('rb') as stream:
                data = stream.readline()
        return json.loads(data) if data else None

    def request(self, value):
        result = self.raw_request((json.dumps(value) + '\n').encode())
        self.assertIsNotNone(result, f'server returned no reply: {self.errors}')
        return result

    def test_identical_duplicate_returns_cached_response_without_reexecution(self):
        request = {'id': 'one-action', 'name': 'behavior_move_base', 'arguments': {'vx': .1}}
        first = self.request(request)
        second = self.request(request)
        self.assertEqual(first, second)
        self.worker.call.assert_called_once()
        self.assertEqual(self.errors, [])

    def test_reusing_id_with_changed_payload_never_executes_new_action(self):
        request = {'id': 'one-action', 'name': 'behavior_move_base', 'arguments': {'vx': .1}}
        first = self.request(request)
        changed = self.request(dict(request, arguments={'vx': .2}))
        self.assertIn('error', changed)
        self.assertEqual(self.request(request), first)
        self.worker.call.assert_called_once()

    def test_error_reply_is_cached_so_partial_failure_is_not_replayed(self):
        self.worker.call.side_effect = RuntimeError('uncertain after partial physical execution')
        request = {'id': 'partial-action', 'name': 'behavior_move_base', 'arguments': {'vx': .1}}
        first = self.request(request)
        self.assertIn('error', first)
        self.assertEqual(self.request(request), first)
        self.worker.call.assert_called_once()

    def test_malformed_request_is_rejected_without_terminating_worker(self):
        result = self.raw_request(b'{invalid-json}\n')
        self.assertIsNotNone(result, f'malformed request terminated worker: {self.errors}')
        self.assertIn('error', result)
        self.assertTrue(self.thread.is_alive())
        self.assertEqual(self.request({'id': 'valid-after-malformed', 'name': 'behavior_state'})['value'], {'step': 1})
        self.worker.call.assert_called_once()

    def test_missing_name_is_rejected_without_terminating_worker(self):
        result = self.request({'id': 'missing-name'})
        self.assertIn('error', result)
        self.assertTrue(self.thread.is_alive())
        self.worker.call.assert_not_called()
        self.assertEqual(self.request({'id': 'valid-after-missing-name', 'name': 'behavior_state'})['value'], {'step': 1})

    def test_invalid_identity_is_rejected_without_terminating_worker(self):
        result = self.request({'id': [], 'name': 'behavior_state'})
        self.assertIn('error', result)
        self.assertTrue(self.thread.is_alive())
        self.worker.call.assert_not_called()
        self.assertEqual(self.request({'id': 'valid-after-invalid-id', 'name': 'behavior_state'})['value'], {'step': 1})


if __name__ == '__main__':
    unittest.main()
