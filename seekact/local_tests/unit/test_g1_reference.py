import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque

import numpy as np
import pytest
import zmq

from services.g1_sonic.camera_service import atomic_json
from services.g1_sonic.executor import validate_job
from services.g1_sonic.reference import from_native_idle, pulse_window, hold_window, pulse_envelope
from services.g1_sonic.state import NAMES, HARDWARE_TO_ISAAC


def fake_row(q=None, dq=None, motion='planner_motion'):
    q = np.zeros(29) if q is None else np.asarray(q)
    dq = np.zeros(29) if dq is None else np.asarray(dq)
    hwq, hwdq = np.zeros(29), np.zeros(29)
    hwq[HARDWARE_TO_ISAAC], hwdq[HARDWARE_TO_ISAAC] = q, dq
    obs = np.zeros(1762)
    obs[4:294] = np.tile(q, 10)
    obs[294:584] = np.tile(dq, 10)
    obs[601:661] = np.tile([1, 0, 0, 1, 0, 0], 10)
    now = time.monotonic()
    return dict(monotonic=now, state_monotonic_ns=int(now*1e9), lowstate_receive_monotonic_ns=int(now*1e9),
                body_q=hwq.tolist(), body_dq=hwdq.tolist(), base_quat=[1, 0, 0, 0], base_ang_vel=[0, 0, 0],
                encoder_source=True, motion=motion, encoder_obs=obs.tolist(), transport_fault=False, log_dropped=0)


def test_roundtrip_preserves_other_joints_and_original_velocity():
    row = fake_row(dq=np.full(29, .01))
    reference = from_native_idle(row)
    peak = pulse_window(reference, 'right', .03, 3., 100)
    i = NAMES.index('right_elbow_joint')
    assert peak['joint_pos'][0, i] == pytest.approx(.03)
    assert np.all(peak['joint_pos'][:, np.arange(29) != i] == 0)
    assert np.allclose(peak['joint_vel'][0], .01)
    end = pulse_window(reference, 'right', .03, 7., 450)
    np.testing.assert_array_equal(end['joint_pos'], reference['joint_pos'])
    np.testing.assert_array_equal(end['joint_vel'], reference['joint_vel'])
    np.testing.assert_array_equal(end['body_quat'], reference['body_quat'])
    t = np.linspace(0, 7, 7001)
    s, ds = pulse_envelope(t)
    assert max(np.abs(ds))*.05 < .05
    assert np.max(np.abs(np.gradient(s, .001)[1:-1]-ds[1:-1])) < 1e-5


@pytest.mark.parametrize('delta', [0, .051, float('nan'), float('inf'), True, '0.03'])
def test_invalid_reference_rejected(delta):
    with pytest.raises(ValueError):
        pulse_window(from_native_idle(fake_row()), 'right', delta, 0, 0)


def test_nonterminal_idle_refused():
    row = fake_row()
    row['encoder_obs'][4] = .1
    with pytest.raises(ValueError, match='terminal'):
        from_native_idle(row)


def test_future_and_stale_jobs_refused():
    now = time.monotonic()
    job = dict(id=uuid.uuid4().hex, action='elbow_pulse', baseline_id='baseline',
               side='right', delta_rad=.03, created_monotonic=now)
    validate_job(job, 'baseline', now)
    for modified in [dict(job, created_monotonic=now+1), dict(job, created_monotonic=now-3), dict(job, gain=10)]:
        with pytest.raises(ValueError):
            validate_job(modified, 'baseline', now)


def test_backend_preview_execution_is_one_shot_and_no_start_tool(tmp_path):
    from services.controller.config import Configuration
    from services.g1_sonic.backend import G1SonicBackend
    control = tmp_path/'control'; control.mkdir(); (control/'reference_requests').mkdir()
    atomic_json(control/'executor_status.json', dict(phase='reference_ready', fault=None, baseline_id='b',
        updated_monotonic=time.monotonic(), reference_lease_remaining_s=60, measured=dict(q=[0]*29)))
    b = G1SonicBackend(Configuration(root=tmp_path/'session', task='test', robot='g1_sonic',
                                   backend_options=dict(control_directory=str(control))))
    plan = b.call('g1_plan_arm_motion', dict(side='right', delta_rad=.03))[0]
    assert not (control/'reference_requests/pending.json').exists()
    result = b.call('g1_execute_reference', dict(plan_id=plan['plan_id']))[0]
    assert result['status'] == 'submitted' and result['measured_success'] is None
    with pytest.raises(ValueError, match='consumed'):
        b.call('g1_execute_reference', dict(plan_id=plan['plan_id']))
    assert not any('start' in t['name'] for t in b.tools())


@pytest.mark.parametrize('delay,stuck', [(0., None), (2., None), (0., 'elbow'), (0., 'wrist_roll')])
def test_executor_handoff_roundtrip_and_feedback_loss(tmp_path, delay, stuck):
    # Complete fake peer on a random localhost port. No DDS / real controller.
    with socket.socket() as port_probe:
        port_probe.bind(('127.0.0.1', 0))
        port = port_probe.getsockname()[1]
    done, freeze = threading.Event(), threading.Event()
    (tmp_path/'controller.log').write_text('Init Done\n')

    def peer():
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b'')
        sub.connect(f'tcp://127.0.0.1:{port}')
        q, dq, motion = np.zeros(29), np.zeros(29), 'planner_motion'
        queue = deque()
        moved = False
        elbow = NAMES.index('right_elbow_joint')
        with (tmp_path/'controller_trace.jsonl').open('w') as f:
            while not done.is_set():
                if sub.poll(10):
                    msg = sub.recv()
                    if msg.startswith(b'pose'):
                        header = json.loads(msg[4:1284].rstrip(b'\0'))
                        offset = 1284
                        fields = {}
                        for field in header['fields']:
                            dtype = {'f32': '<f4', 'i64': '<i8'}[field['dtype']]
                            n = int(np.prod(field['shape']))
                            fields[field['name']] = np.frombuffer(msg, dtype=dtype, count=n, offset=offset).reshape(field['shape'])
                            offset += n*np.dtype(dtype).itemsize
                        queue.append((time.monotonic()+delay, fields['joint_pos'][0], fields['joint_vel'][0]))
                while queue and queue[0][0] <= time.monotonic():
                    _, q, dq = queue.popleft()
                    motion = 'streamed'
                moved = moved or q[elbow] > .02
                if not freeze.is_set():
                    row = fake_row(q, dq, motion)
                    if stuck and moved:
                        blocked = NAMES.index('right_'+stuck+'_joint')
                        row['body_q'][HARDWARE_TO_ISAAC[blocked]] = .1
                    f.write(json.dumps(row)+'\n'); f.flush()
        sub.close(0); ctx.term()

    thread = threading.Thread(target=peer); thread.start()
    log = (tmp_path/'executor.log').open('w')
    process = subprocess.Popen([sys.executable, '-m', 'services.g1_sonic.executor', '--run', str(tmp_path),
        '--port', str(port), '--start-policy', '--reference-hold', '--lease-seconds', '30'], stdout=log, stderr=log)

    def wait_for(predicate, timeout=15):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            assert process.poll() is None, (tmp_path/'executor.log').read_text()
            path = tmp_path/'executor_status.json'
            if path.exists():
                state = json.loads(path.read_text())
                if predicate(state):
                    return state
                assert state['phase'] != 'fault_idle', (tmp_path/'executor.log').read_text()
            time.sleep(.05)
        pytest.fail((tmp_path/'executor.log').read_text())

    try:
        state = wait_for(lambda s: s['phase'] == 'reference_ready')
        job = dict(id=uuid.uuid4().hex, action='elbow_pulse', baseline_id=state['baseline_id'],
                   created_monotonic=time.monotonic(), side='right', delta_rad=.03)
        atomic_json(tmp_path/'reference_requests/pending.json', job)
        state = wait_for(lambda s: s.get('last_completed') is not None, timeout=20)
        result = state['last_completed']
        assert result['native_peak_reference_observed'] and result['native_return_reference_observed']
        assert result['measured_response_detected']
        if stuck:
            assert not result['completed'] and not result['measured_body_return_within_0_05_rad']
            assert result['measured_return_within_0_05_rad'] == (stuck != 'elbow')
            assert state['phase'] == 'reference_review_required'
            assert result['reason'] == 'reference_or_measured_return_not_verified'
            job.update(id=uuid.uuid4().hex, created_monotonic=time.monotonic())
            atomic_json(tmp_path/'reference_requests/pending.json', job)
            deadline = time.monotonic()+2
            rejected_path = tmp_path/'reference_results'/(job['id']+'.json')
            while not rejected_path.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            assert not json.loads(rejected_path.read_text())['completed']
        else:
            assert result['completed'] and result['measured_return_within_0_05_rad']
            if delay:
                assert result['elapsed_s'] > 8.2, 'Must wait for delayed native return, not seven-second sender timer'
        freeze.set()
        state = wait_for(lambda s: s['phase'] == 'fault_idle', timeout=3)
        assert 'stale' in state['fault']
    finally:
        process.send_signal(signal.SIGINT); process.wait(timeout=5)
        done.set(); thread.join(timeout=3); log.close()
