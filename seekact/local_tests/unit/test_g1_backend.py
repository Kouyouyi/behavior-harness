import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np
import pytest

from services.controller.config import Configuration
from services.controller.frontend import Frontend
from services.controller.session import Session
from services.g1_sonic.backend import G1SonicBackend
from services.g1_sonic.state import HARDWARE_TO_ISAAC, checked_state


def backend(tmp_path, **options):
    return G1SonicBackend(Configuration(root=tmp_path/'session', task='g1_test', robot='g1_sonic',
                                         backend_options=options))


def frame(path, age=0):
    path.mkdir()
    frame_id = uuid.uuid4().hex
    folder = path/frame_id
    folder.mkdir()
    meta = dict(frame_id=frame_id, host_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                received_monotonic=time.monotonic()-age, source='live_camera',
                K=[[100, 0, 320], [0, 100, 180], [0, 0, 1]], optical_frame='rectified_left')
    (path/'latest.json').write_text(json.dumps(meta))
    (path/'health.json').write_text(json.dumps(dict(running=True, errors=[], updated_monotonic=time.monotonic())))
    cv2.imwrite(str(folder/'rgb.png'), np.zeros((360, 640, 3), np.uint8))
    cv2.imwrite(str(folder/'depth.png'), np.full((360, 640), 2000, np.uint16))
    cv2.imwrite(str(folder/'valid.png'), np.full((360, 640), 255, np.uint8))


def test_readonly_constructor_and_missing_state(tmp_path):
    b = backend(tmp_path)
    s = b.status()
    assert not s['motion_available'] and 'body_error' in s and 'camera_error' in s
    assert b.mutating == {'g1_execute_reference','g1_vla','g1_hand','g1_move','g1_session_lease'}
    b.close(); b.close()


def test_grasp_plan_submission_preserves_bounded_delta(tmp_path):
    control=tmp_path/'control';control.mkdir();(control/'reference_requests').mkdir()
    status=dict(updated_monotonic=time.monotonic(),phase='reference_ready',fault=None,
                reference_lease_remaining_s=120,baseline_id='test',measured=dict(q=[0]*29),
                posture='clear_arms',reach_profile_enabled=True,grasp_profile_enabled=True)
    (control/'executor_status.json').write_text(json.dumps(status))
    b=backend(tmp_path,control_directory=str(control))
    args=dict(delta_m=[.04,0,.06],rotation_delta=[0,0,0])
    plan=b.call('g1_plan_grasp_step',args)[0]
    assert plan['holds_endpoint'] and not plan['motion_sent']
    sent=b.call('g1_execute_reference',dict(plan_id=plan['plan_id']))[0]
    job=json.loads((control/'reference_requests/pending.json').read_text())
    assert job['id']==sent['job_id'] and job['action']=='grasp_step'
    assert job['delta_m']==args['delta_m'] and job['rotation_delta']==args['rotation_delta']
    with pytest.raises(ValueError):b.call('g1_execute_reference',dict(plan_id=plan['plan_id']))
    b.close()


def test_walk_preview_submits_only_fixed_pair_once(tmp_path):
    control=tmp_path/'control';control.mkdir();(control/'reference_requests').mkdir()
    status=dict(updated_monotonic=time.monotonic(),phase='reference_ready',fault=None,
                reference_lease_remaining_s=120,baseline_id='test',measured=dict(q=[0]*29),
                posture='clear_arms',reach_profile_enabled=True,gait_profile_enabled=True,walk_attempted=False)
    (control/'executor_status.json').write_text(json.dumps(status))
    b=backend(tmp_path,control_directory=str(control))
    plan=b.call('g1_plan_walk',{})[0]
    assert plan['actual_distance_m'] is None and not plan['motion_sent']
    b.call('g1_execute_reference',dict(plan_id=plan['plan_id']))
    job=json.loads((control/'reference_requests/pending.json').read_text())
    assert job['action']=='walk_step' and set(job)=={'id','action','baseline_id','created_monotonic'}
    status.update(walk_attempted=True,updated_monotonic=time.monotonic())
    (control/'executor_status.json').write_text(json.dumps(status))
    with pytest.raises(RuntimeError):b.call('g1_plan_walk',{})
    b.close()


def test_mcp_observation_and_geometry(tmp_path):
    camera = tmp_path/'camera'
    frame(camera)
    b = backend(tmp_path, camera_directory=str(camera))
    f = Frontend(Session(b.config), b)
    try:
        result = f.call('g1_observe', {})
        assert not result.get('isError'), result
        value = json.loads(result['content'][0]['text'])
        assert any(c['type'] == 'image' for c in result['content'])
        p = b.call('g1_pixel_to_position', dict(frame_id=value['frame_id'], x=330, y=180))[0]
        assert p['position_m'] == pytest.approx([.2, 0, 2])
        with pytest.raises(ValueError, match='extrinsics'):
            b.call('g1_pixel_to_position', dict(frame_id=value['frame_id'], x=330, y=180, coordinate_frame='base'))
    finally:
        f.close()


def test_stale_and_stopped_camera_refused(tmp_path):
    camera = tmp_path/'camera'
    frame(camera, age=2)
    b = backend(tmp_path, camera_directory=str(camera))
    with pytest.raises(RuntimeError, match='stale'):
        b.observe()
    health = camera/'health.json'
    health.write_text(json.dumps(dict(running=False, errors=[], updated_monotonic=time.monotonic())))
    with pytest.raises(RuntimeError, match='stopped'):
        b.observe()


@pytest.mark.parametrize('args', [dict(frame_id='../escape', x=0, y=0),
    dict(frame_id='a'*32, x=True, y=0), dict(frame_id='a'*32, x=640, y=0),
    dict(frame_id='a'*32, x=0, y=0, radius=-1)])
def test_pixel_arguments_rejected(tmp_path, args):
    with pytest.raises(ValueError):
        backend(tmp_path).call('g1_pixel_to_position', args)


def test_real_feedback_order_and_freshness():
    now = time.monotonic()
    row = dict(monotonic=now, lowstate_receive_monotonic_ns=int(now*1e9), state_monotonic_ns=int(now*1e9),
               body_q=list(range(29)), body_dq=[0]*29, base_quat=[1, 0, 0, 0], base_ang_vel=[0, 0, 0],
               motion='planner_motion', encoder_source=True)
    result = checked_state(row, now)
    assert result['q'] == HARDWARE_TO_ISAAC
    with pytest.raises(ValueError, match='stale'):
        checked_state(row, now+1)
    with pytest.raises(ValueError, match='fault'):
        checked_state(dict(row, transport_fault=True), now)


def test_g1_workspace_contains_only_available_tools(tmp_path):
    from services.workspace import Setup, build, recipe_for
    build(recipe_for(), Setup('g1_sonic', 'observe_g1', subagents=0, gemini=False), tmp_path/'ws')
    interface = (tmp_path/'ws/.agents/skills/robot-interface/SKILL.md').read_text()
    assert 'g1_observe' in interface and 'rby1_' not in interface
