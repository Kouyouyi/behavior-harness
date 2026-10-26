"""Single-owner hand open/close, interlocks and preservation of other targets."""
from types import SimpleNamespace
import json
import time

import numpy as np
import pytest

from services.g1_sonic.vla import VLAJobs, atomic, submit_hand, result, submit
from services.g1_sonic.vla_runtime import Runtime


def runtime():
    rt=Runtime.__new__(Runtime)
    rt.steps=0;rt.hand_motion=None
    rt.session=SimpleNamespace(pause=lambda:None)
    rt.pose=SimpleNamespace(cancel=lambda:None)
    rt.state=SimpleNamespace(get_msg=lambda:{'fresh':True})
    rt.bridge=SimpleNamespace(feedback=lambda:{},command=lambda l,r:None)
    rt.rtc=SimpleNamespace(legacy=SimpleNamespace(
        feedback_hands=lambda p:{'left_hand':np.ones(6)*.3,'right_hand':np.ones(6)*.8},
        load_poses=lambda:{s:{'open':[0,1,0,0,0,0],'closed':[1]*6} for s in ('left','right')}))
    rt.last_hand_targets={'left_hand_joints':np.ones(6,dtype=np.float32)*.6,'right_hand_joints':np.ones(6,dtype=np.float32)*.9}
    rt.model_path='fixture'
    return rt


@pytest.mark.parametrize('side',['left','right'])
def test_open_preserves_other_target_and_never_returns_body_token(side):
    rt=runtime();other='right' if side=='left' else 'left'
    hold=rt.last_hand_targets[other+'_hand_joints'].copy();rt.release_hand(side)
    for i in range(101):
        action,done=rt.poll()
        assert 'motion_token' not in action
        np.testing.assert_array_equal(action[other+'_hand_joints'],hold)
        rt.apply_hands(action)
        assert done==(i==100)
    np.testing.assert_array_equal(action[side+'_hand_joints'],[0,1,0,0,0,0])
    rt.pause();assert rt.hand_motion is None


def setup(tmp_path):
    rt=runtime();jobs=VLAJobs(tmp_path,'session',rt);jobs.initialized=True
    atomic(tmp_path/'ankle_status.json',dict(session_id='session',phase='ready',fault=None,
        updated_monotonic=time.monotonic(),lease_remaining_seconds=600,vla=jobs.snapshot()))
    return rt,jobs


def tick(jobs,**kw):
    return jobs.tick(time.monotonic(),**(dict(available=True,settled=True,anchor=[0]*64,lease_remaining=600)|kw))


def test_release_job_completion_and_thermal_preemption(tmp_path):
    rt,jobs=setup(tmp_path)
    job=submit_hand(tmp_path,'right','open','User requested release; toy supported on table')['job_id']
    for _ in range(101):assert tick(jobs) is None
    assert result(tmp_path,job)['reason']=='hand_open_command_complete'
    assert jobs.initialized and not jobs.moving
    job=submit_hand(tmp_path,'right','open','Still supported')['job_id'];tick(jobs)
    jobs.pause('ankle_temperature_priority',invalidate=True)
    assert not result(tmp_path,job)['completed'] and not jobs.initialized and rt.hand_motion is None


@pytest.mark.parametrize('changes',[{'hand_control_ready':False},{'active_job':{'id':'busy'}}])
def test_unknown_targets_or_running_refused(tmp_path,changes):
    rt,jobs=setup(tmp_path);p=tmp_path/'ankle_status.json';s=json.loads(p.read_text());s['vla'].update(changes);atomic(p,s)
    with pytest.raises(RuntimeError):submit_hand(tmp_path,'right','open','Supported object')
    assert not (tmp_path/'vla_requests/pending.json').exists()


def test_pause_supersedes_release_and_unsettled_is_rejected(tmp_path):
    rt,jobs=setup(tmp_path)
    job=submit_hand(tmp_path,'right','open','Supported object')['job_id'];submit(tmp_path,'pause');tick(jobs)
    assert result(tmp_path,job)['reason']=='superseded_by_pause'
    job=submit_hand(tmp_path,'right','open','Supported object')['job_id'];tick(jobs,settled=False)
    assert result(tmp_path,job)['reason']=='robot_not_settled'


def test_stale_body_feedback_stops_release(tmp_path):
    rt,jobs=setup(tmp_path)
    job=submit_hand(tmp_path,'right','open','Supported object')['job_id'];tick(jobs)
    rt.state.get_msg=lambda:None
    assert tick(jobs) is None
    assert not result(tmp_path,job)['completed'] and rt.hand_motion is None


@pytest.mark.parametrize('side',['left','right','both'])
@pytest.mark.parametrize('action',['open','close'])
def test_all_hand_commands_via_mcp(tmp_path,side,action):
    from services.controller.config import Configuration
    from services.controller.frontend import Frontend
    from services.controller.session import Session
    from services.g1_sonic.backend import G1SonicBackend
    rt,jobs=setup(tmp_path);before={k:v.copy() for k,v in rt.last_hand_targets.items()}
    writes=[];rt.bridge.command=lambda left,right:writes.append((left.copy(),right.copy()))
    config=Configuration(root=tmp_path/'session',task='hands',robot='g1_sonic',backend_options={'control_directory':str(tmp_path)})
    front=Frontend(Session(config),G1SonicBackend(config))
    try:
        response=front.call('g1_hand',dict(side=side,action=action,evidence='Observed hand/object clearance and requested action'))
        assert not response.get('isError'),response
        job=json.loads(response['content'][0]['text'])['job_id']
        for _ in range(101):assert tick(jobs) is None
        response=front.call('g1_hand_result',dict(job_id=job))
        assert not response.get('isError'),response
        result=json.loads(response['content'][0]['text'])
        assert result['completed'] and result['side']==side and result['hand_action']==action
        assert result['reason']==f'hand_{action}_command_complete'
        assert not result['task_success_verified']
        for index,s in enumerate(('left','right')):
            if side in (s,'both'):
                np.testing.assert_array_equal(writes[-1][index],[0,1,0,0,0,0] if action=='open' else [1]*6)
            else:
                for pair in writes:np.testing.assert_array_equal(pair[index],before[s+'_hand_joints'])
    finally:front.close()


def test_invalid_second_target_prevents_both_hand_writes(tmp_path):
    rt,jobs=setup(tmp_path);writes=[];rt.bridge.command=lambda *args:writes.append(args)
    rt.rtc.legacy.load_poses=lambda:{'left':{'closed':[1]*6},'right':{'closed':[float('nan')]*6}}
    job=submit_hand(tmp_path,'both','close','Both hands clear')['job_id']
    assert tick(jobs) is None
    assert not result(tmp_path,job)['completed'] and not writes
    assert result(tmp_path,job)['steps_published']==0


def test_stale_hand_feedback_stops_further_commands(tmp_path):
    rt,jobs=setup(tmp_path);writes=[];rt.bridge.command=lambda *args:writes.append(args)
    job=submit_hand(tmp_path,'both','close','Both hands clear')['job_id'];tick(jobs)
    def stale(_):raise RuntimeError('stale right hand feedback')
    rt.rtc.legacy.feedback_hands=stale
    assert tick(jobs) is None
    assert len(writes)==1 and not result(tmp_path,job)['completed']


def test_hand_after_movement_does_not_require_pose_reset(tmp_path,monkeypatch):
    from test_g1_locomotion import Driver
    from services.g1_sonic import locomotion
    driver=Driver(tmp_path,monkeypatch);rt,jobs=setup(tmp_path);driver.vla=jobs
    before={k:v.copy() for k,v in rt.last_hand_targets.items()}
    move_job=driver.submit(direction='backward');driver.tick();driver.run_to_end()
    assert locomotion.result(tmp_path,move_job)['completed']
    assert not jobs.initialized and jobs.snapshot()['hand_control_ready']
    for k,v in before.items():np.testing.assert_array_equal(rt.last_hand_targets[k],v)
    driver.refresh();p=tmp_path/'ankle_status.json';state=json.loads(p.read_text());state['vla']=jobs.snapshot();atomic(p,state)
    hand_job=submit_hand(tmp_path,'both','open','Movement settled; object supported')['job_id']
    for _ in range(101):assert tick(jobs) is None
    assert result(tmp_path,hand_job)['completed']
    # Hand control neither changes posture tokens nor reinitializes VLA.
    assert not jobs.initialized


def test_pending_move_and_hand_are_serialized_at_owner_tick(tmp_path,monkeypatch):
    from test_g1_locomotion import Driver
    driver=Driver(tmp_path,monkeypatch);rt,jobs=setup(tmp_path);driver.vla=jobs
    driver.submit();p=tmp_path/'ankle_status.json';state=json.loads(p.read_text());state['vla']=jobs.snapshot();atomic(p,state)
    hand_job=submit_hand(tmp_path,'left','close','Hand clear')['job_id']
    assert driver.tick() is not None and driver.jobs.busy
    assert tick(jobs,available=False) is None
    assert result(tmp_path,hand_job)['reason']=='owner_busy'
    assert rt.hand_motion is None


def test_hand_submit_refuses_active_locomotion(tmp_path):
    rt,jobs=setup(tmp_path);p=tmp_path/'ankle_status.json';state=json.loads(p.read_text())
    state['locomotion']={'active_job':{'id':'moving'}};atomic(p,state)
    with pytest.raises(RuntimeError,match='Movement must stop'):
        submit_hand(tmp_path,'both','close','Hand clear')
    assert not (tmp_path/'vla_requests/pending.json').exists()
