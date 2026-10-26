import json
import time
import uuid
from collections import deque
import socket
import subprocess
import sys
import threading

import zmq

import numpy as np
import pytest

from services.g1_sonic.reach import Kinematics, ReachFlow, window, check_site, OTHER_IDS
from services.g1_sonic.executor import validate_job
from services.g1_sonic.reference import from_native_idle
from services.g1_sonic.state import NAMES
from local_tests.unit.test_g1_reference import fake_row

# Fixed native terminal reference from the successful E022 simulation.
Q = np.array([.071936,.100641,.006323,.078074,-.080004,.020134,.156645,-.352879,.141875,
    .147112,.074404,-.028259,-.009473,-.177348,-.123037,.273276,-.266092,-.111142,.063276,
    -.609014,.664173,1.021968,1.028074,.055853,-.057929,.053045,.194696,.012630,-.134996])


def run_flow(action, reference, stuck=False, **kwargs):
    kin=Kinematics(); now=time.monotonic()
    measured=dict(q=reference['joint_pos'][0].tolist(),dq=[0.]*29)
    flow=ReachFlow(reference,measured,action,kin,now,**kwargs)
    flow.future.result(timeout=5)
    pending=deque(); observed=flow.fields(now,0)
    try:
        for tick in range(1300):
            at=now+tick*.05
            pending.append((at+1.,flow.fields(at,tick*2)))
            while pending and pending[0][0]<=at:
                _,observed=pending.popleft()
            q=observed['joint_pos'][0].copy()
            if stuck and flow.stage=='return':
                q[NAMES.index('right_wrist_roll_joint')]+=.45
            row=fake_row(observed['joint_pos'][0],observed['joint_vel'][0],motion='streamed')
            # Preserve the entire delayed future window in the fake encoder.
            row['encoder_obs'][4:294]=observed['joint_pos'][:10].ravel().tolist()
            row['encoder_obs'][294:584]=observed['joint_vel'][:10].ravel().tolist()
            measured=dict(q=q.tolist(),dq=[0.]*29)
            flow.step(measured,row,at)
            if flow.done: return flow
        pytest.fail('Reach profile did not terminate')
    finally:
        flow.close()


def test_task_space_reach_waits_for_delayed_readback_and_rejects_wrist_error():
    reference=from_native_idle(fake_row(Q))
    clear=run_flow('clear_arms',reference)
    assert clear.result['completed'] and clear.result['awaits_site_observation']
    assert not clear.result['site_clearance_verified']
    np.testing.assert_array_equal(clear.target[OTHER_IDS],reference['joint_pos'][0,OTHER_IDS])
    reference['joint_pos'][:]=clear.target
    passed=run_flow('reach_return',reference)
    assert passed.result['completed'] and passed.result['elapsed_s']>=14
    assert passed.result['reach_displacement_base_m'][0]>.09
    failed=run_flow('reach_return',reference,stuck=True)
    assert not failed.result['completed']
    assert failed.result['return_orientation_error_rad']>.2


def test_non_arm_reference_and_operator_site_evidence_are_enforced(tmp_path):
    reference=from_native_idle(fake_row(Q)); q=Q.copy(); q[OTHER_IDS[0]]+=.01
    with pytest.raises(ValueError,match='non-arm'):
        window(reference,Q,q,1.,0)
    job=dict(id=uuid.uuid4().hex,action='reach_return',baseline_id='b',created_monotonic=time.monotonic())
    with pytest.raises(ValueError): validate_job(job,'b',time.monotonic())
    validate_job(job,'b',time.monotonic(),allow_reach=True)
    with pytest.raises(ValueError): validate_job(dict(job,arbitrary_target=[1,2,3]),'b',time.monotonic(),True)
    path=tmp_path/'operator_clearance.json'; now=time.monotonic()
    note=dict(baseline_id='b',phase='clear_arms',monotonic=now,q=Q.tolist(),feet_bearing_weight=True,both_arms_clear=True)
    path.write_text(json.dumps(note)); check_site(path,'b',dict(q=Q),now)
    for changed in [dict(note,feet_bearing_weight=False),dict(note,both_arms_clear=False),dict(note,monotonic=now-121),dict(note,baseline_id='old')]:
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError): check_site(path,'b',dict(q=Q),now)


def test_grasp_step_holds_endpoint_and_requires_explicit_profile():
    from services.g1_sonic.reach import validate_step, ARM_IDS
    from services.g1_sonic.hand import command_values
    reference=from_native_idle(fake_row(Q))
    clear=run_flow('clear_arms',reference)
    reference['joint_pos'][:]=clear.target
    flow=run_flow('grasp_step',reference,delta_m=[.04,0,.06],rotation_delta=[0,0,0])
    assert flow.result['completed'] and flow.result['holds_endpoint'] and flow.stage=='step'
    assert not flow.result['physical_grasp_tested']
    before=np.array(flow.result['before']['right']['position_m'])
    after=np.array(flow.result['after']['right']['position_m'])
    np.testing.assert_allclose(after-before,[.04,0,.06],atol=.005)
    np.testing.assert_array_equal(flow.target[OTHER_IDS],reference['joint_pos'][0,OTHER_IDS])
    np.testing.assert_array_equal(flow.target[ARM_IDS['left']],reference['joint_pos'][0,ARM_IDS['left']])
    for d,r in [([.11,0,0],[0,0,0]),([0,0,0],[0,.36,0]),([float('nan'),0,0],[0,0,0])]:
        with pytest.raises(ValueError):validate_step(d,r)
    job=dict(id=uuid.uuid4().hex,action='grasp_step',baseline_id='b',created_monotonic=time.monotonic(),
             delta_m=[.04,0,.06],rotation_delta=[0,0,0])
    with pytest.raises(ValueError):validate_job(job,'b',time.monotonic(),True)
    validate_job(job,'b',time.monotonic(),True,True)
    q,dq=command_values([.4,.4,.05,.05,.05,.05],[0,1,0,0,0,0],.5,.2)
    np.testing.assert_allclose(q,[.2,.7,.025,.025,.025,.025]);assert dq==[.2]*6
    with pytest.raises(ValueError):command_values([0]*6,[1.1]*6,1,.2)


def test_planning_rejection_is_distinct_from_failed_physical_verification():
    class RejectIK(Kinematics):
        def solve(self,*args,**kwargs):raise ValueError('unreachable test target')
    reference=from_native_idle(fake_row(Q));kin=RejectIK();now=time.monotonic()
    measured=dict(q=Q.tolist(),dq=[0.]*29)
    flow=ReachFlow(reference,measured,'grasp_step',kin,now,delta_m=[.04,0,.06],rotation_delta=[0,0,0])
    try:
        with pytest.raises(ValueError):flow.future.result(timeout=5)
        flow.step(measured,fake_row(Q),now+.1)
        assert not flow.result['completed'] and flow.result['planning_only_rejection']
        assert not flow.result['motion_started']
        np.testing.assert_array_equal(flow.fields(now+5,0)['joint_pos'],reference['joint_pos'])
        flow.stage='step';flow.fail('native_reference_or_measured_settling_timeout')
        assert flow.result['motion_started'] and not flow.result['planning_only_rejection']
    finally:flow.close()


@pytest.mark.parametrize('next_action',['reach_return','walk_step'])
def test_owner_clear_then_site_gate_then_reach_and_feedback_fault(tmp_path,next_action):
    """Actual owner process and v1 localhost wire; no native controller or DDS."""
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0)); port=probe.getsockname()[1]
    done=threading.Event(); freeze=threading.Event();manual_adjust=threading.Event()
    (tmp_path/'controller.log').write_text('Init Done\n')
    def peer():
        ctx=zmq.Context(); sub=ctx.socket(zmq.SUB);sub.setsockopt(zmq.SUBSCRIBE,b'')
        sub.connect(f'tcp://127.0.0.1:{port}')
        q=np.tile(Q,(55,1));dq=np.zeros((55,29));motion='planner_motion'
        with (tmp_path/'controller_trace.jsonl').open('w') as log:
            while not done.is_set():
                if sub.poll(10):
                    msg=sub.recv()
                    if msg.startswith(b'pose'):
                        head=json.loads(msg[4:1284].rstrip(b'\0'));offset=1284;fields={}
                        for f in head['fields']:
                            dtype={'f32':'<f4','i64':'<i8'}[f['dtype']];n=int(np.prod(f['shape']))
                            fields[f['name']]=np.frombuffer(msg,dtype=dtype,count=n,offset=offset).reshape(f['shape'])
                            offset+=n*np.dtype(dtype).itemsize
                        q,dq=fields['joint_pos'],fields['joint_vel'];motion='streamed'
                if not freeze.is_set():
                    row=fake_row(q[0],np.zeros(29),motion)
                    if manual_adjust.is_set():
                        from services.g1_sonic.state import HARDWARE_TO_ISAAC
                        row['body_dq'][HARDWARE_TO_ISAAC[NAMES.index('left_hip_yaw_joint')]]=2.
                    row['encoder_obs'][4:294]=q[:10].ravel().tolist()
                    row['encoder_obs'][294:584]=dq[:10].ravel().tolist()
                    log.write(json.dumps(row)+'\n');log.flush()
        sub.close(0);ctx.term()
    thread=threading.Thread(target=peer);thread.start()
    with (tmp_path/'owner.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,'-m','services.g1_sonic.executor','--run',str(tmp_path),
            '--port',str(port),'--start-policy','--reference-hold','--reach-profile','--gait-profile','--lease-seconds','90'],stdout=log,stderr=log)
        def wait(predicate,timeout=20):
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                assert proc.poll() is None,(tmp_path/'owner.log').read_text()
                path=tmp_path/'executor_status.json'
                if path.exists():
                    state=json.loads(path.read_text())
                    if predicate(state):return state
                time.sleep(.04)
            pytest.fail((tmp_path/'owner.log').read_text())
        def submit(action,state):
            job=dict(id=uuid.uuid4().hex,action=action,baseline_id=state['baseline_id'],created_monotonic=time.monotonic())
            temp=tmp_path/'reference_requests/new.tmp';temp.write_text(json.dumps(job));temp.replace(tmp_path/'reference_requests/pending.json')
            return job['id']
        try:
            state=wait(lambda s:s['phase']=='reference_ready')
            first=submit('clear_arms',state)
            state=wait(lambda s:s.get('last_completed',{}).get('id')==first if s.get('last_completed') else False)
            assert state['last_completed']['completed'] and state['posture']=='clear_arms'
            if next_action=='walk_step':
                manual_adjust.set();time.sleep(.2)
                state=wait(lambda s:max(map(abs,s['measured']['dq']))>1.5)
                assert state['phase']=='reference_ready' and state['fault'] is None
                while_moving=submit(next_action,state)
                moving_result=tmp_path/'reference_results'/(while_moving+'.json')
                wait(lambda s:moving_result.exists())
                assert 'moving' in json.loads(moving_result.read_text())['reason']
                manual_adjust.clear();time.sleep(1.2)
            missing=submit(next_action,state)
            rejected=tmp_path/'reference_results'/(missing+'.json')
            wait(lambda s:rejected.exists())
            assert not json.loads(rejected.read_text())['completed']
            state=wait(lambda s:s['phase']=='reference_ready')
            # Synthetic site evidence exists only inside this fake-peer test.
            (tmp_path/'operator_clearance.json').write_text(json.dumps(dict(baseline_id=state['baseline_id'],
                phase='clear_arms',monotonic=time.monotonic(),q=state['measured']['q'],feet_bearing_weight=True,both_arms_clear=True)))
            (tmp_path/'operator_walk_clearance.json').write_text(json.dumps(dict(baseline_id=state['baseline_id'],
                monotonic=time.monotonic(),q=state['measured']['q'],feet_weight_bearing=True,
                harness_allows_translation=True,operator_at_stop=True,arms_clear=True,
                forward_corridor_checked=True,minimum_forward_clearance_m=.4)))
            second=submit(next_action,state)
            state=wait(lambda s:s.get('last_completed',{}).get('id')==second if s.get('last_completed') else False,25)
            assert state['last_completed']['completed'] and state['phase']=='reference_ready'
            if next_action=='walk_step':
                assert state['walk_attempted'] and not state['last_completed']['physical_footsteps_verified']
                repeated=submit('walk_step',state)
                rejected=tmp_path/'reference_results'/(repeated+'.json')
                wait(lambda s:rejected.exists())
                assert not json.loads(rejected.read_text())['completed']
            freeze.set()
            state=wait(lambda s:s['phase']=='fault_idle',3)
            assert 'stale' in state['fault']
        finally:
            proc.terminate();proc.wait(timeout=5);done.set();thread.join(timeout=5)
