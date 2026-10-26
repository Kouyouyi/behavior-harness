import json
import uuid

import numpy as np
import pytest

from services.g1_sonic.gait_flow import GaitFlow,check_site,LEGS,NONLEGS
from services.g1_sonic.reference import from_native_idle
from services.g1_sonic.executor import validate_job
from local_tests.unit.test_g1_reference import fake_row
from local_tests.unit.test_g1_reach import Q


def measured(q=Q,dq=None):
    return dict(q=np.asarray(q).tolist(),dq=[0.]*29 if dq is None else dq,
                base_ang_vel=[0.,0.,0.],tilt_rad=0.)


def test_step_requires_native_peak_and_measured_settling():
    ref=from_native_idle(fake_row(Q))
    for omit_left,moving,success in [(False,False,True),(True,False,False),(False,True,False)]:
        flow=GaitFlow(ref,measured(),100.)
        for t in np.arange(0,20,.02):
            packet=flow.fields(100+t,round(t*50))
            q=packet['joint_pos'][0].copy()
            if omit_left:q[9]=Q[9]
            row=fake_row(q,packet['joint_vel'][0],motion='streamed')
            feedback=measured(q,[.4]*29 if moving else None)
            flow.step(feedback,row,100+t)
            if flow.done:break
        assert flow.done and flow.result['completed'] is success
        assert flow.result['actual_distance_m'] is None
        assert not flow.result['physical_footsteps_verified']
        assert not flow.result['automatic_repeat_allowed']


def test_walk_gate_and_profile_guards(tmp_path):
    ref=from_native_idle(fake_row(Q));flow=GaitFlow(ref,measured(),0)
    flow.guard(measured())
    for ids,speed in [(LEGS,6.1),(NONLEGS,1.6)]:
        dq=np.zeros(29);dq[ids[0]]=speed
        with pytest.raises(ValueError,match='speed'):flow.guard(measured(dq=dq.tolist()))
    q=Q.copy();q[LEGS[0]]=flow.high[LEGS[0]]+.01
    with pytest.raises(ValueError,match='envelope'):flow.guard(measured(q))
    job=dict(id=uuid.uuid4().hex,action='walk_step',baseline_id='b',created_monotonic=1.)
    with pytest.raises(ValueError):validate_job(job,'b',1.,True,True)
    validate_job(job,'b',1.,True,True,True)
    path=tmp_path/'clearance.json'
    site=dict(baseline_id='b',monotonic=1.,q=Q.tolist(),feet_weight_bearing=True,
              harness_allows_translation=True,operator_at_stop=True,arms_clear=True,
              forward_corridor_checked=True,minimum_forward_clearance_m=.4)
    path.write_text(json.dumps(site));check_site(path,'b',measured(),1.)
    for change in [dict(harness_allows_translation=False),dict(monotonic=-120.),dict(minimum_forward_clearance_m=.1)]:
        path.write_text(json.dumps(site|change))
        with pytest.raises(ValueError):check_site(path,'b',measured(),1.)
