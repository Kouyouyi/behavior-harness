import numpy as np
import pytest

from services.g1_sonic.walk import pulse,upper_fields,UPPER_IDS,validate_walk
from services.g1_sonic.state import NAMES


def test_walk_pulse_has_finite_duration_and_returns_idle():
    times=np.linspace(0,1.5,1501);speeds=np.array([pulse(t)['speed'] for t in times])
    assert np.max(speeds)<=.1
    assert np.trapz(speeds,times)==pytest.approx(.1,abs=1e-6)
    for t in [-1,0,1.5,2,60]:
        result=pulse(t);assert result['mode']==0 and result['speed']==0 and result['movement']==[0,0,0]
    assert pulse(.75)['mode']==1
    for speed,duration in [(1,.5),(.05,.5),(.1,2),(.1,float('nan'))]:
        with pytest.raises(ValueError):validate_walk(speed,duration)


def test_upper_hold_uses_native_17_channel_order_not_legs():
    names=[NAMES[i] for i in UPPER_IDS]
    assert names[:5]==['waist_yaw_joint','waist_roll_joint','waist_pitch_joint','left_shoulder_pitch_joint','right_shoulder_pitch_joint']
    assert all(not any(s in n for s in ['hip','knee','ankle']) for n in names)
    result=upper_fields(np.arange(29),np.zeros(29))
    assert result['upper_body_position']==UPPER_IDS.tolist() and result['upper_body_velocity']==[0]*17
