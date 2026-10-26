import copy
import subprocess

import pytest

from services.g1_sonic.ankle_thermal import ANKLES,AlarmDetector,announce


def sample(t, temperatures=None):
    temperatures=temperatures or {}
    return dict(monotonic=t,motors=[dict(hardware_index=i,name=name,
        temperature_winding_driver_c=temperatures.get(i,[50,50]),motorstate=0,mode=0)
        for i,name in ANKLES.items()])


def test_strict_91_side_and_channel():
    detector=AlarmDetector()
    assert detector.update(sample(0,{5:[50,91]}),0)==[]
    events=detector.update(sample(.1,{5:[50,91.01]}),.1)
    assert len(events)==1 and events[0]['side']=='left'
    assert not events[0]['motion_enabled']
    events=detector.update(sample(.2,{5:[50,99],10:[92,50]}),.2)
    assert len(events)==1 and events[0]['side']=='right'
    assert events[0]['joints'][0]['hardware_index']==10


def test_native_warning_hysteresis_is_per_motor():
    d=AlarmDetector('native90')
    assert d.update(sample(0,{5:[50,89]}),0)==[]
    assert d.update(sample(.1,{5:[50,90]}),.1)[0]['side']=='left'
    assert d.update(sample(.2,{5:[50,87]}),.2)==[] and d.active['left']
    # A never-triggered pitch at 87 does not keep roll's cleared alarm alive.
    assert d.update(sample(.3,{5:[50,84],4:[50,87]}),.3)[0]['event']=='ankle_alarm_rearmed'
    assert not d.active['left']
    assert d.update(sample(.4,{5:[50,90],11:[90,50]}),.4)[1]['side']=='right'
    with pytest.raises(ValueError):d.update(sample(.4),.5)
    assert d.active['left'] and d.active['right']


def test_continuous_heat_never_repeats():
    d=AlarmDetector();events=[]
    for i in range(200):events+=d.update(sample(i*.1,{4:[95,92],5:[99,50],11:[50,92]}),i*.1)
    assert [e['side'] for e in events]==['left','right']


def test_cool_hysteresis_requires_both_ankles_and_five_seconds():
    d=AlarmDetector();d.update(sample(0,{5:[92,50]}),0)
    for i in range(1,61):
        assert d.update(sample(i*.1,{4:[85,50],5:[84,50]}),i*.1)==[]
    events=[]
    for i in range(61,114):events+=d.update(sample(i*.1),i*.1)
    assert len(events)==1 and events[0]['event']=='ankle_alarm_rearmed'
    event=d.update(sample(11.4,{4:[50,92]}),11.4)[0]
    assert event['episode']==2 and event['side']=='left'


@pytest.mark.parametrize('change', ['missing','nan','wrong_name','stale','duplicate_index'])
def test_invalid_input_cannot_rearm(change):
    d=AlarmDetector();d.update(sample(0,{5:[92,50]}),0)
    s=sample(.1)
    if change=='missing':s['motors'].pop()
    if change=='nan':s['motors'][0]['temperature_winding_driver_c'][0]=float('nan')
    if change=='wrong_name':s['motors'][0]['name']='right_ankle_pitch_joint'
    if change=='stale':s['monotonic']=-1
    if change=='duplicate_index':s['motors'].append(copy.deepcopy(s['motors'][0]))
    with pytest.raises(ValueError):d.update(s,.1)
    assert d.active['left']


def test_missing_interval_and_duplicate_do_not_count_as_cooling():
    d=AlarmDetector();d.update(sample(0,{5:[92,50]}),0)
    d.update(sample(.1),.1)
    assert d.update(sample(9),9)==[]
    with pytest.raises(ValueError):d.update(sample(9),9)
    assert d.active['left']


def test_audio_failure_does_not_report_completion(monkeypatch,tmp_path):
    calls=[]
    def run(command,**kwargs):
        calls.append(command[0])
        if command[0]=='paplay':raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(subprocess,'run',run)
    with pytest.raises(subprocess.CalledProcessError):announce(dict(side='left',episode=1),tmp_path)
    assert calls==['espeak','paplay']


def test_successful_audio_still_cannot_enable_motion(monkeypatch,tmp_path):
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:None)
    result=announce(dict(side='right',episode=1),tmp_path,replay=True)
    assert result['message'].startswith('Replay test.')
    assert result['dispatch_status']=='blocked' and not result['motion_enabled']
