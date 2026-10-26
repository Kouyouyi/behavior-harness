import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from services.g1_sonic.ankle_demo import DemoFlow,guard,validate_request
from services.g1_sonic import ankle_executor as owner


def test_all_source_frames_and_audio_gate():
    tokens=np.arange(117*64,dtype=np.float32).reshape(117,64)/10000
    with pytest.raises(ValueError,match='audio'):DemoFlow(tokens,np.zeros(64),0,audio_completed=False)
    f=DemoFlow(tokens,np.zeros(64),0,audio_completed=True);seen=[]
    for t in np.arange(0,20,.021):
        v=f.sample(float(t),True)
        if f.source_frame is not None:
            seen.append(f.source_frame);np.testing.assert_array_equal(v,tokens[f.source_frame])
        if f.done:break
    assert seen==list(range(117))
    assert f.done and f.return_verified
    np.testing.assert_array_equal(v,tokens[0])


def test_request_and_phase_guard():
    j=dict(id='a'*32,action='ankle_relax',side='right',session_id='s',created_monotonic=10.)
    validate_request(j,'s',10.)
    for bad in (j|dict(side='both'),j|dict(session_id='old'),j|dict(created_monotonic=7.)):
        with pytest.raises(ValueError):validate_request(bad,'s',10.)
    m=dict(q=[0.]*29,dq=[2.5]*29,tilt_rad=0.)
    guard(m,moving=False)  # Manual foot adjustment while holding is not a speed fault.
    with pytest.raises(ValueError,match='speed'):guard(m,moving=True)
    with pytest.raises(ValueError,match='tilt'):guard(m|dict(tilt_rad=.3),moving=False)


def test_vla_without_demo_speed_bounds_retains_feedback_and_pose_checks():
    m=dict(q=[0.]*29,dq=[3.2]*29,tilt_rad=0.)
    guard(m,moving=True,check_speed=False)
    with pytest.raises(ValueError,match='speed'):
        guard(m,moving=True)  # Ankle-demo behavior is unchanged.
    with pytest.raises(ValueError,match='tilt'):
        guard(m|dict(tilt_rad=.3),moving=True,check_speed=False)
    with pytest.raises(ValueError,match='model limit'):
        guard(m|dict(q=[100.]*29),moving=True,check_speed=False)
    with pytest.raises(ValueError,match='Invalid joint feedback'):
        guard(m|dict(dq=[float('nan')]*29),moving=True,check_speed=False)


class ImmediatePool:
    def __init__(self,*a,**kw):pass
    def submit(self,fn,*a,**kw):
        try:return SimpleNamespace(done=lambda:True,result=lambda value=fn(*a,**kw):value)
        except Exception as error:
            def result(error=error):raise error
            return SimpleNamespace(done=lambda:True,result=result)
    def shutdown(self,**kw):pass


@pytest.mark.parametrize('scenario',['normal','native_blend','audio_failure','stale_health','wrong_token','cooled_before_motion',
    'auto_left','auto_right','auto_both','auto_cooled_other','auto_startup_hot','auto_audio_failure',
    'auto_native_repeat','auto_native_repeat10','auto_native_cooled','auto_native_cooled10','auto_native_audio_failure','auto_native_audio_failure10',
    'lease_unlimited','lease_recover','lease_real_fault'])
def test_owner_dispatch_and_faults(tmp_path,monkeypatch,scenario):
    automatic=scenario.startswith('auto_')
    lease_case=scenario.startswith('lease_')
    clock=[100.];handlers={};packets=[];mode=['planner_motion'];token=[np.zeros(64)];handoff=[None]
    monkeypatch.setattr(owner.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(owner.time,'sleep',lambda dt:clock.__setitem__(0,clock[0]+max(dt,.1 if lease_case else .001)))
    monkeypatch.setattr(owner.signal,'signal',lambda sig,handler:handlers.__setitem__(sig,handler))
    class Pool(ImmediatePool):
        def submit(self,fn,*a,**kw):
            f=super().submit(fn,*a,**kw);ready=clock[0]+.12
            f.done=lambda:clock[0]>=ready
            return f
    monkeypatch.setattr(owner,'ThreadPoolExecutor',Pool)
    demo=np.tile(np.linspace(-.5,.5,64,dtype=np.float32),(117,1))
    demo[:,0]+=np.sin(np.linspace(0,1,117))*.1
    monkeypatch.setattr(owner,'load_demos',lambda _:dict(left=demo,right=demo))
    def audio(side,directory,job_id,thermal=False):
        if scenario in ('audio_failure','auto_audio_failure','auto_native_audio_failure','auto_native_audio_failure10'):raise RuntimeError('speaker unavailable')
        return dict(playback_exit_code=0,playback_completed_monotonic=clock[0],playback_started_monotonic=clock[0])
    monkeypatch.setattr(owner,'announce',audio)
    class Pub:
        def setsockopt(self,*a):pass
        def bind(self,*a):pass
        def close(self,*a):pass
        def send(self,p):
            packets.append(p)
            if 'token_state' in p:
                token[0]=p['token_state'][0].copy()
                if handoff[0] is None:handoff[0]=clock[0]
            if 'planner' in p:mode[0]='planner_motion' if p['planner'] else 'temporary_motion'
    monkeypatch.setattr(owner.zmq,'Context',lambda:SimpleNamespace(socket=lambda _:Pub(),term=lambda:None))
    monkeypatch.setattr(owner,'wire_module',lambda:SimpleNamespace(
        pack_pose_message=lambda fields,version:dict(fields,version=version),
        build_command_message=lambda **kw:kw,build_planner_message=lambda *a,**kw:dict(idle=True)))
    class Trace:
        def __init__(self,*a):pass
        def close(self):pass
        def poll(self):
            v=token[0]+(.1 if scenario=='wrong_token' else 0)
            alpha=1.
            if scenario=='native_blend' and handoff[0] is not None:
                u=np.clip((clock[0]-handoff[0])/.6,0,1);alpha=u*u*(3-2*u);v=alpha*v
            return dict(monotonic=clock[0],lowstate_receive_monotonic_ns=int((clock[0]-.001)*1e9),
                state_monotonic_ns=int(clock[0]*1e9),body_q=[0.]*29,
                body_dq=[.4 if scenario=='auto_startup_hot' and clock[0]<120 else 0.]*29,
                base_quat=[1.,0.,0.,0.],base_ang_vel=[0.]*3,transport_fault=False,log_dropped=0,
                motion=mode[0],encoder_source=mode[0]=='planner_motion',token=v.tolist(),blend_alpha=float(alpha))
    monkeypatch.setattr(owner,'TraceReader',Trace)
    (tmp_path/'controller.log').write_text('Init Done\n')
    submitted=[False];armed=[False];thermal_started=[False];lease_attempt=[False];snapshots=[]
    def health(now):
        status_path=tmp_path/'ankle_status.json'
        s=json.loads(status_path.read_text()) if status_path.exists() else {}
        if lease_case:
            snapshots.append(s)
            if s.get('phase')=='fault_idle' and not lease_attempt[0]:
                owner.atomic(tmp_path/'lease_requests/pending.json',dict(id='b'*32,
                    session_id=s['session_id'],seconds=0,created_monotonic=now))
                lease_attempt[0]=True
        if s.get('phase')=='ready' and not submitted[0] and not automatic and not lease_case:
            owner.atomic(tmp_path/'ankle_requests/pending.json',dict(id='a'*32,action='ankle_relax',side='right',
                session_id=s['session_id'],created_monotonic=now));submitted[0]=True
        if s.get('software_completed_sides') and not armed[0] and not automatic:
            owner.atomic(tmp_path/'thermal_arm.json',dict(session_id=s['session_id'],sides=['right']));armed[0]=True
        if s.get('active_job',{}):
            thermal_started[0]|=s['active_job'].get('trigger')=='temperature'
        temperature=92 if armed[0] else 40
        if scenario=='cooled_before_motion' and thermal_started[0]:temperature=40
        if now>(705 if lease_case else 170 if automatic else 145):handlers[owner.signal.SIGTERM]()
        stamp=now-.5 if scenario=='stale_health' and submitted[0] else now
        temps=dict(left=40,right=temperature)
        if automatic:
            hot_sides={'left'} if scenario in ('auto_left','auto_startup_hot') else {'right'} if scenario=='auto_right' else {'left','right'}
            temps={side:92 if side in hot_sides else 40 for side in ('left','right')}
            if scenario=='auto_cooled_other' and thermal_started[0]:temps['right']=40
            if scenario.startswith('auto_native'):
                temps=dict(left=90 if now<105 else 87,right=40)
                if scenario in ('auto_native_cooled','auto_native_cooled10') and now>125:temps['left']=84
        return dict(monotonic=stamp,motors=[dict(hardware_index=i,name=name,
            temperature_winding_driver_c=[40,temps[name.split('_')[0]]],
            motorstate=1 if scenario=='lease_real_fault' and 110<now<111 else 0)
            for i,name in owner.ANKLES.items()])
    owner.run(SimpleNamespace(run=tmp_path,port=19999,start_policy=True,
        lease_seconds=0 if scenario=='lease_unlimited' else 30 if lease_case else 600,
        thermal_auto_sides=['left','right'] if automatic else [],
        thermal_rule='native90' if scenario.startswith('auto_native') else 'strict91',
        thermal_repeat_seconds=10 if scenario.endswith('10') else 30 if scenario.startswith('auto_native') else 0),feedback_source=health)
    events=[json.loads(l) for l in (tmp_path/'ankle_events.jsonl').read_text().splitlines()]
    starts=[e for e in events if e['event']=='motion_started']
    if lease_case:
        assert not starts
        assert not any(p.get('stop') for p in packets)
        recovered=[e for e in events if e['event']=='session_lease_recovered']
        if scenario=='lease_recover':
            assert len(recovered)==1 and not recovered[0]['automatic_motion_resume']
            assert snapshots[-1]['phase']=='ready' and snapshots[-1]['lease_unlimited']
            assert snapshots[-1]['active_job'] is None
        elif scenario=='lease_real_fault':
            assert not recovered and snapshots[-1]['phase']=='fault_idle'
            assert snapshots[-1]['fault']=='Ankle motor fault reported'
            assert snapshots[-1]['lease_remaining_seconds']==0
        else:
            assert not recovered and snapshots[-1]['phase']=='ready'
            assert not any(e['event']=='fault_idle' for e in events)
        return
    if automatic:
        expected=[] if scenario in ('auto_audio_failure','auto_native_audio_failure','auto_native_audio_failure10') else ['left','left'] if scenario=='auto_native_repeat' else ['left','right'] if scenario=='auto_both' else ['right'] if scenario=='auto_right' else ['left']
        if scenario=='auto_native_repeat10':
            assert len(starts)>=3
            expected=['left']*len(starts)
        elif scenario=='auto_native_cooled10':
            expected=['left']*len(starts)
            assert 1<=len(starts)<=2
            assert all(e['monotonic']<125 for e in starts)
        assert [e['side'] for e in starts]==expected
        assert all(e['trigger']=='temperature' for e in starts)
        assert not any(e['event']=='fault_idle' for e in events)
        announcements=[e for e in events if e['event']=='announcement_requested']
        assert len(announcements)==(2 if scenario=='auto_audio_failure' else 1 if scenario in ('auto_native_audio_failure','auto_native_audio_failure10') else len(expected))
        for previous,current in zip(starts,starts[1:]):
            assert any(e['event']=='job_finished' and e['id']==previous['id'] and e['monotonic']<current['monotonic'] for e in events)
        if scenario in ('auto_native_repeat','auto_native_repeat10'):
            interval=10 if scenario.endswith('10') else 30
            for previous,announcement in zip(starts,announcements[1:]):
                done=next(e for e in events if e['event']=='job_finished' and e['id']==previous['id'])
                assert interval<=announcement['monotonic']-done['monotonic']<interval+.1
    elif scenario in ('normal','native_blend'):
        assert len(starts)==2
        assert [e['trigger'] for e in starts]==['manual_test','temperature']
        assert not any(e['event']=='fault_idle' for e in events)
    elif scenario in ('audio_failure','stale_health'):assert not starts
    elif scenario=='cooled_before_motion':assert len(starts)==1
    else:assert any(e['event']=='fault_idle' for e in events)
    for start in starts:
        assert any(e['event']=='audio_completed' and e['id']==start['id'] and e['monotonic']<=start['monotonic'] for e in events)
    if scenario in ('normal','native_blend'):
        rows=[json.loads(l) for l in (tmp_path/'ankle_execution.jsonl').read_text().splitlines()]
        for start in starts:
            assert [r['source_frame'] for r in rows if r['job']['id']==start['id'] and r['source_frame'] is not None]==list(range(117))
