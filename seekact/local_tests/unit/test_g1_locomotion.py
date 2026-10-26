"""Movement contract, shared-owner jobs, native feedback and upper posture hold."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from services.g1_sonic import locomotion as move
from services.g1_sonic.walk import UPPER_HARDWARE, UPPER_IDS
from services.g1_sonic.state import HARDWARE_TO_ISAAC
from test_g1_backend import backend, frame


def command(direction='forward', **kw):
    return move.validate('run',**(dict(direction=direction,frame_id='a'*32,evidence='Free path, hands holding')|kw))


@pytest.mark.parametrize('direction',move.MOTIONS)
def test_vectors_and_turn_ramp(direction):
    c=command(direction);v=move.planner_command(c,1.2,-.4,1.)
    x,y,_=v['movement'];heading=1.6
    body=[math.cos(heading)*x+math.sin(heading)*y,-math.sin(heading)*x+math.cos(heading)*y]
    expected=np.array(move.DIRECTIONS.get(direction,(0.,0.)),float)
    if np.linalg.norm(expected):expected/=np.linalg.norm(expected)
    assert body==pytest.approx(expected)
    turn=15 if direction=='rotate_left' else -15 if direction=='rotate_right' else 0
    assert move.wrap(math.atan2(v['facing'][1],v['facing'][0])-heading)==pytest.approx(math.radians(turn))


@pytest.mark.parametrize('kw',[
    dict(speed=True),dict(speed=float('nan')),dict(speed=.31),dict(speed=.2),
    dict(mode='slowWalk',speed=.1),dict(direction='rotate_left',speed=.2),
    dict(duration_s=0),dict(duration_s=3.1),
    dict(direction='north'),dict(direction=[]),dict(evidence=' '),dict(frame_id='../bad'),
    dict(yaw_rate_deg_s=1),dict(mode='run'),dict(mode='walk',speed=.1),
    dict(direction='rotate_left',duration_s=3),dict(direction='rotate_right',yaw_rate_deg_s=30),
    dict(direction='step_in_place',mode='walk'),dict(extra=1)])
def test_contract_rejects_invalid_values(kw):
    with pytest.raises(ValueError):command(**kw)


@pytest.mark.parametrize('direction',move.MOTIONS)
def test_fixed_speed_preserves_direction_and_idle(direction):
    c=command(direction)
    assert c['speed']==.3
    assert move.planner_command(c,0.,0.,.5)['speed']==.3
    assert move.idle()['speed']==0.


class Driver:
    def __init__(self,path,monkeypatch):
        self.clock=100.;self.path=path;self.jobs=move.LocomotionJobs(path,'session')
        monkeypatch.setattr(move.time,'monotonic',lambda:self.clock)
        self.native=dict(motion='planner_motion',encoder_source=True,locomotion=0,
                         reference_yaw=.2,base_quat=[1,0,0,0],body_q=np.linspace(-.1,.1,29).tolist())
        self.available=True;self.settled=True;self.lease=600;self.vla=None;self.echo=True
        self.refresh()

    def refresh(self):
        move.atomic(self.path/'ankle_status.json',dict(phase='ready',fault=None,session_id='session',
            updated_monotonic=self.clock,lease_remaining_seconds=self.lease,locomotion=self.jobs.snapshot()))

    def submit(self,**kw):
        self.refresh()
        return move.submit(self.path,'run',command(**kw),self.clock)['job_id']

    def tick(self,seconds=.02):
        self.clock+=seconds
        obs=np.zeros(584)
        if self.jobs.upper and self.echo:
            for f in range(10):obs[4+f*29+UPPER_IDS]=self.jobs.upper['upper_body_position']
        self.native['encoder_obs']=obs.tolist()
        packet=self.jobs.tick(self.clock,self.native,self.settled,self.available,self.lease,self.vla)
        if packet:self.native['locomotion']=packet['locomotion']
        self.refresh()
        return packet

    def run_to_end(self):
        for _ in range(800):
            self.tick()
            if not self.jobs.busy:return
        raise AssertionError('job failed to terminate')


def test_upper_capture_native_readback_and_hold_after_completion(tmp_path,monkeypatch):
    d=Driver(tmp_path,monkeypatch);job=d.submit()
    assert d.tick()['locomotion']==0
    expected=[d.native['body_q'][i] for i in UPPER_HARDWARE]
    assert d.jobs.upper['upper_body_position']==pytest.approx(expected)
    d.run_to_end();r=move.result(tmp_path,job)
    assert r['completed'] and r['settled'] and r['source_confirmed']
    assert not r['task_success_verified'] and r['actual_distance_m'] is None
    # Same capture persists after stopping and across subsequent moves, no recapture drift.
    d.native['body_q']=[.5]*29;d.submit(direction='left');d.tick()
    assert d.jobs.upper['upper_body_position']==pytest.approx(expected)
    writes=[]
    wire=SimpleNamespace(build_planner_message=lambda *a,**kw:writes.append((a,kw)))
    d.jobs.packet(wire,d.jobs.idle_command)
    assert writes[0][1]['upper_body_position']==pytest.approx(expected)
    assert writes[0][1]['upper_body_velocity']==[0.]*17
    assert not any('hand' in key for key in writes[0][1])


def test_missing_upper_echo_never_walks(tmp_path,monkeypatch):
    d=Driver(tmp_path,monkeypatch);d.echo=False;job=d.submit();packets=[]
    for _ in range(250):
        p=d.tick()
        if p:packets.append(p)
    assert all(p['locomotion']==0 for p in packets)
    assert not move.result(tmp_path,job)['completed']


@pytest.mark.parametrize('cause',['stop','thermal','source','upper','reference'])
def test_preemption_stops_and_cancels_remaining_turn(tmp_path,monkeypatch,cause):
    d=Driver(tmp_path,monkeypatch);job=d.submit(direction='rotate_left')
    while d.jobs.phase!='running':d.tick()
    d.native['base_quat']=[math.cos(.05),0,0,math.sin(.05)]
    if cause=='stop':move.submit(tmp_path,'stop')
    if cause=='thermal':d.available=False
    if cause=='source':d.native['motion']='temporary_motion';d.clock+=.7
    if cause=='upper':d.echo=False
    if cause=='reference':d.native['reference_yaw']+=.2
    packet=d.tick();assert packet['locomotion']==0 and packet['speed']==0
    assert math.atan2(packet['facing'][1],packet['facing'][0])==pytest.approx(0. if cause=='source' else .1-d.native['reference_yaw'])
    d.native['motion']='planner_motion';d.available=True;d.echo=True
    d.run_to_end();assert not move.result(tmp_path,job)['completed']


def test_stop_precedes_pending_and_does_not_start_vla(tmp_path,monkeypatch):
    d=Driver(tmp_path,monkeypatch);job=d.submit();stop=move.submit(tmp_path,'stop')['job_id']
    assert d.tick() is None
    assert move.result(tmp_path,job)['reason']=='superseded_by_stop'
    assert move.result(tmp_path,stop)['reason']=='no_active_movement'


def test_source_loss_at_deadline_is_not_success(tmp_path,monkeypatch):
    d=Driver(tmp_path,monkeypatch);job=d.submit()
    while d.jobs.phase!='running':d.tick()
    d.tick();assert d.jobs.confirmed
    d.native['encoder_source']=False
    assert d.tick(2.)['locomotion']==0
    d.native['encoder_source']=True;d.run_to_end()
    result=move.result(tmp_path,job)
    assert not result['completed'] and result['reason']=='planner_source_lost'


def test_vla_exclusion_and_posture_handoff(tmp_path,monkeypatch):
    d=Driver(tmp_path,monkeypatch);calls=[]
    d.vla=SimpleNamespace(moving=True,pause=lambda *a,**kw:calls.append((a,kw)))
    job=d.submit();assert d.tick() is None
    assert move.result(tmp_path,job)['reason']=='vla_busy' and not calls
    d.vla.moving=False;d.native['motion']='temporary_motion';d.submit();d.tick()
    assert calls==[(('locomotion_handoff',),dict(invalidate=True))]


@pytest.mark.parametrize('mutation',[dict(session_id='old'),dict(created_monotonic=0.),dict(observed_at=0.)])
def test_expired_requests_cannot_move(tmp_path,monkeypatch,mutation):
    d=Driver(tmp_path,monkeypatch);job=d.submit();p=tmp_path/'move_requests/pending.json'
    move.atomic(p,json.loads(p.read_text())|mutation)
    assert d.tick() is None and not move.result(tmp_path,job)['completed']


def test_duplicate_submission_and_settle_timeout(tmp_path,monkeypatch):
    d=Driver(tmp_path,monkeypatch);job=d.submit()
    with pytest.raises(FileExistsError):d.submit()
    while d.jobs.phase!='running':d.tick()
    d.settled=False
    with pytest.raises(RuntimeError,match='settle'):d.run_to_end()
    assert move.result(tmp_path,job)['reason']=='movement_settle_timeout'


def test_backend_observation_gate_and_shared_tool(tmp_path,monkeypatch):
    control=tmp_path/'control';control.mkdir();d=Driver(control,monkeypatch)
    camera=tmp_path/'camera';frame(camera)
    b=backend(tmp_path,control_directory=str(control),camera_directory=str(camera))
    try:
        observation,_=b.call('g1_observe',{})
        args=dict(action='run',direction='forward_left',frame_id=observation['frame_id'],evidence='clear floor')
        sent,_=b.call('g1_move',args)
        with pytest.raises(ValueError,match='Observe again'):b.call('g1_move',args)
        d.tick();d.run_to_end()
        r,_=b.call('g1_move',dict(action='result',job_id=sent['job_id']))
        assert r['completed']
        assert 'forward_left' in b.call('g1_move',dict(action='status'))[0]['locomotion']['directions']
        assert b.status()['motion_available'] and b.status()['locomotion_available']
        with pytest.raises(ValueError):b.call('g1_move',dict(action='stop',speed=.1))
    finally:b.close()


@pytest.mark.parametrize('interrupt',[None,'thermal','fault'])
def test_actual_owner_serializes_planner_and_thermal_with_upper_hold(tmp_path,monkeypatch,interrupt):
    from services.g1_sonic import ankle_executor as owner
    from test_g1_ankle_demo import ImmediatePool
    clock=[100.];handlers={};packets=[];sent=[]
    native_mode=['planner_motion'];planner=[move.idle()];upper=[{}];tokens=[np.zeros(64)]
    monkeypatch.setattr(owner.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(owner.time,'sleep',lambda dt:clock.__setitem__(0,clock[0]+max(dt,.001)))
    monkeypatch.setattr(owner.signal,'signal',lambda sig,handler:handlers.__setitem__(sig,handler))
    monkeypatch.setattr(owner,'ThreadPoolExecutor',ImmediatePool)
    monkeypatch.setattr(owner,'load_demos',lambda _:dict(left=np.zeros((20,64)),right=np.zeros((20,64))))
    monkeypatch.setattr(owner,'announce',lambda *a,**kw:dict(playback_completed_monotonic=clock[0]))
    class Pub:
        def setsockopt(self,*a):pass
        def bind(self,*a):pass
        def close(self,*a):pass
        def send(self,packet):
            packets.append((clock[0],packet))
            if 'planner' in packet:native_mode[0]='planner_motion' if packet['planner'] else 'temporary_motion'
            if 'locomotion' in packet:
                planner[0]=packet;upper[0]={k:v for k,v in packet.items() if k.startswith('upper_body_')}
            if 'token_state' in packet:tokens[0]=packet['token_state'][0]
    monkeypatch.setattr(owner.zmq,'Context',lambda:SimpleNamespace(socket=lambda _:Pub(),term=lambda:None))
    monkeypatch.setattr(owner,'wire_module',lambda:SimpleNamespace(
        build_planner_message=lambda mode,movement,facing,**kw:dict(locomotion=mode,movement=movement,facing=facing,**kw),
        build_command_message=lambda **kw:kw,pack_pose_message=lambda fields,version:fields))
    class Trace:
        def __init__(self,*a):pass
        def close(self):pass
        def poll(self):
            obs=np.zeros(584)
            if upper[0]:
                for f in range(10):obs[4+f*29+UPPER_IDS]=upper[0]['upper_body_position']
            return dict(monotonic=clock[0],lowstate_receive_monotonic_ns=int(clock[0]*1e9),
                state_monotonic_ns=int(clock[0]*1e9),body_q=[0.]*29,body_dq=[0.]*29,
                base_quat=[1.,0,0,0],base_ang_vel=[0.]*3,transport_fault=False,log_dropped=0,
                motion=native_mode[0],encoder_source=native_mode[0]=='planner_motion',
                token=tokens[0].tolist(),reference_yaw=0.,locomotion=planner[0]['locomotion'],encoder_obs=obs.tolist())
    monkeypatch.setattr(owner,'TraceReader',Trace)
    (tmp_path/'controller.log').write_text('Init Done\n')
    active_at=[None]
    def health(now):
        path=tmp_path/'ankle_status.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        if state.get('phase')=='ready' and not sent:
            sent.append(move.submit(tmp_path,'run',command(),now)['job_id'])
        if planner[0]['locomotion'] and active_at[0] is None:active_at[0]=now
        interrupted=active_at[0] is not None and now-active_at[0]>.5
        if now>125:handlers[owner.signal.SIGTERM]()
        return dict(monotonic=now,motors=[dict(hardware_index=i,name=name,
            temperature_winding_driver_c=[40,92 if interrupt=='thermal' and interrupted else 40],
            motorstate=1 if interrupt=='fault' and interrupted else 0) for i,name in owner.ANKLES.items()])
    owner.run(SimpleNamespace(run=tmp_path,port=19999,start_policy=True,lease_seconds=600,
        locomotion=True,thermal_auto_sides=['left'] if interrupt=='thermal' else [],thermal_rule='native90'),feedback_source=health)
    result=move.result(tmp_path,sent[0]);assert result['completed']==(interrupt is None)
    moving=[(t,p) for t,p in packets if p.get('locomotion',0)>0]
    assert moving and all(len(p['upper_body_position'])==17 and p['upper_body_velocity']==[0.]*17 for _,p in moving)
    if interrupt=='thermal':
        events=[json.loads(line) for line in (tmp_path/'ankle_events.jsonl').read_text().splitlines()]
        announcement=next(e['monotonic'] for e in events if e['event']=='announcement_requested')
        assert all(t<announcement for t,_ in moving)
        assert result['settled'] and result['reason']=='owner_preempted'
    if interrupt=='fault':assert result['reason'].startswith('owner_fault')
