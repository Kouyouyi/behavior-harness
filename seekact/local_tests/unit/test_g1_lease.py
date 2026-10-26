"""Session lifetime and recovery: entirely synthetic, no robot connections."""
import json
import math

import pytest

from services.g1_sonic import lease
from services.g1_sonic.vla import atomic
from test_g1_backend import backend


def state(path, session, now=100., **changes):
    atomic(path/'ankle_status.json',dict(session_id=session.session,phase='ready',fault=None,
        updated_monotonic=now,**session.snapshot(now))|changes)


def test_unlimited_is_default_and_json_safe(tmp_path):
    session=lease.SessionLease(tmp_path,'session');session.start(100.)
    for now in (100.,701.,100.+365*24*3600):
        snapshot=session.snapshot(now)
        assert not session.expired(now)
        assert math.isinf(lease.remaining(snapshot))
        assert snapshot['lease_remaining_seconds'] is None
        json.dumps(snapshot,allow_nan=False)
    assert lease.remaining({})==0
    assert lease.remaining(dict(lease_remaining_seconds=0))==0


@pytest.mark.parametrize('value',[True,None,-1,1,29,float('nan'),float('inf'),'0'])
def test_invalid_duration(value):
    with pytest.raises(ValueError):lease.duration(value)


def test_mcp_expiry_recovery_and_finite_renewal(tmp_path,monkeypatch):
    monkeypatch.setattr(lease.time,'monotonic',lambda:100.)
    session=lease.SessionLease(tmp_path,'session',30);session.start(1.)
    assert session.expired(100.)
    state(tmp_path,session,phase='fault_idle',fault=lease.EXPIRED)
    b=backend(tmp_path,control_directory=str(tmp_path))
    try:
        current=b.call('g1_session_lease',dict(action='status'))[0]
        assert current['lease_remaining_seconds']==0
        sent=b.call('g1_session_lease',dict(action='set',seconds=0))[0]
        assert not sent['automatic_motion_resume']
        assert session.consume(100.,'fault_idle',lease.EXPIRED,True,False)
        done=b.call('g1_session_lease',dict(action='result',job_id=sent['job_id']))[0]
        assert done['completed'] and done['recovered_lease_fault'] and done['lease_unlimited']
        assert not done['automatic_motion_resume']
        assert not session.expired(100000.)
        state(tmp_path,session)
        renewed=lease.submit(tmp_path,3600)
        assert not session.consume(100.,'ready',None,True,False)
        assert lease.result(tmp_path,renewed['job_id'])['completed']
        assert session.remaining(101.)==3599
        assert session.expired(3700.)
    finally:b.close()


@pytest.mark.parametrize('fault,settled,busy',[
    ('Ankle motor fault reported',True,False),('owner_tick_late',True,False),
    ('transport fault',True,False),(lease.EXPIRED,False,False),(lease.EXPIRED,True,True)])
def test_recovery_cannot_clear_other_faults_or_resume_busy_owner(tmp_path,monkeypatch,fault,settled,busy):
    monkeypatch.setattr(lease.time,'monotonic',lambda:100.)
    session=lease.SessionLease(tmp_path,'session',30);session.start(1.)
    state(tmp_path,session,phase='fault_idle',fault=fault)
    sent=lease.submit(tmp_path,0)
    assert not session.consume(100.,'fault_idle',fault,settled,busy)
    assert not lease.result(tmp_path,sent['job_id'])['completed']
    assert session.expired(100.) and session.seconds==30


@pytest.mark.parametrize('patch',[dict(session_id='retired'),dict(created_monotonic=97.),dict(seconds=-1)])
def test_stale_wrong_session_invalid_requests_not_replayed(tmp_path,monkeypatch,patch):
    monkeypatch.setattr(lease.time,'monotonic',lambda:100.)
    session=lease.SessionLease(tmp_path,'session',30);session.start(1.);state(tmp_path,session)
    sent=lease.submit(tmp_path,0);path=tmp_path/'lease_requests/pending.json'
    atomic(path,json.loads(path.read_text())|patch)
    assert not session.consume(100.,'fault_idle',lease.EXPIRED,True,False)
    assert not lease.result(tmp_path,sent['job_id'])['completed']
    assert session.seconds==30 and not path.exists()


def test_old_owner_explicitly_rejected_and_pending_not_overwritten(tmp_path,monkeypatch):
    monkeypatch.setattr(lease.time,'monotonic',lambda:100.)
    session=lease.SessionLease(tmp_path,'session');state(tmp_path,session,lease_control_supported=False)
    with pytest.raises(RuntimeError,match='predates'):lease.submit(tmp_path,0)
    state(tmp_path,session)
    first=lease.submit(tmp_path,0)
    with pytest.raises(FileExistsError):lease.submit(tmp_path,120)
    assert json.loads((tmp_path/'lease_requests/pending.json').read_text())['id']==first['job_id']


def test_unlimited_vla_and_movement_submission(tmp_path,monkeypatch):
    from test_g1_vla import fixture
    from test_g1_locomotion import Driver,command
    from services.g1_sonic import vla,locomotion
    jobs,_=fixture(tmp_path)
    owner=json.loads((tmp_path/'ankle_status.json').read_text())
    atomic(tmp_path/'ankle_status.json',owner|dict(lease_unlimited=True,lease_remaining_seconds=None))
    assert vla.submit(tmp_path,'prepare')['status']=='submitted'
    movement=tmp_path/'movement';movement.mkdir()
    d=Driver(movement,monkeypatch)
    owner=json.loads((movement/'ankle_status.json').read_text())
    atomic(movement/'ankle_status.json',owner|dict(lease_unlimited=True,lease_remaining_seconds=None))
    assert locomotion.submit(movement,'run',command(),d.clock)['status']=='submitted'
