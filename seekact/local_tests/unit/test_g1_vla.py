"""Agent mailbox, bounded jobs and thermal handoff; no robot connections."""
import json
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from services.g1_sonic.vla import VLAJobs, submit, result, atomic, TASKS, validate
from services.g1_sonic.backend import G1SonicBackend
from services.controller.config import Configuration
from services.controller.frontend import Frontend
from services.controller.session import Session


class Runtime:
    model_path='fixture-model'
    def __init__(self):
        self.steps=0;self.commands=[];self.paused=0;self.done=False;self.error=None
    def prepare(self,anchor):self.steps=0;self.mode='prepare'
    def start(self,prompt,steps):self.steps=0;self.prompt=prompt;self.target=steps;self.mode='run'
    def pause(self):self.paused+=1
    def poll(self):
        if self.error:raise RuntimeError(self.error)
        self.steps+=1
        return dict(motion_token=np.zeros(64),left_hand_joints=np.zeros(6),right_hand_joints=np.zeros(6)),self.done
    def apply_hands(self,a):self.commands.append(a)


def fixture(tmp_path):
    runtime=Runtime();jobs=VLAJobs(tmp_path,'session',runtime)
    atomic(tmp_path/'ankle_status.json',dict(session_id='session',phase='ready',fault=None,
        updated_monotonic=time.monotonic(),lease_remaining_seconds=600,vla=jobs.snapshot()))
    return jobs,runtime


def tick(jobs,**kw):
    return jobs.tick(time.monotonic(),**dict(available=True,settled=True,anchor=[0]*64,lease_remaining=600,**kw))


def test_agent_tool_prepare_then_bounded_task(tmp_path):
    jobs,runtime=fixture(tmp_path)
    config=Configuration(root=tmp_path/'session',task='test_vla',robot='g1_sonic',
                         backend_options=dict(control_directory=str(tmp_path)))
    front=Frontend(Session(config),G1SonicBackend(config))
    try:
        def call(args):
            r=front.call('g1_vla',args);assert not r.get('isError'),r
            return json.loads(r['content'][0]['text'])['job_id']
        first=call(dict(action='prepare'));runtime.done=True;tick(jobs)
        assert result(tmp_path,first)['completed'] and jobs.initialized
        job=call(dict(action='run',task='mickey',steps=50,evidence='Mickey on table; hand approach visible'))
        tick(jobs)
        assert runtime.prompt==TASKS['mickey']
        assert result(tmp_path,job)['reason']=='step_budget_complete'
        assert not result(tmp_path,job)['task_success_verified']
        assert len(runtime.commands)==2
    finally:front.close()


def test_pause_wins_over_pending_start(tmp_path):
    jobs,rt=fixture(tmp_path)
    queued=submit(tmp_path,'prepare')['job_id']
    pause=submit(tmp_path,'pause')['job_id']
    assert tick(jobs) is None
    assert result(tmp_path,queued)['reason']=='superseded_by_pause'
    assert result(tmp_path,pause)['completed'] and not rt.commands


def test_thermal_preemption_invalidates_queue_and_initial_pose(tmp_path):
    jobs,rt=fixture(tmp_path);jobs.initialized=True
    job=submit(tmp_path,'run',task='cola',steps=50,evidence='Visible cola in reach')['job_id']
    tick(jobs);assert len(rt.commands)==1
    jobs.pause('ankle_temperature_priority',invalidate=True)
    assert not jobs.initialized and not jobs.moving and rt.paused
    assert tick(jobs) is None and len(rt.commands)==1
    assert result(tmp_path,job)['reason']=='ankle_temperature_priority'
    job2=submit(tmp_path,'run',task='cola',steps=50,evidence='New view')['job_id'];tick(jobs)
    assert result(tmp_path,job2)['reason']=='prepare_required'


@pytest.mark.parametrize('mutation',[{'session_id':'old'},{'created_monotonic':0}])
def test_old_requests_never_move(tmp_path,mutation):
    jobs,rt=fixture(tmp_path);id=submit(tmp_path,'prepare')['job_id']
    p=tmp_path/'vla_requests/pending.json';v=json.loads(p.read_text());v.update(mutation);atomic(p,v)
    assert tick(jobs) is None and not rt.commands
    assert not result(tmp_path,id)['completed']


def test_busy_or_short_lease_and_queue_full(tmp_path):
    jobs,rt=fixture(tmp_path);job=submit(tmp_path,'prepare')['job_id']
    with pytest.raises(FileExistsError):submit(tmp_path,'prepare')
    jobs.tick(time.monotonic(),False,True,[0]*64,600)
    assert result(tmp_path,job)['reason']=='owner_busy'
    job=submit(tmp_path,'prepare')['job_id']
    jobs.tick(time.monotonic(),True,True,[0]*64,2)
    assert result(tmp_path,job)['reason']=='insufficient_owner_lease'
    assert not rt.commands


def test_runtime_error_discards_queue(tmp_path):
    jobs,rt=fixture(tmp_path);rt.error='RTC inference missed deadline'
    job=submit(tmp_path,'prepare')['job_id'];assert tick(jobs) is None
    assert not result(tmp_path,job)['completed'] and rt.paused and not jobs.initialized


def test_request_arriving_during_owner_tick_is_not_mistaken_for_future(tmp_path,monkeypatch):
    jobs,rt=fixture(tmp_path)
    clock=[100.]
    monkeypatch.setattr(time,'monotonic',lambda:clock[0])
    atomic(tmp_path/'ankle_status.json',dict(session_id='session',phase='ready',fault=None,
        updated_monotonic=100.,lease_remaining_seconds=600,vla=jobs.snapshot()))
    tick_started=clock[0]
    clock[0]=100.005
    job=submit(tmp_path,'prepare')['job_id']
    clock[0]=100.010
    rt.done=True
    jobs.tick(tick_started,True,True,[0]*64,600)
    assert result(tmp_path,job)['completed'] and len(rt.commands)==1


def test_age_checked_after_read_still_rejects_expired_and_future(tmp_path,monkeypatch):
    jobs,rt=fixture(tmp_path);clock=[100.]
    monkeypatch.setattr(time,'monotonic',lambda:clock[0])
    for created in (97.,101.):
        atomic(tmp_path/'ankle_status.json',dict(session_id='session',phase='ready',fault=None,
            updated_monotonic=100.,lease_remaining_seconds=600,vla=jobs.snapshot()))
        job=submit(tmp_path,'prepare')['job_id']
        p=tmp_path/'vla_requests/pending.json';request=json.loads(p.read_text())
        request['created_monotonic']=created;atomic(p,request)
        jobs.tick(97.,True,True,[0]*64,600)
        assert not result(tmp_path,job)['completed']
    assert not rt.commands


@pytest.mark.parametrize('arguments',[
    dict(action='run',task='sofa',steps=20,evidence='x'),
    dict(action='run',task='cola',steps=True,evidence='x'),
    dict(action='run',task='cola',steps=1501,evidence='x'),
    dict(action='run',task='cola',steps=20,evidence=''),
    dict(action='prepare',task='cola'),dict(action='resume')])
def test_invalid_tool_arguments(arguments):
    with pytest.raises(ValueError):validate(**arguments)


def test_new_agent_workspace_inherits_vla_tools_and_thermal_skill(tmp_path):
    from services.workspace import Setup,build,recipe_for
    workspace=tmp_path/'agent'
    build(recipe_for(),Setup('g1_sonic',subagents=0),workspace)
    contract=json.loads((workspace/'MCP_CONTRACT.json').read_text())
    b=G1SonicBackend(Configuration(root=tmp_path/'session',task='test',robot='g1_sonic'))
    try:
        assert {t['name'] for t in b.tools()}<=set(contract['tools'])
    finally:b.close()
    skill=(workspace/'.agents/skills/robot-interface/SKILL.md').read_text()
    assert '10-second' in skill and 'gain 1.5' in skill and 'g1_vla(action="run"' in skill
    assert 'Only operator-commissioned small elbow' not in (workspace/'AGENTS.md').read_text()
