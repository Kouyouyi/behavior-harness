"""Bounded agent VLA jobs. No DDS, ZMQ publisher or model imports here."""
from .lease import remaining as lease_remaining
import json
import os
from pathlib import Path
import re
import time
import uuid

import numpy as np

TASKS = {
    'chips': 'pick up the chips on the table',
    'cola': 'pick up the cola on the table',
    'mickey': 'pick up the mickey on the table',
    'milk': 'pick up the milk',
    'pie': 'pick up the pie',
    'wahaha': 'pick up the wahaha',
}
DEFAULT_MODEL = '/media/user/My_PSSD/202_models/home_rtc_quality_bs64_8ep_final'
DEFAULT_RTC = '/home/user/code/Isaac-GR00T-rtc'


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        tmp.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def validate_hand(side, action, evidence):
    if side not in ('left','right','both') or action not in ('open','close'):
        raise ValueError('Choose left/right/both and action=open/close')
    if not isinstance(evidence,str) or not 1 <= len(evidence.strip()) <= 2000:
        raise ValueError('Describe current hand/object contact or supported release geometry')


def submit_hand(directory, side, action, evidence):
    validate_hand(side, action, evidence)
    return submit(directory, 'hand', side=side, hand_action=action, evidence=evidence)


def validate(action, task=None, steps=None, evidence=None, side=None, hand_action=None):
    if action in ('release_hand','hand'):
        if action == 'release_hand' and hand_action is not None:raise ValueError('Legacy release only accepts open')
        validate_hand(side, 'open' if action=='release_hand' else hand_action, evidence)
        if task is not None or steps is not None:
            raise ValueError('Release does not accept task or steps')
        return
    if side is not None or hand_action is not None:
        raise ValueError('side/hand_action apply only to hand control')
    if action not in ('prepare', 'run', 'pause'):
        raise ValueError('VLA action must be prepare, run or pause')
    if action == 'run':
        if task not in TASKS or type(steps) is not int or not 1 <= steps <= 1500:
            raise ValueError('Choose a trained task and 1..1500 steps (50 Hz)')
        if not isinstance(evidence, str) or not 1 <= len(evidence.strip()) <= 2000:
            raise ValueError('Describe the current observed object and handoff geometry')
    elif any(x is not None for x in (task, steps, evidence)):
        raise ValueError('Task, steps and evidence apply only to run')


def status(directory):
    if directory is None:
        raise RuntimeError('No VLA-enabled owner configured')
    value = json.loads((Path(directory)/'ankle_status.json').read_text())
    if not 0 <= time.monotonic()-value['updated_monotonic'] <= .25:
        raise RuntimeError('VLA owner status stale')
    if not value.get('vla'):
        raise RuntimeError('Owner was not started with VLA enabled')
    return value


def submit(directory, action, **arguments):
    validate(action, **arguments)
    owner = status(directory)
    if action != 'pause' and (owner.get('locomotion') or {}).get('active_job'):
        raise RuntimeError('Movement must stop and settle before VLA')
    if action != 'pause' and (owner['phase'] != 'ready' or owner.get('fault')):
        raise RuntimeError('Body owner unavailable or handling ankle relief')
    if action == 'hand':
        if not owner['vla'].get('hand_control_supported'):
            raise RuntimeError('Owner does not support left/right/both open/close; create an updated run')
        if not owner['vla'].get('hand_control_ready') or owner['vla'].get('active_job'):
            raise RuntimeError('Hand control requires idle owner with known hand targets; pause VLA first')
    if action == 'release_hand' and not owner['vla'].get('hand_release_supported'):
        raise RuntimeError('Owner snapshot predates hand release; prepare a new updated run')
    if action == 'release_hand' and (not owner['vla'].get('initialized') or owner['vla'].get('active_job')):
        raise RuntimeError('Hand release requires prepared idle owner; pause VLA first')
    seconds = (arguments.get('steps', 0)/50 + 10) if action == 'run' else 10
    if action != 'pause' and lease_remaining(owner) < seconds:
        raise RuntimeError('Insufficient owner session time')
    job = dict(id=uuid.uuid4().hex, session_id=owner['session_id'],
               created_monotonic=time.monotonic(), action=action, **arguments)
    queue = Path(directory)/'vla_requests'
    temp = queue/(job['id']+'.json')
    try:
        temp.write_text(json.dumps(job, allow_nan=False))
        os.link(temp, queue/('pause.json' if action == 'pause' else 'pending.json'))
    finally:
        temp.unlink(missing_ok=True)
    return dict(job_id=job['id'], status='submitted', task_success_verified=False)


def result(directory, job_id):
    if not isinstance(job_id, str) or not re.fullmatch('[0-9a-f]{32}', job_id):
        raise ValueError('Invalid VLA job ID')
    if directory is None:
        raise RuntimeError('No VLA owner configured')
    path = Path(directory)/'vla_results'/(job_id+'.json')
    if path.exists():
        return json.loads(path.read_text())
    return dict(job_id=job_id, status='pending_or_unknown', owner=status(directory)['vla'])


class VLAJobs:
    """Serial job lifecycle; only the body owner calls tick and consumes actions."""
    def __init__(self, directory, session_id, runtime):
        self.root = Path(directory)
        self.session_id = session_id
        self.runtime = runtime
        for name in ('vla_requests', 'vla_results'):
            (self.root/name).mkdir(exist_ok=True)
        self.job = None
        self.initialized = False
        self.last_result = None
        self.phase = 'uninitialized'
        self.at = None

    @property
    def moving(self):
        return self.job is not None

    def snapshot(self):
        return dict(phase=self.phase, initialized=self.initialized,
                    active_job=self.job, last_result=self.last_result,
                    tasks=TASKS, model_path=str(self.runtime.model_path), hand_release_supported=True,
                    hand_control_supported=True,hand_control_ready=getattr(self.runtime,'last_hand_targets',None) is not None,
                    hand_sides=['left','right','both'],hand_actions=['open','close'])

    def finish(self, reason, completed=False):
        if self.job:
            self.last_result = dict(job_id=self.job['id'], action=self.job['action'],
                task=self.job.get('task'), completed=completed, reason=reason,
                steps_published=self.runtime.steps, task_success_verified=False,
                session_id=self.session_id)
            if self.job['action'] in ('hand','release_hand'):
                self.last_result.update(side=self.job['side'],hand_action=self.job.get('hand_action','open'))
            atomic(self.root/'vla_results'/(self.job['id']+'.json'), self.last_result)
        self.job = None
        self.phase = 'ready' if self.initialized else 'uninitialized'

    def pause(self, reason, invalidate=False):
        self.runtime.pause()  # Detach queue before recording result or moving feet.
        if invalidate:
            self.initialized = False
        self.finish(reason)

    def _request(self, filename):
        path = self.root/'vla_requests'/filename
        if not path.exists():
            return None
        claimed = path.with_suffix('.claimed')
        path.replace(claimed)
        try:
            request = json.loads(claimed.read_text())
            # A request can arrive after the owner's tick-start timestamp.
            # Check age after reading it, never against that earlier tick time.
            checked_at = time.monotonic()
            if not re.fullmatch('[0-9a-f]{32}', request.get('id', '')):
                raise ValueError('Invalid job ID')
            if (request.get('session_id') != self.session_id or
                not 0 <= checked_at-request.get('created_monotonic', -1e10) <= 2):
                raise ValueError('Changed session or expired VLA request')
            allowed = {'id', 'session_id', 'created_monotonic', 'action', 'task', 'steps', 'evidence', 'side', 'hand_action'}
            if set(request)-allowed:
                raise ValueError('Unexpected VLA arguments')
            validate(request['action'], **{k:request[k] for k in ('task','steps','evidence','side','hand_action') if k in request})
            return request
        except (ValueError, KeyError, TypeError) as exc:
            if 'request' in locals() and isinstance(request,dict) and re.fullmatch('[0-9a-f]{32}', request.get('id','')):
                atomic(self.root/'vla_results'/(request['id']+'.json'),
                       dict(job_id=request['id'], completed=False, reason=str(exc)))
            return None
        finally:
            claimed.unlink(missing_ok=True)

    def tick(self, now, available, settled, anchor, lease_remaining):
        pause = self._request('pause.json')
        if pause:
            self.pause('agent_pause')
            atomic(self.root/'vla_results'/(pause['id']+'.json'),
                   dict(job_id=pause['id'], completed=True, reason='queue_discarded_and_endpoint_held',
                        task_success_verified=False))
        request = self._request('pending.json')
        if request:
            reason = None
            if not available or self.job:
                reason = 'owner_busy'
            elif not settled:
                reason = 'robot_not_settled'
            elif lease_remaining < (request.get('steps',0)/50 + 10):
                reason = 'insufficient_owner_lease'
            elif request['action'] in ('run','release_hand') and not self.initialized:
                reason = 'prepare_required'
            elif request['action']=='hand' and getattr(self.runtime,'last_hand_targets',None) is None:
                reason = 'hand_targets_unknown'
            elif pause:
                reason = 'superseded_by_pause'
            if reason:
                atomic(self.root/'vla_results'/(request['id']+'.json'),
                       dict(job_id=request['id'], completed=False, reason=reason))
            else:
                self.job = request
                self.at = now
                try:
                    if request['action'] == 'prepare':
                        self.initialized = False
                        self.runtime.prepare(anchor)
                        self.phase = 'preparing'
                    elif request['action'] == 'hand':
                        self.runtime.control_hands(request['side'],request['hand_action'])
                        self.phase = 'controlling_hands'
                    elif request['action'] == 'release_hand':
                        self.runtime.release_hand(request['side'])
                        self.phase = 'releasing_hand'
                    else:
                        self.runtime.start(TASKS[request['task']],request['steps'])
                        self.phase = 'running'
                except Exception as exc:
                    self.pause('start_failed: '+str(exc))
        if not self.job or not available:
            return None
        try:
            if now-self.at > self.job.get('steps',0)/50+15:
                raise RuntimeError('VLA job deadline exceeded')
            action, done = self.runtime.poll()
            if action is not None:
                self.runtime.apply_hands(action)
            hand_only = self.job['action'] in ('release_hand','hand')
            hand_action = self.job.get('hand_action','open')
            if done:
                preparing = self.job['action']=='prepare'
                if preparing:
                    self.initialized = True
                self.runtime.pause()
                self.finish('initial_pose_complete' if preparing else ('hand_'+hand_action+'_command_complete') if hand_only else 'step_budget_complete', True)
            return None if hand_only else action
        except Exception as exc:
            self.pause('execution_paused: '+str(exc), invalidate=self.phase=='preparing')
            return None
