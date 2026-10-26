"""Sole-thread Isaac owner. The MCP frontend only exchanges bounded requests.

Launch with the existing BEHAVIOR interpreter; never import Isaac in a frontend.
The socket is private to the invoking user. Physics freezes between requests.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import socket
import time
import traceback

from .schema import validate, MUTATING, NATIVE_MOTION_TOOLS, VLA_HANDOFF, VLA_ASSESS
from .timing import WallClockLedger


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


class Worker:
    def __init__(self, config, ledger):
        import numpy as np
        from behavior_ap_sim.environment import BehaviorEnvironment
        self.np = np
        self.config, self.ledger = config, ledger
        self.output = Path(config['output'])
        self.env = BehaviorEnvironment(self.output / 'sim', task=config['task'],
            variant='original', split='public_test', instance_id=301,
            max_steps=config.get('max_steps', 10000), render_profile='fast',
            headless=True, write_video=True, depth_observations=True)
        if self.env.backend.config['seed'] != config.get('seed', 20261020):
            raise ValueError('Requested seed differs from the registered scene seed')
        self.control = None
        self.frames = {}
        self.observations = 0
        self.submitted = self.acknowledged = 0
        self.actions = []
        self.uncertain = None
        self.finish_decision = None
        self.calls = 0
        self.vla = None
        self.action_owner = {'controller': 'native_tools'}
        self.execution = (self.output / 'execution.jsonl').open('x', buffering=1)

    def remaining_wall_seconds(self):
        deadline=self.config.get('deadline_monotonic')
        return None if deadline is None else max(0,deadline-time.monotonic())

    def expire_if_due(self):
        remaining=self.remaining_wall_seconds()
        if remaining is not None and remaining<=0 and not self.env.done:
            self.env.backend.done=True
            self.env.backend.truncated=True
            self.env.backend.termination_reason='wall_time_limit'
        return remaining is not None and remaining<=0

    def array(self, value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        return self.np.asarray(value)

    def start(self):
        from .robot import R1ProControl
        with self.ledger.span('initialization'):
            self.env.start()
            self.sensors = {role: self.env.backend.robot.sensors[key.split('::', 1)[1]]
                for role, key in self.env.backend.evaluator.robot_camera_names.items()}
            # Depth annotators are configured before scene creation. Attaching
            # them to live render products can return empty buffers in Isaac.
            self.env.refresh()
            self.control = R1ProControl(self.env, self.step)
            if self.config.get('vla_tools', {}).get('enabled'):
                from .vla import VlaTools
                self.vla = VlaTools(self, self.config['vla_tools'])
            # Validate depth without moving or giving the Agent privileged state.
            depth_status = {}
            for role, sensor in self.sensors.items():
                obs, _ = sensor.get_obs()
                depth = self.array(obs['depth_linear']).squeeze()
                good = self.np.isfinite(depth) & (depth > 0)
                depth_status[role] = {'shape': list(depth.shape), 'valid_positive_pixels': int(good.sum()),
                    'min_positive_depth_m': float(depth[good].min()) if good.any() else None,
                    'max_positive_depth_m': float(depth[good].max()) if good.any() else None}
                save(self.output / 'initial_depth_diagnostics.json', depth_status)
                if depth.ndim != 2 or depth.shape != self.array(obs['rgb']).shape[:2]:
                    raise RuntimeError(f'Invalid initial depth shape: {role}: {depth.shape}')
                # A wrist can initially look into empty space. Invalid pixels
                # remain invalid; pixel projection refuses them instead of
                # fabricating geometry or rejecting an otherwise valid scene.
                if role == 'head' and not good.any():
                    raise RuntimeError('Head depth has no finite positive measurements')
        save(self.output / 'worker_ready.json', {'status': 'ready', 'pid': os.getpid(),
            'render_profile': 'fast', 'step': self.env.steps, 'depth_linear': True,
            'task': self.config['task'], 'instance_id': 301, 'split': 'public_test',
            'seed': self.env.backend.config['seed']})

    def step(self, action):
        if self.config.get('vla_tools', {}).get('vla_only') and self.action_owner.get('controller') not in ('ap_vla','champion_vla'):
            raise RuntimeError('VLA-only worker rejects non-policy physical actions before admission')
        if self.uncertain or self.finish_decision:
            raise RuntimeError('Physical execution is sealed or acknowledgement uncertain')
        if self.env.done:
            return False
        value = self.np.asarray(action, dtype=self.np.float32)
        if value.shape != (23,) or not self.np.isfinite(value).all():
            raise ValueError('A physical action must be finite23D before admission')
        before = self.env.steps
        seq = self.submitted
        self.execution.write(json.dumps({'event': 'submitted', 'sequence': seq,
            'step_before': before, 'perf_counter_ns': time.perf_counter_ns(),
            'action': value.tolist(), 'owner': getattr(self, 'action_owner', {'controller':'native_tools'})}) + '\n')
        self.execution.flush()
        os.fsync(self.execution.fileno())
        self.submitted += 1
        self.actions.append(value.copy())
        # Invalidate geometry at admission, including a failed step that advances
        # native physics before the backend can increment its step counter.
        self.frames = {}
        self.ledger.emit('action_submitted', sequence=seq, step=before)
        try:
            self.env.backend.action_owner=dict(self.action_owner)
            result = self.env.step(value)
            if not result or self.env.steps != before + 1:
                raise RuntimeError('Submitted physical action lacks an exact one-step acknowledgement')
            self.execution.write(json.dumps({'event': 'ack', 'sequence': seq,
                'step_after': self.env.steps, 'perf_counter_ns': time.perf_counter_ns()}) + '\n')
            self.execution.flush()
            os.fsync(self.execution.fileno())
            self.ledger.emit('action_ack', sequence=seq, step=self.env.steps)
            self.acknowledged += 1
        except BaseException as exc:
            self.uncertain = f'{type(exc).__name__}: {exc}'
            save(self.output / 'uncertain_action.json', {'sequence': seq, 'step_before': before,
                'step_after': self.env.steps, 'error': self.uncertain, 'replay_forbidden': True})
            raise
        return result

    def state(self):
        return {'task': self.config['task'], 'task_description': self.config['prompt'],
            'step': self.env.steps, 'max_steps': self.config.get('max_steps', 10000),
            'official_success': bool(self.env.success), 'done': bool(self.env.done),
            'termination_reason': self.env.termination_reason,
            'remaining_wall_seconds':self.remaining_wall_seconds(),
            'finish_decision': self.finish_decision, 'uncertain_action': self.uncertain,
            'acknowledged_steps': self.acknowledged,
            'robot': self.control.read_state(), 'render_profile': 'fast',
            'vla': self.vla.state() if getattr(self, 'vla', None) else {'enabled': False},
            'physics_frozen_between_requests': True}

    def observe(self):
        with self.ledger.span('observation'):
            self.observations += 1
            ident = f'obs_{self.observations:05d}_step_{self.env.steps:06d}'
            directory = self.output / 'observations' / ident
            record = self.env.capture(directory)
            base_position, base_quat = self.env.backend.robot.get_position_orientation()
            info, frames, images = {}, {}, []
            for role, sensor in self.sensors.items():
                obs, _ = sensor.get_obs()
                depth = self.array(obs['depth_linear']).squeeze().copy()
                K = self.array(sensor.intrinsic_matrix).copy()
                position, quat = sensor.get_position_orientation()
                values = dict(depth=depth, K=K, camera_position=self.array(position),
                    camera_quat=self.array(quat), robot_base_position=self.array(base_position),
                    robot_base_quat=self.array(base_quat))
                self.np.savez_compressed(directory / (role + '_geometry.npz'), **values)
                frames[role] = values
                info[role] = {'width': int(depth.shape[1]), 'height': int(depth.shape[0]),
                    'intrinsics': K.tolist(), 'depth_unit': 'metres',
                    'valid_positive_depth_pixels': int((self.np.isfinite(depth) & (depth > 0)).sum()),
                    'depth_definition': 'axial optical Z, not Euclidean range'}
                images.append({'type': 'image', 'mimeType': 'image/png',
                    'data': base64.b64encode((directory / (role + '.png')).read_bytes()).decode()})
            self.frames = {ident: {'step': self.env.steps, 'cameras': frames}}
            value = {'frame_id': ident, 'step': self.env.steps, 'cameras': info,
                'image_order': list(self.sensors), 'state': self.state()}
            save(directory / 'agent_observation.json', value)
            return value, images

    def call(self, name, args):
        validate(name, args)
        if self.config.get('vla_tools', {}).get('vla_only') and name in NATIVE_MOTION_TOOLS:
            raise ValueError('Direct numeric motion is disabled in VLA-only scheduling')
        self.calls += 1
        self.expire_if_due()
        if name == 'behavior_state':
            return self.state(), []
        if name == 'behavior_observe':
            return self.observe()
        if name == 'behavior_pixel_to_position':
            from .geometry import project_pixel
            frame = self.frames.get(args['frame_id'])
            if self.uncertain or frame is None or frame['step'] != self.env.steps:
                raise ValueError('Stale frame: obtain a fresh observation after every physical action')
            values = frame['cameras'][args['camera']]
            return project_pixel(values['depth'], values['K'], [args['x'], args['y']],
                **{k: v for k, v in values.items() if k not in ('depth', 'K')},
                output_frame=args.get('coordinate_frame', 'base'), radius=args.get('radius', 1)), []
        if name == 'behavior_finish':
            if getattr(self, 'vla', None) and self.vla.pending() and args['outcome'] != 'blocked':
                return {'accepted': False, 'reason': 'Continue the pending VLA job; model waits are not task failures'}, []
            if self.finish_decision:
                return {'accepted': False, 'reason': 'Episode outcome already sealed',
                    'original_decision': self.finish_decision}, []
            if args['outcome'] == 'success' and not self.env.success:
                return {'accepted': False, 'official_success': False,
                    'reason': 'Official BDDL success is false; task remains unfinished'}, []
            if (self.config.get('vla_tools', {}).get('closed_loop') and args['outcome']=='failure'
                    and not self.env.done and (self.config.get('max_steps',10000) is None or self.env.steps < self.config.get('max_steps',10000))):
                return {'accepted':False,'reason':'Temporary stagnation requires visual reassessment and bounded AP/champion recovery; continue until official success or physical budget exhausted. Use blocked only for concrete safety/infrastructure evidence.'}, []
            self.finish_decision = dict(args, step=self.env.steps,
                official_success=bool(self.env.success), perf_counter_ns=time.perf_counter_ns())
            save(self.output / 'agent_finish.json', self.finish_decision)
            return {'accepted': True, **self.finish_decision}, []
        if self.env.done or self.finish_decision or self.uncertain:
            raise RuntimeError('Episode has finished or acknowledgement is uncertain; no further actions')
        vla = getattr(self, 'vla', None)
        if name == VLA_ASSESS:
            if vla is None:raise ValueError('VLA tools are not enabled')
            return vla.assess_interaction(**args), []
        if name == VLA_HANDOFF:
            if vla is None:raise ValueError('VLA tools are not enabled')
            return vla.authorize_champion(**args), []
        if name in ('behavior_active_perception_vla', 'behavior_champion_vla', 'behavior_vla_continue'):
            if vla is None:
                raise ValueError('VLA tools are not enabled in this operator session')
            if name == 'behavior_vla_continue':
                return vla.continue_job(**args)
            return vla.start('ap' if name == 'behavior_active_perception_vla' else 'champion', args)
        if vla:
            if vla.pending() or vla.infrastructure_errors:
                raise RuntimeError('VLA ownership pending/invalid; ordinary motion is blocked')
            vla.epoch += 1  # Measured native motion invalidates champion temporal history.
        self.action_owner = {'controller': 'native_tools', 'tool': name}
        functions = {'behavior_move_base': self.control.move_base, 'behavior_move_arm': self.control.move_arm,
            'behavior_gripper': self.control.gripper, 'behavior_torso': self.control.torso}
        with self.ledger.span('action_execution', tool=name, step_before=self.env.steps):
            result = functions[name](**args)
        return {'motion': result, 'state': self.state()}, []

    def finalize(self, reason):
        from .video import verify_videos, VideoVerificationError
        self.expire_if_due()
        if getattr(self, 'vla', None):
            self.vla.close()
        self.execution.close()
        self.np.save(self.output / 'actions.npy', self.np.asarray(self.actions, dtype=self.np.float32).reshape(-1, 23))
        with self.ledger.span('video_finalize'):
            result = self.env.close(reason)
            try:
                video_evidence = verify_videos(result, self.env.steps)
            except VideoVerificationError as exc:
                save(self.output / 'video_verification_error.json', exc.evidence)
                raise
        record = {'status': 'closed', 'max_steps':self.config.get('max_steps',10000),
            'wall_limit_seconds':self.config.get('wall_limit_seconds'), 'task': self.config['task'], 'steps': self.env.steps,
            'official_success': result['success'], 'submitted': self.submitted,
            'acknowledged': self.acknowledged, 'execution_integrity': self.submitted == self.acknowledged == self.env.steps,
            'uncertain_action': self.uncertain, 'agent_finish': self.finish_decision,
            'tool_requests': self.calls, 'observations': self.observations,
            'termination_reason': result['termination_reason'], 'errors': result['errors'],
            'videos': video_evidence, 'render_profile': 'fast', 'automatic_replay': False}
        if getattr(self, 'vla', None):
            record['vla_evidence'] = json.loads((self.output / 'vla/summary.json').read_text())
        save(self.output / 'episode_closed.json', record)
        return record


def serve(worker, config):
    path = Path(config['socket_path'])
    if path.exists():
        raise FileExistsError('Socket already exists; never attach a new physical episode to an old socket')
    cache = {}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
        os.chmod(path, 0o600)
        if os.geteuid() == 0:
            os.chown(path, config['uid'], config['gid'])
        server.listen(4)
        try:
            while True:
                conn, _ = server.accept()
                close = False
                with conn:
                    conn.settimeout(300)
                    try:
                        with conn.makefile('rb') as stream:
                            raw = stream.readline(1024 * 1024)
                    except OSError as exc:
                        worker.ledger.emit('request_transport_error', error=str(exc))
                        continue
                    try:
                        request = json.loads(raw)
                        if not isinstance(request, dict) or set(request) - {'id', 'name', 'arguments'}:
                            raise ValueError('Invalid request object')
                        ident = request.get('id')
                        if not isinstance(ident, str) or not ident or len(ident) > 128:
                            raise ValueError('Invalid request identity')
                        if not isinstance(request.get('name'), str) or not isinstance(request.get('arguments', {}), dict):
                            raise ValueError('Invalid request name or arguments')
                        payload = json.dumps(request, sort_keys=True, allow_nan=False)
                    except (ValueError, TypeError, UnicodeDecodeError) as exc:
                        reply = {'id': None, 'error': f'Malformed request: {exc}'}
                        try:
                            conn.sendall((json.dumps(reply) + '\n').encode())
                        except OSError:
                            pass
                        continue
                    if ident in cache:
                        old, reply = cache[ident]
                        if old != payload:
                            reply = {'id': ident, 'error': 'Request ID reused with different payload; execution forbidden'}
                    else:
                        try:
                            if request['name'] == '_close':
                                close = True
                                value, images = {'closing': True}, []
                            else:
                                with worker.ledger.span('tool_dispatch', tool=request['name'], request_id=ident):
                                    value, images = worker.call(request['name'], request.get('arguments', {}))
                            reply = {'id': ident, 'value': value, 'images': images}
                        except Exception as exc:
                            reply = {'id': ident, 'error': f'{type(exc).__name__}: {exc}'}
                            worker.ledger.emit('tool_error', request_id=ident, tool=request['name'], error=reply['error'])
                        cache[ident] = payload, reply
                    # Preserve the reply before sending; dropped transport never replays actions.
                    log = {k: v for k, v in reply.items() if k != 'images'}
                    save(worker.output / 'last_response.json', dict(log, request=request))
                    try:
                        conn.sendall((json.dumps(reply, allow_nan=False) + '\n').encode())
                    except (BrokenPipeError, ConnectionResetError, socket.timeout) as exc:
                        worker.ledger.emit('response_transport_error', request_id=ident, error=str(exc))
                if close:
                    return
        finally:
            path.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'worker_timing').exists():
        raise FileExistsError('Existing worker evidence; start a fresh initial episode')
    ledger = WallClockLedger(output / 'worker_timing', expected_categories=(
        'gpu_wait', 'initialization', 'observation', 'action_execution', 'video_finalize'))
    worker = None
    from behavior_ap_sim.resources import gpu_lease
    reason = 'agent_session_ended'
    try:
        lease = gpu_lease(wait=True)
        with ledger.span('gpu_wait'):
            lease.__enter__()
        worker = Worker(config, ledger)
        worker.start()
        serve(worker, config)
    except BaseException as exc:
        reason = 'infrastructure_error'
        save(output / 'worker_error.json', {'type': type(exc).__name__, 'error': str(exc),
            'traceback': traceback.format_exc(), 'automatic_replay': False})
        raise
    finally:
        try:
            if worker is not None:
                worker.finalize(reason)
        except BaseException as exc:
            save(output / 'finalization_error.json', {'type': type(exc).__name__, 'error': str(exc),
                'traceback': traceback.format_exc()})
            raise
        finally:
            try:
                ledger.finish()
            finally:
                try:
                    if worker is not None:
                        try:
                            worker.env.shutdown()
                        except SystemExit as exc:
                            if exc.code not in (None, 0):
                                raise
                finally:
                    if 'lease' in locals():
                        lease.__exit__(None, None, None)


if __name__ == '__main__':
    main()
