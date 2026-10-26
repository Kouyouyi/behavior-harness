"""Characterize the RB-Y1 backend's control tools on the robot (docs/REAL_ROBOT_ADAPTATION.md, Rule 2).

    $PY scripts/rby1_characterize.py <session_root> <part> [<part> ...]

Runs through the real Rby1Backend (the agent's guards), with the operator's arming and a
person at the e-stop. Every move is small, starts from the current pose (normally `ready`)
and returns to it. Results are printed and saved as JSON under
<artifact root>/characterization/<UTC stamp>_<part>.json, to be summarized in RBY1.md and
the control skills.

Parts:
  overhead    wall time of rby1_status, rby1_observe and a 0.01 rad head move (calls only)
  refusals    the exact refusal text at each limit (validation only: nothing moves)
  joints      duration and final error by move size, small to normal (up to 1 rad): head tilt and
              pan, arm wrist, arm elbow, torso yaw, torso height (upright rule kept); and the same
              arm move with the controller's hold time 1.0 s vs 0.2 s
  repeat      the right hand to the same +5 cm pose 5 times from `ready`: spread of the
              reached positions (rby1_move_eef)
  eef         rby1_move_eef 2 / 5 / 10 / 20 cm along x, y, z on the right arm: final error,
              duration, and the deviation of the hand's path (FK of the measured joints,
              sampled during the move) from the straight line
  look        rby1_look_at a few points: predicted vs. reached pointing error
  gripper     open / close duration and openings at two current limits (no object)
  observe     depth validity, intrinsics, stream rates, and pixel_to_position repeatability
              of one image point over 3 frames
  base        forward 0.3 m and back, turn +-0.5 rad (0.1 m/s, 0.2 rad/s): odometry against
              the depth-measured distance to a fixed point straight ahead
              (base_line: only the straight moves; base_turn: only the turns)
"""
from dataclasses import replace
import json
import sys
import threading
import time

import numpy as np

from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from services.controller.config import Configuration  # noqa: E402
from services.rby1.backend import Rby1Backend  # noqa: E402
from services.rby1.kinematics import EE_FRAME, component, pose  # noqa: E402
from services.storage_root import artifact_root  # noqa: E402
from services import arming  # noqa: E402


def timed(fn, *args, **kwargs):
    t0 = time.monotonic()
    value = fn(*args, **kwargs)
    return value, time.monotonic() - t0


def call(backend, name, arguments):
    backend.validate(name, arguments)
    (value, _), seconds = timed(backend.call, name, arguments)
    return value, seconds


def joints(backend, name, goal):
    return call(backend, 'rby1_move_joints', {'component': name, 'positions': np.round(goal, 5).tolist()})


def stats(values):
    v = np.asarray(values, dtype=float)
    return {'n': int(v.size), 'mean': round(float(v.mean()), 4), 'min': round(float(v.min()), 4),
            'max': round(float(v.max()), 4)}


def overhead(backend, out):
    out['status_s'] = stats([timed(backend.call, 'rby1_status', {})[1] for _ in range(10)])
    if backend.camera is not None:
        out['observe_s'] = stats([timed(backend.call, 'rby1_observe', {})[1] for _ in range(5)])
    head = component(backend._snapshot().q, 'head')
    tiny = []
    for sign in (1, -1, 1, -1):
        goal = head + np.array([0.0, 0.01 if sign > 0 else 0.0])
        tiny.append(joints(backend, 'head', goal)[1])
    out['tiny_head_move_s'] = stats(tiny)


def refusals(backend, out):
    q = backend._snapshot().q
    right = component(q, 'right_arm')
    p, quat = pose(backend.kinematics.frame(EE_FRAME['right'], q))
    cases = {
        'joint_limit': ('rby1_move_joints', {'component': 'head', 'positions': [0.6, 0.0]}),
        'step_cap': ('rby1_move_joints', {'component': 'right_arm', 'relative': True,
                                          'positions': [1.3, 0, 0, 0, 0, 0, 0]}),
        'torso_rule': ('rby1_move_joints', {'component': 'torso', 'relative': True, 'positions': [0, 0.1, 0, 0, 0, 0]}),
        'reach': ('rby1_move_eef', {'arm': 'right', 'position': (p + [0.9, 0, 0]).tolist()}),
        'eef_step': ('rby1_move_eef', {'arm': 'right', 'position': [0, 0, 0.3], 'relative': True}),
        'unreachable_orientation': ('rby1_move_eef', {'arm': 'right', 'position': p.tolist(),
                                                      'quaternion': [1.0, 0.0, 0.0, 0.0]}),
        'self_collision': ('rby1_move_joints', {'component': 'right_arm', 'positions':
                                                [-0.082, -0.12, 0.632, -2.249, 0.665, 0.106, -0.921]}),
        'look_behind': ('rby1_look_at', {'point': [-2.0, 0.0, 1.0]}),
        'base_step': ('rby1_base_move', {'forward': 1.5}),
        'gripper_current': ('rby1_gripper', {'side': 'right', 'action': 'close', 'current_a': 3.0}),
    }
    for name, (tool, arguments) in cases.items():
        try:
            backend.validate(tool, arguments)
            out[name] = 'NOT REFUSED'
        except Exception as exc:
            out[name] = f'{type(exc).__name__}: {exc}'
    out['_right_arm_unchanged'] = bool(np.allclose(component(backend._snapshot().q, 'right_arm'), right, atol=0.005))


def joint_sizes(backend, out):
    q = backend._snapshot().q
    plans = {'head_tilt': ('head', [1], (0.05, 0.2, 0.5)), 'head_pan': ('head', [0], (0.05, 0.4)),
             'wrist_right_arm_6': ('right_arm', [6], (0.05, 0.2, 0.5, 1.0)),
             'elbow_right_arm_3': ('right_arm', [3], (0.05, 0.2, 0.5, 1.0)),
             'torso_yaw_torso_5': ('torso', [5], (0.05, 0.2, 0.4)),
             # torso height: a -> a + d on (torso_1, torso_2, torso_3) = (d, -2d, d), the upright rule kept
             'torso_height': ('torso', [1, 2, 3], (0.1, 0.26))}
    for label, (name, indices, sizes) in plans.items():
        start = component(q, name).copy()
        rows = []
        for size in sizes:
            goal = start.copy()
            for index, factor in zip(indices, (1, -2, 1) if len(indices) == 3 else (1,)):
                goal[index] += factor * size
            value, seconds = joints(backend, name, goal)
            back, back_s = joints(backend, name, start)
            rows.append({'size_rad': size, 'seconds': round(seconds, 2), 'planned_s': value['planned_s'],
                         'error_rad': value['max_joint_error_rad'], 'back_s': round(back_s, 2),
                         'back_error_rad': back['max_joint_error_rad']})
        out[label] = rows
    hold = {}
    start = component(q, 'right_arm').copy()
    goal = start.copy()
    goal[6] += 0.2
    for hold_s in (1.0, 0.2):
        backend.options['control_hold_time_s'] = hold_s
        rows = []
        for _ in range(3):
            value, seconds = joints(backend, 'right_arm', goal)
            back, back_s = joints(backend, 'right_arm', start)
            rows.append((seconds, value['max_joint_error_rad'], back_s, back['max_joint_error_rad']))
        rows = np.asarray(rows)
        hold[f'hold_{hold_s}s'] = {'seconds': stats(rows[:, 0]), 'error_rad': stats(rows[:, 1]),
                                   'back_seconds': stats(rows[:, 2]), 'back_error_rad': stats(rows[:, 3])}
    backend.options['control_hold_time_s'] = 1.0
    out['hold_time_0.2_rad_wrist'] = hold


def repeat(backend, out):
    q = backend._snapshot().q
    start_arm = component(q, 'right_arm').copy()
    p0, q0 = pose(backend.kinematics.frame(EE_FRAME['right'], q))
    target = p0 + np.array([0.05, 0.0, 0.0])
    reached, returned = [], []
    for _ in range(5):
        value, _ = call(backend, 'rby1_move_eef', {'arm': 'right', 'position': target.round(5).tolist(),
                                                  'quaternion': q0.round(6).tolist()})
        reached.append(value['right_ee']['position'])
        value, _ = joints(backend, 'right_arm', start_arm)
        returned.append(value['right_ee']['position'])
    reached, returned = np.asarray(reached), np.asarray(returned)
    out['target'] = target.round(4).tolist()
    out['reached_spread_mm'] = round(float(np.max(np.linalg.norm(reached - reached.mean(0), axis=1))) * 1000, 2)
    out['reached_mean_error_mm'] = round(float(np.linalg.norm(reached.mean(0) - target)) * 1000, 2)
    out['returned_spread_mm'] = round(float(np.max(np.linalg.norm(returned - returned.mean(0), axis=1))) * 1000, 2)


def eef(backend, out):
    q = backend._snapshot().q
    start_arm = component(q, 'right_arm').copy()
    p0, q0 = pose(backend.kinematics.frame(EE_FRAME['right'], q))
    rows = []
    for axis, label in ((0, 'x'), (1, 'y'), (2, 'z')):
        for size in (0.02, 0.05, 0.10, 0.20):
            direction = np.zeros(3)
            direction[axis] = size if axis != 1 else -size  # y: outwards (right arm)
            target = p0 + direction
            samples, stop = [], threading.Event()

            def sample():
                while not stop.is_set():
                    samples.append(backend._snapshot().q)
                    time.sleep(0.02)
            arguments = {'arm': 'right', 'position': target.round(5).tolist(), 'quaternion': q0.round(6).tolist()}
            try:
                backend.validate('rby1_move_eef', arguments)
            except ValueError as exc:  # refused before moving: record the reason, keep going
                rows.append({'axis': label, 'size_m': size, 'refused': str(exc)})
                continue
            thread = threading.Thread(target=sample)
            thread.start()
            try:
                value, seconds = call(backend, 'rby1_move_eef', arguments)
            finally:
                stop.set()
                thread.join()
            path = np.array([backend.kinematics.frame(EE_FRAME['right'], s)[:3, 3] for s in samples])
            u = (target - p0) / np.linalg.norm(target - p0)
            rel = path - p0
            along = np.clip(rel @ u, 0.0, np.linalg.norm(target - p0))
            deviation = np.linalg.norm(rel - np.outer(along, u), axis=1)
            rows.append({'axis': label, 'size_m': size, 'error_mm': value['position_error_mm'],
                         'seconds': round(seconds, 2), 'planned_s': value['planned_s'],
                         'path_max_deviation_mm': round(float(deviation.max()) * 1000, 1), 'samples': len(samples)})
            joints(backend, 'right_arm', start_arm)
    out['moves'] = rows


def look(backend, out):
    head0 = component(backend._snapshot().q, 'head').copy()
    rows = []
    for point in ([2.0, 0.0, 1.0], [2.0, 0.6, 1.0], [1.5, -0.4, 0.6], [1.2, 0.0, 0.4]):
        try:
            value, seconds = call(backend, 'rby1_look_at', {'point': point})
            from services.rby1.kinematics import HEAD_CAMERA
            T = backend.kinematics.frame(HEAD_CAMERA, backend._snapshot().q)
            ray = (np.asarray(point) - T[:3, 3]) / np.linalg.norm(np.asarray(point) - T[:3, 3])
            reached = float(np.degrees(np.arccos(np.clip(T[:3, 2] @ ray, -1, 1))))
            rows.append({'point': point, 'predicted_deg': value['predicted_error_deg'],
                         'reached_deg': round(reached, 2), 'seconds': round(seconds, 2)})
        except Exception as exc:
            rows.append({'point': point, 'refused': str(exc)})
    joints(backend, 'head', head0)
    out['points'] = rows


def gripper(backend, out):
    rows = []
    for current in (0.3, 0.8):
        for action in ('close', 'open'):
            value, seconds = call(backend, 'rby1_gripper', {'side': 'right', 'action': action, 'current_a': current})
            rows.append({'action': action, 'current_a': current, 'seconds': round(seconds, 2),
                         'measured_width_m': value['measured_width_m'], 'homed_now': value['homed_now']})
    out['right'] = rows


def observe(backend, out):
    if backend.camera is None:
        out['error'] = backend.camera_error
        return
    frames = []
    for _ in range(3):
        value, seconds = timed(backend.call, 'rby1_observe', {})
        frames.append((value[0], seconds))
        time.sleep(0.3)
    meta = json.loads((backend.frames_root/frames[0][0]['frame_id']/'meta.json').read_text())
    entry = meta['cameras']['head_camera']
    out['image'] = [entry['width'], entry['height']]
    out['K'] = entry['K']
    import cv2 as cv
    depth = cv.imread(str(backend.frames_root/frames[0][0]['frame_id']/entry['depth']), cv.IMREAD_UNCHANGED)
    out['depth_valid_fraction'] = round(float((depth > 0).mean()), 3)
    u, v = entry['width'] // 2, int(entry['height'] * 0.75)  # below the centre: the floor or a table
    points = []
    for value, _ in frames:
        reply, _ = backend.call('rby1_pixel_to_position', {'frame_id': value['frame_id'], 'camera': 'head_camera',
                                                           'pixels': [[u, v]], 'radius': 5})
        points.append(reply['points'][0])
    good = np.array([p['position'] for p in points if 'position' in p])
    out['pixel'] = [u, v]
    out['positions'] = good.round(4).tolist()
    if len(good) > 1:
        out['position_spread_mm'] = round(float(np.max(np.linalg.norm(good - good.mean(0), axis=1))) * 1000, 1)
    if backend.stream is not None:
        time.sleep(3.0)
        out['stream_index'] = json.loads((backend.stream/'index.json').read_text())


def base(backend, out, moves=({'forward': 0.3}, {'forward': -0.3}, {'turn': 0.5}, {'turn': -0.5})):
    def ahead():
        """Median depth-measured distance to the scene straight ahead (a patch at the image centre)."""
        value, _ = backend.call('rby1_observe', {})
        meta = json.loads((backend.frames_root/value['frame_id']/'meta.json').read_text())['cameras']['head_camera']
        import cv2 as cv
        depth = cv.imread(str(backend.frames_root/value['frame_id']/meta['depth']), cv.IMREAD_UNCHANGED)
        h, w = depth.shape
        patch = depth[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20]
        valid = patch[patch > 0]
        return float(np.median(valid)) / 1000.0 if valid.size else None, value['frame_id']
    rows = []
    for arguments in moves:
        before, frame_before = ahead() if 'forward' in arguments else (None, None)
        value, seconds = call(backend, 'rby1_base_move', {**arguments, 'max_linear': 0.1, 'max_angular': 0.2})
        time.sleep(0.5)
        after, frame_after = ahead() if 'forward' in arguments else (None, None)
        row = {'commanded': arguments, 'odometry': value['measured_by_odometry'], 'seconds': round(seconds, 2)}
        if before is not None and after is not None:
            row['camera_forward_m'] = round(before - after, 4)
            row['frames'] = [frame_before, frame_after]
        rows.append(row)
        time.sleep(0.5)
    out['moves'] = rows


PARTS = {'overhead': overhead, 'refusals': refusals, 'joints': joint_sizes, 'repeat': repeat, 'eef': eef,
         'look': look, 'gripper': gripper, 'observe': observe, 'base': base,
         'base_line': lambda b, out: base(b, out, ({'forward': 0.3}, {'forward': -0.3})),
         'base_turn': lambda b, out: base(b, out, ({'turn': 0.5}, {'turn': -0.5}))}


def main():
    root, parts = sys.argv[1], sys.argv[2:]
    unknown = set(parts) - set(PARTS)
    if not parts or unknown:
        sys.exit(f'parts: {" ".join(PARTS)}' + (f' (unknown: {sorted(unknown)})' if unknown else ''))
    config = Configuration.load(f'{root}/operator-input.json')
    camera = bool({'overhead', 'observe', 'base', 'base_line', 'base_turn'} & set(parts))
    options = {'camera': camera, 'grippers': 'gripper' in parts, 'stream_enabled': 'observe' in parts}
    backend = Rby1Backend(replace(config, backend_options=options))
    folder = artifact_root()/'characterization'
    folder.mkdir(parents=True, exist_ok=True)
    try:
        if set(parts) != {'refusals'} and not arming.read(config.root)['armed']:
            print('not armed: the operator arms first (rby1 arm 30)', flush=True)
            return
        if camera:
            deadline = time.time() + 30
            while backend.camera is not None and backend.camera.read() is None and time.time() < deadline:
                time.sleep(0.2)
        for part in parts:
            out = {'part': part, 'started': time.time(), 'start_pose': {n: np.round(component(
                backend._snapshot().q, n), 4).tolist() for n in ('torso', 'right_arm', 'left_arm', 'head')}}
            try:
                PARTS[part](backend, out)
            except Exception as exc:
                out['STOPPED'] = f'{type(exc).__name__}: {exc}'
                out['robot_state'] = getattr(exc, 'robot_state', None)
            path = folder/f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{part}.json"
            path.write_text(json.dumps(out, indent=1, default=str))
            print(json.dumps(out, default=str), flush=True)
            if 'STOPPED' in out:
                break
    finally:
        backend.close()
        print('closed', flush=True)


if __name__ == '__main__':
    main()
