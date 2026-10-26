"""Operator-supervised motion check through the real Rby1Backend (same guards the agent gets).

    $PY scripts/configure_session.py --task motion_check --task-text "motion check" --robot rby1
    $PY scripts/rby1_motion_check.py <session_root> <step>

It connects, waits up to 5 minutes for the operator to arm the session (`rby1 arm 5`, a
person at the e-stop), runs one step and closes. Every move is small and returns to the
pose the step started from; a move that does not arrive stops the step. Nothing moves
before the operator arms; the first armed step powers the drives on (they then hold).
Run each step once with nothing near the robot and record the printed numbers in
RBY1.md "Motion check". Re-run after any firmware or rby1_sdk upgrade.

Steps:
  status       connect and print status (no arming, nothing moves, nothing is powered)
  disarmed     motion refused while disarmed (moves nothing)
  head         head pan +0.1 rad and back, tilt +0.1 rad and back
  wrists       both arms' last joint +0.1 rad and back
  joints_all   every arm joint by 0.05 / 0.1 rad, direction chosen by FK to move the hand
               away from the body; all back
  torso        torso_5 (chest yaw) +0.05 rad and back; torso_3 (lean) +0.05 rad and back
  body         torso and both arms to `half`, then `ready`, head to `zero` (two rby1_move_body
               calls); the starting pose is NOT restored (`ready` is the neutral pose)
  base         forward / back 5 cm at 0.05 m/s, turn +-0.1 rad at 0.1 rad/s (wheel odometry)
  ik           each arm's end effector up 2 cm and back, forward 2 cm and back (rby1_move_eef)
  gripper      home both grippers (each is pushed fully closed, then fully open), then close and
               open each fully through rby1_gripper
  disarm_test  head tilt towards +0.3 rad at 0.1 rad/s; `rby1 disarm` mid-motion must stop it
               (PASS = cancelled, stopped short, then held within 0.005 rad from 0.5 s to 3 s)
  disarm_arm   the same with the right elbow (right_arm_3) +0.5 rad at the arms' normal 0.3 rad/s
"""
from dataclasses import replace
import json
import subprocess
import sys
import threading
import time

import numpy as np

from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from services.controller.config import Configuration  # noqa: E402
from services.rby1.backend import Rby1Backend  # noqa: E402
from services.rby1.kinematics import EE_FRAME, component, pose, with_component  # noqa: E402
from services import arming  # noqa: E402

COMPONENTS = ('torso', 'right_arm', 'left_arm', 'head')
STEPS = {
    # step: list of (component, relative delta); out on even steps, back to the start on odd ones
    'head': [('head', [0.1, 0.0]), ('head', [0.0, 0.0]), ('head', [0.0, 0.1]), ('head', [0.0, 0.0])],
    'wrists': [('right_arm', [0, 0, 0, 0, 0, 0, 0.1]), ('right_arm', [0] * 7),
               ('left_arm', [0, 0, 0, 0, 0, 0, 0.1]), ('left_arm', [0] * 7)],
    'torso': [('torso', [0, 0, 0, 0, 0, 0.05]), ('torso', [0] * 6), ('torso', [0, 0, 0, 0.05, 0, 0]),
              ('torso', [0] * 6)],
}


def pose_now(backend):
    q = backend._snapshot().q
    return {name: np.round(component(q, name), 4).tolist() for name in COMPONENTS}


def show(label, value):
    keep = {k: value[k] for k in ('component', 'seconds', 'planned_s', 'max_joint_error_rad', 'settled', 'finish',
                                  'target', 'position_error_mm') if k in value}
    print(label, json.dumps(keep), flush=True)


def move(backend, name, goal, label):
    arguments = {'component': name, 'positions': np.round(np.asarray(goal, dtype=float), 4).tolist()}
    backend.validate('rby1_move_joints', arguments)
    value, _ = backend.call('rby1_move_joints', arguments)
    show(label, value)
    time.sleep(0.3)


def disarm_test(backend, config, start, name='head', index=1, delta=0.3):
    """Disarm mid-motion: one joint of one component moves by delta; disarm once it has moved 0.05 rad."""
    origin = float(start[name][index])
    goal = np.asarray(start[name], dtype=float)
    goal[index] += delta
    arguments = {'component': name, 'positions': np.round(goal, 4).tolist()}
    backend.validate('rby1_move_joints', arguments)
    outcome = {}

    def joint():
        return float(component(backend._snapshot().q, name)[index])

    def run():
        try:
            outcome['value'] = backend.call('rby1_move_joints', arguments)[0]
        except Exception as exc:
            outcome['error'] = exc
    worker = threading.Thread(target=run)
    t0 = time.monotonic()
    worker.start()
    while worker.is_alive() and abs(joint() - origin) < 0.05 and time.monotonic() - t0 < 10:
        time.sleep(0.01)
    q_disarm, t_disarm = joint(), time.monotonic()
    print(f'{name} joint {index} moved {q_disarm - origin:+.3f} rad after {t_disarm - t0:.2f} s; running rby1 disarm',
          flush=True)
    cli = subprocess.run([str(PROJECT/'scripts'/'rby1'), 'disarm'], capture_output=True, text=True, timeout=30)
    print('rby1 disarm:', (cli.stdout + cli.stderr).strip(), flush=True)
    worker.join(30)
    error = outcome.get('error')
    print(f"backend call returned {time.monotonic() - t_disarm:.2f} s after disarm started: "
          f"{type(error).__name__ if error else 'NO ERROR'}: {error}", flush=True)
    q_stop = joint()
    print('hold errors:', (getattr(error, 'robot_state', None) or {}).get('hold_errors'), flush=True)
    samples, times = [], []
    t_stop = time.monotonic()
    for _ in range(60):  # 3 s: does it hold?
        samples.append(joint())
        times.append(time.monotonic() - t_stop)
        time.sleep(0.05)
    samples = np.asarray(samples)
    for t_mark in (0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0):
        k = min(range(len(times)), key=lambda i: abs(times[i] - t_mark))
        print(f'  t={times[k]:4.2f} s {samples[k] - origin:+.4f}', flush=True)
    settled = samples[np.asarray(times) >= 0.5]
    print(f'at disarm {q_disarm - origin:+.4f} rad from start; at stop {q_stop - origin:+.4f}; '
          f'settled {settled.mean() - origin:+.4f}; goal {delta:+.4f}', flush=True)
    print(f'overrun after the disarm {settled.mean() - q_disarm:+.4f} rad; hold drift after 0.5 s: '
          f'{settled.max() - settled.min():.4f} rad', flush=True)
    stopped_short = abs(q_stop - origin) < abs(delta) - 0.05
    held = (settled.max() - settled.min()) < 0.005
    ok = error is not None and getattr(error, 'control_certain', False) and stopped_short and held
    print(f"{'PASS' if ok else 'FAIL'}: stopped_short={stopped_short} held={held}", flush=True)


def outward_sign(backend, side, index, delta):
    """+1 or -1: the direction of this arm joint that moves the hand away from the body midline (|y| up)."""
    q = backend._snapshot().q
    name = f'{side}_arm'
    y = {}
    for sign in (1, -1):
        arm = component(q, name).copy()
        arm[index] += sign * delta
        y[sign] = abs(backend.kinematics.frame(EE_FRAME[side], with_component(q, name, arm))[1, 3])
    return 1 if y[1] >= y[-1] else -1


def joints_all(backend, start):
    for side in ('right', 'left'):
        name = f'{side}_arm'
        base = np.asarray(start[name])
        for index in range(7):
            delta = 0.05 if index < 4 else 0.1
            sign = outward_sign(backend, side, index, delta)
            goal = base.copy()
            goal[index] += sign * delta
            move(backend, name, goal, f'{name} j{index} {sign * delta:+.2f}:')
            move(backend, name, base, f'{name} j{index} back:')
    print('PASS: every arm joint moved and returned', flush=True)


def ik_check(backend, start):
    for side in ('right', 'left'):
        p0, q0 = pose(backend.kinematics.frame(EE_FRAME[side], backend._snapshot().q))
        for label, d in (('up 2 cm', [0, 0, 0.02]), ('forward 2 cm', [0.02, 0, 0])):
            for target, what in ((p0 + np.asarray(d), label), (p0, 'back')):
                arguments = {'arm': side, 'position': np.round(target, 4).tolist(),
                             'quaternion': np.round(q0, 5).tolist()}
                backend.validate('rby1_move_eef', arguments)
                value, _ = backend.call('rby1_move_eef', arguments)
                show(f'{side} {what}:', value)
                time.sleep(0.3)
        name = f'{side}_arm'
        drift = float(np.max(np.abs(component(backend._snapshot().q, name) - np.asarray(start[name]))))
        print(f'{name} joint drift after IK round trips: {drift:.4f} rad', flush=True)
        if drift > 0.01:
            move(backend, name, start[name], f'{name} exact restore:')
    print('PASS: IK moves reached their targets and returned', flush=True)


def main():
    root, step = sys.argv[1], sys.argv[2]
    config = Configuration.load(f'{root}/operator-input.json')
    options = {'stream_enabled': False, 'camera': False, 'grippers': step == 'gripper'}
    if step == 'disarm_test':
        options['head_speed_radps'] = 0.1  # slow: plenty of time to disarm mid-motion
    backend = Rby1Backend(replace(config, backend_options=options))
    try:
        if step == 'status':
            print(json.dumps(backend.status(), indent=1, default=str), flush=True)
            return
        if step == 'disarmed':
            try:
                backend.validate('rby1_move_joints', {'component': 'head', 'positions': [0.1, 0], 'relative': True})
                print('FAIL: motion was not refused while disarmed', flush=True)
            except PermissionError as exc:
                print('OK refused while disarmed:', exc, flush=True)
            return
        print(f'waiting for the operator to arm {config.root.name} (rby1 arm 5) ...', flush=True)
        deadline = time.time() + 300
        while not arming.read(config.root)['armed']:
            if time.time() > deadline:
                print('not armed within 5 minutes; nothing moved', flush=True)
                return
            time.sleep(0.5)
        print('armed by the operator; starting', step, flush=True)
        start = pose_now(backend)
        print('start', json.dumps(start), flush=True)
        if step == 'disarm_test':
            disarm_test(backend, config, start)
            return
        if step == 'disarm_arm':
            disarm_test(backend, config, start, 'right_arm', 3, 0.5)
            return
        if step == 'body':
            for preset in ('half', 'ready'):
                targets = {'torso': {'preset': preset}, 'right_arm': {'preset': preset},
                           'left_arm': {'preset': preset}, 'head': {'preset': 'zero'}}
                backend.validate('rby1_move_body', {'targets': targets})
                value, _ = backend.call('rby1_move_body', {'targets': targets})
                show(f'body {preset}:', value)
                time.sleep(0.5)
            return
        if step == 'base':
            for label, arguments in (('forward 5 cm', {'forward': 0.05}), ('back 5 cm', {'forward': -0.05}),
                                     ('turn +0.1 rad', {'turn': 0.1}), ('turn -0.1 rad', {'turn': -0.1})):
                arguments = {**arguments, 'max_linear': 0.05, 'max_angular': 0.1}
                backend.validate('rby1_base_move', arguments)
                value, _ = backend.call('rby1_base_move', arguments)
                print(label, json.dumps({k: value[k] for k in ('commanded', 'measured_by_odometry', 'segments')}),
                      flush=True)
                time.sleep(1.0)
            return
        if step == 'gripper':
            for arguments in ({'side': 'right', 'action': 'close'}, {'side': 'left', 'action': 'close'},
                              {'side': 'right', 'action': 'open'}, {'side': 'left', 'action': 'open'}):
                backend.validate('rby1_gripper', arguments)
                value, _ = backend.call('rby1_gripper', arguments)
                print(json.dumps(value), flush=True)
            return
        if step == 'ik':
            ik_check(backend, start)
            return
        if step == 'joints_all':
            joints_all(backend, start)
            return
        for i, (name, delta) in enumerate(STEPS[step]):
            goal = np.asarray(start[name]) + np.asarray(delta)
            move(backend, name, goal, f'{name} {delta}:')
            time.sleep(0.5)
        end = pose_now(backend)
        drift = {n: float(np.max(np.abs(np.asarray(end[n]) - np.asarray(start[n])))) for n in start}
        print('largest change start->end per component (rad):', json.dumps({k: round(v, 4) for k, v in drift.items()}),
              flush=True)
    except Exception as exc:
        print(f'STOPPED: {type(exc).__name__}: {exc}', flush=True)
        print('robot_state', json.dumps(getattr(exc, 'robot_state', None), default=str), flush=True)
    finally:
        backend.close()
        print('closed', flush=True)


if __name__ == '__main__':
    main()
