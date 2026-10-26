"""Operator-supervised health check of the RB-Y1 through the real Rby1Backend, at its slow defaults.

    $PY scripts/configure_session.py --task health_check --task-text "health check" --robot rby1
    $PY scripts/rby1_health_check.py <session_root> [part ...]

One arming covers the whole run (`rby1 arm 30`, a person at the e-stop). Every move goes
through the agent's tools and guards, while a sampler records joint position, velocity,
torque and drive temperature at 25 Hz. Per move it prints the duration, the final error,
the peak velocity and acceleration (smoothness) and the highest torque as a fraction of
the joint's rating. Raw samples and results go to runtime/characterization/health_*.json.

Parts (default: all, in this order):
  rest      no motion: faults, temperatures, holding torque vs the vendor gravity model
  ready     torso, arms and head to `ready` / `zero` (one rby1_move_body)
  joints    each arm joint 0.1 rad out and back, torso squat +-0.1 rad (torso_1, torso_2 = -2x),
            torso_3 lean +0.05, torso_5 yaw +-0.1, head pan / tilt +-0.15
  sweep     a larger, slow arm move (shoulder 0.5 rad out and back) for the velocity profile
  guards    the new guards on the hardware, none of which can push anything: (1) with the
            right shoulder's rating lowered so that its guard sits 4 N m above its starting
            load, a shoulder lift that raises the load is stopped and the joints are
            re-targeted where they stopped; (2) the watchdog with its guard set
            below the standing load re-targets in place (must move < 2 mrad); (3) a
            0.01 mrad "exact" tolerance makes a normal move count as blocked: the in-place
            re-target after a finished command must not move the arm
  gripper   close and open each gripper (homes both first)
  restore   back to the pose the run started from
  base      (only when named) forward 0.1 m and back, turn +-0.2 rad at the new caps: clear
            the floor around the base first
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
from services.rby1.backend import BODY, Rby1Backend  # noqa: E402
from services.rby1.kinematics import COMPONENTS, JOINT_NAMES, component  # noqa: E402
from services import arming  # noqa: E402

PARTS = ('rest', 'ready', 'joints', 'sweep', 'guards', 'gripper', 'restore', 'base')
DEFAULT = PARTS[:-1]
ALL = ('torso', 'right_arm', 'left_arm', 'head')
BODY_SLICE = slice(2, 22)
results = {}


def pose_now(backend):
    q = backend._snapshot().q
    return {name: np.round(component(q, name), 4).tolist() for name in ALL}


class Sampler:
    """Joint state at 25 Hz while a backend call runs in this thread's caller."""

    def __init__(self, backend):
        self.backend, self.rows, self.stop = backend, [], threading.Event()

    def __enter__(self):
        def loop():
            while not self.stop.is_set():
                s = self.backend._snapshot()
                self.rows.append((time.monotonic(), s.q.copy(), s.velocity.copy(), s.torque.copy(),
                                  s.temperature.copy()))
                time.sleep(0.04)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()

    def summary(self, limit):
        t = np.array([r[0] for r in self.rows])
        v = np.array([r[2] for r in self.rows])
        tau = np.array([r[3] for r in self.rows])
        temp = np.array([r[4] for r in self.rows])
        load = np.abs(tau) / limit
        j = int(np.unravel_index(np.argmax(load), load.shape)[1])
        moving = np.max(np.abs(v[:, BODY_SLICE.start:24]), axis=0)
        k = int(np.argmax(moving)) + BODY_SLICE.start
        acc = np.abs(np.diff(v[:, k]) / np.maximum(np.diff(t), 1e-3)) if len(t) > 2 else np.array([0.0])
        return {'peak_load': f'{JOINT_NAMES[j]} {load[:, j].max():.0%}',
                'fastest_joint': JOINT_NAMES[k], 'peak_velocity_radps': round(float(moving[k - 2]), 3),
                'peak_accel_radps2': round(float(np.percentile(acc, 95)), 2),
                'hottest_c': float(temp.max())}


def call(backend, tool, arguments, label):
    backend.validate(tool, arguments)
    with Sampler(backend) as sampler:
        try:
            value, _ = backend.call(tool, arguments)
            error = None
        except Exception as exc:
            value, error = {}, exc
    summary = sampler.summary(backend.kinematics.torque_limit)
    keep = {k: value[k] for k in ('seconds', 'planned_s', 'max_joint_error_rad', 'settled', 'blocked_rad', 'warning',
                                  'position_error_mm', 'measured_by_odometry', 'measured_width_m') if k in value}
    row = {'label': label, **keep, **summary}
    if error is not None:
        row['stopped'] = f'{type(error).__name__}: {error}'
    results.setdefault('moves', []).append(row)
    print(json.dumps(row), flush=True)
    time.sleep(0.3)
    if error is not None:
        raise error
    return value


def move(backend, name, goal, label):
    return call(backend, 'rby1_move_joints', {'component': name,
                                              'positions': np.round(np.asarray(goal, float), 4).tolist()}, label)


def rest(backend):
    snap = backend._snapshot()
    model = backend.kinematics.gravity(snap.q)
    limit = backend.kinematics.torque_limit
    rows = {}
    for j in range(2, 22):
        rows[JOINT_NAMES[j]] = {'measured': round(float(snap.torque[j]), 1), 'model': round(float(model[j]), 1),
                                'load': f'{abs(snap.torque[j]) / limit[j]:.0%}', 'temp_c': float(snap.temperature[j])}
    results['rest'] = {'manager': backend.robot.manager_state(), 'all_ready': bool(np.all(snap.ready)),
                       'torques': rows}
    print(json.dumps(results['rest']), flush=True)
    big = {n: r for n, r in rows.items() if abs(r['model']) > 5}
    ratio = {n: round(r['measured'] / r['model'], 2) for n, r in big.items()}
    print('measured / model on loaded joints:', json.dumps(ratio), flush=True)
    results['rest']['ratio'] = ratio


def joints(backend):
    start = pose_now(backend)
    for side in ('right', 'left'):
        name = f'{side}_arm'
        base = np.asarray(start[name])
        lower, upper = backend._limits(name)
        for index in range(7):
            goal = base.copy()
            goal[index] += 0.1 if base[index] + 0.1 <= upper[index] else -0.1
            move(backend, name, goal, f'{name} j{index} {goal[index] - base[index]:+.1f}')
            move(backend, name, base, f'{name} j{index} back')
    torso = np.asarray(start['torso'])
    for label, delta in (('torso squat +0.1', [0, 0.1, -0.2, 0.1, 0, 0]), ('torso squat -0.1', [0, -0.1, 0.2, -0.1, 0, 0]),
                         ('torso_3 lean +0.05', [0, 0, 0, 0.05, 0, 0]), ('torso_5 yaw +0.1', [0, 0, 0, 0, 0, 0.1]),
                         ('torso_5 yaw -0.1', [0, 0, 0, 0, 0, -0.1])):
        move(backend, 'torso', torso + np.asarray(delta), label)
        move(backend, 'torso', torso, label + ' back')
    head = np.asarray(start['head'])
    for label, delta in (('head pan +0.15', [0.15, 0]), ('head pan -0.15', [-0.15, 0]), ('head tilt +0.15', [0, 0.15])):
        move(backend, 'head', head + np.asarray(delta), label)
        move(backend, 'head', head, label + ' back')


def sweep(backend):
    base = np.asarray(pose_now(backend)['right_arm'])
    goal = base.copy()
    goal[0] -= 0.5  # shoulder pitch: the arm swings forward and up
    move(backend, 'right_arm', goal, 'right shoulder -0.5 (sweep)')
    move(backend, 'right_arm', base, 'right shoulder back (sweep)')


def guards(backend):
    o = backend.options
    limit = backend.kinematics.torque_limit
    out = {}
    # (1) torque guard trips mid-move on a shoulder lift that raises the shoulder load
    base = np.asarray(pose_now(backend)['right_arm'])
    j = COMPONENTS['right_arm'][0]  # right_arm_0
    tau0 = abs(float(backend._snapshot().torque[j]))
    saved, saved_static, rated = o['torque_guard_fraction'], o['static_torque_fraction'], float(limit[j])
    limit[j] = (tau0 + 4.0) / saved  # the guard now trips 4 N m above the starting shoulder load
    o['static_torque_fraction'] = 10.0  # the lowered rating is only for the live guard
    goal = base.copy()
    goal[0] -= 0.6
    print(f'guard 1: right_arm_0 holds {tau0:.1f} N m; guard at {tau0 + 4:.1f} N m; lifting', flush=True)
    try:
        move(backend, 'right_arm', goal, 'guard 1: shoulder lift against a low torque guard')
        out['torque_guard'] = 'FAIL: the move finished without tripping'
    except Exception as exc:
        moved = float(component(backend._snapshot().q, 'right_arm')[0] - base[0])
        event = backend.events[-1] if backend.events else None
        ok = 'of its rating' in str(exc) and event and event['what'].startswith('re-targeted')
        out['torque_guard'] = {'result': 'PASS' if ok else 'FAIL', 'stopped': str(exc)[:200],
                               'shoulder_moved_rad': round(moved, 3), 'relief': event}
    finally:
        limit[j], o['static_torque_fraction'] = rated, saved_static
    print('guard 1:', json.dumps(out['torque_guard'], default=str), flush=True)
    time.sleep(1.0)
    move(backend, 'right_arm', base, 'guard 1: back')
    # (2) watchdog re-targets in place when a standing load is over its guard
    snap = backend._snapshot()
    loads = np.abs(snap.torque) / limit
    k = int(np.argmax(loads))
    saved_s = o['watchdog_torque_s']
    o['torque_guard_fraction'], o['watchdog_torque_s'] = loads[k] - 0.05, 0.0
    before = len(backend.events)
    try:
        backend.over_since = None
        for _ in range(3):
            backend._watch_once()
            time.sleep(0.1)
    finally:
        o['torque_guard_fraction'], o['watchdog_torque_s'] = saved, saved_s
        backend.over_since = None
    event = backend.events[-1] if len(backend.events) > before else None
    ok = bool(event) and event['moved_rad'] < 0.002
    out['watchdog'] = {'result': 'PASS' if ok else 'FAIL', 'joint': JOINT_NAMES[k], 'load': f'{loads[k]:.0%}',
                       'relief': event}
    print('guard 2:', json.dumps(out['watchdog'], default=str), flush=True)
    time.sleep(1.0)
    # (3) blocked detection after a finished command: the in-place re-target must not move the arm
    saved_e, saved_t = o['joint_exact_rad'], o['exact_timeout_s']
    o['joint_exact_rad'], o['exact_timeout_s'] = 0.00001, 0.3
    goal = base.copy()
    goal[6] += 0.1
    try:
        value = move(backend, 'right_arm', goal, 'guard 3: wrist +0.1 with a 0.01 mrad exact tolerance')
    finally:
        o['joint_exact_rad'], o['exact_timeout_s'] = saved_e, saved_t
    event = backend.events[-1]
    ok = value.get('settled') is False and 'blocked_rad' in value and event['moved_rad'] < 0.002
    out['blocked'] = {'result': 'PASS' if ok else 'FAIL', 'blocked_rad': value.get('blocked_rad'), 'relief': event}
    print('guard 3:', json.dumps(out['blocked'], default=str), flush=True)
    move(backend, 'right_arm', base, 'guard 3: back')
    results['guards'] = out


def base_part(backend):
    for label, arguments in (('base forward 0.1', {'forward': 0.1}), ('base back 0.1', {'forward': -0.1}),
                             ('base turn +0.2', {'turn': 0.2}), ('base turn -0.2', {'turn': -0.2})):
        call(backend, 'rby1_base_move', arguments, label)
        time.sleep(0.7)


def gripper(backend):
    for side in ('right', 'left'):
        for action in ('close', 'open'):
            call(backend, 'rby1_gripper', {'side': side, 'action': action}, f'gripper {side} {action}')


def main():
    root, parts = sys.argv[1], sys.argv[2:] or list(DEFAULT)
    unknown = set(parts) - set(PARTS)
    if unknown:
        sys.exit(f'unknown parts {sorted(unknown)}; choose from {PARTS}')
    config = Configuration.load(f'{root}/operator-input.json')
    backend = Rby1Backend(replace(config, backend_options={'stream_enabled': False, 'camera': False,
                                                           'grippers': 'gripper' in parts}))
    out = Path(root).parent.parent/'characterization'/f'health_{time.strftime("%Y%m%dT%H%M%S")}.json'
    try:
        start = pose_now(backend)
        results['start'] = start
        print('start', json.dumps(start), flush=True)
        if 'rest' in parts:
            rest(backend)
        moving = [p for p in parts if p != 'rest']
        if moving:
            print(f'waiting for the operator to arm {config.root.name} (rby1 arm 30) ...', flush=True)
            deadline = time.time() + 600
            while not arming.read(config.root)['armed']:
                if time.time() > deadline:
                    print('not armed within 10 minutes; nothing moved', flush=True)
                    return
                time.sleep(0.5)
            print('armed by the operator; parts:', moving, flush=True)
        for part in moving:
            print(f'--- {part}', flush=True)
            if part == 'ready':
                targets = {c: {'preset': 'ready'} for c in BODY} | {'head': {'preset': 'zero'}}
                call(backend, 'rby1_move_body', {'targets': targets}, 'body to ready')
            elif part == 'restore':
                targets = {c: {'positions': start[c]} for c in ALL}
                call(backend, 'rby1_move_body', {'targets': targets}, 'body back to the start pose')
            elif part == 'base':
                base_part(backend)
            else:
                globals()[part](backend)
        print('DONE', flush=True)
    except Exception as exc:
        results['stopped'] = f'{type(exc).__name__}: {exc}'
        print(f'STOPPED: {type(exc).__name__}: {exc}', flush=True)
    finally:
        results['safety'] = backend._safety_report(backend._snapshot())
        results['events'] = backend.events
        backend.close()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1, default=str))
        print('results:', out, flush=True)


if __name__ == '__main__':
    main()
