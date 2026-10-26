"""Operator-supervised characterization of the controller's own command modes (controller_modes.py).

    $PY scripts/rby1_controller_check.py <session_root> [part ...]

The session must be configured with backend option controller_commands = true. One arming
covers the run (`rby1 arm`, a person at the e-stop). Every move goes through the real backend
and its guards, starts from `ready` and is small (a few centimetres in free space in front of
the robot); each part ends back at `ready`. Per move it records the duration, the final
position error, the deviation of the hand path from the straight line, how much each part
moved (torso use), torso roll (torso_0 / torso_4), the peak joint load and any guard stop.
Results: runtime/characterization/controller_check_*.json.

Parts (default: all, in this order):
  ready      rest -> half -> ready (joint space, as before)
  optimal    rby1_optimal_control, body: right hand +5 cm forward, back; left hand +5 cm up,
             back; both hands +4 cm forward together, back; right hand 10 cm down (the torso
             may join), back
  cartesian  rby1_cartesian: right_arm +5 cm forward and back, +0.3 rad about the hand's own
             axis and back; body: left hand +5 cm sideways and back
  eef        rby1_move_eef in controller mode: right hand +3 cm up and back
  impedance  rby1_cartesian_impedance right +3 cm and back; rby1_link_impedance left hold
             +2 cm for 3 s; rby1_joint_impedance right wrist +0.1 rad and back
  jog        rby1_jog right_arm_6 +0.1 / -0.1, head_0 +0.1 / -0.1; rby1_joint_group torso_5
             +0.1 and back
  stop       rby1_hold mode stop_command while idle (the StopCommand path)
"""
from dataclasses import replace
import json
import sys
import time

import numpy as np

from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT/'scripts'))
from rby1_health_check import Sampler  # noqa: E402
from services.controller.config import Configuration  # noqa: E402
from services.rby1.backend import BODY, Rby1Backend  # noqa: E402
from services.rby1.kinematics import EE_FRAME, JOINT_NAMES, component  # noqa: E402
from services import arming  # noqa: E402

PARTS = ('ready', 'optimal', 'cartesian', 'eef', 'impedance', 'jog', 'stop')
results = {'moves': []}


def hands(backend, q):
    frames = backend.kinematics.frames(list(EE_FRAME.values()), q)
    return {side: frames[f][:3, 3].copy() for side, f in EE_FRAME.items()}


def run(backend, tool, arguments, label):
    """One tool call with 25 Hz sampling; returns the reply, or None after a handled stop (recorded)."""
    start = backend._snapshot().q.copy()
    h0 = hands(backend, start)
    row = {'label': label, 'tool': tool}
    t0 = time.monotonic()
    with Sampler(backend) as sampler:
        try:
            value, _ = backend.call(tool, arguments)
        except Exception as exc:
            value, row['stopped'] = None, f'{type(exc).__name__}: {str(exc)[:300]}'
    row['seconds'] = round(time.monotonic() - t0, 2)
    end = backend._snapshot().q.copy()
    h1 = hands(backend, end)
    qs = np.array([r[1] for r in sampler.rows]) if sampler.rows else np.array([end])
    tau = np.array([r[3] for r in sampler.rows]) if sampler.rows else None
    for side in ('right', 'left'):
        path = np.array([hands(backend, q)[side] for q in qs[::2]])
        a, b = h0[side], h1[side]
        d = b - a
        if np.linalg.norm(d) > 0.005:
            u = d / np.linalg.norm(d)
            off = path - a - np.outer((path - a) @ u, u)
            row[f'{side}_path_deviation_mm'] = round(float(np.max(np.linalg.norm(off, axis=1))) * 1000, 1)
        row[f'{side}_moved_m'] = np.round(d, 4).tolist()
    row['torso_change_rad'] = round(float(np.max(np.abs(component(end, 'torso') - component(start, 'torso')))), 4)
    row['torso_roll_max_rad'] = round(float(np.max(np.abs(qs[:, [2, 6]]))), 4)
    for c in ('right_arm', 'left_arm', 'head'):
        row[f'{c}_change_rad'] = round(float(np.max(np.abs(component(end, c) - component(start, c)))), 4)
    if tau is not None and len(tau):
        load = np.abs(tau) / backend.kinematics.torque_limit
        j = int(np.unravel_index(np.argmax(load), load.shape)[1])
        row['peak_load'] = f'{JOINT_NAMES[j]} {load[:, j].max():.0%}'
    if value:
        for key in ('finish', 'targets', 'joint_error_rad', 'position_error_mm', 'mode'):
            if key in value:
                row[key] = value[key]
    results['moves'].append(row)
    print(json.dumps(row, default=str), flush=True)
    time.sleep(0.5)
    return value


def to_ready(backend, label='back to ready'):
    t = {c: {'preset': 'ready'} for c in BODY} | {'head': {'preset': 'zero'}}
    run(backend, 'rby1_move_body', {'targets': t}, label)


def rel(link, d, **extra):
    return {'link': link, 'position': list(d), 'relative': True, **extra}


def part_ready(backend):
    for preset in ('half', 'ready'):
        t = {c: {'preset': preset} for c in BODY} | {'head': {'preset': 'zero'}}
        run(backend, 'rby1_move_body', {'targets': t}, f'body to {preset}')


def part_optimal(backend):
    for label, targets in (('OC right +5 cm forward', [rel('ee_right', [0.05, 0, 0])]),
                           ('OC left +5 cm up', [rel('ee_left', [0, 0, 0.05])]),
                           ('OC both +4 cm forward', [rel('ee_right', [0.04, 0, 0]), rel('ee_left', [0.04, 0, 0])]),
                           ('OC right 10 cm down (torso may join)', [rel('ee_right', [0, 0, -0.10])])):
        run(backend, 'rby1_optimal_control', {'component': 'body', 'targets': targets}, label)
        to_ready(backend, label + ': back (joint space)')


def part_cartesian(backend):
    from scipy.spatial.transform import Rotation
    q = backend._snapshot().q
    T = backend.kinematics.frame('ee_right', q)
    turned = (Rotation.from_matrix(T[:3, :3]) * Rotation.from_rotvec([0, 0, 0.3])).as_quat().tolist()
    run(backend, 'rby1_cartesian', {'component': 'right_arm', 'targets': [rel('ee_right', [0.05, 0, 0])]},
        'cartesian right_arm +5 cm forward')
    run(backend, 'rby1_cartesian', {'component': 'right_arm', 'targets': [rel('ee_right', [-0.05, 0, 0])]},
        'cartesian right_arm back 5 cm')
    run(backend, 'rby1_cartesian', {'component': 'right_arm', 'targets': [
        {'link': 'ee_right', 'position': [0, 0, 0], 'relative': True, 'quaternion': turned}]},
        'cartesian right_arm turn 0.3 rad about the hand z')
    to_ready(backend, 'cartesian: back (joint space)')
    run(backend, 'rby1_cartesian', {'component': 'body', 'targets': [rel('ee_left', [0, 0.05, 0])]},
        'cartesian body left +5 cm sideways')
    run(backend, 'rby1_cartesian', {'component': 'body', 'targets': [rel('ee_left', [0, -0.05, 0])]},
        'cartesian body left back 5 cm')
    to_ready(backend, 'cartesian body: back (joint space)')


def part_eef(backend):
    run(backend, 'rby1_move_eef', {'arm': 'right', 'position': [0, 0, 0.03], 'relative': True}, 'move_eef (cartesian) +3 cm up')
    run(backend, 'rby1_move_eef', {'arm': 'right', 'position': [0, 0, -0.03], 'relative': True}, 'move_eef (cartesian) back')
    to_ready(backend, 'eef: back (joint space)')


def part_impedance(backend):
    run(backend, 'rby1_cartesian_impedance', {'arm': 'right', 'position': [0.03, 0, 0], 'relative': True},
        'cartesian impedance right +3 cm forward')
    run(backend, 'rby1_cartesian_impedance', {'arm': 'right', 'position': [-0.03, 0, 0], 'relative': True},
        'cartesian impedance right back')
    run(backend, 'rby1_link_impedance', {'arm': 'left', 'position': [0, 0, 0.02], 'relative': True, 'seconds': 3},
        'link impedance left +2 cm up, 3 s')
    q = component(backend._snapshot().q, 'right_arm')
    run(backend, 'rby1_joint_impedance', {'arm': 'right', 'positions': (q + [0, 0, 0, 0, 0, 0, 0.1]).tolist()},
        'joint impedance right wrist +0.1')
    run(backend, 'rby1_joint_impedance', {'arm': 'right', 'positions': q.tolist()}, 'joint impedance right wrist back')
    to_ready(backend, 'impedance: back (joint space)')


def part_jog(backend):
    for joint, value in (('right_arm_6', 0.1), ('right_arm_6', -0.1), ('head_0', 0.1), ('head_0', -0.1)):
        run(backend, 'rby1_jog', {'joint': joint, 'mode': 'relative', 'value': value}, f'jog {joint} {value:+.1f}')
    t5 = float(component(backend._snapshot().q, 'torso')[5])
    run(backend, 'rby1_joint_group', {'joints': ['torso_5'], 'positions': [t5 + 0.1]}, 'joint group torso_5 +0.1')
    run(backend, 'rby1_joint_group', {'joints': ['torso_5'], 'positions': [t5]}, 'joint group torso_5 back')
    to_ready(backend, 'jog: back (joint space)')


def part_stop(backend):
    run(backend, 'rby1_hold', {'mode': 'stop_command'}, 'hold with stop_command (idle)')
    to_ready(backend, 'after stop_command: move again (joint space)')


def main():
    args = [a for a in sys.argv[1:]]
    root, parts = args[0], args[1:] or list(PARTS)
    config = Configuration.load(f'{root}/operator-input.json')
    if not config.backend_options.get('controller_commands'):
        sys.exit('configure the session with backend option controller_commands = true')
    print(f'waiting for the operator to arm {config.root.name} ...', flush=True)
    deadline = time.time() + 900
    while not arming.read(config.root)['armed']:
        if time.time() > deadline:
            sys.exit('not armed within 15 minutes; nothing moved')
        time.sleep(0.5)
    backend = Rby1Backend(replace(config, backend_options={**config.backend_options, 'stream_enabled': False,
                                                           'camera': False, 'grippers': False}))
    out = Path(root).parent.parent/'characterization'/f'controller_check_{time.strftime("%Y%m%dT%H%M%S")}.json'
    try:
        for part in parts:
            print(f'--- {part}', flush=True)
            globals()[f'part_{part}'](backend)
            stopped = [m for m in results['moves'] if 'stopped' in m]
            if stopped:
                raise SystemExit(f"a move was stopped ({stopped[0]['label']}: {stopped[0]['stopped'][:160]}); "
                                 'ending the run so it can be inspected')
        print('DONE', flush=True)
    except BaseException as exc:
        results['stopped'] = f'{type(exc).__name__}: {exc}'
        print(f'STOPPED: {type(exc).__name__}: {exc}', flush=True)
    finally:
        results['events'] = backend.events
        backend.close()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1, default=str))
        print('results:', out, flush=True)


if __name__ == '__main__':
    main()
