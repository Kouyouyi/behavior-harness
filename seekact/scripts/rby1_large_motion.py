"""Operator-supervised large-motion test of the RB-Y1: squats, arm stretches, long hand paths.

    $PY scripts/rby1_large_motion.py <session_root> [part ...]

Everything goes through the agent's tools and guards (limits, squat cap, self-collision,
gravity load, torque guard, in-place re-target) at the backend's slow defaults, from the
`ready` posture and back to it, one arm at a time. Moves larger than 1.0 rad on a joint are
split into chained calls. A refusal by the backend is recorded and the part goes on with
the next posture (refusals are results too). Joint state is sampled at 25 Hz as in
rby1_health_check.py; results go to runtime/characterization/large_motion_*.json.

Parts (default: all, in this order):
  clearance  head camera at pan -0.47 / 0 / +0.47: depth points (not the robot itself, not
             the floor) inside 1.0 m of the base, per zone; refuses to go on if anything is
             inside the arms' reach (0.95 m) unless --ignore-clearance; frames are saved
  squat      torso ready -> 45 -> 60 -> 69 deg (the cap) -> back to ready with the arms in ready,
             then a forward lean at ready (torso_3 +0.15)
  lean       at each torso height (zero, ready, mid, low, lowest): lean +0.10 and +0.15, and at
             +0.15 each hand 0.08 m forward and back (rby1_move_eef); then arms `park` at
             `mid` with lean +0.2 and +0.3
  stretch    each arm: forward horizontal, forward-up 45 deg, overhead, back; forward-out
             45 deg, sideways 60 deg, back; forearm roll +-1.0, wrist roll +-1.2; then both
             arms forward together (these are the arm presets forward, forward_up, overhead,
             forward_out and side)
  reach      right hand through a 0.2 m square in front of the shoulder and a 0.25 m reach
             forward with rby1_move_eef, path deviation from the straight line measured
  low        squat 60 deg + right hand forward-down to table height, a 0.1 m descent there
             with rby1_move_eef, and back
  restore    back to the pose the run started from
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
from rby1_health_check import Sampler, pose_now  # noqa: E402
from services.controller.config import Configuration  # noqa: E402
from services.rby1.backend import BODY, PRESETS, Rby1Backend  # noqa: E402
from services.rby1.kinematics import EE_FRAME, HEAD_CAMERA, component, pose  # noqa: E402
from services import arming  # noqa: E402

PARTS = ('clearance', 'squat', 'lean', 'stretch', 'reach', 'low', 'restore')
REACH_M = 0.95          # arm tips stay within this of the base origin in every posture below
MAX_CHUNK_RAD = 1.0
MIRROR = np.array([1, -1, -1, 1, -1, 1, -1])  # right-arm angles -> the left arm's mirror image
READY_ARM = np.array(PRESETS['right_arm']['ready'])
READY_TORSO = np.array(PRESETS['torso']['ready'])


def torso(a, lean=0.0, yaw=0.0):
    return np.array([0, a, -2 * a, a + lean, 0, yaw])


STRETCH = [  # (label, right-arm posture); the left arm uses the mirror image
    ('forward horizontal', [-1.57, -0.3, 0, -0.25, 0, 0.25, 0]),
    ('forward-up 45 deg', [-2.36, -0.3, 0, -0.25, 0, 0.25, 0]),
    ('overhead', [-2.9, -0.3, 0, -0.35, 0, 0.3, 0]),
    ('forward-up 45 deg (down)', [-2.36, -0.3, 0, -0.25, 0, 0.25, 0]),
    ('forward horizontal (down)', [-1.57, -0.3, 0, -0.25, 0, 0.25, 0]),
    ('ready', READY_ARM.tolist()),
    ('forward-out 45 deg', [-1.2, -0.8, 0, -0.4, 0, 0.3, 0]),
    ('sideways 60 deg', [0, -1.05, 0, -0.25, 0, 0, 0]),
    # sideways 90 deg ([0, -1.57, 0, -0.25, 0, 0, 0]) was dropped: on 2026-10-08 the left shoulder roll hit
    # the torque guard (81 % of 70 N m) on the way up; the right peaked at 73 %
    ('forward-out 45 deg (back)', [-1.2, -0.8, 0, -0.4, 0, 0.3, 0]),
    ('ready', READY_ARM.tolist()),
    ('forearm roll +1.0', (READY_ARM + [0, 0, 0, 0, 1.0, 0, 0]).tolist()),
    ('forearm roll -1.0', (READY_ARM + [0, 0, 0, 0, -1.0, 0, 0]).tolist()),
    ('wrist roll +1.2', (READY_ARM + [0, 0, 0, 0, 0, 0, 1.2]).tolist()),
    ('wrist roll -1.2', (READY_ARM + [0, 0, 0, 0, 0, 0, -1.2]).tolist()),
    ('ready', READY_ARM.tolist()),
]
results = {'moves': [], 'refused': [], 'postures': {}}


def record(backend, label, row):
    snap = backend._snapshot()
    frames = backend.kinematics.frames(list(EE_FRAME.values()), snap.q)
    hands = {side: np.round(frames[f][:3, 3], 3).tolist() for side, f in EE_FRAME.items()}
    load = np.abs(snap.torque) / backend.kinematics.torque_limit
    top = np.argsort(-load)[:2]
    from services.rby1.kinematics import JOINT_NAMES
    row.update(hands=hands, holding_load={JOINT_NAMES[j]: f'{load[j]:.0%}' for j in top})
    results['postures'][label] = {'hands': hands, 'holding_load': row['holding_load'],
                                  **{c: np.round(component(snap.q, c), 3).tolist() for c in BODY}}


def run(backend, tool, arguments, label):
    """One call with sampling; a backend refusal (ValueError) is recorded and returned as None."""
    try:
        backend.validate(tool, arguments)
    except ValueError as exc:
        results['refused'].append({'label': label, 'reason': str(exc)[:300]})
        print(json.dumps({'label': label, 'refused': str(exc)[:300]}), flush=True)
        return None
    with Sampler(backend) as sampler:
        value, _ = backend.call(tool, arguments)
    summary = sampler.summary(backend.kinematics.torque_limit)
    row = {'label': label, **{k: value[k] for k in ('seconds', 'planned_s', 'max_joint_error_rad', 'settled',
                                                     'blocked_rad', 'warning', 'position_error_mm') if k in value},
           **summary}
    if tool == 'rby1_move_eef':
        row['path_deviation_mm'] = path_deviation(backend, sampler, arguments['arm'], value)
    record(backend, label, row)
    results['moves'].append(row)
    print(json.dumps(row), flush=True)
    time.sleep(0.3)
    return value


def path_deviation(backend, sampler, arm, value):
    points = np.array([backend.kinematics.frame(EE_FRAME[arm], q)[:3, 3] for _, q, *_ in sampler.rows[::2]])
    a, b = points[0], np.asarray(value['target_position'])
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    off = points - a - np.outer((points - a) @ d, d)
    return round(float(np.max(np.linalg.norm(off, axis=1))) * 1000, 1)


def body(backend, targets, label):
    """rby1_move_body to {component: positions}, split so no joint moves more than MAX_CHUNK_RAD per call."""
    q = backend._snapshot().q
    start = {c: component(q, c) for c in targets}
    step = max(float(np.max(np.abs(np.asarray(g) - start[c]))) for c, g in targets.items())
    n = max(1, int(np.ceil(step / MAX_CHUNK_RAD)))
    value = None
    for i in range(1, n + 1):
        chunk = {c: {'positions': np.round(start[c] + i / n * (np.asarray(g) - start[c]), 4).tolist()}
                 for c, g in targets.items()}
        value = run(backend, 'rby1_move_body', {'targets': chunk}, label if n == 1 else f'{label} ({i}/{n})')
        if value is None:
            return None
    return value


def clearance(backend, ignore):
    """Depth points inside REACH_M + 0.05 of the base, excluding the robot (FK link origins) and the floor."""
    kin = backend.kinematics
    report = {}
    for pan in (-0.47, 0.0, 0.47):
        run(backend, 'rby1_move_joints', {'component': 'head', 'positions': [pan, 0.35]}, f'look pan {pan:+.2f}')
        time.sleep(0.5)
        value, _ = backend.call('rby1_observe', {})
        directory = backend.workspace/value['directory']
        meta = json.loads((directory/'meta.json').read_text())['cameras']['head_camera']
        import cv2 as cv
        depth = cv.imread(str(directory/meta['depth']), cv.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
        K, T = np.asarray(meta['K']), np.asarray(meta['T_base_cam'])
        v, u = np.mgrid[0:depth.shape[0]:4, 0:depth.shape[1]:4]
        z = depth[v, u]
        ok = z > 0.2
        cam = np.stack([(u[ok] - K[0, 2]) * z[ok] / K[0, 0], (v[ok] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], axis=1)
        pts = cam @ T[:3, :3].T + T[:3, 3]
        q = backend._snapshot().q
        links = kin.frames([n for n in kin.links if n != 'base'], q)
        origins = np.array([t[:3, 3] for t in links.values()])
        near_self = np.min(np.linalg.norm(pts[:, None, :] - origins[None, ::2, :], axis=2), axis=1) < 0.18
        keep = (~near_self) & (pts[:, 2] > 0.12) & (pts[:, 2] < 2.0)
        pts = pts[keep]
        r = np.linalg.norm(pts[:, :2], axis=1)
        inside = pts[r < REACH_M + 0.05]
        report[f'pan {pan:+.2f}'] = {'frame': value['directory'], 'points_within_1m': int(len(inside)),
                                     'nearest_m': round(float(r.min()), 2) if len(r) else None,
                                     'nearest_point': np.round(pts[np.argmin(r)], 2).tolist() if len(r) else None}
        print('clearance', pan, json.dumps(report[f'pan {pan:+.2f}']), flush=True)
    run(backend, 'rby1_move_joints', {'component': 'head', 'preset': 'zero'}, 'head zero')
    results['clearance'] = report
    blocked = [k for k, v in report.items() if v['points_within_1m'] > 30]
    if blocked and not ignore:
        raise SystemExit(f'something is within {REACH_M + 0.05} m of the base in {blocked}: not stretching the arms '
                         '(check the frames; rerun with --ignore-clearance if it is the robot itself)')


def squat(backend):
    arms = {'right_arm': READY_ARM, 'left_arm': READY_ARM * MIRROR}
    body(backend, {'torso': READY_TORSO, **arms, 'head': [0.0, 0.0]}, 'ready')
    for label, a in (('squat 45 deg (mid)', 0.7854), ('squat 60 deg (low)', 1.047), ('squat 69 deg (lowest, cap)', 1.2),
                     ('squat 60 deg (up)', 1.047), ('squat 45 deg (up)', 0.7854), ('ready (up)', 0.5236)):
        body(backend, {'torso': torso(a)}, label)
    body(backend, {'torso': torso(0.5236, lean=0.15)}, 'lean +0.15 at ready')
    body(backend, {'torso': torso(0.5236, lean=0.3)}, 'lean +0.3 at ready')
    body(backend, {'torso': READY_TORSO}, 'ready (unlean)')
    body(backend, {'torso': torso(0.5236, yaw=0.4)}, 'chest yaw +0.4')
    body(backend, {'torso': torso(0.5236, yaw=-0.4)}, 'chest yaw -0.4')
    body(backend, {'torso': READY_TORSO}, 'ready (yaw back)')


def lean(backend):
    body(backend, {'torso': READY_TORSO, 'right_arm': READY_ARM, 'left_arm': READY_ARM * MIRROR, 'head': [0.0, 0.0]},
         'ready')
    for name, a in (('zero', 0.0), ('ready', 0.5236), ('mid', 0.7854), ('low', 1.047), ('lowest', 1.2)):
        body(backend, {'torso': torso(a)}, name)
        body(backend, {'torso': torso(a, lean=0.1)}, f'{name} lean +0.10')
        if body(backend, {'torso': torso(a, lean=0.15)}, f'{name} lean +0.15') is not None:
            for side in ('right', 'left'):
                if run(backend, 'rby1_move_eef', {'arm': side, 'position': [0.08, 0.0, 0.0], 'relative': True},
                       f'{name} lean +0.15 {side} reach +0.08') is not None:  # only come back from a reach made
                    run(backend, 'rby1_move_eef', {'arm': side, 'position': [-0.08, 0.0, 0.0], 'relative': True},
                        f'{name} lean +0.15 {side} back')
        body(backend, {'torso': torso(a)}, f'{name} unlean')
    body(backend, {'torso': torso(0.7854)}, 'mid')
    body(backend, {'right_arm': PRESETS['right_arm']['park'], 'left_arm': PRESETS['left_arm']['park']}, 'arms park')
    for amount in (0.2, 0.3):
        body(backend, {'torso': torso(0.7854, lean=amount)}, f'mid, arms park, lean +{amount}')
    body(backend, {'torso': torso(0.7854)}, 'mid unlean')
    body(backend, {'right_arm': READY_ARM, 'left_arm': READY_ARM * MIRROR}, 'arms ready')


def stretch(backend):
    for side, sign in (('right', np.ones(7)), ('left', MIRROR)):
        for label, posture in STRETCH:
            body(backend, {f'{side}_arm': np.asarray(posture) * sign}, f'{side} {label}')
    both = np.array([-1.57, -0.35, 0, -0.25, 0, 0.25, 0])
    body(backend, {'right_arm': both, 'left_arm': both * MIRROR}, 'both forward horizontal')
    body(backend, {'right_arm': READY_ARM, 'left_arm': READY_ARM * MIRROR}, 'both ready')


def reach(backend):
    _, quat = pose(backend.kinematics.frame(EE_FRAME['right'], backend._snapshot().q))
    path = [('to square corner', [0.5, -0.3, 1.2]), ('square left 0.2', [0.5, -0.1, 1.2]),
            ('square up 0.2', [0.5, -0.1, 1.4]), ('square right 0.2', [0.5, -0.3, 1.4]),
            ('square down 0.2', [0.5, -0.3, 1.2]), ('forward 0.15', [0.65, -0.3, 1.2]),
            ('back 0.15', [0.5, -0.3, 1.2])]
    for label, p in path:
        run(backend, 'rby1_move_eef', {'arm': 'right', 'position': p, 'quaternion': np.round(quat, 5).tolist()},
            f'eef {label}')
    body(backend, {'right_arm': READY_ARM}, 'right ready')


def low(backend):
    body(backend, {'torso': torso(1.047)}, 'squat 60 deg')
    body(backend, {'right_arm': [-0.9, -0.15, 0, -0.6, 0, 0.3, 0]}, 'right forward-down (table height)')
    p, quat = pose(backend.kinematics.frame(EE_FRAME['right'], backend._snapshot().q))
    for label, dz in (('eef down 0.1', -0.1), ('eef up 0.1', 0.0)):
        run(backend, 'rby1_move_eef', {'arm': 'right', 'position': np.round(p + [0, 0, dz], 4).tolist(),
                                       'quaternion': np.round(quat, 5).tolist()}, label)
    body(backend, {'right_arm': READY_ARM}, 'right ready')
    body(backend, {'torso': READY_TORSO}, 'torso ready')


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    root, parts = args[0], args[1:] or list(PARTS)
    ignore = '--ignore-clearance' in sys.argv
    config = Configuration.load(f'{root}/operator-input.json')
    backend = Rby1Backend(replace(config, backend_options={'stream_enabled': False, 'grippers': False}))
    out = Path(root).parent.parent/'characterization'/f'large_motion_{time.strftime("%Y%m%dT%H%M%S")}.json'
    try:
        results['start'] = start = pose_now(backend)
        if not arming.read(config.root)['armed']:
            print('not armed: nothing moves', flush=True)
            return
        for part in parts:
            print(f'--- {part}', flush=True)
            if part == 'clearance':
                clearance(backend, ignore)
            elif part == 'restore':
                body(backend, {c: start[c] for c in (*BODY, 'head')}, 'back to the start pose')
            else:
                globals()[part](backend)
        print('DONE', flush=True)
    except BaseException as exc:
        results['stopped'] = f'{type(exc).__name__}: {exc}'
        print(f'STOPPED: {type(exc).__name__}: {exc}', flush=True)
    finally:
        results['safety'] = backend._safety_report(backend._snapshot())
        results['events'] = backend.events
        backend.close()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1, default=str))
        print('start pose was', json.dumps(start), flush=True)
        print('results:', out, flush=True)


if __name__ == '__main__':
    main()
