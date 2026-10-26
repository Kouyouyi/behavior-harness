"""Commissioning references: terminal native idle and one bounded elbow round trip.

Uses the existing CoMotion adapter for SONIC's original 1762-dimensional encoder.
No IK, VLA, root translation, hand motion or low-level motor command is generated.
"""
import importlib.util
from pathlib import Path

import numpy as np
# Import before the operator's real-time loop starts; the reused adapter needs it.
from scipy.spatial.transform import Rotation  # noqa: F401

from .state import NAMES, JOINTS, checked_state

ADAPTER = Path('/home/user/code/CoMotion/experiments/shared_control/shared_control/reference.py')
RATE = 50
LOOKAHEAD = 55
MAX_DELTA = .05
DURATION = 7.


def from_native_idle(row):
    measured = checked_state(row)
    if row['motion'] != 'planner_motion' or not row['encoder_source']:
        raise ValueError('Expected measured native planner idle')
    if measured['tilt_rad'] > .25 or max(abs(v) for v in measured['dq']) > 1:
        raise ValueError('Native standing not settled')
    spec = importlib.util.spec_from_file_location('rax_comotion_reference', ADAPTER)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    ref = adapter.from_idle_observation(row['encoder_obs'], {'root_qpos': [0, 0, 0, *row['base_quat']]})
    return {key: ref[key].copy() for key in ('joint_pos', 'joint_vel', 'body_quat')}


def check_pulse(side, delta_rad):
    if side not in ('left', 'right'):
        raise ValueError('side must be left or right')
    if type(delta_rad) not in (int, float) or not np.isfinite(delta_rad) or not 0 < abs(delta_rad) <= MAX_DELTA:
        raise ValueError(f'delta_rad must be finite, nonzero, and at most {MAX_DELTA} rad')


def pulse_envelope(t):
    """One second entry, two second rise, one hold, two return, one settle."""
    t = np.asarray(t, dtype=float)
    u = np.where(t < 3, (t-1)/2, (6-t)/2)
    u = np.clip(u, 0, 1)
    s = u**3*(10+u*(-15+6*u))
    derivative = 30*u*u*(1-u)**2/2
    derivative *= np.where(t < 3, 1, -1)
    return s, derivative


def pulse_window(reference, side, delta_rad, elapsed, frame):
    check_pulse(side, delta_rad)
    if not np.isfinite(elapsed) or elapsed < 0 or type(frame) is not int or frame < 0:
        raise ValueError('Invalid trajectory clock/frame')
    index = NAMES.index(side+'_elbow_joint')
    q = np.repeat(reference['joint_pos'][:1], LOOKAHEAD, axis=0)
    dq = np.repeat(reference['joint_vel'][:1], LOOKAHEAD, axis=0)
    amplitude, velocity = pulse_envelope(elapsed+np.arange(LOOKAHEAD)/RATE)
    q[:, index] += delta_rad*amplitude
    dq[:, index] += delta_rad*velocity
    joint = JOINTS[index]
    if np.any(q[:, index] < joint['lower']+.1) or np.any(q[:, index] > joint['upper']-.1):
        raise ValueError('Elbow reference too close to model limit')
    return dict(joint_pos=q.astype(np.float32), joint_vel=dq.astype(np.float32),
                body_quat=np.repeat(reference['body_quat'][:1], LOOKAHEAD, axis=0).astype(np.float32),
                frame_index=np.arange(frame, frame+LOOKAHEAD, dtype=np.int64))


def hold_window(reference, frame):
    return {key: np.repeat(reference[key][:1], LOOKAHEAD, axis=0).astype(np.float32)
            for key in ('joint_pos', 'joint_vel', 'body_quat')} | {
                'frame_index': np.arange(frame, frame+LOOKAHEAD, dtype=np.int64)}
