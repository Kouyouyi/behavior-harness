"""Current-observation measurements in the frames actually supported by this G1.

Matches the official observe/localize/act/check contract; never manufactures
pelvis or level coordinates from uncalibrated camera extrinsics.
"""
import math
import numpy as np


def wrap_degrees(angle):
    return (angle+180.) % 360.-180.


def heading(observation):
    state = observation.get('robot_state') or {}
    q = np.asarray(state.get('base_quat_wxyz', []), dtype=float)
    if (q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6
            or state.get('measured') is not True
            or not 0 <= state.get('lowstate_age_s', float('inf')) <= .5):
        raise ValueError('Fresh measured body orientation required')
    w, x, y, z = q/np.linalg.norm(q)
    return math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def measure(rgb, depth, observation):
    """Measurement quality only; no reference image or historical depth comparison."""
    try:
        yaw = heading(observation)
    except (ValueError, TypeError) as exc:
        return dict(ok=False, measured={}, why=str(exc))
    if (rgb is None or depth is None or rgb.ndim != 3 or depth.ndim != 2
            or rgb.shape[:2] != depth.shape):
        return dict(ok=False, measured={}, why='current_rgbd_unavailable')
    valid = np.isfinite(depth) & (depth >= 250) & (depth <= 5000)
    fraction = float(valid.mean())
    result = dict(frame_id=observation['frame_id'], heading_deg=yaw,
                  depth_valid_fraction=fraction, coordinate_frame='camera_optical',
                  camera_to_robot_extrinsics_validated=False,
                  position_feedback_available=False, collision_clearance_verified=False)
    if fraction < .10:
        return dict(ok=False, measured=result, why='insufficient_current_depth')
    result['depth_quantiles_m'] = (np.percentile(depth[valid], [10, 50, 90])/1000).tolist()
    return dict(ok=True, measured=result, why='current_observation_available')


def turn_correction(error_deg, step):
    """Bounded correction using the taught turn's measured response as a local model.

    At most half its duration, at most one second; the caller re-measures after
    each correction and caps attempts. This is not a calibrated position servo.
    """
    command = step['command']
    response = abs(step['measured_yaw_delta_deg'])
    if not math.isfinite(error_deg) or response < 2:
        raise ValueError('No usable measured turn response')
    duration = min(1., command['duration_s']*.5,
                   max(.1, command['duration_s']*abs(error_deg)/response))
    return dict(direction='rotate_left' if error_deg > 0 else 'rotate_right',
                duration_s=duration, yaw_rate_deg_s=command['yaw_rate_deg_s'])
