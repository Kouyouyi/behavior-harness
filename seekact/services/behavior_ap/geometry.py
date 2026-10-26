"""Pure NumPy projection of synchronized OmniGibson RGB-D observations.

OmniGibson ``VisionSensor.depth_linear`` is ``distance_to_image_plane``:
depth is optical axial Z, not Euclidean range along the viewing ray. Its
``intrinsic_matrix`` uses pixel (u right, v down) coordinates. The official
``eval/utils/obs_utils.py:depth_to_pcd`` appends a 180-degree X rotation to
the camera pose, converting optical [x, y, z] to USD [x, -y, -z]. Public
OmniGibson poses use quaternion [x, y, z, w], as in transform_utils.quat2mat.

This module never accesses simulator state, changes a pose, or verifies frame
freshness. The worker must bind depth, K, and both world poses to one frame ID.
"""
from __future__ import annotations

import numpy as np


def _numeric_array(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a numeric array of shape {shape}")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _rotation_xyzw(value, name):
    quaternion = _numeric_array(value, (4,), name)
    scale = np.max(np.abs(quaternion))
    if scale == 0:
        raise ValueError(f"{name} must be a nonzero quaternion")
    # Normalize without overflowing for a finite, scaled quaternion.
    quaternion = quaternion / scale
    quaternion = quaternion / np.linalg.norm(quaternion)
    x, y, z, w = quaternion
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def project_pixel(depth_linear, K, pixel_xy, *, camera_position=None,
                  camera_quat=None, robot_base_position=None, robot_base_quat=None,
                  output_frame="base", radius=2, min_depth=0.0, max_depth=None):
    """Project one RGB pixel using median finite positive axial depth, in meters.

    ``pixel_xy`` is (u, v), not array (row, column). Fractional pixel centers
    are allowed inside [0, width-1] / [0, height-1]. The depth patch is centered
    on the nearest integer pixel, clipped at image boundaries, with a square
    radius in pixels. The returned point follows the requested original ray;
    the patch supplies its median Z. Invalid center depth can be replaced by
    valid neighbors, but no depth is fabricated if the complete patch is bad.

    ``camera_position`` and ``camera_quat`` are the USD camera's world pose;
    ``robot_base_position`` and ``robot_base_quat`` are the base's world pose.
    Both poses are required for ``output_frame='base'``. For
    ``'camera_optical'`` (or alias ``'camera'``), no world poses are needed.

    Returns JSON-compatible evidence including the selected point/frame,
    optical point, actual sampled median depth, sample count, and patch bounds.
    No frame ID is accepted here: the calling worker validates synchronization.
    """
    if output_frame == "camera":
        output_frame = "camera_optical"
    if output_frame not in ("base", "camera_optical"):
        raise ValueError("output_frame must be 'base' or 'camera_optical'")
    if type(radius) is not int or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    if isinstance(min_depth, (bool, np.bool_)) or not np.isscalar(min_depth):
        raise ValueError("min_depth must be a finite nonnegative number")
    try:
        minimum = float(min_depth)
    except (ValueError, TypeError):
        raise ValueError("min_depth must be a finite nonnegative number") from None
    if not np.isfinite(minimum) or minimum < 0:
        raise ValueError("min_depth must be a finite nonnegative number")
    maximum = None
    if max_depth is not None:
        if isinstance(max_depth, (bool, np.bool_)) or not np.isscalar(max_depth):
            raise ValueError("max_depth must be finite and greater than min_depth")
        try:
            maximum = float(max_depth)
        except (ValueError, TypeError):
            raise ValueError("max_depth must be finite and greater than min_depth") from None
        if not np.isfinite(maximum) or maximum <= minimum:
            raise ValueError("max_depth must be finite and greater than min_depth")

    depth = np.asarray(depth_linear)
    if depth.ndim != 2 or not all(depth.shape) or depth.dtype.kind not in "fiu":
        raise ValueError("depth_linear must be a nonempty 2D numeric axial-depth image")
    intrinsic = _numeric_array(K, (3, 3), "K")
    if (intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or not np.allclose(intrinsic[2], [0, 0, 1], rtol=0, atol=1e-12)):
        raise ValueError("K must be a pinhole intrinsic matrix with positive focal lengths and last row [0,0,1]")
    pixel = _numeric_array(pixel_xy, (2,), "pixel_xy")
    u, v = float(pixel[0]), float(pixel[1])
    height, width = depth.shape
    if not (0 <= u <= width - 1 and 0 <= v <= height - 1):
        raise ValueError(f"pixel_xy is outside image bounds: width={width}, height={height}")
    center_x, center_y = int(np.floor(u + 0.5)), int(np.floor(v + 0.5))
    left, right = max(0, center_x - radius), min(width, center_x + radius + 1)
    top, bottom = max(0, center_y - radius), min(height, center_y + radius + 1)
    patch = depth[top:bottom, left:right].astype(np.float64, copy=False)
    valid = np.isfinite(patch) & (patch > minimum) & (patch > 0)
    if maximum is not None:
        valid &= patch <= maximum
    count = int(np.count_nonzero(valid))
    if count == 0:
        raise ValueError("No finite positive depth remains in the requested pixel neighborhood")
    sampled_depth = float(np.median(patch[valid]))
    try:
        optical_ray = np.linalg.solve(intrinsic, np.array([u, v, 1.0]))
    except np.linalg.LinAlgError:
        raise ValueError("K is singular") from None
    if not np.isfinite(optical_ray).all() or abs(optical_ray[2]) <= 1e-12:
        raise ValueError("K produced an invalid optical ray")
    optical_point = optical_ray / optical_ray[2] * sampled_depth

    if output_frame == "camera_optical":
        point = optical_point
    else:
        camera_p = _numeric_array(camera_position, (3,), "camera_position")
        base_p = _numeric_array(robot_base_position, (3,), "robot_base_position")
        camera_rotation = _rotation_xyzw(camera_quat, "camera_quat")
        base_rotation = _rotation_xyzw(robot_base_quat, "robot_base_quat")
        usd_camera_point = optical_point * np.array([1.0, -1.0, -1.0])
        world_point = camera_rotation @ usd_camera_point + camera_p
        point = base_rotation.T @ (world_point - base_p)
    if not np.isfinite(point).all():
        raise ValueError("Projection produced a nonfinite point")
    return {
        "point": point.tolist(),
        "frame": output_frame,
        "unit": "meters",
        "camera_optical_point": optical_point.tolist(),
        "sampled_depth_m": sampled_depth,
        "depth_convention": "distance_to_image_plane_axial_z",
        "depth_statistic": "median_of_finite_positive_depth",
        "pixel_xy": [u, v],
        "sample_center_xy": [center_x, center_y],
        "sample_radius_pixels": radius,
        "sample_window_xyxy_exclusive": [left, top, right, bottom],
        "valid_depth_samples": count,
        "sample_window_pixels": int(patch.size),
        "min_depth_exclusive_m": minimum,
        "max_depth_inclusive_m": maximum,
    }
