"""Privileged, rendered instance-mask visibility scorer for single-env OmniGibson.

Only use outside planner / policy input construction. Pixel visibility is an
evaluation proxy; it does not establish that a VLM can recognize the object.
"""
import numpy as np
def array(value):
    if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
    return np.asarray(value)

ROLES = ("head", "left_wrist", "right_wrist")


def measure_target_pixels(segmentation, labels, target_name, min_pixels=16):
    if not isinstance(target_name, str) or not target_name:
        raise ValueError("Exact simulator object name required")
    if not isinstance(min_pixels, int) or min_pixels < 1:
        raise ValueError("Positive integer pixel threshold required")
    mask = array(segmentation)
    if mask.ndim != 2 or not mask.size or mask.dtype.kind not in "iu":
        raise ValueError("Expected nonempty integer HxW instance mask")
    if not isinstance(labels, dict) or not all(isinstance(v, str) for v in labels.values()):
        raise ValueError("Expected OmniGibson remapped ID-to-object-name metadata")
    mapping = {int(k): v for k, v in labels.items()}
    unknown = set(int(x) for x in np.unique(mask)) - set(mapping) - {0, 1}
    if unknown:
        raise ValueError(f"Instance mask has unmapped IDs: {sorted(unknown)}")
    ids = [k for k, v in mapping.items() if v == target_name]
    pixels = int(np.isin(mask, ids).sum()) if ids else 0
    return {"visible": pixels >= min_pixels, "pixel_count": pixels,
            "image_pixels": int(mask.size), "pixel_fraction": pixels / mask.size,
            "threshold_pixels": min_pixels}


class RenderedTargetScorer:
    def __init__(self, robot, target, camera_names, min_pixels=16):
        if set(camera_names) != set(ROLES):
            raise ValueError("Exactly head / left_wrist / right_wrist cameras required")
        if not isinstance(min_pixels, int) or min_pixels < 1:
            raise ValueError("Positive integer pixel threshold required")
        self.robot, self.target, self.min_pixels = robot, target, min_pixels
        self.sensors = {}
        for role in ROLES:
            # Official get_robot_camera_names returns robot::sensor names.
            robot_name, sensor_name = camera_names[role].split("::", 1)
            if robot_name != robot.name:
                raise ValueError("Scoring camera must belong to this robot")
            self.sensors[role] = robot.sensors[sensor_name]

    def enable(self):
        """Call once during setup, then render and refresh onboard RGB before trial."""
        for sensor in self.sensors.values():
            sensor.add_modality("seg_semantic")
            sensor.add_modality("seg_instance")

    def __call__(self):
        cameras = {}
        for role, sensor in self.sensors.items():
            obs, info = sensor.get_obs()
            cameras[role] = measure_target_pixels(
                obs["seg_instance"], info["seg_instance"], self.target.name, self.min_pixels)
        base, _ = self.robot.get_position_orientation()
        target, _ = self.target.get_position_orientation()
        return {
            "base_position": array(base).tolist(), "target_position": array(target).tolist(),
            "target_visible": any(x["visible"] for x in cameras.values()),
            "visibility_evidence": {
                "method": "exact object-name instance segmentation in current onboard cameras",
                "cameras": cameras,
                "any_camera_visible_at_pixel_threshold": {
                    str(n): any(x["pixel_count"] >= n for x in cameras.values()) for n in (1,16,64)},
                "note": "Rendered pixel visibility, not verified semantic recognizability."
            }
        }


def evaluate_approach(evidence, max_distance=1.0):
    """Private terminal criterion; never supplied to planner or policy."""
    base = np.asarray(evidence['base_position'], dtype=float)
    target = np.asarray(evidence['target_position'], dtype=float)
    if base.shape != (3,) or target.shape != (3,) or not np.isfinite([base, target]).all():
        raise ValueError('Finite 3D positions required for arrival audit')
    if not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError('Positive finite approach threshold required')
    distance = float(np.linalg.norm((target - base)[:2]))
    return {**evidence, 'horizontal_distance_m': distance,
            'max_horizontal_distance_m': max_distance,
            'find_and_approach_success': evidence.get('target_visible') is True and distance <= max_distance,
            'criterion': 'planner reached claim plus exact target >=16 rendered pixels in any onboard camera plus horizontal base distance <= threshold',
            'physical_reachability_verified': False}
