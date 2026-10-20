"""Stable observation and action contracts for the G1 SONIC baselines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class G1SonicActionSpec:
    """The exported action layout consumed by the HUMANO SONIC adapter."""

    motion_token_dim: int = 64
    left_hand_dim: int = 7
    right_hand_dim: int = 7

    @property
    def total_dim(self) -> int:
        return self.motion_token_dim + self.left_hand_dim + self.right_hand_dim

    @property
    def source_keys(self) -> tuple[str, str, str]:
        return (
            "action.motion_token",
            "teleop.left_hand_joints",
            "teleop.right_hand_joints",
        )


@dataclass(frozen=True)
class G1SonicObservationSpec:
    """GR00T-aligned state layout consumed by every Track B policy."""

    joint_state_dim: int = 43
    projected_gravity_dim: int = 3
    state_key: str = "observation.state"
    projected_gravity_key: str = "observation.projected_gravity"

    @property
    def total_dim(self) -> int:
        return self.joint_state_dim + self.projected_gravity_dim


@dataclass(frozen=True)
class G1SonicDatasetManifest:
    """Validated metadata needed by the conversion and training entrypoints."""

    codebase_version: str
    robot_type: str
    fps: int
    state_dim: int
    action_dim: int
    image_key: str
    state_key: str


_ACTION_SPEC = G1SonicActionSpec()
_OBSERVATION_SPEC = G1SonicObservationSpec()
_SUPPORTED_FPS = frozenset({30, 50})
_EXPECTED_FEATURES: dict[str, tuple[str, tuple[int, ...]]] = {
    "observation.images.ego_view": ("video", (360, 640, 3)),
    "observation.state": ("float64", (43,)),
    "observation.projected_gravity": ("float64", (3,)),
    "action.motion_token": ("float64", (64,)),
    "teleop.left_hand_joints": ("float32", (7,)),
    "teleop.right_hand_joints": ("float32", (7,)),
}


def _feature_shape(feature: Mapping[str, Any]) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in feature["shape"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("feature metadata must include an integer shape") from exc


def validate_dataset_info(info: Mapping[str, Any]) -> G1SonicDatasetManifest:
    """Validate the v2.1 G1 SONIC fields required by the ACT adapter."""

    for key, expected in (("codebase_version", "v2.1"), ("robot_type", "g1_sonic")):
        actual = info.get(key)
        if actual != expected:
            raise ValueError(f"{key} must be {expected!r}, got {actual!r}")
    fps = info.get("fps")
    if fps not in _SUPPORTED_FPS:
        raise ValueError(f"fps must be one of {sorted(_SUPPORTED_FPS)}, got {fps!r}")

    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("features metadata must be a mapping")
    for key, (expected_dtype, expected_shape) in _EXPECTED_FEATURES.items():
        feature = features.get(key)
        if not isinstance(feature, Mapping):
            raise ValueError(f"missing required feature {key!r}")
        actual_dtype = feature.get("dtype")
        actual_shape = _feature_shape(feature)
        if actual_dtype != expected_dtype or actual_shape != expected_shape:
            raise ValueError(
                f"feature {key!r} must have dtype/shape {expected_dtype!r}/{expected_shape!r}, "
                f"got {actual_dtype!r}/{actual_shape!r}"
            )

    return G1SonicDatasetManifest(
        codebase_version="v2.1",
        robot_type="g1_sonic",
        fps=int(fps),
        state_dim=_OBSERVATION_SPEC.total_dim,
        action_dim=_ACTION_SPEC.total_dim,
        image_key="observation.images.ego_view",
        state_key="observation.state",
    )


def _as_vector(values: Any, key: str, expected_dim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (expected_dim,):
        raise ValueError(f"{key} must have shape ({expected_dim},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{key} must contain only finite values")
    return array


def compose_action(row: Mapping[str, Any], spec: G1SonicActionSpec = _ACTION_SPEC) -> np.ndarray:
    """Compose the explicit 78D motion-token + hand action vector."""

    motion = _as_vector(row.get(spec.source_keys[0]), spec.source_keys[0], spec.motion_token_dim)
    left_hand = _as_vector(row.get(spec.source_keys[1]), spec.source_keys[1], spec.left_hand_dim)
    right_hand = _as_vector(row.get(spec.source_keys[2]), spec.source_keys[2], spec.right_hand_dim)
    return np.concatenate((motion, left_hand, right_hand)).astype(np.float32, copy=False)


def compose_state(
    row: Mapping[str, Any],
    spec: G1SonicObservationSpec = _OBSERVATION_SPEC,
) -> np.ndarray:
    """Append projected gravity to the 43D joint state in GR00T modality order."""

    joint_state = _as_vector(row.get(spec.state_key), spec.state_key, spec.joint_state_dim)
    projected_gravity = _as_vector(
        row.get(spec.projected_gravity_key),
        spec.projected_gravity_key,
        spec.projected_gravity_dim,
    )
    return np.concatenate((joint_state, projected_gravity)).astype(np.float32, copy=False)


def validate_episode_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate monotonic global indices and frame indices reset at episode boundaries."""

    previous_index: int | None = None
    previous_episode: int | None = None
    expected_frame = 0
    for row_number, row in enumerate(rows):
        try:
            episode = int(row["episode_index"])
            frame = int(row["frame_index"])
            index = int(row["index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"row {row_number} is missing integer episode/index fields") from exc

        if previous_index is not None and index != previous_index + 1:
            raise ValueError(f"index must increase by one at row {row_number}")
        if previous_episode == episode:
            if frame != expected_frame:
                raise ValueError(f"frame_index must be contiguous at row {row_number}")
        elif previous_episode is not None:
            if episode != previous_episode + 1 or frame != 0:
                raise ValueError(f"episode_index/frame_index boundary is invalid at row {row_number}")
        elif frame != 0:
            raise ValueError("the first episode must start at frame_index 0")

        previous_index = index
        previous_episode = episode
        expected_frame = frame + 1
