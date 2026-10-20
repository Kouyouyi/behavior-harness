"""Simulator-independent validation for one observation-grounded GPT decision.

Validation never clips a command or changes control ownership. The simulator
must execute the returned decision once, against the observation and remaining
budget used here. Task success remains the environment evaluator's decision.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


_ACTIONS = frozenset({"observe", "move_base", "look", "look_head", "handoff", "run_vla", "stop"})
_FIELDS = frozenset({
    "action", "observation_id", "reasoning", "target_visible",
    "manipulation_ready", "vx_mps", "vy_mps", "yaw_rps", "steps",
    "trunk_qpos", "head_yaw_rad", "head_pitch_rad", "stop_reason",
})
_ARITHMETIC_EPSILON = 1e-12


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected a number, not {type(value).__name__}")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name}: expected a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name}: expected a finite number")
    return result


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name}: expected an integer")
    if value < minimum:
        raise ValueError(f"{name}: must be >= {minimum}")
    return value


@dataclass(frozen=True)
class ControlLimits:
    """Host-owned limits; GPT cannot enlarge these through its decision."""

    max_base_speed_mps: float = 0.30
    max_yaw_rate_rps: float = 0.60
    max_control_steps: int = 60
    max_vla_steps: int = 300
    max_trunk_delta_rad: float = 0.25
    max_head_delta_rad: float = 0.5
    max_head_yaw_rad: float = 1.57
    min_head_pitch_rad: float = -0.8
    max_head_pitch_rad: float = 0.8

    def __post_init__(self) -> None:
        for name in (
            "max_base_speed_mps", "max_yaw_rate_rps", "max_trunk_delta_rad",
            "max_head_delta_rad", "max_head_yaw_rad",
        ):
            value = _number(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name}: must be nonnegative")
        for name in ("max_control_steps", "max_vla_steps"):
            _integer(getattr(self, name), name, 1)
        min_pitch = _number(self.min_head_pitch_rad, "min_head_pitch_rad")
        max_pitch = _number(self.max_head_pitch_rad, "max_head_pitch_rad")
        if min_pitch > max_pitch:
            raise ValueError("head pitch limits: minimum exceeds maximum")


def _joint_vector(value: Any, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{name}: expected four joint values")
    return [_number(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _joint_bounds(value: Any) -> list[tuple[float, float]]:
    """Runtime provides four [lower, upper] pairs in trunk joint order."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError("trunk_limits: expected four [lower, upper] pairs")
    bounds = []
    for index, pair in enumerate(value):
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise ValueError(f"trunk_limits[{index}]: expected [lower, upper]")
        lower = _number(pair[0], f"trunk_limits[{index}][0]")
        upper = _number(pair[1], f"trunk_limits[{index}][1]")
        if lower > upper:
            raise ValueError(f"trunk_limits[{index}]: lower exceeds upper")
        bounds.append((lower, upper))
    return bounds


def validate_decision(
    decision: Mapping[str, Any],
    current_observation_id: str,
    limits: ControlLimits,
    trunk_qpos: Sequence[float] | None,
    trunk_limits: Sequence[Sequence[float]] | None,
    phase: str,
    remaining_steps: int,
    *,
    head_enabled: bool = False,
    head_state: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Return a new normalized decision, or raise ValueError before execution.

    ``trunk_qpos`` and ``trunk_limits`` are the runtime's current four joints
    and four [lower, upper] pairs, respectively. They are required for ``look``.
    Base velocities use the robot-local frame; the translation speed cap applies
    to the vector norm, including diagonal motion. Ordinary VLA chunks require
    phase ``vla`` and do not perform another handoff or policy reset.
    ``look_head`` controls a separately enabled virtual camera gimbal using
    absolute yaw/pitch targets. ``head_state`` must be its current [yaw, pitch].
    """
    if not isinstance(limits, ControlLimits):
        raise ValueError("limits: expected ControlLimits")
    if not isinstance(decision, Mapping):
        raise ValueError("decision: expected an object")
    unknown = set(decision) - _FIELDS
    if unknown:
        raise ValueError(f"decision: unknown fields {sorted(map(str, unknown))}")
    missing = _FIELDS - set(decision)
    if missing:
        raise ValueError(f"decision: missing fields {sorted(missing)}")

    result = dict(decision)
    for name in ("action", "observation_id", "reasoning", "stop_reason"):
        if not isinstance(result[name], str):
            raise ValueError(f"{name}: expected a string")
    action = result["action"]
    if action not in _ACTIONS:
        raise ValueError(f"action: unsupported action {action!r}")
    if not isinstance(current_observation_id, str) or not current_observation_id:
        raise ValueError("current_observation_id: expected a nonempty string")
    if result["observation_id"] != current_observation_id:
        raise ValueError("observation_id: stale or unknown observation; observe again")
    for name in ("target_visible", "manipulation_ready"):
        if type(result[name]) is not bool:
            raise ValueError(f"{name}: expected a boolean")
    for name in ("vx_mps", "vy_mps", "yaw_rps", "head_yaw_rad", "head_pitch_rad"):
        result[name] = _number(result[name], name)
    steps = result["steps"] = _integer(result["steps"], "steps", 0)
    remaining_steps = _integer(remaining_steps, "remaining_steps", 0)
    if not isinstance(result["trunk_qpos"], list):
        raise ValueError("trunk_qpos: expected a list")

    if action == "look":
        result["trunk_qpos"] = _joint_vector(result["trunk_qpos"], "trunk_qpos")
    elif result["trunk_qpos"]:
        raise ValueError(f"trunk_qpos: must be [] for {action}")
    else:
        result["trunk_qpos"] = []
    if action != "move_base" and any(result[name] != 0.0 for name in ("vx_mps", "vy_mps", "yaw_rps")):
        raise ValueError(f"base velocities: must all be zero for {action}")
    if action != "look_head" and any(result[name] != 0.0 for name in ("head_yaw_rad", "head_pitch_rad")):
        raise ValueError(f"head targets: must both be zero for {action}")

    if action in {"move_base", "look", "look_head", "run_vla"}:
        maximum = limits.max_vla_steps if action == "run_vla" else limits.max_control_steps
        if not 1 <= steps <= maximum:
            raise ValueError(f"steps: {action} requires 1..{maximum}")
        if steps > remaining_steps:
            raise ValueError(f"steps: exceeds remaining episode budget ({remaining_steps})")
    elif steps != 0:
        raise ValueError(f"steps: must be zero for {action}")

    if action == "move_base":
        if math.hypot(result["vx_mps"], result["vy_mps"]) > limits.max_base_speed_mps:
            raise ValueError(f"base speed: exceeds {limits.max_base_speed_mps} m/s")
        if abs(result["yaw_rps"]) > limits.max_yaw_rate_rps:
            raise ValueError(f"yaw rate: exceeds {limits.max_yaw_rate_rps} rad/s")
    elif action == "look":
        current = _joint_vector(trunk_qpos, "current trunk_qpos")
        bounds = _joint_bounds(trunk_limits)
        for index, (target, present, (lower, upper)) in enumerate(zip(result["trunk_qpos"], current, bounds)):
            if not lower <= target <= upper:
                raise ValueError(f"trunk_qpos[{index}]: outside runtime limits [{lower}, {upper}]")
            if abs(target - present) > limits.max_trunk_delta_rad + _ARITHMETIC_EPSILON:
                raise ValueError(f"trunk_qpos[{index}]: delta exceeds {limits.max_trunk_delta_rad} rad")
    elif action == "look_head":
        if head_enabled is not True:
            raise ValueError("look_head: virtual head is not enabled")
        if not isinstance(head_state, Sequence) or isinstance(head_state, (str, bytes)) or len(head_state) != 2:
            raise ValueError("head_state: expected current [yaw, pitch]")
        current_head = [_number(item, f"head_state[{index}]") for index, item in enumerate(head_state)]
        head_bounds = (
            ("head_yaw_rad", -limits.max_head_yaw_rad, limits.max_head_yaw_rad),
            ("head_pitch_rad", limits.min_head_pitch_rad, limits.max_head_pitch_rad),
        )
        for present, (name, lower, upper) in zip(current_head, head_bounds):
            target = result[name]
            if not lower <= target <= upper:
                raise ValueError(f"{name}: outside head limits [{lower}, {upper}]")
            if abs(target - present) > limits.max_head_delta_rad + _ARITHMETIC_EPSILON:
                raise ValueError(f"{name}: delta exceeds {limits.max_head_delta_rad} rad")
    elif action == "handoff":
        if phase != "gpt":
            raise ValueError("handoff: requires phase='gpt'; use run_vla to continue VLA without clearing its chunk")
        if not result["target_visible"] or not result["manipulation_ready"]:
            raise ValueError("handoff: target_visible and manipulation_ready must both be true")
    elif action == "run_vla" and phase != "vla":
        raise ValueError("run_vla: requires phase='vla'; hand off first")

    if action == "stop":
        if not result["stop_reason"].strip():
            raise ValueError("stop_reason: a stopped run needs a nonempty reason")
    elif result["stop_reason"]:
        raise ValueError(f"stop_reason: must be empty for {action}")
    return result
